"""Deterministic, non-LLM verification checks.

The Critic gives a *semantic* signal (a second model's judgement); these checks
give a *structural* signal that costs nothing and can't hallucinate. The gate
combines both. Concretely we check:

  - grounding:   is a committed value actually present in the source text?
  - arithmetic:  does subtotal + tax reconcile to total?
  - date format: does invoice_date parse as ISO 8601?

Each check contributes a per-field factor in [0,1]; the gate multiplies the
Critic confidence by this factor, so a structurally impossible value can never
be waved through on the Critic's word alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date

from reflect_mcp.schema import EXTRACTION_FIELDS, InvoiceData

# Fields we can meaningfully ground by substring search in the source.
_TEXT_GROUNDED_FIELDS = {"vendor_name", "invoice_number", "currency"}
_NUMERIC_FIELDS = {"subtotal", "tax", "total"}

_ARITHMETIC_TOLERANCE = 0.02  # currency rounding slack


@dataclass
class HeuristicReport:
    """Per-field deterministic factor [0,1] + human-readable reasons."""

    factors: dict[str, float] = dc_field(default_factory=dict)
    reasons: dict[str, list[str]] = dc_field(default_factory=dict)

    def penalize(self, field: str, factor: float, reason: str) -> None:
        # keep the harshest factor seen for a field
        self.factors[field] = min(self.factors.get(field, 1.0), factor)
        self.reasons.setdefault(field, []).append(reason)

    def factor_for(self, field: str) -> float:
        return self.factors.get(field, 1.0)

    def reason_for(self, field: str) -> str:
        return "; ".join(self.reasons.get(field, []))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _number_in_text(value: float, text: str) -> bool:
    # Match 1234.5, 1,234.50, 1234, etc. Compare on the numeric value.
    candidates = re.findall(r"\d[\d,]*\.?\d*", text)
    for c in candidates:
        try:
            if abs(float(c.replace(",", "")) - value) < 0.005:
                return True
        except ValueError:
            continue
    return False


def run_heuristics(text: str, data: InvoiceData) -> HeuristicReport:
    """Compute deterministic per-field verification factors."""
    report = HeuristicReport()
    norm_text = _normalize(text)

    for field in EXTRACTION_FIELDS:
        value = getattr(data, field)
        if value is None:
            # A null is an honest abstention -- no structural penalty here.
            continue

        if field in _TEXT_GROUNDED_FIELDS:
            if _normalize(str(value)) not in norm_text:
                report.penalize(field, 0.25, f"value {value!r} not found in source text")

        if field in _NUMERIC_FIELDS:
            if not _number_in_text(float(value), text):
                report.penalize(field, 0.35, f"amount {value} not found in source text")

        if field == "invoice_date":
            try:
                date.fromisoformat(str(value))
            except ValueError:
                report.penalize(field, 0.2, f"date {value!r} is not ISO 8601 (YYYY-MM-DD)")

    # Arithmetic reconciliation: subtotal + tax should equal total.
    if data.subtotal is not None and data.tax is not None and data.total is not None:
        if abs((data.subtotal + data.tax) - data.total) > _ARITHMETIC_TOLERANCE:
            msg = f"subtotal({data.subtotal}) + tax({data.tax}) != total({data.total})"
            # The total is the most likely culprit; penalize it hard and leave the
            # components to their own grounding/critic signals rather than
            # over-flagging fields that may well be correct.
            report.penalize("total", 0.3, msg)

    return report
