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

from reflect_mcp.schema import EXTRACTION_FIELDS, InvoiceData, ProvenanceSpan

# Fields we can meaningfully ground by substring search in the source.
_TEXT_GROUNDED_FIELDS = {"vendor_name", "invoice_number", "currency"}
_NUMERIC_FIELDS = {"subtotal", "tax", "total"}

_ARITHMETIC_TOLERANCE = 0.02  # currency rounding slack


@dataclass
class HeuristicReport:
    """Per-field deterministic factor [0,1] + human-readable reasons."""

    factors: dict[str, float] = dc_field(default_factory=dict)
    reasons: dict[str, list[str]] = dc_field(default_factory=dict)
    provenance: dict[str, list[ProvenanceSpan]] = dc_field(default_factory=dict)

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


def _find_text_span(value: str, original_text: str) -> ProvenanceSpan | None:
    """Case-insensitive search for *value* in the original (un-normalized) text."""
    pos = original_text.lower().find(value.lower())
    if pos == -1:
        return None
    return ProvenanceSpan(start=pos, end=pos + len(value))


def _find_number_span(value: float, text: str) -> ProvenanceSpan | None:
    """Find the first numeric token in *text* whose value matches."""
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        try:
            if abs(float(m.group().replace(",", "")) - value) < 0.005:
                return ProvenanceSpan(start=m.start(), end=m.end())
        except ValueError:
            continue
    return None


def _number_in_text(value: float, text: str) -> bool:
    return _find_number_span(value, text) is not None


def run_heuristics(text: str, data: InvoiceData) -> HeuristicReport:
    """Compute deterministic per-field verification factors."""
    report = HeuristicReport()
    norm_text = _normalize(text)

    for field in EXTRACTION_FIELDS:
        value = getattr(data, field)
        if value is None:
            continue

        if field in _TEXT_GROUNDED_FIELDS:
            span = _find_text_span(str(value), text)
            if span is not None:
                report.provenance[field] = [span]
            else:
                report.provenance[field] = []
                if _normalize(str(value)) not in norm_text:
                    report.penalize(field, 0.25, f"value {value!r} not found in source text")

        if field in _NUMERIC_FIELDS:
            span = _find_number_span(float(value), text)
            if span is not None:
                report.provenance[field] = [span]
            else:
                report.provenance[field] = []
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
