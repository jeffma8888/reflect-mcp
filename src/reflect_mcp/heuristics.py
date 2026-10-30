"""Deterministic, non-LLM verification checks.

The Critic gives a *semantic* signal (a second model's judgement); these checks
give a *structural* signal that costs nothing and can't hallucinate. The gate
combines both. Concretely we check:

  - grounding:   is a committed value actually present in the source text?
  - arithmetic:  does subtotal + tax reconcile to total?
  - date format: does invoice_date parse as ISO 8601?
  - date grounding: does that ISO date appear in the source in some supported
                 written form (ISO, numeric slash/dot, or English month name)?
  - amount-role grounding: a grounded subtotal/tax/total must match one of the
                 numbers the source *labels* for that role (e.g. "Total: 75.00"),
                 so a swapped amount or a stray number can't pass on presence alone.

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

# English month names, full and 3-letter, mapped to month numbers. Hard-coded
# (not `calendar.month_name`) because that table is locale-dependent.
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, 1)}
_MONTHS.update({name[:3]: i for i, name in enumerate(_MONTH_NAMES, 1)})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))  # longest first

# Supported written date forms. Digit-boundary lookarounds stop a date token
# from matching inside a longer digit run (e.g. "12026-03-145").
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})([/.])(\d{1,2})\2(\d{4})(?!\d)")
_MONTH_FIRST_RE = re.compile(
    rf"(?<![a-z])({_MONTH_RE})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", re.IGNORECASE
)
_DAY_FIRST_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s+({_MONTH_RE})\s+(\d{{4}})(?!\d)", re.IGNORECASE
)

# Amount-role labels. The letter lookarounds stop a label matching inside a word
# ("Internet" is not "net"); the `sub` lookbehinds keep "Sub-total" from also
# counting as a total label.
_ROLE_LABELS = {
    "subtotal": r"sub-?total|sub total|net",
    "tax": r"tax|vat|gst",
    "total": r"(?<!sub)(?<!sub-)(?<!sub )(?:grand total|amount due|total)",
}
# A labelled number sits on the label's line, at most 25 non-digit chars after
# it, optionally currency-prefixed, written with exactly two decimals. The strict
# money form keeps IDs and quantities ("Order 12", "Total: 15") from labelling.
_LABELLED_AMOUNT_RES = {
    role: re.compile(
        rf"(?<![a-z])(?:{labels})(?![a-z])[^\d\n]{{0,25}}?(?:[$€£]\s*)?"
        r"((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?!\d)",
        re.IGNORECASE,
    )
    for role, labels in _ROLE_LABELS.items()
}


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


def _labelled_amounts(text: str, role: str) -> list[float]:
    """Distinct numbers the source labels for *role*, in first-occurrence order."""
    values: list[float] = []
    for m in _LABELLED_AMOUNT_RES[role].finditer(text):
        value = float(m.group(1).replace(",", ""))
        if value not in values:
            values.append(value)
    return values


def _check_amount_role(field: str, value: float, text: str, report: HeuristicReport) -> None:
    """Penalize a grounded amount that differs from every number labelled for its role.

    Presence alone can't tell a subtotal from a tax (and arithmetic is
    commutative), so a swap or a stray "STORE #12" total would pass. The check
    abstains when the source labels nothing for the role: no label, no evidence.
    """
    labelled = _labelled_amounts(text, field)
    if not labelled or any(abs(v - value) < 0.005 for v in labelled):
        return
    shown = ", ".join(f"{v:.2f}" for v in labelled)
    report.penalize(field, 0.3, f"source labels {field} as {shown}, extraction says {value}")


def _date_readings(match: re.Match[str]) -> set[tuple[int, int, int]]:
    """All (year, month, day) readings of one matched date token.

    Numeric slash/dot dates are ambiguous between month-first and day-first, so
    both readings are returned: ambiguity must never *cause* a penalty, because
    this heuristic exists to catch invented dates, not to adjudicate locale.
    """
    if match.re is _ISO_DATE_RE:
        y, m, d = match.group(1, 2, 3)
        return {(int(y), int(m), int(d))}
    if match.re is _NUMERIC_DATE_RE:
        a, b, y = int(match.group(1)), int(match.group(3)), int(match.group(4))
        return {(y, a, b), (y, b, a)}
    if match.re is _MONTH_FIRST_RE:
        mon, d, y = match.group(1, 2, 3)
    elif match.re is _DAY_FIRST_RE:
        d, mon, y = match.group(1, 2, 3)
    else:  # a new pattern was added without a reader here
        raise AssertionError(f"unhandled date pattern: {match.re.pattern}")
    return {(int(y), _MONTHS[mon.lower()], int(d))}


def _find_date_span(iso_value: date, text: str) -> ProvenanceSpan | None:
    """Find the first token in *text* that writes *iso_value* in a supported form."""
    target = (iso_value.year, iso_value.month, iso_value.day)
    matches = [
        m
        for pattern in (_ISO_DATE_RE, _NUMERIC_DATE_RE, _MONTH_FIRST_RE, _DAY_FIRST_RE)
        for m in pattern.finditer(text)
        if target in _date_readings(m)
    ]
    if not matches:
        return None
    first = min(matches, key=lambda m: m.start())
    return ProvenanceSpan(start=first.start(), end=first.end())


def _check_date(value: str, text: str, report: HeuristicReport) -> None:
    """Validate ISO format, then ground the date in the source.

    A non-ISO value is already penalized harder (0.2) than an ungrounded one
    (0.3), so grounding is only attempted on a parseable date — stacking a
    second reason on an unparseable value would add noise, not signal.
    """
    report.provenance["invoice_date"] = []
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        report.penalize("invoice_date", 0.2, f"date {value!r} is not ISO 8601 (YYYY-MM-DD)")
        return
    span = _find_date_span(parsed, text)
    if span is not None:
        report.provenance["invoice_date"] = [span]
    else:
        report.penalize("invoice_date", 0.3, f"date {value!r} not found in source text")


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
                # Only a found amount gets the role check, so an ungrounded one
                # keeps its single, clearer "not found" reason.
                _check_amount_role(field, float(value), text, report)
            else:
                report.provenance[field] = []
                report.penalize(field, 0.35, f"amount {value} not found in source text")

        if field == "invoice_date":
            _check_date(str(value), text, report)

    # Arithmetic reconciliation: subtotal + tax should equal total.
    if data.subtotal is not None and data.tax is not None and data.total is not None:
        if abs((data.subtotal + data.tax) - data.total) > _ARITHMETIC_TOLERANCE:
            msg = f"subtotal({data.subtotal}) + tax({data.tax}) != total({data.total})"
            # The total is the most likely culprit; penalize it hard and leave the
            # components to their own grounding/critic signals rather than
            # over-flagging fields that may well be correct.
            report.penalize("total", 0.3, msg)

    return report
