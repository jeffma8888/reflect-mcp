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
  - party-role grounding: a grounded vendor_name must not appear *only* on
                 counterparty lines ("Bill to", "Ship to", ...), which name the buyer.
  - date-role grounding: a grounded invoice_date must not appear *only* as a
                 due/payment/ship/delivery date ("Due date: ..."), which an invoice
                 states but which is not the invoice's own date.
  - identifier-role grounding: a grounded invoice_number must not appear *only*
                 on PO/order/customer/account lines when another line carries an
                 invoice-number label ("Invoice No: ..."), so a copied PO number fails.
  - currency-code shape: currency must be an ISO 4217-shaped code (3 uppercase
                 letters), so "$" or "US Dollars" can't pass on presence alone.
  - omitted total: an explicit null total on a source that labels a total is
                 penalized, so a dropped total can't pass as an abstention.

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

# ISO 4217 shape only (no code list): `[A-Z]` is ASCII, used with `fullmatch`.
_CURRENCY_CODE_RE = re.compile(r"[A-Z]{3}")


def _lookahead(alternatives: list[str] | tuple[str, ...]) -> str:
    """`(?=[...])` on the alternatives' first letters. `re` can't prefix-optimize a flat
    case-insensitive alternation, so without this every char position tries every branch."""
    return f"(?=[{''.join(sorted({alt[0] for alt in alternatives}))}])"


# English month names, full and 3-letter, mapped to month numbers. Hard-coded
# (not `calendar.month_name`) because that table is locale-dependent.
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
# Unique 3-letter prefix + optional rest ("jan(?:uary)?"): one branch per month, and
# the greedy rest tries the long form first, as the old longest-first alternation did.
_MONTH_BRANCHES = tuple(f"{n[:3]}(?:{n[3:]})?" if n[3:] else n for n in _MONTH_NAMES)
_MONTH_RE = _lookahead(_MONTH_NAMES) + "(?:" + "|".join(_MONTH_BRANCHES) + ")"
# Per-month readers built from the same branches and flag as the date patterns, so
# any token `_MONTH_RE` matches ("APRİL", "ſep") reads back; a `lower()` dict key can't.
_MONTH_READERS = tuple(re.compile(b, re.IGNORECASE) for b in _MONTH_BRANCHES)

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
_DATE_PATTERNS = (_ISO_DATE_RE, _NUMERIC_DATE_RE, _MONTH_FIRST_RE, _DAY_FIRST_RE)

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
    role: re.compile(  # lookahead letters read off the labels, so none can be excluded
        _lookahead(re.sub(r"\(\?<!.*?\)|\(\?:|\)", "", labels).split("|"))
        + rf"(?<![a-z])(?:{labels})(?![a-z])[^\d\n]{{0,25}}?(?:[$€£]\s*)?"
        r"((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?!\d)",
        re.IGNORECASE,
    )
    for role, labels in _ROLE_LABELS.items()
}


# Counterparty labels: a line that names the *buyer/recipient*, not the seller.
# Only explicit "<verb> to" forms; bare "customer"/"client" lines can name the vendor.
_COUNTERPARTY_LINE_RE = re.compile(
    r"^\s*(?:bill(?:ed)?|ship(?:ped)?|sold|deliver(?:ed)?)\s*-?\s*to\b", re.IGNORECASE
)


# Date-role labels: label text before a date token that marks it as a due,
# payment, ship or delivery date -- a date the invoice states, but not its date.
# Whole words only, so "Shipment ref 2026-07-31" stays unlabelled.
_OTHER_DATE_LABEL_RE = re.compile(
    r"(?<![a-z])(?:due|pay(?:ment)?\s+by|ship(?:ped|ping)?|deliver(?:y|ed))(?![a-z])",
    re.IGNORECASE,
)


# Identifier-role labels: a line that starts with another document's identifier
# (PO, order, customer, account). Whole words only ("Orderly" is a name), and never
# when "invoice" follows, because "Customer invoice no:" labels the invoice itself.
_OTHER_ID_LINE_RE = re.compile(
    r"^\s*(?:p\.?o\.?|purchase\s+order|order|customer|client|account|acct)(?![a-z])"
    r"(?![^\w\n]*invoice(?![a-z]))",
    re.IGNORECASE,
)
# An invoice-number label: "invoice", an optional No/Number/#/ID and colon, then a
# token with a digit -- so "Invoice date: ..." and a bare "INVOICE" title don't count.
_INVOICE_NUMBER_LABEL_RE = re.compile(
    r"(?<![a-z])invoice(?:[^\S\n]*(?:no\.?|number|#|id))?[^\S\n]*:?[^\S\n]*#?(?=[\w-]*\d)\w",
    re.IGNORECASE,
)


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
    """Case-insensitive search for *value* in the original (un-normalized) text.

    Searches the original text itself rather than `text.lower()`: lowering can change
    length ("İ" becomes two chars), which would shift every later span off its value.
    """
    match = re.search(re.escape(value), original_text, re.IGNORECASE)
    if match is None:
        return None
    return ProvenanceSpan(start=match.start(), end=match.end())


def _contains_ci(value: str, line: str) -> bool:
    """True when *line* contains *value* under the same rule `_find_text_span` grounds with.

    A `lower()` substring test would miss case variants across "İ" (lowering it adds
    a combining dot), so a value grounded by `re.I` would see no lines and abstain.
    """
    return re.search(re.escape(value), line, re.IGNORECASE) is not None


def _find_number_span(value: float, text: str) -> ProvenanceSpan | None:
    """Find the first numeric token in *text* whose value matches."""
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        try:
            if abs(float(m.group().replace(",", "")) - value) < 0.005:
                return ProvenanceSpan(start=m.start(), end=m.end())
        except ValueError:
            continue
    return None


def _labelled_amounts(text: str, role: str) -> list[float]:
    """Distinct numbers the source labels for *role*, in first-occurrence order."""
    values: list[float] = []
    for m in _LABELLED_AMOUNT_RES[role].finditer(text):
        value = float(m.group(1).replace(",", ""))
        if value not in values:
            values.append(value)
    return values


def _check_party_role(value: str, text: str, report: HeuristicReport) -> None:
    """Penalize a vendor_name the source shows only on counterparty lines.

    Abstains when any line containing the value is not a counterparty line, so a
    vendor named in both the header and a "Bill to" line still passes.
    """
    lines = [ln for ln in text.split("\n") if _contains_ci(value, ln)]
    if lines and all(_COUNTERPARTY_LINE_RE.match(ln) for ln in lines):
        report.penalize(
            "vendor_name", 0.3,
            f"vendor {value!r} appears only on a bill-to/ship-to/sold-to line (the buyer, not the seller)",
        )


def _check_identifier_role(value: str, text: str, report: HeuristicReport) -> None:
    """Penalize an invoice_number the source shows only on PO/order/customer/account lines.

    A copied PO number is grounded by substring, so presence can't catch it. The
    check abstains unless some line carries an invoice-number label (an invoice
    identified only by "Order 5567" has no better candidate), and when any line
    containing the value is not an other-id line.
    """
    lines = [ln for ln in text.split("\n") if _contains_ci(value, ln)]
    if not _INVOICE_NUMBER_LABEL_RE.search(text):
        return
    if lines and all(_OTHER_ID_LINE_RE.match(ln) for ln in lines):
        report.penalize(
            "invoice_number", 0.3,
            f"invoice number {value!r} appears only on a PO/order/customer/account line, not the invoice-number line",
        )


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
    return {(int(y), _month_number(mon), int(d))}


def _month_number(token: str) -> int:
    """Month number of a token `_MONTH_RE` matched, read by the rule that matched it."""
    for number, reader in enumerate(_MONTH_READERS, 1):
        if reader.fullmatch(token):
            return number
    raise AssertionError(f"unreadable month token: {token!r}")


def _find_date_span(iso_value: date, text: str) -> ProvenanceSpan | None:
    """Find the first token in *text* that writes *iso_value* in a supported form."""
    target = (iso_value.year, iso_value.month, iso_value.day)
    matches = [
        m
        for pattern in _DATE_PATTERNS
        for m in pattern.finditer(text)
        if target in _date_readings(m)
    ]
    if not matches:
        return None
    first = min(matches, key=lambda m: m.start())
    return ProvenanceSpan(start=first.start(), end=first.end())


def _date_role_labels(target: tuple[int, int, int], text: str) -> list[bool]:
    """For each token writing *target*, whether its label is a due/ship/delivery role.

    A token's label is the text between the previous date token on its line (or
    the line start) and the token itself, so in "Invoice date: X   Due date: Y"
    the "Due" label belongs to Y alone and X stays neutral.
    """
    labelled: list[bool] = []
    for line in text.split("\n"):
        tokens = sorted(
            (m for pattern in _DATE_PATTERNS for m in pattern.finditer(line)),
            key=lambda m: m.start(),
        )
        label_start = 0
        for token in tokens:
            if target in _date_readings(token):
                label = line[label_start : token.start()]
                labelled.append(_OTHER_DATE_LABEL_RE.search(label) is not None)
            label_start = token.end()
    return labelled


def _check_date_role(iso_value: date, value: str, text: str, report: HeuristicReport) -> None:
    """Penalize an invoice_date that the source writes only as a due/ship/delivery date.

    Models confuse an invoice's dates more than any other field, and a copied due
    date passes plain date grounding because it *is* in the source. The check
    abstains when any token writing the value is not role-labelled: the same day
    can be both the invoice date and the ship date, and no label means no evidence.
    """
    labelled = _date_role_labels((iso_value.year, iso_value.month, iso_value.day), text)
    if labelled and all(labelled):
        report.penalize(
            "invoice_date",
            0.3,
            f"date {value!r} appears only as a due/ship/delivery date, not the invoice date",
        )


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
        _check_date_role(parsed, value, text, report)
    else:
        report.penalize("invoice_date", 0.3, f"date {value!r} not found in source text")


def run_heuristics(text: str, data: InvoiceData) -> HeuristicReport:
    """Compute deterministic per-field verification factors."""
    report = HeuristicReport()
    norm_text = _normalize(text)

    for field in EXTRACTION_FIELDS:
        value = getattr(data, field)
        if value is None:
            # An explicit null total beside a labelled one is an omission, not an
            # abstention; an omitted key (partial-data validate) stays unchecked.
            if field == "total" and field in data.model_fields_set:
                if labelled := _labelled_amounts(text, field):
                    shown = ", ".join(f"{v:.2f}" for v in labelled)
                    report.penalize(field, 0.3, f"source labels total as {shown}, extraction left it null")
            continue

        if field == "currency" and not _CURRENCY_CODE_RE.fullmatch(str(value)):
            # A malformed code is worse than an ungrounded one, so it gets one
            # reason and no grounding: "$" being in the source proves nothing.
            report.provenance[field] = []
            report.penalize(field, 0.2, f"currency {value!r} is not an ISO 4217 code (3 uppercase letters)")
            continue

        if field in _TEXT_GROUNDED_FIELDS:
            span = _find_text_span(str(value), text)
            if span is not None:
                report.provenance[field] = [span]
                if field == "vendor_name":
                    _check_party_role(str(value), text, report)
                elif field == "invoice_number":
                    _check_identifier_role(str(value), text, report)
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
