"""Regression tests for iter-84 review finding: role checks must select lines with the
same case-insensitive rule that grounds the span.

Once `_find_text_span` grounds with `re.IGNORECASE`, a case variant of a value on an
"İ" line gets a span. If the buyer/PO-line checks still matched lines by `lower()`
substring they would see zero lines and abstain, passing a wrong value as clean.
"""

from __future__ import annotations

import pytest

from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.schema import InvoiceData


def _factors(text: str, **data: str) -> dict[str, float]:
    return run_heuristics(text, InvoiceData(**data)).factors


def test_buyer_copied_as_vendor_on_dotted_capital_i_line_is_penalized() -> None:
    text = "Bill to: İSTANBUL CO\nInvoice No: A-1"
    report = run_heuristics(text, InvoiceData(vendor_name="Istanbul Co"))
    assert report.factors.get("vendor_name") == pytest.approx(0.3)
    assert "bill-to" in report.reason_for("vendor_name")


def test_po_number_copied_as_invoice_number_on_dotted_capital_i_line_is_penalized() -> None:
    text = "Invoice No: A-1\nPO: İX-9"
    report = run_heuristics(text, InvoiceData(invoice_number="ix-9"))
    assert report.factors.get("invoice_number") == pytest.approx(0.3)
    assert "PO/order" in report.reason_for("invoice_number")


def test_vendor_on_header_and_bill_to_line_still_passes() -> None:
    assert "vendor_name" not in _factors("İSTANBUL CO\nBill to: Acme", vendor_name="Istanbul Co")


@pytest.mark.parametrize(
    ("text", "data", "field"),
    [
        ("Bill to: ACME CORP\nInvoice No: A-1", {"vendor_name": "acme corp"}, "vendor_name"),
        ("Invoice No: A-1\nPO: PX-9", {"invoice_number": "px-9"}, "invoice_number"),
    ],
)
def test_ascii_case_variants_keep_their_role_penalty(text: str, data: dict[str, str], field: str) -> None:
    assert _factors(text, **data).get(field) == pytest.approx(0.3)
