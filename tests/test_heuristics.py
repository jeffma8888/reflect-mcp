"""Deterministic (non-LLM) verification checks."""

from __future__ import annotations

from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.schema import InvoiceData

SOURCE = "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: 100.00\nTax: 8.50\nTotal: 108.50\nUSD"


def _good() -> InvoiceData:
    return InvoiceData(
        vendor_name="ACME CORP", invoice_number="INV-1001", invoice_date="2026-03-14",
        subtotal=100.0, tax=8.5, total=108.5, currency="USD",
    )


def test_clean_extraction_gets_no_penalty():
    report = run_heuristics(SOURCE, _good())
    assert all(report.factor_for(f) == 1.0 for f in ("vendor_name", "invoice_number", "total"))


def test_ungrounded_vendor_is_penalized():
    data = _good()
    data.vendor_name = "Hallucinated Inc"  # not in source
    report = run_heuristics(SOURCE, data)
    assert report.factor_for("vendor_name") < 0.5
    assert "not found" in report.reason_for("vendor_name")


def test_ungrounded_amount_is_penalized():
    data = _good()
    data.total = 999.99  # not in source
    report = run_heuristics(SOURCE, data)
    assert report.factor_for("total") < 0.5


def test_arithmetic_mismatch_flags_total_hard():
    data = _good()
    data.total = 200.00  # 100 + 8.50 != 200 (and 200 not in source)
    report = run_heuristics(SOURCE, data)
    assert report.factor_for("total") <= 0.35


def test_bad_date_format_penalized():
    data = _good()
    data.invoice_date = "03/14/2026"  # not ISO 8601
    report = run_heuristics(SOURCE, data)
    assert report.factor_for("invoice_date") < 0.5


def test_null_fields_are_not_penalized():
    data = InvoiceData(vendor_name="ACME CORP")  # rest null
    report = run_heuristics(SOURCE, data)
    for f in ("subtotal", "tax", "total", "currency", "invoice_date"):
        assert report.factor_for(f) == 1.0


# --- Provenance span tests (engineer-owned) ---


def test_text_grounded_field_gets_correct_span():
    report = run_heuristics(SOURCE, _good())
    spans = report.provenance.get("vendor_name", [])
    assert len(spans) == 1
    assert SOURCE[spans[0].start : spans[0].end] == "ACME CORP"


def test_numeric_field_gets_correct_span():
    report = run_heuristics(SOURCE, _good())
    spans = report.provenance.get("total", [])
    assert len(spans) == 1
    assert SOURCE[spans[0].start : spans[0].end] == "108.50"


def test_ungrounded_field_gets_empty_provenance():
    data = _good()
    data.vendor_name = "Hallucinated Inc"
    report = run_heuristics(SOURCE, data)
    assert report.provenance.get("vendor_name") == []


def test_null_field_has_no_provenance():
    data = InvoiceData(vendor_name="ACME CORP")
    report = run_heuristics(SOURCE, data)
    assert "subtotal" not in report.provenance
