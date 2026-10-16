"""HITL gate decision logic (pure, no LLM)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflect_mcp.gate import GateConfig, decide
from reflect_mcp.heuristics import HeuristicReport, run_heuristics
from reflect_mcp.schema import CritiqueResult, InvoiceData

SOURCE = "ACME CORP\nInvoice #INV-1001\n2026-03-14\nSubtotal 100.00\nTax 8.50\nTotal 108.50\nUSD"


def _data():
    return InvoiceData(
        vendor_name="ACME CORP", invoice_number="INV-1001", invoice_date="2026-03-14",
        subtotal=100.0, tax=8.5, total=108.5, currency="USD",
    )


def _critique(overall=0.95, per_field=0.95, overrides=None):
    overrides = overrides or {}
    verdicts = []
    for f in ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency"):
        conf = overrides.get(f, per_field)
        verdicts.append({"field": f, "confidence": conf, "grounded": conf >= 0.7, "rationale": f"{f} check"})
    return CritiqueResult.model_validate({"field_verdicts": verdicts, "overall_confidence": overall})


def test_high_confidence_passes_clean():
    result = decide(_data(), _critique(), run_heuristics(SOURCE, _data()), "t1")
    assert result.status == "ok"
    assert result.flagged_fields == []
    assert all(c >= 0.7 for c in result.field_confidence.values())


def test_low_field_confidence_flags_that_field():
    crit = _critique(overall=0.9, overrides={"total": 0.2})
    result = decide(_data(), crit, run_heuristics(SOURCE, _data()), "t2")
    assert result.status == "needs_review"
    assert [f.field for f in result.flagged_fields] == ["total"]


def test_low_overall_confidence_escalates_even_if_fields_ok():
    crit = _critique(overall=0.5)  # below overall_threshold
    result = decide(_data(), crit, run_heuristics(SOURCE, _data()), "t3")
    assert result.status == "needs_review"
    assert result.flagged_fields == []  # no single field failed, but overall too low
    assert "0.75" not in result.notes  # human-readable percent, not raw float
    assert "review" in result.notes.lower()


def test_null_field_is_never_flagged():
    data = _data()
    data.tax = None  # honest abstention
    crit = _critique(overall=0.9, overrides={"tax": 0.0})  # even 0 confidence
    result = decide(data, crit, run_heuristics(SOURCE, data), "t4")
    assert all(f.field != "tax" for f in result.flagged_fields)


def test_heuristic_factor_caps_confidence():
    # Critic is fooled (high conf) but the value isn't in the source: combined must drop.
    data = _data()
    data.vendor_name = "Hallucinated Inc"
    crit = _critique(overall=0.95, overrides={"vendor_name": 0.99})
    result = decide(data, crit, run_heuristics(SOURCE, data), "t5")
    assert result.status == "needs_review"
    assert result.field_confidence["vendor_name"] < 0.7


def test_gate_config_validates_thresholds():
    with pytest.raises(ValidationError):
        GateConfig(field_threshold=1.5)


def test_custom_thresholds_are_respected():
    # A permissive gate accepts what the default would flag.
    crit = _critique(overall=0.6, overrides={"total": 0.55})
    permissive = GateConfig(field_threshold=0.5, overall_threshold=0.5)
    result = decide(_data(), crit, run_heuristics(SOURCE, _data()), "t6", permissive)
    assert result.status == "ok"
