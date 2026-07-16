"""Schema validation + robust JSON parsing."""

from __future__ import annotations

import pytest

from reflect_mcp.parsing import ParseError, load_prompt, parse_json_object
from reflect_mcp.schema import EXTRACTION_FIELDS, CritiqueResult, InvoiceData


def test_extraction_fields_are_stable():
    assert EXTRACTION_FIELDS == [
        "vendor_name", "invoice_number", "invoice_date",
        "subtotal", "tax", "total", "currency",
    ]


def test_invoice_data_defaults_to_null():
    data = InvoiceData()
    assert all(getattr(data, f) is None for f in EXTRACTION_FIELDS)


def test_invoice_data_ignores_extra_keys():
    data = InvoiceData.model_validate({"vendor_name": "X", "bogus": 1})
    assert data.vendor_name == "X"
    assert not hasattr(data, "bogus")


def test_critique_verdict_lookup():
    crit = CritiqueResult.model_validate(
        {"field_verdicts": [{"field": "total", "confidence": 0.4, "grounded": False}],
         "overall_confidence": 0.5}
    )
    assert crit.verdict_for("total").confidence == 0.4
    assert crit.verdict_for("missing") is None


def test_confidence_bounds_enforced():
    with pytest.raises(Exception):
        CritiqueResult.model_validate({"field_verdicts": [], "overall_confidence": 1.5})


@pytest.mark.parametrize(
    "raw",
    [
        '{"vendor_name": "ACME"}',
        '```json\n{"vendor_name": "ACME"}\n```',
        'Sure! Here is the JSON:\n{"vendor_name": "ACME"}\nHope that helps.',
        '```\n{"vendor_name": "ACME"}\n```',
    ],
)
def test_parse_json_object_recovers_from_wrapping(raw):
    assert parse_json_object(raw)["vendor_name"] == "ACME"


def test_parse_json_object_rejects_garbage():
    with pytest.raises(ParseError):
        parse_json_object("no json here at all")


def test_prompt_files_load_and_mention_injection_awareness():
    gen = load_prompt("generator.txt")
    crit = load_prompt("critic.txt")
    assert "null" in gen.lower()
    # both prompts must warn about treating document content as data, not commands
    assert "command" in gen.lower()
    assert "data" in crit.lower()
