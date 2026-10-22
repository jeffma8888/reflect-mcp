"""Black-box behavioral tests for iter-35: field-level provenance spans.

Covers all 6 Expected Behaviors from pm.md:
  1. ExtractionResult gains field_provenance: dict[str, list[ProvenanceSpan]]
     where ProvenanceSpan has start: int and end: int.
  2. Text-grounded fields (vendor_name, invoice_number, currency) get a span
     whose start:end slice equals the value (case-insensitive).
  3. Numeric fields (subtotal, tax, total) get a span whose start:end slice
     contains the matching number string.
  4. Field value NOT found in source → field_provenance[field] is [].
  5. Fields with None values do not appear in field_provenance.
  6. JSON serialization (CLI stdout) includes field_provenance.

Isolation contract: tests drive the public interface (pipeline.extract,
run_extract) and inspect observable output. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from reflect_mcp.schema import ExtractionResult
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pipeline(gen_json: str, crit_json: str):
    """Build a full pipeline from raw generator + critic JSON strings."""
    from reflect_mcp.critic import Critic
    from reflect_mcp.generator import Generator
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.scripted import ScriptedLLMClient

    gen = Generator(ScriptedLLMClient([gen_json]))
    crit = Critic(ScriptedLLMClient([crit_json]))
    return ReflectivePipeline(gen, crit)


def _extract_with_source(source: str, extraction: dict,
                         critic_conf: float = 0.95, overall_conf: float = 0.95):
    """Run the full pipeline and return the ExtractionResult."""
    gen_json = json.dumps(extraction)
    crit_json = critique_all(critic_conf, overall_conf)
    pipeline = _build_pipeline(gen_json, crit_json)
    return pipeline.extract(source)


# ---------------------------------------------------------------------------
# EB-1: ExtractionResult has field_provenance with ProvenanceSpan(start, end)
# ---------------------------------------------------------------------------

class TestEB1FieldProvenanceExists:
    """EB-1: ExtractionResult.field_provenance is a dict of field -> list[ProvenanceSpan]."""

    def test_field_provenance_present_in_result(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        assert hasattr(result, "field_provenance")
        assert isinstance(result.field_provenance, dict)

    def test_field_provenance_values_are_lists(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        for field, spans in result.field_provenance.items():
            assert isinstance(spans, list), f"field_provenance[{field!r}] should be a list"

    def test_provenance_span_has_start_and_end(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        found_span = False
        for field, spans in result.field_provenance.items():
            for span in spans:
                found_span = True
                assert hasattr(span, "start"), f"ProvenanceSpan for {field} missing 'start'"
                assert hasattr(span, "end"), f"ProvenanceSpan for {field} missing 'end'"
                assert isinstance(span.start, int)
                assert isinstance(span.end, int)
        assert found_span, "Expected at least one provenance span in a correct extraction"

    def test_field_provenance_in_model_dump(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        dumped = result.model_dump()
        assert "field_provenance" in dumped
        assert isinstance(dumped["field_provenance"], dict)

    def test_provenance_keys_are_field_names(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        for key in result.field_provenance:
            assert key in CORRECT_EXTRACTION, \
                f"Unexpected provenance key {key!r} not in extraction fields"


# ---------------------------------------------------------------------------
# EB-2: Text-grounded fields get correct spans (case-insensitive)
# ---------------------------------------------------------------------------

class TestEB2TextGroundedSpans:
    """EB-2: vendor_name, invoice_number, currency get spans where
    source[start:end] matches the value (case-insensitive)."""

    def test_vendor_name_span_matches_source(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("vendor_name", [])
        assert len(spans) == 1, "Expected exactly one span for vendor_name"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert sliced.lower() == CORRECT_EXTRACTION["vendor_name"].lower()

    def test_invoice_number_span_matches_source(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("invoice_number", [])
        assert len(spans) == 1, "Expected exactly one span for invoice_number"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert sliced.lower() == CORRECT_EXTRACTION["invoice_number"].lower()

    def test_currency_span_matches_source(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("currency", [])
        assert len(spans) == 1, "Expected exactly one span for currency"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert sliced.lower() == CORRECT_EXTRACTION["currency"].lower()

    def test_span_offsets_are_nonnegative(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        for field in ("vendor_name", "invoice_number", "currency"):
            spans = result.field_provenance.get(field, [])
            for span in spans:
                assert span.start >= 0
                assert span.end >= span.start
                assert span.end <= len(SOURCE)

    def test_case_insensitive_match(self):
        """A value that differs only in case from the source should still get a span."""
        modified = dict(CORRECT_EXTRACTION)
        modified["vendor_name"] = "acme corp"
        result = _extract_with_source(SOURCE, modified)
        spans = result.field_provenance.get("vendor_name", [])
        assert len(spans) == 1
        sliced = SOURCE[spans[0].start:spans[0].end]
        assert sliced.lower() == "acme corp"


# ---------------------------------------------------------------------------
# EB-3: Numeric field spans contain the matching number string
# ---------------------------------------------------------------------------

class TestEB3NumericSpans:
    """EB-3: subtotal, tax, total get spans where source[start:end] contains
    the matching number string."""

    def test_subtotal_span_contains_number(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("subtotal", [])
        assert len(spans) == 1, "Expected exactly one span for subtotal"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert "100" in sliced, f"Expected '100' in span slice {sliced!r}"

    def test_tax_span_contains_number(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("tax", [])
        assert len(spans) == 1, "Expected exactly one span for tax"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert "8.5" in sliced or "8.50" in sliced, \
            f"Expected '8.5' or '8.50' in span slice {sliced!r}"

    def test_total_span_contains_number(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        spans = result.field_provenance.get("total", [])
        assert len(spans) == 1, "Expected exactly one span for total"
        span = spans[0]
        sliced = SOURCE[span.start:span.end]
        assert "108.5" in sliced or "108.50" in sliced, \
            f"Expected '108.5' or '108.50' in span slice {sliced!r}"

    def test_numeric_span_offsets_within_source(self):
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        for field in ("subtotal", "tax", "total"):
            spans = result.field_provenance.get(field, [])
            for span in spans:
                assert 0 <= span.start < span.end <= len(SOURCE), \
                    f"{field} span ({span.start},{span.end}) out of bounds"


# ---------------------------------------------------------------------------
# EB-4: Value not found in source → empty list
# ---------------------------------------------------------------------------

class TestEB4UnfoundValueEmptyList:
    """EB-4: When a field value is NOT in the source text, provenance is []."""

    def test_vendor_not_in_source_gives_empty_provenance(self):
        source = "Some random document with no matching fields at all.\n99.99"
        extraction = dict(CORRECT_EXTRACTION)
        extraction["vendor_name"] = "NONEXISTENT VENDOR"
        result = _extract_with_source(source, extraction)
        spans = result.field_provenance.get("vendor_name", [])
        assert spans == [], f"Expected empty list for vendor_name not in source, got {spans}"

    def test_currency_not_in_source_gives_empty_provenance(self):
        source = "Invoice from ACME CORP, total 100.00"
        extraction = dict(CORRECT_EXTRACTION)
        extraction["currency"] = "EUR"
        result = _extract_with_source(source, extraction)
        spans = result.field_provenance.get("currency", [])
        assert spans == [], f"Expected empty list for currency 'EUR' not in source"

    def test_numeric_not_in_source_gives_empty_provenance(self):
        source = "ACME CORP\nInvoice #INV-1001\nPaid in USD"
        extraction = dict(CORRECT_EXTRACTION)
        extraction["subtotal"] = 999.99
        extraction["tax"] = 77.77
        extraction["total"] = 1077.76
        result = _extract_with_source(source, extraction)
        for field in ("subtotal", "tax", "total"):
            spans = result.field_provenance.get(field, [])
            assert spans == [], \
                f"Expected empty provenance for {field} not in source, got {spans}"


# ---------------------------------------------------------------------------
# EB-5: None fields absent from field_provenance
# ---------------------------------------------------------------------------

class TestEB5NoneFieldsAbsent:
    """EB-5: Fields with None values do not appear in field_provenance."""

    def test_none_vendor_not_in_provenance(self):
        extraction = dict(CORRECT_EXTRACTION)
        extraction["vendor_name"] = None
        gen_json = json.dumps(extraction)
        crit_json = critique_all(0.95, 0.95)
        pipeline = _build_pipeline(gen_json, crit_json)
        result = pipeline.extract(SOURCE)
        assert "vendor_name" not in result.field_provenance

    def test_none_total_not_in_provenance(self):
        extraction = dict(CORRECT_EXTRACTION)
        extraction["total"] = None
        gen_json = json.dumps(extraction)
        crit_json = critique_all(0.95, 0.95)
        pipeline = _build_pipeline(gen_json, crit_json)
        result = pipeline.extract(SOURCE)
        assert "total" not in result.field_provenance

    def test_multiple_none_fields_absent(self):
        extraction = dict(CORRECT_EXTRACTION)
        extraction["vendor_name"] = None
        extraction["currency"] = None
        extraction["tax"] = None
        gen_json = json.dumps(extraction)
        crit_json = critique_all(0.95, 0.95)
        pipeline = _build_pipeline(gen_json, crit_json)
        result = pipeline.extract(SOURCE)
        for field in ("vendor_name", "currency", "tax"):
            assert field not in result.field_provenance, \
                f"None field {field!r} should not appear in field_provenance"

    def test_non_none_fields_still_present(self):
        """Even when some fields are None, the non-None fields get provenance."""
        extraction = dict(CORRECT_EXTRACTION)
        extraction["vendor_name"] = None
        gen_json = json.dumps(extraction)
        crit_json = critique_all(0.95, 0.95)
        pipeline = _build_pipeline(gen_json, crit_json)
        result = pipeline.extract(SOURCE)
        assert "vendor_name" not in result.field_provenance
        assert "invoice_number" in result.field_provenance


# ---------------------------------------------------------------------------
# EB-6: JSON serialization includes field_provenance
# ---------------------------------------------------------------------------

class TestEB6JSONSerialization:
    """EB-6: CLI output JSON includes field_provenance dict."""

    def test_cli_extract_output_has_field_provenance(self):
        from reflect_mcp.cli import run_extract

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(0.95, 0.95)

        def spy_factory(*args, **kwargs):
            return _build_pipeline(gen_json, crit_json)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract([])

        output = json.loads(captured.getvalue())
        assert "field_provenance" in output, "CLI output must include field_provenance"
        assert isinstance(output["field_provenance"], dict)

    def test_cli_field_provenance_spans_are_serializable(self):
        from reflect_mcp.cli import run_extract

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(0.95, 0.95)

        def spy_factory(*args, **kwargs):
            return _build_pipeline(gen_json, crit_json)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract([])

        output = json.loads(captured.getvalue())
        for field, spans in output["field_provenance"].items():
            assert isinstance(spans, list)
            for span in spans:
                assert "start" in span
                assert "end" in span
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)

    def test_cli_batch_output_has_field_provenance(self):
        from reflect_mcp.cli import run_extract

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(0.95, 0.95)

        def spy_factory(*args, **kwargs):
            from reflect_mcp.critic import Critic
            from reflect_mcp.generator import Generator
            from reflect_mcp.pipeline import ReflectivePipeline
            from reflect_mcp.scripted import ScriptedLLMClient
            return ReflectivePipeline(
                Generator(ScriptedLLMClient([gen_json])),
                Critic(ScriptedLLMClient([crit_json])),
            )

        jsonl = json.dumps({"text": SOURCE}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        lines = [l for l in captured.getvalue().strip().split("\n") if l.strip()]
        assert len(lines) == 1
        output = json.loads(lines[0])
        assert "field_provenance" in output

    def test_field_provenance_key_in_extraction_result_keys(self):
        """The full set of ExtractionResult keys includes field_provenance."""
        result = _extract_with_source(SOURCE, CORRECT_EXTRACTION)
        dumped = result.model_dump()
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(dumped.keys())
