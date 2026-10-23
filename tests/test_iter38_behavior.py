"""Black-box behavioral tests for iter-38: batch serial-path simplification.

Covers Expected Behaviors from pm.md:
  1. Serial batch produces identical output (valid lines).
  2. Error lines produce identical JSON error envelopes (same wording).
  3. Exit-code rollup is preserved (error=3 > needs_review=1 > ok=0).
  4. Empty lines are still skipped in serial batch.
  5. _run_batch body shrinks (structural — not testable black-box).
  6. Concurrent batch path unchanged (already covered by iter-34 tests).
  7. All existing tests pass (verified by full suite run).

Isolation contract: tests drive the public CLI interface (run_extract) and
inspect observable output. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from reflect_mcp.schema import ExtractionResult
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch_pipeline(n_lines: int, critic_conf: float = 0.95, overall_conf: float = 0.95):
    from reflect_mcp.critic import Critic
    from reflect_mcp.generator import Generator
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.scripted import ScriptedLLMClient

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(critic_conf, overall_conf)
    return ReflectivePipeline(
        Generator(ScriptedLLMClient([gen_json] * n_lines)),
        Critic(ScriptedLLMClient([crit_json] * n_lines)),
    )


def _make_jsonl(*texts: str) -> str:
    return "\n".join(json.dumps({"text": t}) for t in texts) + "\n"


def _run_batch_with_spy(cli_args: list[str], input_text: str, n_lines: int = 3,
                        critic_conf: float = 0.95, overall_conf: float = 0.95):
    from reflect_mcp.cli import run_extract

    spy_calls: list[tuple] = []

    def spy_factory(*args, **kwargs):
        spy_calls.append((args, kwargs))
        return _make_batch_pipeline(n_lines, critic_conf, overall_conf)

    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
        with patch("sys.stdin", io.StringIO(input_text)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract(cli_args)

    return code, captured.getvalue(), spy_calls


def _parse_output_lines(stdout: str) -> list[dict]:
    lines = [line for line in stdout.strip().split("\n") if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# EB-1: Serial batch produces identical output
# ---------------------------------------------------------------------------

class TestEB1SerialBatchOutput:
    """EB-1: serial batch with valid lines produces correct ExtractionResult JSON."""

    def test_valid_lines_produce_ok_status(self):
        jsonl = _make_jsonl(SOURCE, SOURCE)
        code, stdout, _ = _run_batch_with_spy(["--batch"], jsonl, n_lines=2)
        results = _parse_output_lines(stdout)
        assert len(results) == 2
        for r in results:
            assert r["status"] == "ok"
            assert r["data"]["vendor_name"] == "ACME CORP"
            assert r["data"]["total"] == 108.5

    def test_output_has_all_extraction_result_keys(self):
        jsonl = _make_jsonl(SOURCE)
        code, stdout, _ = _run_batch_with_spy(["--batch"], jsonl, n_lines=1)
        results = _parse_output_lines(stdout)
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(results[0].keys())

    def test_serial_and_concurrent_produce_equivalent_results(self):
        """Serial and concurrent paths produce equivalent output for same input."""
        jsonl = _make_jsonl(SOURCE, SOURCE)
        code_serial, stdout_serial, _ = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=2
        )
        code_conc, stdout_conc, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "1"], jsonl, n_lines=2
        )
        serial_results = _parse_output_lines(stdout_serial)
        conc_results = _parse_output_lines(stdout_conc)
        assert len(serial_results) == len(conc_results)
        for s, c in zip(serial_results, conc_results):
            assert s["status"] == c["status"]
            assert s["data"] == c["data"]
        assert code_serial == code_conc


# ---------------------------------------------------------------------------
# EB-2: Error envelopes produce identical JSON error envelopes
# ---------------------------------------------------------------------------

class TestEB2ErrorEnvelopes:
    """EB-2: error envelopes have correct status and error_message wording."""

    def test_missing_text_key_error_message(self):
        """JSON object without 'text' key produces error with 'missing "text" key'."""
        from reflect_mcp.cli import run_extract

        pipeline = MagicMock()
        input_lines = json.dumps({"no_text": "oops"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert 'missing "text" key' in results[0]["error_message"]

    def test_invalid_json_error_message(self):
        """Non-JSON line produces error envelope with relevant message."""
        from reflect_mcp.cli import run_extract

        pipeline = MagicMock()
        input_lines = "this is not json\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert len(results[0]["error_message"]) > 0

    def test_pipeline_failure_error_message(self):
        """Pipeline exception on extraction produces error with exception text."""
        from reflect_mcp.cli import run_extract

        def fail_extract(text, **kwargs):
            raise RuntimeError("simulated pipeline failure")

        pipeline = MagicMock()
        pipeline.extract = fail_extract

        jsonl = _make_jsonl("doc one")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "simulated pipeline failure" in results[0]["error_message"]

    def test_serial_and_concurrent_same_error_envelope_for_missing_key(self):
        """Serial and concurrent paths produce the same error envelope for missing 'text' key."""
        from reflect_mcp.cli import run_extract

        input_lines = json.dumps({"no_text": "oops"}) + "\n"

        def run_with_args(args):
            pipeline = MagicMock()
            with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
                with patch("sys.stdin", io.StringIO(input_lines)):
                    captured = io.StringIO()
                    with patch("sys.stdout", captured):
                        code = run_extract(args)
            return code, _parse_output_lines(captured.getvalue())

        code_serial, results_serial = run_with_args(["--batch"])
        code_conc, results_conc = run_with_args(["--batch", "--batch-concurrency", "1"])

        assert results_serial[0]["status"] == results_conc[0]["status"]
        assert results_serial[0]["error_message"] == results_conc[0]["error_message"]

    def test_empty_json_object_error(self):
        """Empty JSON object {} is missing 'text' key → error envelope."""
        from reflect_mcp.cli import run_extract

        pipeline = MagicMock()
        input_lines = "{}\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert 'missing "text" key' in results[0]["error_message"]


# ---------------------------------------------------------------------------
# EB-3: Exit-code rollup preserved
# ---------------------------------------------------------------------------

class TestEB3ExitCodeRollup:
    """EB-3: exit-code rollup: error=3, needs_review=1, ok=0, worst wins."""

    def test_all_ok_exits_0(self):
        jsonl = _make_jsonl(SOURCE, SOURCE, SOURCE)
        code, _, _ = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=3, critic_conf=0.95, overall_conf=0.95
        )
        assert code == 0

    def test_single_ok_exits_0(self):
        jsonl = _make_jsonl(SOURCE)
        code, _, _ = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=1, critic_conf=0.95, overall_conf=0.95
        )
        assert code == 0

    def test_needs_review_exits_1(self):
        jsonl = _make_jsonl("some short text")
        code, stdout, _ = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=1, critic_conf=0.3, overall_conf=0.3
        )
        assert code == 1

    def test_error_exits_3(self):
        from reflect_mcp.cli import run_extract

        def fail(text, **kwargs):
            raise RuntimeError("boom")

        pipeline = MagicMock()
        pipeline.extract = fail

        jsonl = _make_jsonl("doc")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3

    def test_error_beats_needs_review(self):
        """error (3) > needs_review (1)."""
        from reflect_mcp.cli import run_extract

        call_count = 0

        review_result = MagicMock(spec=ExtractionResult)
        review_result.status = "needs_review"
        review_result.model_dump.return_value = {
            "status": "needs_review", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.5,
            "flagged_fields": [{"field": "total", "value": 108.5,
                                "confidence": 0.3, "reason": "low"}],
            "critic_summary": "review", "trace_id": "t1", "notes": "review",
        }

        def mixed_extract(text, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("extraction failed")
            return review_result

        pipeline = MagicMock()
        pipeline.extract = mixed_extract

        jsonl = _make_jsonl("doc one", "doc two", "doc three")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3

    def test_error_beats_ok(self):
        """error (3) > ok (0)."""
        from reflect_mcp.cli import run_extract

        call_count = 0

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        def mixed_extract(text, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first line fails")
            return ok_result

        pipeline = MagicMock()
        pipeline.extract = mixed_extract

        jsonl = _make_jsonl("doc one", "doc two")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3

    def test_needs_review_beats_ok(self):
        """needs_review (1) > ok (0)."""
        from reflect_mcp.cli import run_extract

        call_count = 0

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        review_result = MagicMock(spec=ExtractionResult)
        review_result.status = "needs_review"
        review_result.model_dump.return_value = {
            "status": "needs_review", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.5,
            "flagged_fields": [{"field": "total", "value": 108.5,
                                "confidence": 0.3, "reason": "low"}],
            "critic_summary": "review", "trace_id": "t1", "notes": "review",
        }

        def mixed_extract(text, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ok_result
            return review_result

        pipeline = MagicMock()
        pipeline.extract = mixed_extract

        jsonl = _make_jsonl("doc one", "doc two")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 1

    def test_malformed_line_exits_3(self):
        """Malformed JSONL counts as error → exit 3."""
        from reflect_mcp.cli import run_extract

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        pipeline = MagicMock()
        pipeline.extract = MagicMock(return_value=ok_result)

        input_lines = "not json\n" + json.dumps({"text": "valid"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3


# ---------------------------------------------------------------------------
# EB-4: Empty lines skipped
# ---------------------------------------------------------------------------

class TestEB4EmptyLinesSkipped:
    """EB-4: blank lines in JSONL input are silently skipped."""

    def test_blank_lines_produce_no_output(self):
        """Blank lines between valid lines produce no output lines."""
        from reflect_mcp.cli import run_extract

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        pipeline = MagicMock()
        pipeline.extract = MagicMock(return_value=ok_result)

        input_lines = json.dumps({"text": "doc one"}) + "\n\n\n" + json.dumps({"text": "doc two"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert pipeline.extract.call_count == 2

    def test_only_blank_lines_exits_0_no_output(self):
        code, stdout, _ = _run_batch_with_spy(["--batch"], "\n\n\n", n_lines=0)
        assert code == 0
        stripped = stdout.strip()
        assert stripped == ""

    def test_trailing_blank_lines_ignored(self):
        from reflect_mcp.cli import run_extract

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        pipeline = MagicMock()
        pipeline.extract = MagicMock(return_value=ok_result)

        input_lines = json.dumps({"text": "doc one"}) + "\n\n\n\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert pipeline.extract.call_count == 1

    def test_cr_lf_blank_lines_skipped(self):
        """Lines containing only \\r\\n are also skipped."""
        from reflect_mcp.cli import run_extract

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        pipeline = MagicMock()
        pipeline.extract = MagicMock(return_value=ok_result)

        input_lines = json.dumps({"text": "doc one"}) + "\r\n\r\n" + json.dumps({"text": "doc two"}) + "\r\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert pipeline.extract.call_count == 2
