"""Black-box behavioral tests for iter-29: batch extraction via JSONL I/O.

Covers all 8 Expected Behaviors from pm.md:
  1. --batch reads JSONL (stdin or --file), runs pipeline.extract per line,
     prints one JSON result per line to stdout.
  2. Pipeline instance constructed ONCE and reused across all JSONL lines.
  3. Per-line extraction error emits {"status":"error","error_message":"..."}
     for that line and continues processing remaining lines.
  4. Malformed JSONL (not JSON or missing "text" key) emits error for that
     line and continues.
  5. Exit code: 0 if all ok, 1 if any needs_review (no errors), 3 if any
     error. Highest severity wins.
  6. Empty input → exit 0, no output.
  7. --batch composes with --field-threshold / --overall-threshold.
  8. Without --batch, existing single-document behavior unchanged.

Isolation contract: tests drive the public CLI interface (run_extract) and
inspect observable output. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from reflect_mcp.gate import GateConfig
from reflect_mcp.schema import ExtractionResult
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch_pipeline(n_lines: int, critic_conf: float = 0.95, overall_conf: float = 0.95):
    """Build a pipeline with enough scripted responses for n_lines extractions."""
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
    """Build JSONL input from one or more text strings."""
    return "\n".join(json.dumps({"text": t}) for t in texts) + "\n"


def _run_batch_with_spy(cli_args: list[str], input_text: str, n_lines: int = 3,
                        critic_conf: float = 0.95, overall_conf: float = 0.95):
    """Run run_extract --batch with a spy on build_default_pipeline.

    Returns (exit_code, stdout_text, spy_calls).
    """
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


def _run_batch_simple(cli_args: list[str], input_text: str):
    """Run run_extract --batch without mocking build_default_pipeline.

    Returns (exit_code, stdout_text).
    """
    from reflect_mcp.cli import run_extract

    with patch("sys.stdin", io.StringIO(input_text)):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            code = run_extract(cli_args)

    return code, captured.getvalue()


def _parse_output_lines(stdout: str) -> list[dict]:
    """Parse JSONL output into list of dicts, skipping blank lines."""
    lines = [line for line in stdout.strip().split("\n") if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# EB-1: --batch reads JSONL and emits one result per line
# ---------------------------------------------------------------------------

class TestEB1BatchReadsJSONL:
    """EB-1: --batch reads JSONL, runs pipeline.extract per line, prints one JSON per line."""

    def test_three_lines_produce_three_results(self):
        jsonl = _make_jsonl("doc one", "doc two", "doc three")
        code, stdout, _spy = _run_batch_with_spy(["--batch"], jsonl, n_lines=3)
        results = _parse_output_lines(stdout)
        assert len(results) == 3

    def test_each_result_is_valid_json_with_status(self):
        jsonl = _make_jsonl("doc one", "doc two")
        code, stdout, _spy = _run_batch_with_spy(["--batch"], jsonl, n_lines=2)
        results = _parse_output_lines(stdout)
        for r in results:
            assert "status" in r
            assert r["status"] in ("ok", "needs_review")

    def test_each_result_has_extraction_result_keys(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, _spy = _run_batch_with_spy(["--batch"], jsonl, n_lines=1)
        results = _parse_output_lines(stdout)
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(results[0].keys())

    def test_batch_with_file_flag(self, tmp_path):
        """--batch --file reads JSONL from file instead of stdin."""
        from reflect_mcp.cli import run_extract

        jsonl_file = tmp_path / "batch.jsonl"
        jsonl_file.write_text(_make_jsonl(SOURCE, SOURCE), encoding="utf-8")

        spy_calls: list[tuple] = []

        def spy_factory(*args, **kwargs):
            spy_calls.append((args, kwargs))
            return _make_batch_pipeline(2)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract(["--batch", "--file", str(jsonl_file)])

        assert code == 0
        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2

    def test_single_line_batch(self):
        jsonl = _make_jsonl("single doc")
        code, stdout, _spy = _run_batch_with_spy(["--batch"], jsonl, n_lines=1)
        results = _parse_output_lines(stdout)
        assert len(results) == 1
        assert results[0]["status"] in ("ok", "needs_review")


# ---------------------------------------------------------------------------
# EB-2: Pipeline reuse — built ONCE for entire batch
# ---------------------------------------------------------------------------

class TestEB2PipelineReuse:
    """EB-2: build_default_pipeline called exactly once for multi-line batch."""

    def test_single_build_call_for_three_lines(self):
        jsonl = _make_jsonl("doc one", "doc two", "doc three")
        code, stdout, spy_calls = _run_batch_with_spy(["--batch"], jsonl, n_lines=3)
        assert len(spy_calls) == 1, "Pipeline must be constructed once, not per-line"

    def test_single_build_call_for_one_line(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, spy_calls = _run_batch_with_spy(["--batch"], jsonl, n_lines=1)
        assert len(spy_calls) == 1


# ---------------------------------------------------------------------------
# EB-3: Per-line extraction error → error envelope, continue processing
# ---------------------------------------------------------------------------

class TestEB3PerLineErrorIsolation:
    """EB-3: extraction exception on one line emits error, continues processing."""

    def test_error_on_middle_line_continues(self):
        from reflect_mcp.cli import run_extract

        call_count = 0
        results_to_return = []

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"
        ok_result.model_dump.return_value = {
            "status": "ok", "data": CORRECT_EXTRACTION,
            "field_confidence": {}, "overall_confidence": 0.95,
            "flagged_fields": [], "critic_summary": "ok",
            "trace_id": "t1", "notes": "",
        }

        def mock_extract(text, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("extraction failed on line 2")
            return ok_result

        pipeline = MagicMock()
        pipeline.extract = mock_extract

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            jsonl = _make_jsonl("doc one", "doc two", "doc three")
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 3, "All 3 lines must produce output"
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"
        assert "error_message" in results[1]
        assert results[2]["status"] == "ok"

    def test_error_envelope_has_error_message(self):
        from reflect_mcp.cli import run_extract

        def always_fail(text, **kwargs):
            raise RuntimeError("boom")

        pipeline = MagicMock()
        pipeline.extract = always_fail

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            jsonl = _make_jsonl("doc one")
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "error_message" in results[0]
        assert len(results[0]["error_message"]) > 0


# ---------------------------------------------------------------------------
# EB-4: Malformed JSONL → error envelope, continue processing
# ---------------------------------------------------------------------------

class TestEB4MalformedJSONL:
    """EB-4: malformed JSONL (not JSON or missing 'text') emits error, continues."""

    def test_not_valid_json(self):
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

        input_lines = "this is not json\n" + json.dumps({"text": "valid doc"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert "error_message" in results[0]
        assert results[1]["status"] == "ok"

    def test_missing_text_key(self):
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

        input_lines = json.dumps({"no_text_key": "oops"}) + "\n" + json.dumps({"text": "valid"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert "error_message" in results[0]
        assert results[1]["status"] == "ok"

    def test_empty_json_object(self):
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

        input_lines = "{}\n" + json.dumps({"text": "valid"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert results[1]["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-5: Exit codes — highest severity wins
# ---------------------------------------------------------------------------

class TestEB5ExitCodes:
    """EB-5: exit 0 (all ok), 1 (any needs_review, no error), 3 (any error)."""

    def test_all_ok_exits_0(self):
        jsonl = _make_jsonl(SOURCE, SOURCE)
        code, stdout, _spy = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=2, critic_conf=0.95, overall_conf=0.95
        )
        assert code == 0
        results = _parse_output_lines(stdout)
        for r in results:
            assert r["status"] == "ok"

    def test_needs_review_exits_1(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, _spy = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=1, critic_conf=0.3, overall_conf=0.3
        )
        assert code == 1
        results = _parse_output_lines(stdout)
        assert results[0]["status"] == "needs_review"

    def test_error_exits_3(self):
        from reflect_mcp.cli import run_extract

        def always_fail(text, **kwargs):
            raise RuntimeError("boom")

        pipeline = MagicMock()
        pipeline.extract = always_fail

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            jsonl = _make_jsonl("doc one")
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3

    def test_error_beats_needs_review(self):
        """error (3) > needs_review (1): if both present, exit 3."""
        from reflect_mcp.cli import run_extract

        call_count = 0

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "needs_review"
        ok_result.model_dump.return_value = {
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
            return ok_result

        pipeline = MagicMock()
        pipeline.extract = mixed_extract

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            jsonl = _make_jsonl("doc one", "doc two", "doc three")
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch"])

        assert code == 3, "error (3) must beat needs_review (1)"

    def test_malformed_line_exits_3(self):
        """Malformed JSONL is also an error → exit 3."""
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
# EB-6: Empty input → exit 0, no output
# ---------------------------------------------------------------------------

class TestEB6EmptyInput:
    """EB-6: empty input → exit 0, no output lines."""

    def test_empty_stdin_exits_0(self):
        code, stdout, spy_calls = _run_batch_with_spy(["--batch"], "", n_lines=0)
        assert code == 0

    def test_empty_stdin_no_output(self):
        code, stdout, _spy = _run_batch_with_spy(["--batch"], "", n_lines=0)
        stripped = stdout.strip()
        assert stripped == "", f"Expected no output but got: {stripped!r}"

    def test_whitespace_only_exits_0(self):
        code, stdout, _spy = _run_batch_with_spy(["--batch"], "\n\n\n", n_lines=0)
        assert code == 0


# ---------------------------------------------------------------------------
# EB-7: --batch composes with threshold flags
# ---------------------------------------------------------------------------

class TestEB7ThresholdComposition:
    """EB-7: --batch composes with --field-threshold / --overall-threshold."""

    def test_field_threshold_forwarded_in_batch(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, spy_calls = _run_batch_with_spy(
            ["--batch", "--field-threshold", "0.85"], jsonl, n_lines=1
        )
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert isinstance(gate_config, GateConfig)
        assert gate_config.field_threshold == pytest.approx(0.85)

    def test_overall_threshold_forwarded_in_batch(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, spy_calls = _run_batch_with_spy(
            ["--batch", "--overall-threshold", "0.80"], jsonl, n_lines=1
        )
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_both_thresholds_forwarded_in_batch(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, spy_calls = _run_batch_with_spy(
            ["--batch", "--field-threshold", "0.85", "--overall-threshold", "0.80"],
            jsonl, n_lines=1,
        )
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert gate_config.field_threshold == pytest.approx(0.85)
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_thresholds_affect_batch_results(self):
        """With a very high threshold, confident extractions become needs_review."""
        jsonl = _make_jsonl("doc one")
        code, stdout, _spy = _run_batch_with_spy(
            ["--batch", "--field-threshold", "0.99", "--overall-threshold", "0.99"],
            jsonl, n_lines=1, critic_conf=0.95, overall_conf=0.95,
        )
        assert code == 1
        results = _parse_output_lines(stdout)
        assert results[0]["status"] == "needs_review"


# ---------------------------------------------------------------------------
# EB-8: Without --batch, existing behavior unchanged
# ---------------------------------------------------------------------------

class TestEB8NoBatchUnchanged:
    """EB-8: Without --batch, single-document behavior works as before."""

    def test_single_document_without_batch(self):
        from reflect_mcp.cli import run_extract

        spy_calls: list[tuple] = []

        def spy_factory(*args, **kwargs):
            spy_calls.append((args, kwargs))
            return _make_batch_pipeline(1)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract([])

        assert code == 0
        parsed = json.loads(captured.getvalue())
        assert parsed["status"] == "ok"
        assert "data" in parsed

    def test_single_document_is_not_jsonl(self):
        """Without --batch, input is raw text, not JSONL."""
        from reflect_mcp.cli import run_extract

        spy_calls: list[tuple] = []

        def spy_factory(*args, **kwargs):
            spy_calls.append((args, kwargs))
            return _make_batch_pipeline(1)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract([])

        assert code == 0
        output = captured.getvalue().strip()
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed["status"] in ("ok", "needs_review")
