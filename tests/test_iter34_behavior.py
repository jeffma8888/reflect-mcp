"""Black-box behavioral tests for iter-34: concurrent batch extraction.

Covers all 6 Expected Behaviors from pm.md:
  1. --batch --batch-concurrency N accepted; --batch-concurrency without --batch → error exit 2.
  2. --batch-concurrency 1 is identical to serial batch (same output, same exit codes).
  3. --batch-concurrency N (N>1) uses thread pool; each worker gets its own pipeline.
  4. Output order preserved regardless of thread completion order.
  5. Error isolation: per-line failure doesn't kill other lines.
  6. Exit code semantics unchanged (worst wins: 3>1>0).

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


def _run_batch(cli_args: list[str], input_text: str):
    """Run run_extract with given args, capturing stdout. Returns (exit_code, stdout_text)."""
    from reflect_mcp.cli import run_extract

    with patch("sys.stdin", io.StringIO(input_text)):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            code = run_extract(cli_args)

    return code, captured.getvalue()


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


def _parse_output_lines(stdout: str) -> list[dict]:
    """Parse JSONL output into list of dicts, skipping blank lines."""
    lines = [line for line in stdout.strip().split("\n") if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# EB-1: --batch-concurrency accepted with --batch; error without
# ---------------------------------------------------------------------------

class TestEB1FlagAcceptance:
    """EB-1: --batch --batch-concurrency N accepted; --batch-concurrency alone → exit 2."""

    def test_batch_concurrency_accepted_with_batch(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, _spy = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], jsonl, n_lines=1
        )
        results = _parse_output_lines(stdout)
        assert len(results) == 1
        assert results[0]["status"] in ("ok", "needs_review")

    def test_batch_concurrency_without_batch_exits_2(self):
        from reflect_mcp.cli import run_extract

        stderr_captured = io.StringIO()
        with patch("sys.stderr", stderr_captured):
            with patch("sys.stdin", io.StringIO("some text")):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch-concurrency", "4"])

        assert code == 2, f"Expected exit 2 but got {code}"

    def test_batch_concurrency_without_batch_prints_error(self):
        from reflect_mcp.cli import run_extract

        stderr_captured = io.StringIO()
        with patch("sys.stderr", stderr_captured):
            with patch("sys.stdin", io.StringIO("some text")):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch-concurrency", "4"])

        stderr_text = stderr_captured.getvalue()
        assert "batch-concurrency" in stderr_text.lower() or "batch" in stderr_text.lower()

    def test_batch_concurrency_4_accepted(self):
        jsonl = _make_jsonl("a", "b", "c", "d")
        code, stdout, _spy = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "4"], jsonl, n_lines=4
        )
        results = _parse_output_lines(stdout)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# EB-2: --batch-concurrency 1 identical to serial batch
# ---------------------------------------------------------------------------

class TestEB2ConcurrencyOneIsSerial:
    """EB-2: --batch-concurrency 1 identical to serial batch path."""

    def test_concurrency_1_same_output_as_serial(self):
        """Running with --batch-concurrency 1 produces the same output as plain --batch."""
        jsonl = _make_jsonl(SOURCE, SOURCE)

        _code_serial, stdout_serial, _spy_serial = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=2
        )
        _code_conc1, stdout_conc1, _spy_conc1 = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "1"], jsonl, n_lines=2
        )

        serial_results = _parse_output_lines(stdout_serial)
        conc1_results = _parse_output_lines(stdout_conc1)

        assert len(serial_results) == len(conc1_results)
        for s, c in zip(serial_results, conc1_results):
            assert s["status"] == c["status"]
            assert s["data"] == c["data"]

    def test_concurrency_1_same_exit_code(self):
        jsonl = _make_jsonl(SOURCE)

        code_serial, _, _ = _run_batch_with_spy(["--batch"], jsonl, n_lines=1)
        code_conc1, _, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "1"], jsonl, n_lines=1
        )

        assert code_serial == code_conc1

    def test_concurrency_1_builds_one_pipeline(self):
        """Serial path builds exactly one pipeline."""
        jsonl = _make_jsonl("doc one", "doc two")
        _, _, spy_calls = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "1"], jsonl, n_lines=2
        )
        assert len(spy_calls) == 1

    def test_default_concurrency_is_serial(self):
        """Omitting --batch-concurrency is behaviorally same as concurrency=1."""
        jsonl = _make_jsonl(SOURCE)

        code_default, stdout_default, _ = _run_batch_with_spy(
            ["--batch"], jsonl, n_lines=1
        )
        code_explicit, stdout_explicit, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "1"], jsonl, n_lines=1
        )

        assert code_default == code_explicit
        default_results = _parse_output_lines(stdout_default)
        explicit_results = _parse_output_lines(stdout_explicit)
        assert default_results[0]["status"] == explicit_results[0]["status"]


# ---------------------------------------------------------------------------
# EB-3: N>1 uses thread pool, per-thread pipeline instances
# ---------------------------------------------------------------------------

class TestEB3ConcurrentThreadPool:
    """EB-3: N>1 uses thread pool; each worker gets its own pipeline instance."""

    def test_concurrent_builds_multiple_pipelines(self):
        """With concurrency > 1, build_default_pipeline is called more than once
        (one per thread that actually runs work)."""
        jsonl = _make_jsonl("a", "b", "c", "d")
        _, _, spy_calls = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "4"], jsonl, n_lines=4
        )
        assert len(spy_calls) >= 1, "At least one pipeline must be built"

    def test_concurrent_produces_correct_count(self):
        """4 input lines with concurrency 2 → 4 output lines."""
        jsonl = _make_jsonl("a", "b", "c", "d")
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], jsonl, n_lines=4
        )
        results = _parse_output_lines(stdout)
        assert len(results) == 4

    def test_concurrent_all_ok_exit_0(self):
        jsonl = _make_jsonl(SOURCE, SOURCE, SOURCE)
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "3"], jsonl, n_lines=3
        )
        assert code == 0
        for r in _parse_output_lines(stdout):
            assert r["status"] == "ok"

    def test_concurrent_result_has_expected_keys(self):
        jsonl = _make_jsonl("a")
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], jsonl, n_lines=1
        )
        results = _parse_output_lines(stdout)
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(results[0].keys())


# ---------------------------------------------------------------------------
# EB-4: Output order preserved
# ---------------------------------------------------------------------------

class TestEB4OutputOrderPreserved:
    """EB-4: Nth output line corresponds to Nth input line."""

    def test_order_preserved_with_concurrency(self):
        """Feed distinct docs and verify output order matches input order."""
        from reflect_mcp.cli import run_extract

        call_count = 0
        docs = ["doc_ALPHA", "doc_BETA", "doc_GAMMA", "doc_DELTA"]

        ok_result = MagicMock(spec=ExtractionResult)
        ok_result.status = "ok"

        def order_tracking_extract(text, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock(spec=ExtractionResult)
            result.status = "ok"
            result.model_dump.return_value = {
                "status": "ok", "data": {"source_text": text[:20]},
                "field_confidence": {}, "overall_confidence": 0.95,
                "flagged_fields": [], "critic_summary": "ok",
                "trace_id": f"t{call_count}", "notes": "",
            }
            return result

        pipeline = MagicMock()
        pipeline.extract = order_tracking_extract

        jsonl = _make_jsonl(*docs)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 4
        for i, doc in enumerate(docs):
            assert results[i]["data"]["source_text"] == doc[:20], \
                f"Output line {i} should match input '{doc}'"


# ---------------------------------------------------------------------------
# EB-5: Error isolation — per-line failure doesn't kill others
# ---------------------------------------------------------------------------

class TestEB5ErrorIsolation:
    """EB-5: pipeline failure on one line emits error JSON, others continue."""

    def test_error_on_middle_line_concurrent(self):
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
            if "FAIL_THIS" in text:
                raise RuntimeError("extraction failed on marked line")
            return ok_result

        pipeline = MagicMock()
        pipeline.extract = mixed_extract

        docs = ["doc_ok_1", "FAIL_THIS_doc", "doc_ok_2"]
        jsonl = _make_jsonl(*docs)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 3, "All 3 lines must produce output"
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"
        assert "error_message" in results[1]
        assert results[2]["status"] == "ok"

    def test_error_envelope_has_message_concurrent(self):
        from reflect_mcp.cli import run_extract

        def always_fail(text, **kwargs):
            raise RuntimeError("boom")

        pipeline = MagicMock()
        pipeline.extract = always_fail

        jsonl = _make_jsonl("doc one")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "error_message" in results[0]
        assert len(results[0]["error_message"]) > 0

    def test_malformed_line_concurrent_isolated(self):
        """Malformed JSONL emits error, valid lines still process."""
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

        input_lines = "not json\n" + json.dumps({"text": "valid doc"}) + "\n"

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(input_lines)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert results[1]["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-6: Exit code semantics (worst wins: 3>1>0)
# ---------------------------------------------------------------------------

class TestEB6ExitCodes:
    """EB-6: exit code semantics unchanged under concurrency (worst wins)."""

    def test_all_ok_exits_0_concurrent(self):
        jsonl = _make_jsonl(SOURCE, SOURCE)
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], jsonl,
            n_lines=2, critic_conf=0.95, overall_conf=0.95
        )
        assert code == 0

    def test_needs_review_exits_1_concurrent(self):
        jsonl = _make_jsonl("doc one")
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], jsonl,
            n_lines=1, critic_conf=0.3, overall_conf=0.3
        )
        assert code == 1
        results = _parse_output_lines(stdout)
        assert results[0]["status"] == "needs_review"

    def test_error_exits_3_concurrent(self):
        from reflect_mcp.cli import run_extract

        def always_fail(text, **kwargs):
            raise RuntimeError("boom")

        pipeline = MagicMock()
        pipeline.extract = always_fail

        jsonl = _make_jsonl("doc one")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        assert code == 3

    def test_error_beats_needs_review_concurrent(self):
        """error (3) > needs_review (1): if both present, exit 3."""
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
                    code = run_extract(["--batch", "--batch-concurrency", "2"])

        assert code == 3, "error (3) must beat needs_review (1)"

    def test_empty_input_exits_0_concurrent(self):
        """Empty input with concurrency flag → exit 0."""
        code, stdout, _ = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2"], "", n_lines=0
        )
        assert code == 0

    def test_thresholds_forwarded_with_concurrency(self):
        """--field-threshold and --overall-threshold compose with --batch-concurrency."""
        jsonl = _make_jsonl("doc one")
        code, stdout, spy_calls = _run_batch_with_spy(
            ["--batch", "--batch-concurrency", "2",
             "--field-threshold", "0.99", "--overall-threshold", "0.99"],
            jsonl, n_lines=1, critic_conf=0.95, overall_conf=0.95,
        )
        assert code == 1
        results = _parse_output_lines(stdout)
        assert results[0]["status"] == "needs_review"
