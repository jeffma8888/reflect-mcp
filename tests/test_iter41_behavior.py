"""Black-box behavioral tests for iter-41: heuristic-only verification mode.

Covers all 8 Expected Behaviors from pm.md:
  1. ReflectivePipeline.__init__ accepts critic=None; when None, _verify synthesizes
     a neutral CritiqueResult with overall_confidence=1.0, per-field 1.0, empty summary.
  2. build_default_pipeline accepts critic_enabled=False; when False, no critic client
     is instantiated and the pipeline's critic is None.
  3. extract CLI verb accepts --heuristics-only flag.
  4. validate CLI verb accepts --heuristics-only flag.
  5. Heuristic penalty surfaces correctly in heuristic-only mode
     (1.0 x 0.25 = 0.25 < 0.70 -> needs_review).
  6. critic_summary is empty string "" in heuristic-only mode.
  7. --heuristics-only appears in --help output of both extract and validate.
  8. Batch mode (serial + concurrent) works with --heuristics-only.

Isolation contract: tests drive the public package API and CLI entry points,
inspecting observable output. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline, build_default_pipeline
from reflect_mcp.schema import ExtractionResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_response(overrides: dict | None = None) -> str:
    data = dict(CORRECT_EXTRACTION)
    if overrides:
        data.update(overrides)
    return json.dumps(data)


def _make_heuristic_only_pipeline(
    gen_responses: list[str] | None = None,
    n_extractions: int = 1,
) -> ReflectivePipeline:
    if gen_responses is None:
        gen_responses = [_gen_response()] * n_extractions
    gen = Generator(ScriptedLLMClient(gen_responses))
    return ReflectivePipeline(gen, critic=None)


def _parse_output_lines(stdout: str) -> list[dict]:
    lines = [line for line in stdout.strip().split("\n") if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# EB-1: ReflectivePipeline.__init__ accepts critic=None
# ---------------------------------------------------------------------------

class TestEB1CriticNone:
    """EB-1: ReflectivePipeline with critic=None synthesizes neutral critique."""

    def test_constructor_accepts_none_critic(self):
        pipeline = _make_heuristic_only_pipeline()
        assert pipeline.critic is None

    def test_extract_produces_valid_result(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        assert isinstance(result, ExtractionResult)
        assert result.status in ("ok", "needs_review")

    def test_extract_ok_with_correct_data(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        assert result.status == "ok"

    def test_overall_confidence_is_one(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        assert result.overall_confidence == 1.0

    def test_all_fields_confidence_one(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        for field, conf in result.field_confidence.items():
            assert conf == 1.0, f"field {field} expected 1.0 but got {conf}"

    def test_non_null_fields_have_verdicts(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        data = result.data
        non_null_fields = {
            f for f in CORRECT_EXTRACTION if getattr(data, f, None) is not None
        }
        assert set(result.field_confidence.keys()) == non_null_fields

    def test_null_field_not_flagged(self):
        gen = _gen_response({"currency": None})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        flagged_names = [f.field for f in result.flagged_fields]
        assert "currency" not in flagged_names


# ---------------------------------------------------------------------------
# EB-2: build_default_pipeline(critic_enabled=False)
# ---------------------------------------------------------------------------

class TestEB2BuildDefaultPipelineCriticDisabled:
    """EB-2: build_default_pipeline(critic_enabled=False) skips critic."""

    def test_signature_accepts_critic_enabled(self):
        import inspect
        sig = inspect.signature(build_default_pipeline)
        assert "critic_enabled" in sig.parameters

    def test_critic_enabled_defaults_true(self):
        import inspect
        sig = inspect.signature(build_default_pipeline)
        param = sig.parameters["critic_enabled"]
        assert param.default is True

    def test_critic_disabled_returns_pipeline(self):
        with patch("reflect_mcp.providers.build_client") as mock_build:
            mock_build.return_value = ScriptedLLMClient([_gen_response()])
            pipeline = build_default_pipeline(critic_enabled=False)
        assert isinstance(pipeline, ReflectivePipeline)

    def test_critic_disabled_pipeline_has_none_critic(self):
        with patch("reflect_mcp.providers.build_client") as mock_build:
            mock_build.return_value = ScriptedLLMClient([_gen_response()])
            pipeline = build_default_pipeline(critic_enabled=False)
        assert pipeline.critic is None

    def test_critic_disabled_does_not_build_critic_client(self):
        with patch("reflect_mcp.providers.build_client") as mock_build:
            mock_build.return_value = ScriptedLLMClient([_gen_response()])
            build_default_pipeline(critic_enabled=False)
        call_roles = [call.args[0] for call in mock_build.call_args_list]
        assert "critic" not in call_roles, (
            "build_client('critic') should not be called when critic_enabled=False"
        )


# ---------------------------------------------------------------------------
# EB-3: extract CLI verb accepts --heuristics-only
# ---------------------------------------------------------------------------

class TestEB3ExtractHeuristicsOnlyFlag:
    """EB-3: extract accepts --heuristics-only and builds pipeline with critic_enabled=False."""

    def test_extract_heuristics_only_accepted(self):
        from reflect_mcp.cli import run_extract

        pipeline = _make_heuristic_only_pipeline()

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--heuristics-only"])

        assert code == 0

    def test_extract_heuristics_only_output_valid_json(self):
        from reflect_mcp.cli import run_extract

        pipeline = _make_heuristic_only_pipeline()

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_extract(["--heuristics-only"])

        result = json.loads(captured.getvalue())
        assert result["status"] == "ok"

    def test_extract_heuristics_only_passes_critic_enabled_false(self):
        from reflect_mcp.cli import run_extract

        pipeline = _make_heuristic_only_pipeline()
        spy_kwargs = {}

        def spy_factory(**kwargs):
            spy_kwargs.update(kwargs)
            return pipeline

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_extract(["--heuristics-only"])

        assert spy_kwargs.get("critic_enabled") is False


# ---------------------------------------------------------------------------
# EB-4: validate CLI verb accepts --heuristics-only
# ---------------------------------------------------------------------------

class TestEB4ValidateHeuristicsOnlyFlag:
    """EB-4: validate accepts --heuristics-only with the same effect."""

    def test_validate_heuristics_only_accepted(self):
        from reflect_mcp.cli import run_validate

        pipeline = _make_heuristic_only_pipeline(gen_responses=[])

        data_json = json.dumps(CORRECT_EXTRACTION)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_validate(["--heuristics-only", "--data", data_json])

        assert code == 0

    def test_validate_heuristics_only_output_valid_json(self):
        from reflect_mcp.cli import run_validate

        pipeline = _make_heuristic_only_pipeline(gen_responses=[])

        data_json = json.dumps(CORRECT_EXTRACTION)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_validate(["--heuristics-only", "--data", data_json])

        result = json.loads(captured.getvalue())
        assert result["status"] == "ok"

    def test_validate_heuristics_only_passes_critic_enabled_false(self):
        from reflect_mcp.cli import run_validate

        pipeline = _make_heuristic_only_pipeline(gen_responses=[])
        spy_kwargs = {}

        def spy_factory(**kwargs):
            spy_kwargs.update(kwargs)
            return pipeline

        data_json = json.dumps(CORRECT_EXTRACTION)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_validate(["--heuristics-only", "--data", data_json])

        assert spy_kwargs.get("critic_enabled") is False


# ---------------------------------------------------------------------------
# EB-5: Heuristic penalty surfaces correctly in heuristic-only mode
# ---------------------------------------------------------------------------

class TestEB5HeuristicPenaltySurfaces:
    """EB-5: ungrounded field → combined 1.0 × 0.25 = 0.25 < 0.70 → needs_review."""

    def test_ungrounded_vendor_flagged(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        assert result.status == "needs_review"

    def test_flagged_field_is_vendor_name(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        flagged_names = [f.field for f in result.flagged_fields]
        assert "vendor_name" in flagged_names

    def test_vendor_name_confidence_below_threshold(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        assert result.field_confidence["vendor_name"] < 0.70

    def test_vendor_name_confidence_is_heuristic_factor(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        assert result.field_confidence["vendor_name"] == 0.25

    def test_other_fields_still_ok(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        for field, conf in result.field_confidence.items():
            if field != "vendor_name":
                assert conf == 1.0, f"{field} should be 1.0 but got {conf}"

    def test_arithmetic_penalty_flags_total(self):
        gen = _gen_response({"total": 999.99})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        assert result.status == "needs_review"
        flagged_names = [f.field for f in result.flagged_fields]
        assert "total" in flagged_names


# ---------------------------------------------------------------------------
# EB-6: critic_summary is empty string "" in heuristic-only mode
# ---------------------------------------------------------------------------

class TestEB6CriticSummaryEmpty:
    """EB-6: critic_summary in ExtractionResult is '' when critic=None."""

    def test_critic_summary_empty_string(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        assert result.critic_summary == ""

    def test_critic_summary_not_none(self):
        pipeline = _make_heuristic_only_pipeline()
        result = pipeline.extract(SOURCE)
        assert result.critic_summary is not None

    def test_critic_summary_empty_when_penalty_present(self):
        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])
        result = pipeline.extract(SOURCE)
        assert result.critic_summary == ""


# ---------------------------------------------------------------------------
# EB-7: --heuristics-only in --help output of both subcommands
# ---------------------------------------------------------------------------

class TestEB7HelpOutput:
    """EB-7: --heuristics-only appears in help text for extract and validate."""

    def test_extract_help_contains_flag(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "reflect_mcp", "extract", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd="/Users/jincm/projects/reflect-mcp",
        )
        assert r.returncode == 0
        assert "--heuristics-only" in r.stdout

    def test_validate_help_contains_flag(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "reflect_mcp", "validate", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd="/Users/jincm/projects/reflect-mcp",
        )
        assert r.returncode == 0
        assert "--heuristics-only" in r.stdout

    def test_extract_help_describes_heuristics_only(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "reflect_mcp", "extract", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd="/Users/jincm/projects/reflect-mcp",
        )
        lower = r.stdout.lower()
        assert "critic" in lower or "heuristic" in lower, (
            "Help text for --heuristics-only should mention critic or heuristics"
        )

    def test_validate_help_describes_heuristics_only(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "reflect_mcp", "validate", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd="/Users/jincm/projects/reflect-mcp",
        )
        lower = r.stdout.lower()
        assert "critic" in lower or "heuristic" in lower


# ---------------------------------------------------------------------------
# EB-8: Batch mode works with --heuristics-only
# ---------------------------------------------------------------------------

class TestEB8BatchModeHeuristicsOnly:
    """EB-8: --batch and --batch-concurrency compose with --heuristics-only."""

    def _make_jsonl(self, *texts: str) -> str:
        return "\n".join(json.dumps({"text": t}) for t in texts) + "\n"

    def test_serial_batch_heuristics_only(self):
        from reflect_mcp.cli import run_extract

        n = 3
        gen_responses = [_gen_response()] * n
        pipeline = _make_heuristic_only_pipeline(gen_responses=gen_responses)

        jsonl = self._make_jsonl(SOURCE, SOURCE, SOURCE)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--heuristics-only"])

        assert code == 0
        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 3
        for r in results:
            assert r["status"] == "ok"

    def test_concurrent_batch_heuristics_only(self):
        from reflect_mcp.cli import run_extract

        n = 3
        spy_kwargs_list = []

        def spy_factory(**kwargs):
            spy_kwargs_list.append(kwargs)
            gen_responses = [_gen_response()] * n
            return _make_heuristic_only_pipeline(gen_responses=gen_responses)

        jsonl = self._make_jsonl(SOURCE, SOURCE, SOURCE)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract([
                        "--batch", "--batch-concurrency", "2", "--heuristics-only"
                    ])

        assert code == 0
        results = _parse_output_lines(captured.getvalue())
        assert len(results) == 3

    def test_batch_heuristics_only_passes_critic_enabled_false(self):
        from reflect_mcp.cli import run_extract

        spy_kwargs_list = []

        def spy_factory(**kwargs):
            spy_kwargs_list.append(kwargs)
            return _make_heuristic_only_pipeline()

        jsonl = self._make_jsonl(SOURCE)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_extract(["--batch", "--heuristics-only"])

        assert any(
            kw.get("critic_enabled") is False for kw in spy_kwargs_list
        ), "batch path should pass critic_enabled=False"

    def test_concurrent_batch_heuristics_only_passes_critic_enabled_false(self):
        from reflect_mcp.cli import run_extract

        spy_kwargs_list = []

        def spy_factory(**kwargs):
            spy_kwargs_list.append(kwargs)
            gen_responses = [_gen_response()] * 2
            return _make_heuristic_only_pipeline(gen_responses=gen_responses)

        jsonl = self._make_jsonl(SOURCE, SOURCE)

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    run_extract([
                        "--batch", "--batch-concurrency", "2", "--heuristics-only"
                    ])

        assert any(
            kw.get("critic_enabled") is False for kw in spy_kwargs_list
        ), "concurrent batch path should pass critic_enabled=False"

    def test_serial_batch_heuristics_only_penalty_surfaces(self):
        from reflect_mcp.cli import run_extract

        gen = _gen_response({"vendor_name": "GLOBEX INC"})
        pipeline = _make_heuristic_only_pipeline(gen_responses=[gen])

        jsonl = self._make_jsonl(SOURCE)

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(jsonl)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_extract(["--batch", "--heuristics-only"])

        assert code == 1
        results = _parse_output_lines(captured.getvalue())
        assert results[0]["status"] == "needs_review"
