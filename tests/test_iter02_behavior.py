"""Black-box behavioral tests for iter-02: structured error envelope on pipeline failure.

Covers all 6 Expected Behaviors from pm.md:
  1. Pipeline exception → JSON {"status":"error","error_message":"..."} on stdout, exit 3
  2. Error envelope is valid JSON parseable by json.loads(), no extra text
  3. error_message contains exception string, not full traceback
  4. Existing exit codes preserved: 0=ok, 1=needs_review, 2=usage error
  5. MCP ExtractionResult return type unchanged (no "error" status)
  6. All offline via ScriptedLLMClient / mocked pipeline, no network

Isolation contract: tests drive the public CLI interface (run_extract) and
inspect observable output. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli_with_pipeline_error(
    exception: Exception,
    input_text: str = SOURCE,
) -> tuple[int, str]:
    """Run `run_extract` with a pipeline whose .extract() raises `exception`.

    Returns (exit_code, stdout_text).
    """
    from reflect_mcp.cli import run_extract

    mock_pipeline = MagicMock()
    mock_pipeline.extract.side_effect = exception

    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=mock_pipeline):
        with patch("sys.stdin", io.StringIO(input_text)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract([])
    return code, captured.getvalue()


def _run_cli_with_constructor_error(exception: Exception) -> tuple[int, str]:
    """Run `run_extract` where build_default_pipeline() itself raises."""
    from reflect_mcp.cli import run_extract

    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=exception):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract([])
    return code, captured.getvalue()


# ---------------------------------------------------------------------------
# EB-1: Pipeline exception → JSON error on stdout, exit code 3
# ---------------------------------------------------------------------------

class TestEB1ErrorEnvelopeAndExitCode:
    """EB-1: pipeline exception produces JSON error envelope on stdout + exit 3."""

    def test_extract_raises_runtime_error(self) -> None:
        code, output = _run_cli_with_pipeline_error(
            RuntimeError("model endpoint unreachable")
        )
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_extract_raises_value_error(self) -> None:
        code, output = _run_cli_with_pipeline_error(
            ValueError("invalid JSON in model response")
        )
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_constructor_raises(self) -> None:
        """build_default_pipeline() itself raises (e.g. bad provider config)."""
        code, output = _run_cli_with_constructor_error(
            RuntimeError("REFLECT_PROVIDER not set")
        )
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_envelope_has_exactly_two_keys(self) -> None:
        """Spec: envelope is {"status": "error", "error_message": "..."}."""
        code, output = _run_cli_with_pipeline_error(
            RuntimeError("some failure")
        )
        parsed = json.loads(output)
        assert set(parsed.keys()) == {"status", "error_message"}


# ---------------------------------------------------------------------------
# EB-2: Error envelope is valid JSON, no extra text
# ---------------------------------------------------------------------------

class TestEB2ValidJSON:
    """EB-2: error envelope is valid JSON with no extra text before or after."""

    def test_no_extra_text_before_json(self) -> None:
        _, output = _run_cli_with_pipeline_error(RuntimeError("fail"))
        stripped = output.strip()
        assert stripped.startswith("{"), (
            f"Expected JSON to start with '{{', got: {stripped[:50]!r}"
        )

    def test_no_extra_text_after_json(self) -> None:
        _, output = _run_cli_with_pipeline_error(RuntimeError("fail"))
        stripped = output.strip()
        assert stripped.endswith("}"), (
            f"Expected JSON to end with '}}', got: ...{stripped[-50:]!r}"
        )

    def test_single_json_object_on_stdout(self) -> None:
        """Stdout contains exactly one JSON value, nothing else."""
        _, output = _run_cli_with_pipeline_error(RuntimeError("fail"))
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        re_encoded = json.dumps(parsed)
        assert json.loads(re_encoded) == parsed


# ---------------------------------------------------------------------------
# EB-3: error_message is exception string, not traceback
# ---------------------------------------------------------------------------

class TestEB3NoTraceback:
    """EB-3: error_message contains the exception string, not a traceback."""

    def test_error_message_matches_exception(self) -> None:
        msg = "provider returned HTTP 429"
        _, output = _run_cli_with_pipeline_error(RuntimeError(msg))
        parsed = json.loads(output)
        assert msg in parsed["error_message"]

    def test_no_traceback_keyword(self) -> None:
        _, output = _run_cli_with_pipeline_error(
            RuntimeError("something broke")
        )
        parsed = json.loads(output)
        assert "Traceback" not in parsed["error_message"]

    def test_no_file_line_pattern(self) -> None:
        _, output = _run_cli_with_pipeline_error(
            RuntimeError("something broke")
        )
        parsed = json.loads(output)
        assert 'File "' not in parsed["error_message"]

    def test_empty_exception_message(self) -> None:
        """Exception() with no args: envelope still produced, exit code 3."""
        code, output = _run_cli_with_pipeline_error(Exception())
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed


# ---------------------------------------------------------------------------
# EB-4: Existing exit codes preserved (0, 1, 2)
# ---------------------------------------------------------------------------

class TestEB4ExistingExitCodes:
    """EB-4: exit 0 for ok, 1 for needs_review, 2 for usage errors — unchanged."""

    def _make_fake_pipeline_factory(self, critic_conf: float, overall_conf: float):
        from reflect_mcp.pipeline import ReflectivePipeline
        from reflect_mcp.generator import Generator
        from reflect_mcp.critic import Critic
        from reflect_mcp.scripted import ScriptedLLMClient

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(critic_conf, overall_conf)

        def factory() -> ReflectivePipeline:
            return ReflectivePipeline(
                Generator(ScriptedLLMClient([gen_json])),
                Critic(ScriptedLLMClient([crit_json])),
            )
        return factory

    def test_exit_0_for_ok(self) -> None:
        from reflect_mcp.cli import run_extract

        with patch("reflect_mcp.pipeline.build_default_pipeline",
                    self._make_fake_pipeline_factory(0.95, 0.95)):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                with patch("sys.stdout", io.StringIO()):
                    code = run_extract([])
        assert code == 0

    def test_exit_1_for_needs_review(self) -> None:
        from reflect_mcp.cli import run_extract

        with patch("reflect_mcp.pipeline.build_default_pipeline",
                    self._make_fake_pipeline_factory(0.30, 0.30)):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                with patch("sys.stdout", io.StringIO()):
                    code = run_extract([])
        assert code == 1

    def test_exit_2_for_no_input(self) -> None:
        from reflect_mcp.cli import run_extract

        with patch("sys.stdin", io.StringIO("")):
            with patch("sys.stderr", io.StringIO()):
                with patch("sys.stdout", io.StringIO()):
                    code = run_extract([])
        assert code == 2

    def test_exit_2_for_missing_file(self) -> None:
        from reflect_mcp.cli import run_extract

        with patch("sys.stderr", io.StringIO()):
            with patch("sys.stdout", io.StringIO()):
                code = run_extract(["--file", "/tmp/__no_such_file_iter02__.txt"])
        assert code == 2


# ---------------------------------------------------------------------------
# EB-5: MCP ExtractionResult unchanged (no "error" status literal)
# ---------------------------------------------------------------------------

class TestEB5MCPUnchanged:
    """EB-5: MCP ExtractionResult has no 'error' status — envelope is CLI-only."""

    def test_extraction_result_status_literals(self) -> None:
        """ExtractionResult.status should accept 'ok' and 'needs_review' only."""
        from reflect_mcp.schema import ExtractionResult
        import inspect

        sig = inspect.signature(ExtractionResult)
        status_param = sig.parameters.get("status")
        assert status_param is not None, "ExtractionResult must have a 'status' parameter"

        hints = ExtractionResult.__annotations__
        if "status" in hints:
            status_hint = hints["status"]
            hint_str = str(status_hint)
            assert "error" not in hint_str, (
                f"ExtractionResult.status should NOT include 'error': {hint_str}"
            )


# ---------------------------------------------------------------------------
# EB-6: All offline, no network (implicit but verified)
# ---------------------------------------------------------------------------

class TestEB6Offline:
    """EB-6: all tests run offline with mocked pipeline, no API keys needed."""

    def test_error_path_offline_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error path works with no provider env vars set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("REFLECT_PROVIDER", raising=False)

        code, output = _run_cli_with_pipeline_error(
            RuntimeError("simulated failure")
        )
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
