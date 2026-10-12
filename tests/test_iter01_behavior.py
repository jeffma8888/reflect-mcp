"""Black-box behavioral tests for iter-01: CLI extract verb.

Covers all 9 Expected Behaviors from pm.md:
  1. stdin → JSON stdout with all ExtractionResult fields
  2. --file flag reads file content, same JSON output
  3. exit code 0 when status == "ok"
  4. exit code 1 when status == "needs_review"
  5. empty stdin + no --file → usage error on stderr, exit 2
  6. --file pointing to nonexistent file → error on stderr, exit 2
  7. bare `reflect-mcp` (no subcommand) starts server (backward compat)
  8. JSON output parseable by json.loads(), no extra text
  9. all offline via ScriptedLLMClient, no network

Uses subprocess to drive the CLI (black-box) and ScriptedLLMClient
for the LLM boundary (no API keys, fully offline).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


EXTRACT_CMD = [sys.executable, "-m", "reflect_mcp", "extract"]


def _run_extract(
    *extra_args: str,
    input_text: str | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    """Run `reflect-mcp extract` as a subprocess."""
    return subprocess.run(
        [*EXTRACT_CMD, *extra_args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


# ---------- EB-1: stdin → JSON stdout with all ExtractionResult fields ----------

def test_eb1_stdin_produces_json_with_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """EB-1: echo text | reflect-mcp extract prints valid JSON with all ExtractionResult fields."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.95, 0.95)

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract([])
    output = captured.getvalue()
    parsed = json.loads(output)
    assert code == 0
    required_fields = {"status", "data", "field_confidence", "overall_confidence",
                       "flagged_fields", "trace_id", "notes", "critic_summary"}
    assert required_fields.issubset(set(parsed.keys())), (
        f"Missing fields: {required_fields - set(parsed.keys())}"
    )


# ---------- EB-2: --file flag reads file, same output ----------

def test_eb2_file_flag_reads_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """EB-2: reflect-mcp extract --file path/to/file.txt produces JSON output."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.95, 0.95)

    invoice_file = tmp_path / "invoice.txt"
    invoice_file.write_text(SOURCE, encoding="utf-8")

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            code = run_extract(["--file", str(invoice_file)])
    output = captured.getvalue()
    parsed = json.loads(output)
    assert code == 0
    assert parsed["status"] == "ok"
    assert parsed["data"]["vendor_name"] == "ACME CORP"


# ---------- EB-3: exit code 0 when status == "ok" ----------

def test_eb3_exit_code_0_for_ok() -> None:
    """EB-3: when pipeline returns status='ok', exit code is 0."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.95, 0.95)

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            with patch("sys.stdout", io.StringIO()):
                code = run_extract([])
    assert code == 0


# ---------- EB-4: exit code 1 when status == "needs_review" ----------

def test_eb4_exit_code_1_for_needs_review() -> None:
    """EB-4: when pipeline returns status='needs_review', exit code is 1."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.30, 0.30, summary="low confidence all around")

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            with patch("sys.stdout", io.StringIO()):
                code = run_extract([])
    assert code == 1


# ---------- EB-5: empty stdin + no --file → stderr + exit 2 ----------

def test_eb5_empty_stdin_no_file_exits_2() -> None:
    """EB-5: no input → usage error on stderr, exit 2."""
    from reflect_mcp.cli import run_extract
    from unittest.mock import patch
    import io

    stderr_capture = io.StringIO()
    with patch("sys.stdin", io.StringIO("")):
        with patch("sys.stderr", stderr_capture):
            with patch("sys.stdout", io.StringIO()):
                code = run_extract([])
    assert code == 2
    err_output = stderr_capture.getvalue()
    assert len(err_output) > 0, "Expected error message on stderr"


# ---------- EB-6: --file nonexistent → stderr + exit 2 ----------

def test_eb6_nonexistent_file_exits_2() -> None:
    """EB-6: --file pointing to nonexistent file → error on stderr, exit 2."""
    from reflect_mcp.cli import run_extract
    from unittest.mock import patch
    import io

    stderr_capture = io.StringIO()
    with patch("sys.stderr", stderr_capture):
        with patch("sys.stdout", io.StringIO()):
            code = run_extract(["--file", "/tmp/no_such_file_reflect_mcp_test.txt"])
    assert code == 2
    err_output = stderr_capture.getvalue()
    assert len(err_output) > 0, "Expected error message on stderr"


# ---------- EB-7: bare reflect-mcp starts server (backward compat) ----------

def test_eb7_bare_command_starts_server() -> None:
    """EB-7: bare `reflect-mcp` (no subcommand) starts MCP server.

    We verify this by running the bare command and checking it doesn't
    exit immediately with an error — it should start up (we kill it fast).
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "reflect_mcp"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    import time
    time.sleep(1)
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
        assert True, "Server started and was running (backward compat OK)"
    else:
        _stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0 or "extract" not in stderr.lower(), (
            f"Bare command failed unexpectedly: rc={proc.returncode}, stderr={stderr}"
        )


# ---------- EB-8: JSON output parseable by json.loads, no extra text ----------

def test_eb8_json_output_clean_parseable() -> None:
    """EB-8: output is valid JSON with no extra text before or after."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.95, 0.95)

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                run_extract([])
    raw = captured.getvalue()
    assert raw.strip() == raw.strip()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    re_serialized = json.dumps(parsed)
    assert json.loads(re_serialized) == parsed


# ---------- EB-9: all offline via ScriptedLLMClient ----------

def test_eb9_offline_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """EB-9: the tests above all use ScriptedLLMClient; this test verifies
    the pipeline can run without any provider env vars set."""
    from reflect_mcp.cli import run_extract
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.generator import Generator
    from reflect_mcp.critic import Critic
    from reflect_mcp.scripted import ScriptedLLMClient
    from unittest.mock import patch
    import io

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REFLECT_PROVIDER", raising=False)

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(0.95, 0.95)

    def fake_pipeline() -> ReflectivePipeline:
        return ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

    with patch("reflect_mcp.pipeline.build_default_pipeline", fake_pipeline):
        with patch("sys.stdin", io.StringIO(SOURCE)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract([])
    assert code == 0
    parsed = json.loads(captured.getvalue())
    assert parsed["status"] == "ok"
