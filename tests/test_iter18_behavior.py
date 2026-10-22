"""Black-box behavioral tests for iter-18: validate verb (verify-only mode).

Covers all 6 Expected Behaviors from pm.md:
  1. `reflect-mcp validate --file f --data '{...}'` constructs InvoiceData, runs
     Critic+heuristics+gate (no Generator), prints ExtractionResult JSON to stdout.
  2. `--data` is required; omitting it prints usage error to stderr, exit 2.
  3. Invalid JSON or bad InvoiceData fields in --data → stderr error, exit 2.
  4. Exit codes: 0=ok, 1=needs_review, 2=input error, 3=pipeline error.
  5. `--field-threshold` / `--overall-threshold` forward to GateConfig, identical to extract.
  6. MCP `validate(text, data, ...)` tool returns ExtractionResult dict.

Isolation contract: tests drive the public CLI interface (run_validate) and
the public pipeline/MCP surface. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from reflect_mcp.gate import GateConfig
from reflect_mcp.schema import InvoiceData
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DATA_JSON = json.dumps(CORRECT_EXTRACTION)

PARTIAL_DATA = {"vendor_name": "ACME CORP", "total": 108.5}
PARTIAL_DATA_JSON = json.dumps(PARTIAL_DATA)


def _make_working_pipeline(critic_conf: float = 0.95, overall_conf: float = 0.95):
    """Build a ReflectivePipeline with scripted LLM clients (Critic only needed for validate)."""
    from reflect_mcp.critic import Critic
    from reflect_mcp.generator import Generator
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.scripted import ScriptedLLMClient

    gen_json = json.dumps(CORRECT_EXTRACTION)
    crit_json = critique_all(critic_conf, overall_conf)
    return ReflectivePipeline(
        Generator(ScriptedLLMClient([gen_json])),
        Critic(ScriptedLLMClient([crit_json])),
    )


def _run_validate_with_spy(cli_args: list[str], input_text: str = SOURCE):
    """Run run_validate with a spy on build_default_pipeline.

    Returns (exit_code, stdout_text, stderr_text, spy_calls).
    """
    from reflect_mcp.cli import run_validate

    spy_calls: list[tuple] = []

    def spy_factory(*args, **kwargs):
        spy_calls.append((args, kwargs))
        return _make_working_pipeline()

    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
        with patch("sys.stdin", io.StringIO(input_text)):
            captured_out = io.StringIO()
            captured_err = io.StringIO()
            with patch("sys.stdout", captured_out), patch("sys.stderr", captured_err):
                try:
                    code = run_validate(cli_args)
                except SystemExit as exc:
                    code = exc.code

    return code, captured_out.getvalue(), captured_err.getvalue(), spy_calls


def _run_validate_simple(cli_args: list[str], input_text: str = SOURCE):
    """Run run_validate without mocking build_default_pipeline.

    Returns (exit_code, stdout_text, stderr_text).
    """
    from reflect_mcp.cli import run_validate

    with patch("sys.stdin", io.StringIO(input_text)):
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with patch("sys.stdout", captured_out), patch("sys.stderr", captured_err):
            try:
                code = run_validate(cli_args)
            except SystemExit as exc:
                code = exc.code

    return code, captured_out.getvalue(), captured_err.getvalue()


# ---------------------------------------------------------------------------
# EB-1: validate --file --data → ExtractionResult JSON, no Generator
# ---------------------------------------------------------------------------

class TestEB1ValidateProducesExtractionResult:
    """EB-1: validate constructs InvoiceData, runs verify-only, prints ExtractionResult JSON."""

    def test_validate_prints_json_to_stdout(self, tmp_path):
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        code, stdout, _stderr, _spy = _run_validate_with_spy(
            ["--file", str(invoice_file), "--data", VALID_DATA_JSON]
        )
        assert code == 0
        parsed = json.loads(stdout)
        assert "status" in parsed
        assert parsed["status"] in ("ok", "needs_review")

    def test_validate_result_has_extraction_result_keys(self, tmp_path):
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        code, stdout, _stderr, _spy = _run_validate_with_spy(
            ["--file", str(invoice_file), "--data", VALID_DATA_JSON]
        )
        assert code == 0
        parsed = json.loads(stdout)
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(parsed.keys())

    def test_validate_data_reflects_provided_fields(self, tmp_path):
        """The output data should contain the fields from the provided --data."""
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        code, stdout, _stderr, _spy = _run_validate_with_spy(
            ["--file", str(invoice_file), "--data", VALID_DATA_JSON]
        )
        assert code == 0
        parsed = json.loads(stdout)
        assert parsed["data"]["vendor_name"] == "ACME CORP"
        assert parsed["data"]["total"] == pytest.approx(108.5)

    def test_validate_stdin_mode(self):
        """validate reads text from stdin when --file is not given."""
        code, stdout, _stderr, _spy = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON], input_text=SOURCE
        )
        assert code == 0
        parsed = json.loads(stdout)
        assert parsed["status"] in ("ok", "needs_review")

    def test_validate_does_not_call_generator(self, tmp_path):
        """The Generator should not be invoked during validate (verify-only)."""
        from reflect_mcp.critic import Critic
        from reflect_mcp.generator import Generator
        from reflect_mcp.pipeline import ReflectivePipeline
        from reflect_mcp.scripted import ScriptedLLMClient

        crit_json = critique_all(0.95, 0.95)
        gen_client = ScriptedLLMClient([])
        crit_client = ScriptedLLMClient([crit_json])
        pipeline = ReflectivePipeline(Generator(gen_client), Critic(crit_client))

        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    from reflect_mcp.cli import run_validate
                    code = run_validate(["--file", str(invoice_file), "--data", VALID_DATA_JSON])

        assert code == 0


# ---------------------------------------------------------------------------
# EB-2: --data is required; omitting it → exit 2
# ---------------------------------------------------------------------------

class TestEB2DataRequired:
    """EB-2: --data is required; omitting it prints usage error to stderr, exit 2."""

    def test_missing_data_exits_2(self, tmp_path):
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--file", str(invoice_file)]
        )
        assert code == 2

    def test_missing_data_prints_to_stderr(self, tmp_path):
        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--file", str(invoice_file)]
        )
        assert code == 2
        assert stderr.strip() != "", "Missing --data should produce stderr output"


# ---------------------------------------------------------------------------
# EB-3: invalid JSON or bad InvoiceData fields → exit 2
# ---------------------------------------------------------------------------

class TestEB3InvalidDataExits2:
    """EB-3: invalid JSON or bad InvoiceData fields → stderr error, exit 2."""

    def test_invalid_json_exits_2(self):
        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--data", "not valid json"]
        )
        assert code == 2

    def test_invalid_json_stderr(self):
        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--data", "not valid json"]
        )
        assert stderr.strip() != "", "Invalid JSON should produce stderr output"

    def test_bad_invoice_fields_exits_2(self):
        """InvoiceData validation rejects bad types (e.g., total as a string)."""
        bad_data = json.dumps({"total": "not a number"})
        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--data", bad_data]
        )
        assert code == 2

    def test_bad_invoice_fields_stderr(self):
        bad_data = json.dumps({"total": "not a number"})
        code, _stdout, stderr, _spy = _run_validate_with_spy(
            ["--data", bad_data]
        )
        assert stderr.strip() != "", "Bad InvoiceData fields should produce stderr output"


# ---------------------------------------------------------------------------
# EB-4: exit codes: 0=ok, 1=needs_review, 2=input error, 3=pipeline error
# ---------------------------------------------------------------------------

class TestEB4ExitCodes:
    """EB-4: exit codes follow the extract contract."""

    def test_ok_exits_0(self):
        code, stdout, _stderr, _spy = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON]
        )
        assert code == 0
        parsed = json.loads(stdout)
        assert parsed["status"] == "ok"

    def test_needs_review_exits_1(self):
        """Low confidence triggers needs_review → exit 1."""
        from reflect_mcp.cli import run_validate
        from reflect_mcp.critic import Critic
        from reflect_mcp.generator import Generator
        from reflect_mcp.pipeline import ReflectivePipeline
        from reflect_mcp.scripted import ScriptedLLMClient

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(0.3, 0.3)
        pipeline = ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )

        with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=pipeline):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_validate(["--data", VALID_DATA_JSON])

        assert code == 1
        parsed = json.loads(captured.getvalue())
        assert parsed["status"] == "needs_review"

    def test_input_error_exits_2(self):
        code, _stdout, _stderr, _spy = _run_validate_with_spy(
            ["--data", "bad json"]
        )
        assert code == 2

    def test_pipeline_error_exits_3(self):
        """Pipeline exception → exit 3 with error envelope."""
        from reflect_mcp.cli import run_validate

        def broken_factory(*args, **kwargs):
            raise RuntimeError("critic exploded")

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=broken_factory):
            with patch("sys.stdin", io.StringIO(SOURCE)):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    code = run_validate(["--data", VALID_DATA_JSON])

        assert code == 3
        parsed = json.loads(captured.getvalue())
        assert parsed["status"] == "error"
        assert "error_message" in parsed


# ---------------------------------------------------------------------------
# EB-5: --field-threshold / --overall-threshold forward to GateConfig
# ---------------------------------------------------------------------------

class TestEB5ThresholdForwarding:
    """EB-5: threshold flags forward to GateConfig, identical to extract."""

    def test_field_threshold_forwarded(self):
        code, _stdout, _stderr, spy_calls = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON, "--field-threshold", "0.85"]
        )
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert isinstance(gate_config, GateConfig)
        assert gate_config.field_threshold == pytest.approx(0.85)

    def test_overall_threshold_forwarded(self):
        code, _stdout, _stderr, spy_calls = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON, "--overall-threshold", "0.80"]
        )
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert isinstance(gate_config, GateConfig)
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_both_thresholds_combined(self):
        code, _stdout, _stderr, spy_calls = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON, "--field-threshold", "0.85", "--overall-threshold", "0.80"]
        )
        assert code == 0
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert gate_config.field_threshold == pytest.approx(0.85)
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_no_thresholds_no_gate_config(self):
        """Without threshold flags, build_default_pipeline called with no gate_config."""
        code, _stdout, _stderr, spy_calls = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON]
        )
        assert code == 0
        _args, kwargs = spy_calls[0]
        assert "gate_config" not in kwargs or kwargs.get("gate_config") is None

    def test_field_threshold_zero_forwarded(self):
        """--field-threshold 0.0 is a valid boundary and must be forwarded."""
        code, _stdout, _stderr, spy_calls = _run_validate_with_spy(
            ["--data", VALID_DATA_JSON, "--field-threshold", "0.0"]
        )
        assert code == 0
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None, "gate_config must be forwarded even for 0.0"
        assert gate_config.field_threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EB-6: MCP validate tool returns ExtractionResult dict
# ---------------------------------------------------------------------------

class TestEB6MCPValidateTool:
    """EB-6: MCP validate(text, data, ...) tool returns ExtractionResult dict."""

    def test_pipeline_validate_method_exists(self):
        """ReflectivePipeline has a validate() method."""
        from reflect_mcp.pipeline import ReflectivePipeline
        assert hasattr(ReflectivePipeline, "validate")
        assert callable(getattr(ReflectivePipeline, "validate"))

    def test_pipeline_validate_returns_extraction_result(self):
        """pipeline.validate(text, data) returns an ExtractionResult."""
        pipeline = _make_working_pipeline()
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = pipeline.validate(SOURCE, data)
        assert result.status in ("ok", "needs_review")
        expected_keys = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "critic_summary", "trace_id", "notes",
        }
        assert expected_keys == set(result.model_dump().keys())

    def test_pipeline_validate_high_confidence_ok(self):
        """High-confidence validate returns status='ok'."""
        pipeline = _make_working_pipeline(critic_conf=0.95, overall_conf=0.95)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = pipeline.validate(SOURCE, data)
        assert result.status == "ok"

    def test_pipeline_validate_low_confidence_needs_review(self):
        """Low-confidence validate returns status='needs_review'."""
        pipeline = _make_working_pipeline(critic_conf=0.3, overall_conf=0.3)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = pipeline.validate(SOURCE, data)
        assert result.status == "needs_review"

    def test_pipeline_validate_result_data_matches_input(self):
        """The result data should match the provided InvoiceData."""
        pipeline = _make_working_pipeline()
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = pipeline.validate(SOURCE, data)
        d = result.model_dump()
        assert d["data"]["vendor_name"] == "ACME CORP"
        assert d["data"]["total"] == pytest.approx(108.5)

    def test_pipeline_validate_result_model_dump(self):
        """The result can be serialized to dict (as the MCP tool would do)."""
        pipeline = _make_working_pipeline()
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = pipeline.validate(SOURCE, data)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["status"] in ("ok", "needs_review")
