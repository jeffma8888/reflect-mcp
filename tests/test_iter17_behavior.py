"""Black-box behavioral tests for iter-17: CLI --field-threshold / --overall-threshold flags.

Covers all 5 Expected Behaviors from pm.md:
  1. --field-threshold 0.85 passes GateConfig(field_threshold=0.85) to build_default_pipeline
  2. --overall-threshold 0.80 passes GateConfig(overall_threshold=0.80) to build_default_pipeline
  3. Both flags combined into one GateConfig
  4. Omitting both flags uses defaults — build_default_pipeline called with no gate_config
  5. Out-of-range value (--field-threshold 1.5 or -0.1) produces exit 3 + JSON error envelope

Isolation contract: tests drive the public CLI interface (run_extract) and
inspect observable output / call args. No implementation source was read.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from reflect_mcp.gate import GateConfig
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_working_pipeline(critic_conf: float = 0.95, overall_conf: float = 0.95):
    """Build a fully wired ReflectivePipeline with scripted LLM clients."""
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


def _run_with_spy(cli_args: list[str], input_text: str = SOURCE):
    """Run run_extract with a spy on build_default_pipeline.

    Returns (exit_code, stdout_text, spy_calls) where spy_calls is a list
    of (args, kwargs) tuples recorded from build_default_pipeline invocations.
    """
    from reflect_mcp.cli import run_extract

    spy_calls: list[tuple] = []

    def spy_factory(*args, **kwargs):
        spy_calls.append((args, kwargs))
        return _make_working_pipeline()

    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
        with patch("sys.stdin", io.StringIO(input_text)):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract(cli_args)

    return code, captured.getvalue(), spy_calls


def _run_extract_simple(cli_args: list[str], input_text: str = SOURCE):
    """Run run_extract without mocking build_default_pipeline.

    Returns (exit_code, stdout_text).
    """
    from reflect_mcp.cli import run_extract

    with patch("sys.stdin", io.StringIO(input_text)):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            code = run_extract(cli_args)

    return code, captured.getvalue()


# ---------------------------------------------------------------------------
# EB-1: --field-threshold 0.85 forwarded as GateConfig
# ---------------------------------------------------------------------------

class TestEB1FieldThresholdForwarded:
    """EB-1: --field-threshold is forwarded as GateConfig to build_default_pipeline."""

    def test_field_threshold_forwarded(self):
        code, _output, spy_calls = _run_with_spy(["--field-threshold", "0.85"])
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None, "gate_config must be passed to build_default_pipeline"
        assert isinstance(gate_config, GateConfig)
        assert gate_config.field_threshold == pytest.approx(0.85)

    def test_field_threshold_preserves_default_overall(self):
        """When only --field-threshold is given, overall_threshold stays at default 0.75."""
        _code, _output, spy_calls = _run_with_spy(["--field-threshold", "0.85"])
        _args, kwargs = spy_calls[0]
        gate_config = kwargs["gate_config"]
        assert gate_config.overall_threshold == pytest.approx(0.75)

    def test_field_threshold_zero_forwarded(self):
        """--field-threshold 0.0 is a valid boundary and must be forwarded (not dropped)."""
        code, _output, spy_calls = _run_with_spy(["--field-threshold", "0.0"])
        assert code == 0
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None, "gate_config must be forwarded even for 0.0"
        assert gate_config.field_threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EB-2: --overall-threshold 0.80 forwarded as GateConfig
# ---------------------------------------------------------------------------

class TestEB2OverallThresholdForwarded:
    """EB-2: --overall-threshold is forwarded as GateConfig to build_default_pipeline."""

    def test_overall_threshold_forwarded(self):
        code, _output, spy_calls = _run_with_spy(["--overall-threshold", "0.80"])
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None, "gate_config must be passed to build_default_pipeline"
        assert isinstance(gate_config, GateConfig)
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_overall_threshold_preserves_default_field(self):
        """When only --overall-threshold is given, field_threshold stays at default 0.70."""
        _code, _output, spy_calls = _run_with_spy(["--overall-threshold", "0.80"])
        _args, kwargs = spy_calls[0]
        gate_config = kwargs["gate_config"]
        assert gate_config.field_threshold == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# EB-3: both flags combined into one GateConfig
# ---------------------------------------------------------------------------

class TestEB3BothFlagsCombined:
    """EB-3: both --field-threshold and --overall-threshold combine into one GateConfig."""

    def test_both_thresholds_forwarded(self):
        code, _output, spy_calls = _run_with_spy([
            "--field-threshold", "0.85",
            "--overall-threshold", "0.80",
        ])
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        gate_config = kwargs.get("gate_config")
        assert gate_config is not None
        assert isinstance(gate_config, GateConfig)
        assert gate_config.field_threshold == pytest.approx(0.85)
        assert gate_config.overall_threshold == pytest.approx(0.80)

    def test_combined_is_single_gate_config_kwarg(self):
        """Both values packaged in ONE gate_config kwarg, not separate threshold kwargs."""
        _, _, spy_calls = _run_with_spy([
            "--field-threshold", "0.85",
            "--overall-threshold", "0.80",
        ])
        _args, kwargs = spy_calls[0]
        assert "gate_config" in kwargs
        assert "field_threshold" not in kwargs
        assert "overall_threshold" not in kwargs


# ---------------------------------------------------------------------------
# EB-4: omitting both flags uses existing defaults
# ---------------------------------------------------------------------------

class TestEB4DefaultsWhenOmitted:
    """EB-4: omitting both flags → build_default_pipeline called without gate_config."""

    def test_no_flags_no_gate_config(self):
        """Without threshold flags, build_default_pipeline is called with no gate_config."""
        code, _output, spy_calls = _run_with_spy([])
        assert code == 0
        assert len(spy_calls) == 1
        _args, kwargs = spy_calls[0]
        assert "gate_config" not in kwargs or kwargs.get("gate_config") is None

    def test_defaults_produce_ok_with_high_confidence(self):
        """With critic confidence 0.95 and default thresholds, status is 'ok'."""
        code, output, _ = _run_with_spy([])
        assert code == 0
        parsed = json.loads(output)
        assert parsed["status"] == "ok"

    def test_file_flag_without_thresholds_no_gate_config(self, tmp_path):
        """--file alone does not trigger gate_config forwarding."""
        from reflect_mcp.cli import run_extract

        invoice_file = tmp_path / "invoice.txt"
        invoice_file.write_text(SOURCE, encoding="utf-8")

        spy_calls: list[tuple] = []

        def spy_factory(*args, **kwargs):
            spy_calls.append((args, kwargs))
            return _make_working_pipeline()

        with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=spy_factory):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = run_extract(["--file", str(invoice_file)])

        assert code == 0
        _args, kwargs = spy_calls[0]
        assert "gate_config" not in kwargs or kwargs.get("gate_config") is None


# ---------------------------------------------------------------------------
# EB-5: out-of-range threshold → exit 3 + JSON error envelope
# ---------------------------------------------------------------------------

class TestEB5OutOfRangeErrorEnvelope:
    """EB-5: out-of-range threshold → exit 3, JSON error envelope on stdout."""

    def test_field_threshold_above_one(self):
        code, output = _run_extract_simple(["--field-threshold", "1.5"])
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_field_threshold_negative(self):
        code, output = _run_extract_simple(["--field-threshold", "-0.1"])
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_overall_threshold_above_one(self):
        code, output = _run_extract_simple(["--overall-threshold", "1.5"])
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_overall_threshold_negative(self):
        code, output = _run_extract_simple(["--overall-threshold", "-0.1"])
        assert code == 3
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert "error_message" in parsed

    def test_error_envelope_has_expected_keys(self):
        """Error envelope matches iter-02 convention: exactly status + error_message."""
        code, output = _run_extract_simple(["--field-threshold", "1.5"])
        parsed = json.loads(output)
        assert set(parsed.keys()) == {"status", "error_message"}

    def test_error_message_not_traceback(self):
        """error_message is a clean string, not a full traceback."""
        _, output = _run_extract_simple(["--field-threshold", "1.5"])
        parsed = json.loads(output)
        assert "Traceback" not in parsed["error_message"]
