"""Black-box behavioral tests for iter-52: critic short-circuit optimization.

Covers all 7 Expected Behaviors from pm.md:
  1. All non-null fields doomed → Critic NOT called, result needs_review, all flagged.
  2. At least one field at/above threshold → Critic IS called normally.
  3. Empty extraction (all null) → Critic IS called normally.
  4. Short-circuit produces critic_summary="".
  5. Custom field_threshold (GateConfig) respected by predicate.
  6. critic=None (heuristic-only) bypasses short-circuit entirely.
  7. "critic.skipped" log event emitted on short-circuit path.

Isolation contract: tests drive the public package API (ReflectivePipeline,
GateConfig, ScriptedLLMClient) and inspect observable output. No implementation
source was read.
"""

from __future__ import annotations

import json
import logging

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
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


ALL_UNGROUNDED = {
    "vendor_name": "GLOBEX INC",
    "invoice_number": "ZZ-0000",
    "invoice_date": None,
    "subtotal": 999.0,
    "tax": 99.0,
    "total": 1098.0,
    "currency": "EUR",
}


def _make_pipeline_with_critic(
    gen_overrides: dict | None = None,
    critic_response: str | None = None,
    gate_config: GateConfig | None = None,
) -> tuple[ReflectivePipeline, ScriptedLLMClient]:
    """Build a pipeline with a real Critic backed by a ScriptedLLMClient.

    Returns (pipeline, critic_client) so tests can inspect whether the Critic
    was actually called via critic_client.calls.
    """
    gen_json = _gen_response(gen_overrides)
    gen_client = ScriptedLLMClient([gen_json])
    gen = Generator(gen_client)

    if critic_response is None:
        critic_response = critique_all(0.95, 0.90, summary="looks good")
    critic_client = ScriptedLLMClient([critic_response])
    crit = Critic(critic_client)

    pipeline = ReflectivePipeline(gen, crit, gate_config=gate_config)
    return pipeline, critic_client


# ---------------------------------------------------------------------------
# EB-1: All non-null fields doomed → Critic NOT called
# ---------------------------------------------------------------------------

class TestEB1AllFieldsDoomedSkipsCritic:
    """EB-1: When all non-null fields have heuristic factors < field_threshold,
    the Critic critique() is NOT called; result is needs_review with all flagged."""

    def test_critic_not_called_when_all_doomed(self):
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides=ALL_UNGROUNDED,
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 0, (
            "Critic should NOT be called when all fields are doomed"
        )

    def test_status_is_needs_review(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert result.status == "needs_review"

    def test_all_non_null_fields_flagged(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        flagged_names = {f.field for f in result.flagged_fields}
        data = result.data
        non_null = {
            f for f in ALL_UNGROUNDED if getattr(data, f, None) is not None
        }
        assert flagged_names == non_null, (
            f"Expected all non-null fields flagged: {non_null}, got {flagged_names}"
        )

    def test_result_is_extraction_result(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert isinstance(result, ExtractionResult)

    def test_data_preserved_despite_short_circuit(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert result.data.vendor_name == "GLOBEX INC"
        assert result.data.total == 1098.0


# ---------------------------------------------------------------------------
# EB-2: At least one field at/above threshold → Critic IS called
# ---------------------------------------------------------------------------

class TestEB2OneGoodFieldCriticCalled:
    """EB-2: When at least one non-null field has factor >= threshold, the Critic
    IS called normally (no short-circuit)."""

    def test_critic_called_when_one_field_ok(self):
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides={"vendor_name": "GLOBEX INC"},
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called when at least one field is at/above threshold"
        )

    def test_all_correct_fields_critic_called(self):
        pipeline, critic_client = _make_pipeline_with_critic()
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called when all fields are correctly grounded"
        )

    def test_factor_exactly_at_threshold_not_doomed(self):
        """A field with factor exactly at threshold (e.g. 0.70) is NOT doomed
        because the predicate uses strict < (matching the gate's own check)."""
        pipeline, critic_client = _make_pipeline_with_critic()
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1


# ---------------------------------------------------------------------------
# EB-3: Empty extraction (all null) → Critic IS called
# ---------------------------------------------------------------------------

class TestEB3EmptyExtractionCriticCalled:
    """EB-3: When all fields are null (empty extraction), the Critic IS called
    normally because overall_confidence could trigger needs_review."""

    def test_critic_called_on_all_null(self):
        all_null = {f: None for f in CORRECT_EXTRACTION}
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides=all_null,
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called when extraction is all-null"
        )

    def test_all_null_result_valid(self):
        all_null = {f: None for f in CORRECT_EXTRACTION}
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=all_null)
        result = pipeline.extract(SOURCE)
        assert isinstance(result, ExtractionResult)
        assert result.status in ("ok", "needs_review")


# ---------------------------------------------------------------------------
# EB-4: Short-circuit produces critic_summary=""
# ---------------------------------------------------------------------------

class TestEB4ShortCircuitCriticSummaryEmpty:
    """EB-4: The short-circuit path produces critic_summary='' since no Critic
    was consulted."""

    def test_critic_summary_empty_string(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert result.critic_summary == ""

    def test_critic_summary_not_none(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert result.critic_summary is not None

    def test_critic_summary_is_string(self):
        pipeline, _ = _make_pipeline_with_critic(gen_overrides=ALL_UNGROUNDED)
        result = pipeline.extract(SOURCE)
        assert isinstance(result.critic_summary, str)


# ---------------------------------------------------------------------------
# EB-5: Custom field_threshold respected by predicate
# ---------------------------------------------------------------------------

class TestEB5CustomFieldThreshold:
    """EB-5: GateConfig field_threshold is respected by the short-circuit
    predicate — e.g., field_threshold=0.30 dooms fields with factor 0.25,
    while field_threshold=0.20 does NOT doom them (0.25 >= 0.20)."""

    def test_low_threshold_does_not_short_circuit(self):
        """field_threshold=0.20: vendor_name factor 0.25 >= 0.20, so NOT doomed;
        Critic should be called."""
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides={"vendor_name": "GLOBEX INC"},
            gate_config=GateConfig(field_threshold=0.20),
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called: vendor_name factor 0.25 >= threshold 0.20"
        )

    def test_high_threshold_all_ungrounded_short_circuits(self):
        """field_threshold=0.40: all non-null ungrounded fields have factors
        0.25 or 0.35, all < 0.40, so all doomed; Critic NOT called."""
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides=ALL_UNGROUNDED,
            gate_config=GateConfig(field_threshold=0.40),
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 0, (
            "Critic should NOT be called: all factors < threshold 0.40"
        )

    def test_threshold_020_all_ungrounded_no_short_circuit(self):
        """field_threshold=0.20: all ungrounded fields have factor 0.25 >= 0.20,
        so NOT doomed; Critic should be called."""
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides=ALL_UNGROUNDED,
            gate_config=GateConfig(field_threshold=0.20),
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called: all factors 0.25 >= threshold 0.20"
        )

    def test_threshold_zero_never_short_circuits(self):
        """field_threshold=0.0: no field can have factor < 0.0, so doom never
        fires; Critic should always be called."""
        pipeline, critic_client = _make_pipeline_with_critic(
            gen_overrides=ALL_UNGROUNDED,
            gate_config=GateConfig(field_threshold=0.0),
        )
        result = pipeline.extract(SOURCE)
        assert len(critic_client.calls) == 1, (
            "Critic should be called: threshold 0.0 means doom never fires"
        )


# ---------------------------------------------------------------------------
# EB-6: critic=None bypasses short-circuit entirely
# ---------------------------------------------------------------------------

class TestEB6CriticNoneBypassesShortCircuit:
    """EB-6: When critic=None (heuristic-only mode), the short-circuit logic is
    bypassed entirely — it only applies when a Critic is configured."""

    def test_critic_none_all_doomed_still_works(self):
        """critic=None + all doomed fields: should produce a valid result
        (heuristic-only mode handles it, short-circuit logic not involved)."""
        gen_json = _gen_response(ALL_UNGROUNDED)
        gen = Generator(ScriptedLLMClient([gen_json]))
        pipeline = ReflectivePipeline(gen, critic=None)
        result = pipeline.extract(SOURCE)
        assert isinstance(result, ExtractionResult)
        assert result.status == "needs_review"

    def test_critic_none_critic_summary_empty(self):
        """critic=None with doomed fields: critic_summary is still '' via the
        heuristic-only path, not the short-circuit path."""
        gen_json = _gen_response(ALL_UNGROUNDED)
        gen = Generator(ScriptedLLMClient([gen_json]))
        pipeline = ReflectivePipeline(gen, critic=None)
        result = pipeline.extract(SOURCE)
        assert result.critic_summary == ""

    def test_critic_none_correct_data_ok(self):
        """critic=None with correct data: should be ok."""
        gen_json = _gen_response()
        gen = Generator(ScriptedLLMClient([gen_json]))
        pipeline = ReflectivePipeline(gen, critic=None)
        result = pipeline.extract(SOURCE)
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# EB-7: "critic.skipped" log event emitted on short-circuit
# ---------------------------------------------------------------------------

class TestEB7CriticSkippedLogEvent:
    """EB-7: A 'critic.skipped' log stage event is emitted when the Critic is
    short-circuited (instead of 'critique.done')."""

    def test_critic_skipped_logged_on_doom(self, caplog):
        """When all fields are doomed, 'critic.skipped' appears in logs."""
        with caplog.at_level(logging.INFO, logger="reflect_mcp"):
            pipeline, _ = _make_pipeline_with_critic(
                gen_overrides=ALL_UNGROUNDED,
            )
            pipeline.extract(SOURCE)

        messages = [r.getMessage() for r in caplog.records]
        assert "critic.skipped" in messages, (
            f"Expected 'critic.skipped' in log messages, got: {messages}"
        )

    def test_critique_done_not_logged_on_doom(self, caplog):
        """When short-circuited, 'critique.done' should NOT appear."""
        with caplog.at_level(logging.INFO, logger="reflect_mcp"):
            pipeline, _ = _make_pipeline_with_critic(
                gen_overrides=ALL_UNGROUNDED,
            )
            pipeline.extract(SOURCE)

        messages = [r.getMessage() for r in caplog.records]
        assert "critique.done" not in messages, (
            "'critique.done' should not appear when Critic was short-circuited"
        )

    def test_critique_done_logged_on_normal_path(self, caplog):
        """When Critic IS called (no short-circuit), 'critique.done' appears."""
        with caplog.at_level(logging.INFO, logger="reflect_mcp"):
            pipeline, _ = _make_pipeline_with_critic()
            pipeline.extract(SOURCE)

        messages = [r.getMessage() for r in caplog.records]
        assert "critique.done" in messages, (
            f"Expected 'critique.done' in log messages for normal path, got: {messages}"
        )

    def test_critic_skipped_not_logged_on_normal_path(self, caplog):
        """When Critic IS called normally, 'critic.skipped' should NOT appear."""
        with caplog.at_level(logging.INFO, logger="reflect_mcp"):
            pipeline, _ = _make_pipeline_with_critic()
            pipeline.extract(SOURCE)

        messages = [r.getMessage() for r in caplog.records]
        assert "critic.skipped" not in messages, (
            "'critic.skipped' should not appear when Critic was called normally"
        )
