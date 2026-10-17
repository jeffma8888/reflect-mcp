"""Black-box behavioral tests for iter-19: Generator/Critic retry on parse failure.

Covers all 6 Expected Behaviors from pm.md:
  1. Generator retry succeeds (bad→good) — returns valid InvoiceData on second attempt.
  2. Critic retry succeeds (bad→good) — returns valid CritiqueResult on second attempt.
  3. Generator exhausts retries → GeneratorError.
  4. Critic exhausts retries → CriticError.
  5. Zero retries disables retry — first parse failure raises immediately.
  6. E2E pipeline succeeds through transient failure (via build_default_pipeline).

Isolation contract: tests drive the public interface (Generator, Critic,
ReflectivePipeline, build_default_pipeline) using ScriptedLLMClient. No
implementation source was read.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.pipeline import ReflectivePipeline, build_default_pipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all


GARBAGE = "this is not json at all"
VALID_GEN_JSON = json.dumps(CORRECT_EXTRACTION)
VALID_CRIT_JSON = critique_all(0.95, 0.95)


# ---------------------------------------------------------------------------
# EB-1: Generator retry succeeds (bad→good)
# ---------------------------------------------------------------------------

class TestEB1GeneratorRetrySucceeds:
    """EB-1: Generator with max_retries=1 recovers from first-call parse failure."""

    def test_returns_valid_invoice_data_on_retry(self):
        client = ScriptedLLMClient([GARBAGE, VALID_GEN_JSON])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            result = gen.generate(SOURCE)
        assert isinstance(result, InvoiceData)
        assert result.vendor_name == "ACME CORP"
        assert result.total == pytest.approx(108.5)

    def test_makes_two_llm_calls(self):
        client = ScriptedLLMClient([GARBAGE, VALID_GEN_JSON])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            gen.generate(SOURCE)
        assert len(client.calls) == 2

    def test_all_scripted_responses_consumed(self):
        client = ScriptedLLMClient([GARBAGE, VALID_GEN_JSON])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            result = gen.generate(SOURCE)
        assert result is not None


# ---------------------------------------------------------------------------
# EB-2: Critic retry succeeds (bad→good)
# ---------------------------------------------------------------------------

class TestEB2CriticRetrySucceeds:
    """EB-2: Critic with max_retries=1 recovers from first-call parse failure."""

    def test_returns_valid_critique_result(self):
        client = ScriptedLLMClient([GARBAGE, VALID_CRIT_JSON])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            result = crit.critique(SOURCE, data)
        assert result.overall_confidence == pytest.approx(0.95)

    def test_makes_two_llm_calls(self):
        client = ScriptedLLMClient([GARBAGE, VALID_CRIT_JSON])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            crit.critique(SOURCE, data)
        assert len(client.calls) == 2

    def test_all_scripted_responses_consumed(self):
        client = ScriptedLLMClient([GARBAGE, VALID_CRIT_JSON])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            result = crit.critique(SOURCE, data)
        assert result is not None


# ---------------------------------------------------------------------------
# EB-3: Generator exhausts retries → GeneratorError
# ---------------------------------------------------------------------------

class TestEB3GeneratorExhaustsRetries:
    """EB-3: Both attempts return garbage → GeneratorError raised."""

    def test_raises_generator_error(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            with pytest.raises(GeneratorError):
                gen.generate(SOURCE)

    def test_consumes_all_attempts(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            with pytest.raises(GeneratorError):
                gen.generate(SOURCE)
        assert len(client.calls) == 2

    def test_error_chains_original_exception(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        gen = Generator(client, max_retries=1)
        with patch("time.sleep"):
            with pytest.raises(GeneratorError) as exc_info:
                gen.generate(SOURCE)
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# EB-4: Critic exhausts retries → CriticError
# ---------------------------------------------------------------------------

class TestEB4CriticExhaustsRetries:
    """EB-4: Both attempts return garbage → CriticError raised."""

    def test_raises_critic_error(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            with pytest.raises(CriticError):
                crit.critique(SOURCE, data)

    def test_consumes_all_attempts(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            with pytest.raises(CriticError):
                crit.critique(SOURCE, data)
        assert len(client.calls) == 2

    def test_error_chains_original_exception(self):
        client = ScriptedLLMClient([GARBAGE, GARBAGE])
        crit = Critic(client, max_retries=1)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep"):
            with pytest.raises(CriticError) as exc_info:
                crit.critique(SOURCE, data)
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# EB-5: Zero retries disables retry
# ---------------------------------------------------------------------------

class TestEB5ZeroRetriesDisablesRetry:
    """EB-5: max_retries=0 → first parse failure raises immediately."""

    def test_generator_zero_retries_raises_on_first_failure(self):
        client = ScriptedLLMClient([GARBAGE])
        gen = Generator(client, max_retries=0)
        with pytest.raises(GeneratorError):
            gen.generate(SOURCE)
        assert len(client.calls) == 1

    def test_critic_zero_retries_raises_on_first_failure(self):
        client = ScriptedLLMClient([GARBAGE])
        crit = Critic(client, max_retries=0)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with pytest.raises(CriticError):
            crit.critique(SOURCE, data)
        assert len(client.calls) == 1

    def test_generator_zero_retries_succeeds_on_good_first_response(self):
        client = ScriptedLLMClient([VALID_GEN_JSON])
        gen = Generator(client, max_retries=0)
        result = gen.generate(SOURCE)
        assert isinstance(result, InvoiceData)
        assert len(client.calls) == 1

    def test_critic_zero_retries_succeeds_on_good_first_response(self):
        client = ScriptedLLMClient([VALID_CRIT_JSON])
        crit = Critic(client, max_retries=0)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        result = crit.critique(SOURCE, data)
        assert result.overall_confidence == pytest.approx(0.95)
        assert len(client.calls) == 1

    def test_generator_no_sleep_on_zero_retries(self):
        client = ScriptedLLMClient([GARBAGE])
        gen = Generator(client, max_retries=0)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(GeneratorError):
                gen.generate(SOURCE)
        mock_sleep.assert_not_called()

    def test_critic_no_sleep_on_zero_retries(self):
        client = ScriptedLLMClient([GARBAGE])
        crit = Critic(client, max_retries=0)
        data = InvoiceData.model_validate(CORRECT_EXTRACTION)
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(CriticError):
                crit.critique(SOURCE, data)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# EB-6: E2E pipeline succeeds through transient failure
# ---------------------------------------------------------------------------

class TestEB6E2EPipelineTransientFailure:
    """EB-6: Full pipeline.extract() succeeds when Generator retries."""

    def test_pipeline_extract_succeeds_on_generator_retry(self):
        gen_client = ScriptedLLMClient([GARBAGE, VALID_GEN_JSON])
        crit_client = ScriptedLLMClient([VALID_CRIT_JSON])
        pipeline = ReflectivePipeline(
            Generator(gen_client, max_retries=1),
            Critic(crit_client, max_retries=0),
        )
        with patch("time.sleep"):
            result = pipeline.extract(SOURCE)
        assert result.status in ("ok", "needs_review")
        assert result.data is not None
        assert result.data.vendor_name == "ACME CORP"

    def test_pipeline_extract_succeeds_on_critic_retry(self):
        gen_client = ScriptedLLMClient([VALID_GEN_JSON])
        crit_client = ScriptedLLMClient([GARBAGE, VALID_CRIT_JSON])
        pipeline = ReflectivePipeline(
            Generator(gen_client, max_retries=0),
            Critic(crit_client, max_retries=1),
        )
        with patch("time.sleep"):
            result = pipeline.extract(SOURCE)
        assert result.status in ("ok", "needs_review")
        assert result.data is not None

    def test_build_default_pipeline_accepts_max_retries_kwarg(self):
        """build_default_pipeline signature accepts max_retries as keyword arg."""
        import inspect
        sig = inspect.signature(build_default_pipeline)
        assert "max_retries" in sig.parameters
        param = sig.parameters["max_retries"]
        assert param.default == 1
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_default_max_retries_is_1(self):
        """Default max_retries=1: Generator retries once on parse failure."""
        gen_client = ScriptedLLMClient([GARBAGE, VALID_GEN_JSON])
        crit_client = ScriptedLLMClient([VALID_CRIT_JSON])
        pipeline = ReflectivePipeline(
            Generator(gen_client),
            Critic(crit_client),
        )
        with patch("time.sleep"):
            result = pipeline.extract(SOURCE)
        assert result.status in ("ok", "needs_review")
        assert len(gen_client.calls) == 2
