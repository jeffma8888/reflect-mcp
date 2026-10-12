"""Black-box behavioral tests for iter-03: SDK client reuse across calls.

Covers all 8 Expected Behaviors from pm.md:
  1. AnthropicClient.__init__ creates anthropic.Anthropic() once; .complete() reuses
  2. OpenAIClient.__init__ creates openai.OpenAI() once; .complete() reuses
  3. BedrockClient.__init__ creates boto3.client() once; .complete() reuses
  4. Two .complete() calls on same instance use same SDK client (identity check)
  5. LLMClient protocol signature unchanged
  6. Lazy SDK imports: helpful error when SDK package not installed
  7. ScriptedLLMClient unaffected (no per-call construction)
  8. All existing tests pass unchanged (verified by full suite run)

Isolation contract: tests drive the public provider interface and inspect
observable output/attributes. No implementation source was read.
"""

from __future__ import annotations

import inspect
import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from reflect_mcp.scripted import ScriptedLLMClient


# ---------------------------------------------------------------------------
# Helpers — mock SDK modules and responses
# ---------------------------------------------------------------------------

def _make_fake_anthropic():
    """Return (fake_module, fake_client) for anthropic SDK."""
    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_module.Anthropic.return_value = fake_client
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"result": "ok"}')]
    fake_client.messages.create.return_value = fake_response
    return fake_module, fake_client


def _make_fake_openai():
    """Return (fake_module, fake_client) for openai SDK."""
    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_module.OpenAI.return_value = fake_client
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"result": "ok"}'))]
    fake_client.chat.completions.create.return_value = fake_response
    return fake_module, fake_client


def _make_fake_boto3():
    """Return (fake_module, fake_client) for boto3 SDK."""
    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_module.client.return_value = fake_client
    body_bytes = json.dumps({"content": [{"text": '{"result": "ok"}'}]}).encode()
    fake_client.invoke_model.return_value = {"body": io.BytesIO(body_bytes)}
    return fake_module, fake_client


def _find_client_attr(instance):
    """Discover the stored SDK client attribute on a provider instance."""
    for name, val in instance.__dict__.items():
        if "client" in name.lower():
            return name, val
    return None, None


# ---------------------------------------------------------------------------
# EB-1: AnthropicClient.__init__ creates SDK client once
# ---------------------------------------------------------------------------

class TestEB1AnthropicClientReuse:
    """EB-1: AnthropicClient.__init__ creates anthropic.Anthropic() once."""

    def test_sdk_constructor_called_once_in_init(self) -> None:
        fake_module, fake_client = _make_fake_anthropic()
        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from reflect_mcp.providers import AnthropicClient
            provider = AnthropicClient(model="test-model")
        assert fake_module.Anthropic.call_count == 1

    def test_sdk_constructor_not_called_again_on_complete(self) -> None:
        fake_module, fake_client = _make_fake_anthropic()
        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from reflect_mcp.providers import AnthropicClient
            provider = AnthropicClient(model="test-model")
            count_after_init = fake_module.Anthropic.call_count
            try:
                provider.complete(system="test", user="test")
            except Exception:
                pass
            assert fake_module.Anthropic.call_count == count_after_init


# ---------------------------------------------------------------------------
# EB-2: OpenAIClient.__init__ creates SDK client once
# ---------------------------------------------------------------------------

class TestEB2OpenAIClientReuse:
    """EB-2: OpenAIClient.__init__ creates openai.OpenAI() once."""

    def test_sdk_constructor_called_once_in_init(self) -> None:
        fake_module, fake_client = _make_fake_openai()
        with patch.dict("sys.modules", {"openai": fake_module}):
            from reflect_mcp.providers import OpenAIClient
            provider = OpenAIClient(model="test-model")
        assert fake_module.OpenAI.call_count == 1

    def test_sdk_constructor_not_called_again_on_complete(self) -> None:
        fake_module, fake_client = _make_fake_openai()
        with patch.dict("sys.modules", {"openai": fake_module}):
            from reflect_mcp.providers import OpenAIClient
            provider = OpenAIClient(model="test-model")
            count_after_init = fake_module.OpenAI.call_count
            try:
                provider.complete(system="test", user="test")
            except Exception:
                pass
            assert fake_module.OpenAI.call_count == count_after_init


# ---------------------------------------------------------------------------
# EB-3: BedrockClient.__init__ creates SDK client once
# ---------------------------------------------------------------------------

class TestEB3BedrockClientReuse:
    """EB-3: BedrockClient.__init__ creates boto3.client() once."""

    def test_sdk_constructor_called_once_in_init(self) -> None:
        fake_module, fake_client = _make_fake_boto3()
        with patch.dict("sys.modules", {"boto3": fake_module}):
            from reflect_mcp.providers import BedrockClient
            provider = BedrockClient(model="test-model")
        assert fake_module.client.call_count == 1

    def test_sdk_constructor_not_called_again_on_complete(self) -> None:
        fake_module, fake_client = _make_fake_boto3()
        with patch.dict("sys.modules", {"boto3": fake_module}):
            from reflect_mcp.providers import BedrockClient
            provider = BedrockClient(model="test-model")
            count_after_init = fake_module.client.call_count
            try:
                provider.complete(system="test", user="test")
            except Exception:
                pass
            assert fake_module.client.call_count == count_after_init


# ---------------------------------------------------------------------------
# EB-4: Two .complete() calls use the same SDK client object (identity)
# ---------------------------------------------------------------------------

class TestEB4ClientIdentity:
    """EB-4: consecutive .complete() calls use the same SDK client object."""

    def test_anthropic_client_identity(self) -> None:
        fake_module, _ = _make_fake_anthropic()
        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from reflect_mcp.providers import AnthropicClient
            provider = AnthropicClient(model="test-model")
            _, client_before = _find_client_attr(provider)
            assert client_before is not None, "No client attribute found on AnthropicClient"
            try:
                provider.complete(system="s", user="u")
            except Exception:
                pass
            _, client_after = _find_client_attr(provider)
            assert client_before is client_after

    def test_openai_client_identity(self) -> None:
        fake_module, _ = _make_fake_openai()
        with patch.dict("sys.modules", {"openai": fake_module}):
            from reflect_mcp.providers import OpenAIClient
            provider = OpenAIClient(model="test-model")
            _, client_before = _find_client_attr(provider)
            assert client_before is not None, "No client attribute found on OpenAIClient"
            try:
                provider.complete(system="s", user="u")
            except Exception:
                pass
            _, client_after = _find_client_attr(provider)
            assert client_before is client_after

    def test_bedrock_client_identity(self) -> None:
        fake_module, _ = _make_fake_boto3()
        with patch.dict("sys.modules", {"boto3": fake_module}):
            from reflect_mcp.providers import BedrockClient
            provider = BedrockClient(model="test-model")
            _, client_before = _find_client_attr(provider)
            assert client_before is not None, "No client attribute found on BedrockClient"
            try:
                provider.complete(system="s", user="u")
            except Exception:
                pass
            _, client_after = _find_client_attr(provider)
            assert client_before is client_after


# ---------------------------------------------------------------------------
# EB-5: LLMClient protocol signature unchanged
# ---------------------------------------------------------------------------

class TestEB5ProtocolSignature:
    """EB-5: LLMClient protocol complete(*, system: str, user: str) -> str."""

    def test_complete_signature_has_system_and_user_keyword_only(self) -> None:
        from reflect_mcp.providers import LLMClient
        sig = inspect.signature(LLMClient.complete)
        params = sig.parameters
        assert "system" in params, "LLMClient.complete must have 'system' parameter"
        assert "user" in params, "LLMClient.complete must have 'user' parameter"
        assert params["system"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["user"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_complete_return_annotation_is_str(self) -> None:
        from reflect_mcp.providers import LLMClient
        sig = inspect.signature(LLMClient.complete)
        assert sig.return_annotation is str or sig.return_annotation == "str"


# ---------------------------------------------------------------------------
# EB-6: Lazy SDK imports — helpful error on missing SDK
# ---------------------------------------------------------------------------

class TestEB6LazyImportGuard:
    """EB-6: ImportError with helpful message when SDK not installed."""

    def test_anthropic_missing_raises_with_message(self) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            from reflect_mcp.providers import AnthropicClient
            with pytest.raises(Exception, match="(?i)anthropic|pip install"):
                AnthropicClient(model="test-model")

    def test_openai_missing_raises_with_message(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            from reflect_mcp.providers import OpenAIClient
            with pytest.raises(Exception, match="(?i)openai|pip install"):
                OpenAIClient(model="test-model")

    def test_boto3_missing_raises_with_message(self) -> None:
        with patch.dict("sys.modules", {"boto3": None}):
            from reflect_mcp.providers import BedrockClient
            with pytest.raises(Exception, match="(?i)boto3|pip install"):
                BedrockClient(model="test-model")


# ---------------------------------------------------------------------------
# EB-7: ScriptedLLMClient unaffected
# ---------------------------------------------------------------------------

class TestEB7ScriptedUnaffected:
    """EB-7: ScriptedLLMClient still works (no per-call construction)."""

    def test_scripted_client_complete(self) -> None:
        client = ScriptedLLMClient(["hello world"])
        result = client.complete(system="sys", user="usr")
        assert result == "hello world"

    def test_scripted_client_no_client_attr(self) -> None:
        """ScriptedLLMClient has no _client attribute — no SDK construction."""
        client = ScriptedLLMClient(["response"])
        _, val = _find_client_attr(client)
        assert val is None, "ScriptedLLMClient should not have a client attribute"


# ---------------------------------------------------------------------------
# EB-8: All existing tests pass (verified by running full suite)
# ---------------------------------------------------------------------------

class TestEB8ExistingSuitePasses:
    """EB-8: ScriptedLLMClient pipeline still works — existing seam unchanged."""

    def test_full_pipeline_with_scripted_client(self) -> None:
        """Existing pipeline through ScriptedLLMClient still produces a result."""
        from reflect_mcp.pipeline import ReflectivePipeline
        from reflect_mcp.generator import Generator
        from reflect_mcp.critic import Critic
        from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

        gen_json = json.dumps(CORRECT_EXTRACTION)
        crit_json = critique_all(0.95, 0.95)
        pipe = ReflectivePipeline(
            Generator(ScriptedLLMClient([gen_json])),
            Critic(ScriptedLLMClient([crit_json])),
        )
        result = pipe.extract(SOURCE)
        assert result.status in ("ok", "needs_review")
        assert result.data is not None
