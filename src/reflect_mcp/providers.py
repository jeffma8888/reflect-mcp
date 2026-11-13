"""LLM boundary + provider switching.

`LLMClient` is the *single seam* the whole pipeline talks to. Tests and the
offline eval swap in `ScriptedLLMClient` (see `scripted.py`); production picks a
real provider via env vars:

    REFLECT_PROVIDER        anthropic | openai | bedrock   (default: anthropic)
    REFLECT_GENERATOR_MODEL model id for the Generator
    REFLECT_CRITIC_MODEL    model id for the Critic (should DIFFER from generator)

Provider SDKs are imported lazily so the base install and the offline test suite
never need `anthropic` / `openai` / `boto3` installed.
"""

from __future__ import annotations

import importlib
import json
import os
from types import ModuleType
from typing import Callable, Protocol, runtime_checkable

# Sensible defaults; a reviewer can override any of these via env.
_DEFAULTS = {
    "anthropic": {
        "generator": "claude-3-5-sonnet-latest",
        "critic": "claude-3-5-haiku-latest",  # different model == real cross-check
    },
    "openai": {
        "generator": "gpt-4o",
        "critic": "gpt-4o-mini",
    },
    "bedrock": {
        "generator": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "critic": "anthropic.claude-3-5-haiku-20241022-v1:0",
    },
}


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-in/text-out contract. Callers expect a JSON string back."""

    def complete(self, *, system: str, user: str) -> str: ...


def _model_for(role: str, provider: str) -> str:
    env_key = f"REFLECT_{role.upper()}_MODEL"
    if os.getenv(env_key):
        return os.environ[env_key]
    return _DEFAULTS[provider][role]


def _require(module: str, provider: str, extra: str) -> ModuleType:
    """Lazy SDK import: one guard keeps every install hint alike; honours sys.modules stubs."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - only hit without SDK
        raise RuntimeError(
            f"{provider} provider selected but the `{module}` package is not "
            f"installed. Run: pip install 'reflect-mcp[{extra}]'"
        ) from exc


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = _require("anthropic", "Anthropic", "anthropic").Anthropic()

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class OpenAIClient:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = _require("openai", "OpenAI", "openai").OpenAI()

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class BedrockClient:
    """AWS Bedrock (Claude) -- echoes the original production stack directly."""

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        boto3 = _require("boto3", "Bedrock", "bedrock")
        self._client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

    def complete(self, *, system: str, user: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = self._client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return "".join(b.get("text", "") for b in payload.get("content", []))


# Keyed exactly like `_DEFAULTS`, so adding a provider is one class + one entry.
_CLIENTS: dict[str, Callable[[str], LLMClient]] = {"anthropic": AnthropicClient, "openai": OpenAIClient,
                                                  "bedrock": BedrockClient}


def build_client(role: str) -> LLMClient:
    """Build a real provider client for `role` in {"generator", "critic"}."""
    provider = os.getenv("REFLECT_PROVIDER", "anthropic").lower()
    if provider not in _DEFAULTS:
        raise ValueError(f"Unknown REFLECT_PROVIDER={provider!r}. Choose one of: {', '.join(_DEFAULTS)}.")
    return _CLIENTS[provider](_model_for(role, provider))
