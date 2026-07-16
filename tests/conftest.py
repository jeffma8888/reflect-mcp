"""Shared test fixtures.

Everything here keeps tests OFFLINE: the LLM boundary is a `ScriptedLLMClient`
that returns canned JSON, so `pytest` needs no API keys and never touches the
network. If any code path tried a real call, the scripted client would raise.
"""

from __future__ import annotations

import json

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.scripted import ScriptedLLMClient

# A clean source document all the happy-path tests can share.
SOURCE = (
    "ACME CORP\n"
    "Invoice #INV-1001\n"
    "Date: 2026-03-14\n"
    "Subtotal: $100.00\n"
    "Tax: $8.50\n"
    "Total: $108.50\n"
    "Paid in USD"
)

CORRECT_EXTRACTION = {
    "vendor_name": "ACME CORP",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-03-14",
    "subtotal": 100.0,
    "tax": 8.5,
    "total": 108.5,
    "currency": "USD",
}


def critique_all(confidence: float, overall: float, summary: str = "ok") -> str:
    verdicts = [
        {"field": f, "confidence": confidence, "grounded": True, "rationale": "matches source"}
        for f in CORRECT_EXTRACTION
    ]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": overall, "summary": summary})


@pytest.fixture
def source() -> str:
    return SOURCE


@pytest.fixture
def make_pipeline():
    """Factory: build a full pipeline from raw generator + critic JSON strings."""

    def _build(generator_raw: str, critic_raw: str) -> ReflectivePipeline:
        gen = Generator(ScriptedLLMClient([generator_raw]))
        crit = Critic(ScriptedLLMClient([critic_raw]))
        return ReflectivePipeline(gen, crit)

    return _build
