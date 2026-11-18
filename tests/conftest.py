"""Shared test fixtures.

Everything here keeps tests OFFLINE: the LLM boundary is a `ScriptedLLMClient`
that returns canned JSON, so `pytest` needs no API keys and never touches the
network. If any code path tried a real call, the scripted client would raise.
"""

from __future__ import annotations

import json
import os

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


@pytest.fixture(autouse=True)
def _clear_operator_gate_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic against an operator's exported MCP gate floor.

    README tells operators to export `REFLECT_FIELD_THRESHOLD` /
    `REFLECT_OVERALL_THRESHOLD` for their server; if those leaked into tests,
    every MCP-path test would see a raised floor and a clean checkout would go
    red. Every `REFLECT_*_THRESHOLD` name is cleared (prefix/suffix match, so a
    future floor var is covered too). Tests that want a floor `setenv` it in
    their own body, which runs after this fixture.
    """
    for name in list(os.environ):
        if name.startswith("REFLECT_") and name.endswith("_THRESHOLD"):
            monkeypatch.delenv(name, raising=False)
