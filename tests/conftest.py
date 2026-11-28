"""Shared test fixtures.

Everything here keeps tests OFFLINE: the LLM boundary is a `ScriptedLLMClient`
that returns canned JSON, so `pytest` needs no API keys and never touches the
network. If any code path tried a real call, the scripted client would raise.
The MCP tools' process-lifetime provider-client memo is cleared before each test,
so no test sees a client (or fake SDK) memoised by an earlier one.
"""

from __future__ import annotations

import json
import os

import pytest

from reflect_mcp import parsing, tools
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


@pytest.fixture(autouse=True)
def _clear_client_memo() -> None:
    """Start each test with an empty MCP client memo; tests swap SDKs and factories."""
    tools._CLIENT_MEMO.clear()


# Tests that observe the real retry back-off keep it: a `sleeps` recorder
# fixture pins `[0.5]`, and iter86 EB5a reads the constant itself.
_REAL_DELAY_FIXTURES = frozenset({"sleeps"})
_REAL_DELAY_TESTS = frozenset({"test_eb5a_complete_parsed_exists_and_owns_delay"})


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_retry_delay: keep parsing's real retry back-off sleep for this test"
    )


@pytest.fixture(autouse=True)
def _zero_retry_delay(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the 0.5 s retry back-off in-process; it proves nothing but wall-clock."""
    if (
        request.node.get_closest_marker("real_retry_delay") is not None
        or _REAL_DELAY_FIXTURES.intersection(request.fixturenames)
        or getattr(request.node, "originalname", request.node.name) in _REAL_DELAY_TESTS
    ):
        return
    monkeypatch.setattr(parsing, "_RETRY_DELAY_SECONDS", 0.0)
