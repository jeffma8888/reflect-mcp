"""Generator + Critic parse/validate behavior at the LLM boundary."""

from __future__ import annotations

import json

import pytest

from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.scripted import ScriptedLLMClient
from reflect_mcp.schema import InvoiceData


def test_generator_parses_typed_json():
    raw = json.dumps({"vendor_name": "ACME CORP", "total": 108.5, "currency": "USD"})
    gen = Generator(ScriptedLLMClient([raw]))
    data = gen.generate("some text")
    assert isinstance(data, InvoiceData)
    assert data.vendor_name == "ACME CORP"
    assert data.subtotal is None  # absent -> null, never invented


def test_generator_tolerates_code_fences():
    raw = "```json\n{\"vendor_name\": \"ACME\"}\n```"
    gen = Generator(ScriptedLLMClient([raw]))
    assert gen.generate("t").vendor_name == "ACME"


def test_generator_raises_on_garbage():
    gen = Generator(ScriptedLLMClient(["not json"]))
    with pytest.raises(GeneratorError):
        gen.generate("t")


def test_critic_parses_critique():
    raw = json.dumps({
        "field_verdicts": [{"field": "total", "confidence": 0.9, "grounded": True}],
        "overall_confidence": 0.9, "summary": "looks good",
    })
    crit = Critic(ScriptedLLMClient([raw]))
    result = crit.critique("t", InvoiceData(total=108.5))
    assert result.overall_confidence == 0.9
    assert result.verdict_for("total").confidence == 0.9


def test_critic_raises_on_garbage():
    crit = Critic(ScriptedLLMClient(["definitely not json"]))
    with pytest.raises(CriticError):
        crit.critique("t", InvoiceData())


def test_scripted_client_raises_when_exhausted():
    client = ScriptedLLMClient([])
    with pytest.raises(RuntimeError):
        client.complete(system="s", user="u")
