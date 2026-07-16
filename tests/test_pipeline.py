"""End-to-end pipeline over the mocked LLM boundary (fully offline)."""

from __future__ import annotations

import json

from tests.conftest import CORRECT_EXTRACTION, critique_all


def test_happy_path_returns_ok(make_pipeline, source):
    pipeline = make_pipeline(json.dumps(CORRECT_EXTRACTION), critique_all(0.95, 0.95))
    result = pipeline.extract(source)
    assert result.status == "ok"
    assert result.flagged_fields == []
    assert result.data.total == 108.5
    assert result.overall_confidence == 0.95
    assert result.trace_id.startswith("trace-")


def test_result_is_json_serializable(make_pipeline, source):
    pipeline = make_pipeline(json.dumps(CORRECT_EXTRACTION), critique_all(0.95, 0.95))
    result = pipeline.extract(source)
    dumped = result.model_dump()
    # The MCP tool returns this dict; it must round-trip through JSON.
    assert json.loads(json.dumps(dumped, default=str))["status"] == "ok"


def test_pipeline_makes_exactly_two_llm_calls(make_pipeline, source):
    from reflect_mcp.critic import Critic
    from reflect_mcp.generator import Generator
    from reflect_mcp.pipeline import ReflectivePipeline
    from reflect_mcp.scripted import ScriptedLLMClient

    gen_client = ScriptedLLMClient([json.dumps(CORRECT_EXTRACTION)])
    crit_client = ScriptedLLMClient([critique_all(0.95, 0.95)])
    pipeline = ReflectivePipeline(Generator(gen_client), Critic(crit_client))
    pipeline.extract(source)
    assert len(gen_client.calls) == 1  # one generate call
    assert len(crit_client.calls) == 1  # one critique call
    # The critic must be shown BOTH the source and the candidate extraction.
    assert "ACME CORP" in crit_client.calls[0]["user"]
    assert "108.5" in crit_client.calls[0]["user"]
