"""The reflective pipeline: generate -> heuristics -> critique -> decide.

This is the orchestration seam. It is deliberately free of any provider or MCP
import so it can be unit-tested end to end with `ScriptedLLMClient` and zero
network / API keys.
"""

from __future__ import annotations

from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig, decide
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.logging_config import log_stage, new_trace_id
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import ExtractionResult, InvoiceData


class ReflectivePipeline:
    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        gate_config: GateConfig | None = None,
        heuristics_enabled: bool = True,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.gate_config = gate_config or GateConfig()
        self.heuristics_enabled = heuristics_enabled

    def extract(self, text: str) -> ExtractionResult:
        trace_id = new_trace_id()
        log_stage("generate.start", trace_id, input_chars=len(text))

        data = self.generator.generate(text)
        log_stage(
            "generate.done",
            trace_id,
            fields={k: v for k, v in data.model_dump().items() if v is not None},
        )

        return self._verify(text, data, trace_id)

    def validate(self, text: str, data: InvoiceData) -> ExtractionResult:
        """Run Critic + heuristics + gate on pre-extracted data, skipping the Generator."""
        trace_id = new_trace_id()
        log_stage("validate.start", trace_id, input_chars=len(text))
        return self._verify(text, data, trace_id)

    def _verify(self, text: str, data: InvoiceData, trace_id: str) -> ExtractionResult:
        """Shared verification path: heuristics → critique → gate."""
        if self.heuristics_enabled:
            heuristics = run_heuristics(text, data)
        else:
            heuristics = run_heuristics(text, data.__class__())
        log_stage("heuristics.done", trace_id, factors=heuristics.factors)

        critique = self.critic.critique(text, data)
        log_stage(
            "critique.done",
            trace_id,
            overall_confidence=critique.overall_confidence,
            field_verdicts={v.field: v.confidence for v in critique.field_verdicts},
        )

        result = decide(data, critique, heuristics, trace_id, self.gate_config)
        log_stage(
            "decide.done",
            trace_id,
            status=result.status,
            flagged=[f.field for f in result.flagged_fields],
            overall_confidence=result.overall_confidence,
        )
        return result


def build_default_pipeline(
    gate_config: GateConfig | None = None,
    *,
    max_retries: int = 1,
) -> ReflectivePipeline:
    """Wire up a pipeline against real providers (env-configured).

    Imported lazily so tests never touch provider SDKs.
    """
    from reflect_mcp.providers import build_client

    gen_client: LLMClient = build_client("generator")
    crit_client: LLMClient = build_client("critic")
    return ReflectivePipeline(
        Generator(gen_client, max_retries=max_retries),
        Critic(crit_client, max_retries=max_retries),
        gate_config,
    )
