"""MCP server exposing the reflective extraction pipeline as `extract` and `validate` tools.

Run it directly (`python -m reflect_mcp.server` or the `reflect-mcp` console
script) and wire it into Claude Desktop / Cursor / any MCP client. See the
README for the `claude_desktop_config.json` snippet.

Provider is chosen via env (REFLECT_PROVIDER + provider credentials). This module
is intentionally the ONLY one that imports `mcp`, so the rest of the package --
and the whole test suite -- has no hard dependency on the MCP SDK.
"""

from __future__ import annotations

from typing import Any

from reflect_mcp.gate import GateConfig
from reflect_mcp.logging_config import get_logger

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - only hit without the server extra
    raise SystemExit(
        "The MCP SDK is not installed. Install the server extra:\n"
        "    pip install 'reflect-mcp[server]'\n"
        "and a provider extra, e.g. 'reflect-mcp[anthropic]'."
    ) from exc

mcp = FastMCP("reflect-mcp")


@mcp.tool()
def extract(text: str, field_threshold: float = 0.70, overall_threshold: float = 0.75) -> dict[str, Any]:
    """Extract structured invoice/receipt data from messy text, verified by a
    second-model critic and a human-in-the-loop confidence gate.

    Returns a JSON object with:
      - status: "ok" | "needs_review"
      - data: the typed extraction (nulls for absent fields)
      - field_confidence: per-field combined confidence in [0,1]
      - overall_confidence: the critic's overall confidence
      - flagged_fields: fields below threshold that a human should confirm
      - field_provenance: per-field character-offset spans into the source text
      - critic_summary: the critic's one-line human-readable assessment
      - trace_id / notes: observability + the decision rationale

    When status == "needs_review", DO NOT use `data` blindly -- confirm the
    fields listed in `flagged_fields` first. The tool surfaces uncertainty
    instead of returning a confident guess.
    """
    from reflect_mcp.pipeline import build_default_pipeline

    config = GateConfig(field_threshold=field_threshold, overall_threshold=overall_threshold)
    pipeline = build_default_pipeline(gate_config=config)
    result = pipeline.extract(text)
    return result.model_dump()


@mcp.tool()
def validate(
    text: str, data: dict[str, Any], field_threshold: float = 0.70, overall_threshold: float = 0.75
) -> dict[str, Any]:
    """Verify pre-extracted data against source text without running the Generator.

    Runs the Critic + deterministic heuristics + confidence gate on data you
    already have (from another LLM, a rules engine, or manual entry).

    Parameters:
      - text: the source document to verify against
      - data: dict of pre-extracted fields (InvoiceData shape)
      - field_threshold / overall_threshold: confidence gate controls

    Returns the same ExtractionResult envelope as `extract`.
    """
    from reflect_mcp.pipeline import build_default_pipeline
    from reflect_mcp.schema import InvoiceData

    invoice = InvoiceData.model_validate(data)
    config = GateConfig(field_threshold=field_threshold, overall_threshold=overall_threshold)
    pipeline = build_default_pipeline(gate_config=config)
    result = pipeline.validate(text, invoice)
    return result.model_dump()


def main() -> None:
    get_logger().info("reflect-mcp server starting", extra={"fields": {"stage": "server.start"}})
    mcp.run()


if __name__ == "__main__":
    main()
