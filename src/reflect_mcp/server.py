"""MCP server exposing the reflective extraction pipeline as `extract` and `validate` tools.

Run it directly (`python -m reflect_mcp.server` or the `reflect-mcp` console
script) and wire it into Claude Desktop / Cursor / any MCP client. See the
README for the `claude_desktop_config.json` snippet.

Provider is chosen via env (REFLECT_PROVIDER + provider credentials). This module
is intentionally the ONLY one that imports `mcp`, so the rest of the package --
and the whole test suite -- has no hard dependency on the MCP SDK. It only
declares the tool surface; the bodies live in `reflect_mcp.tools` so they are
tested offline.
"""

from __future__ import annotations

from typing import Any

from reflect_mcp import tools
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
def extract(
    text: str,
    field_threshold: float = 0.70,
    overall_threshold: float = 0.75,
    heuristics_only: bool = False,
) -> dict[str, Any]:
    """Extract structured invoice/receipt data from messy text, verified by a
    second-model critic and a human-in-the-loop confidence gate.

    Returns a JSON object with:
      - status: "ok" | "needs_review" | "error"
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

    When status == "error", the object is { status, error_message } and no
    extraction happened (empty input, bad thresholds, provider/parse failure).
    Set heuristics_only=True to skip the Critic LLM call and verify with
    deterministic heuristics only.
    """
    return tools.extract_tool(text, field_threshold, overall_threshold, heuristics_only)


@mcp.tool()
def validate(
    text: str,
    data: dict[str, Any],
    field_threshold: float = 0.70,
    overall_threshold: float = 0.75,
    heuristics_only: bool = False,
) -> dict[str, Any]:
    """Verify pre-extracted data against source text without running the Generator.

    Runs the Critic + deterministic heuristics + confidence gate on data you
    already have (from another LLM, a rules engine, or manual entry).

    Parameters:
      - text: the source document to verify against
      - data: dict of pre-extracted fields (InvoiceData shape)
      - field_threshold / overall_threshold: confidence gate controls
      - heuristics_only: skip the Critic LLM call; heuristics + gate only

    Returns the same ExtractionResult envelope as `extract`, or
    { status: "error", error_message } when the input or `data` is invalid or
    the pipeline fails.
    """
    return tools.validate_tool(text, data, field_threshold, overall_threshold, heuristics_only)


def main() -> None:
    get_logger().info("reflect-mcp server starting", extra={"fields": {"stage": "server.start"}})
    mcp.run()


if __name__ == "__main__":
    main()
