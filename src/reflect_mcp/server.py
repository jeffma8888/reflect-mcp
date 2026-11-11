"""MCP server exposing the reflective extraction pipeline as `extract` and `validate` tools.

Run it directly (`python -m reflect_mcp.server` or the `reflect-mcp` console
script) and wire it into Claude Desktop / Cursor / any MCP client. See the
README for the `claude_desktop_config.json` snippet.

Provider is chosen via env (REFLECT_PROVIDER + provider credentials). This module
is intentionally the ONLY one that imports `mcp`, so the rest of the package --
and the whole test suite -- has no hard dependency on the MCP SDK. It only
declares the tool surface; the bodies live in `reflect_mcp.tools` so they are
tested offline. The tools are `async` and hand that sync body to
`asyncio.to_thread` because FastMCP runs a sync tool inline on its event loop,
which would serialize parallel calls and stall pings/cancellations meanwhile.

Each tool returns a `CallToolResult` annotated with `tools.ToolResult`, so
`tools/list` advertises an outputSchema with a required `status` enum and a
`status:"error"` row reaches the host as `isError: true`. The structuredContent
is built explicitly: annotating `-> ToolResult` alone would make FastMCP
re-dump the dict with `None` defaults that fail the published schema.

`FastMCP()` takes no version, so the low-level server's version is set to
`__version__` explicitly; otherwise the `initialize` handshake's
`serverInfo.version` reports the MCP SDK's version instead of this build's.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from reflect_mcp import __version__, tools
from reflect_mcp.logging_config import get_logger
from reflect_mcp.tools import ToolResult

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
except ImportError as exc:  # pragma: no cover - only hit without the server extra
    raise SystemExit(
        "The MCP SDK is not installed. Install the server extra:\n"
        "    pip install 'reflect-mcp[server]'\n"
        "and a provider extra, e.g. 'reflect-mcp[anthropic]'."
    ) from exc

mcp = FastMCP("reflect-mcp")
if hasattr(mcp, "_mcp_server"):  # not every FastMCP-compatible object exposes the low-level server
    mcp._mcp_server.version = __version__


def _structured(result: dict[str, Any]) -> CallToolResult:
    """Wrap a tool-body dict so hosts get it unchanged as structuredContent and text.

    `isError` mirrors `status == "error"` so a host that branches on the MCP
    error flag never mistakes a failed extraction for a result.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        structuredContent=result,
        isError=result["status"] == "error",
    )


@mcp.tool()
async def extract(
    text: str,
    field_threshold: float | None = None,
    overall_threshold: float | None = None,
    heuristics_only: bool = False,
) -> Annotated[CallToolResult, ToolResult]:
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

    field_threshold / overall_threshold can only be raised above the
    operator's floor (env REFLECT_FIELD_THRESHOLD / REFLECT_OVERALL_THRESHOLD,
    defaults 0.70 / 0.75). Omit them (or pass null) for 0.70 / 0.75, or the
    operator floor if higher. An explicit value below the floor is raised to
    it and `notes` says so; an omitted one never adds a note.
    """
    return _structured(
        await asyncio.to_thread(
            tools.extract_tool, text, field_threshold, overall_threshold, heuristics_only
        )
    )


@mcp.tool()
async def validate(
    text: str,
    data: dict[str, Any],
    field_threshold: float | None = None,
    overall_threshold: float | None = None,
    heuristics_only: bool = False,
) -> Annotated[CallToolResult, ToolResult]:
    """Verify pre-extracted data against source text without running the Generator.

    Runs the Critic + deterministic heuristics + confidence gate on data you
    already have (from another LLM, a rules engine, or manual entry).

    Parameters:
      - text: the source document to verify against
      - data: dict of pre-extracted InvoiceData fields, or a stored reflect-mcp
        result envelope (e.g. an earlier `extract` result), whose `data`
        object is what gets verified
      - field_threshold / overall_threshold: confidence gate controls; they can
        only be raised above the operator's floor (env REFLECT_FIELD_THRESHOLD /
        REFLECT_OVERALL_THRESHOLD, defaults 0.70 / 0.75). Omit them (or pass
        null) for 0.70 / 0.75, or the operator floor if higher. An explicit
        value below the floor is raised to it with a note in `notes`; an
        omitted one never adds a note
      - heuristics_only: skip the Critic LLM call; heuristics + gate only

    Returns the same ExtractionResult envelope as `extract`, or
    { status: "error", error_message } when the input or `data` is invalid or
    the pipeline fails.
    """
    return _structured(
        await asyncio.to_thread(
            tools.validate_tool, text, data, field_threshold, overall_threshold, heuristics_only
        )
    )


def main() -> None:
    get_logger().info("reflect-mcp server starting", extra={"fields": {"stage": "server.start"}})
    mcp.run()


if __name__ == "__main__":
    main()
