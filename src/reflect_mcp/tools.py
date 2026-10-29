"""MCP tool bodies, kept free of the MCP SDK so the suite can run them offline.

`server.py` is the only module that imports `mcp`, and it is excluded from
coverage, so any logic living there was never tested. The tool functions here
hold that logic instead; `server.py` only declares the tool surface and
delegates. They mirror the CLI's contract: every outcome comes back as a dict
whose `status` is `"ok"`, `"needs_review"` or `"error"`, so an agent can branch
on `status` alone instead of parsing exception text. They never raise.

The pipeline factory is looked up on `reflect_mcp.pipeline` at call time
(not bound at import), so tests can monkeypatch `build_default_pipeline`.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from reflect_mcp import pipeline as pipeline_module
from reflect_mcp.gate import GateConfig
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData

EMPTY_INPUT_MESSAGE = "input is empty"


def _error(message: str) -> dict[str, Any]:
    """Build the structured error envelope, matching the CLI's `status:"error"` shape."""
    return {"status": "error", "error_message": message}


def _gate_config(field_threshold: float, overall_threshold: float) -> GateConfig | dict[str, Any]:
    """Validate thresholds up front so a bad value never builds a pipeline (no LLM call)."""
    try:
        return GateConfig(field_threshold=field_threshold, overall_threshold=overall_threshold)
    except ValidationError as exc:
        return _error(f"invalid thresholds: {exc}")


def _build_pipeline(gate_config: GateConfig, heuristics_only: bool) -> ReflectivePipeline:
    """Call the factory, passing `critic_enabled` only when the caller opted out of the Critic.

    Omitting the kwarg in the default case mirrors the CLI's `_build_pipeline_kwargs`.
    """
    kwargs: dict[str, Any] = {"gate_config": gate_config}
    if heuristics_only:
        kwargs["critic_enabled"] = False
    return pipeline_module.build_default_pipeline(**kwargs)


def extract_tool(
    text: str,
    field_threshold: float = 0.70,
    overall_threshold: float = 0.75,
    heuristics_only: bool = False,
) -> dict[str, Any]:
    """Run the full Generator -> Critic -> gate pipeline; return a result or error dict."""
    if not isinstance(text, str) or not text.strip():
        return _error(EMPTY_INPUT_MESSAGE)
    gate_config = _gate_config(field_threshold, overall_threshold)
    if isinstance(gate_config, dict):
        return gate_config
    try:
        result = _build_pipeline(gate_config, heuristics_only).extract(text)
    except Exception as exc:  # noqa: BLE001 - the envelope is the contract, not a traceback
        return _error(str(exc))
    return result.model_dump()


def validate_tool(
    text: str,
    data: dict[str, Any],
    field_threshold: float = 0.70,
    overall_threshold: float = 0.75,
    heuristics_only: bool = False,
) -> dict[str, Any]:
    """Verify pre-extracted *data* against *text* (no Generator); return a result or error dict."""
    if not isinstance(text, str) or not text.strip():
        return _error(EMPTY_INPUT_MESSAGE)
    gate_config = _gate_config(field_threshold, overall_threshold)
    if isinstance(gate_config, dict):
        return gate_config
    try:
        invoice = InvoiceData.model_validate(data)
    except ValidationError as exc:
        return _error(f"data failed InvoiceData validation: {exc}")
    try:
        result = _build_pipeline(gate_config, heuristics_only).validate(text, invoice)
    except Exception as exc:  # noqa: BLE001 - the envelope is the contract, not a traceback
        return _error(str(exc))
    return result.model_dump()
