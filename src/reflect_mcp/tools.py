"""MCP tool bodies, kept free of the MCP SDK so the suite can run them offline.

`server.py` is the only module that imports `mcp`, and it is excluded from
coverage, so any logic living there was never tested. The tool functions here
hold that logic instead; `server.py` only declares the tool surface and
delegates. They mirror the CLI's contract: every outcome comes back as a dict
whose `status` is `"ok"`, `"needs_review"` or `"error"`, so an agent can branch
on `status` alone instead of parsing exception text. They never raise.

The pipeline factory is looked up on `reflect_mcp.pipeline` at call time
(not bound at import), so tests can monkeypatch `build_default_pipeline`.

Over MCP the threshold arguments are filled in by the *calling agent*, not the
human, so they can only tighten the thresholds: the operator pins a floor with
`REFLECT_FIELD_THRESHOLD` / `REFLECT_OVERALL_THRESHOLD` (defaults 0.70 / 0.75)
and any lower argument is raised to it, with the raise recorded in `notes`.
"""

from __future__ import annotations

import os
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


FIELD_FLOOR_ENV = "REFLECT_FIELD_THRESHOLD"
OVERALL_FLOOR_ENV = "REFLECT_OVERALL_THRESHOLD"
_PACKAGE_DEFAULTS = GateConfig()


def _env_floor(name: str, default: float) -> float:
    """Read one operator floor from the environment at call time.

    Unset/blank falls back to the package default so the gate is safe by
    default. A malformed value raises instead of being ignored: silently
    dropping a misspelt floor would hand the agent back the bypass it closes.
    """
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        value = float(raw)
        in_range = 0.0 <= value <= 1.0  # False for nan and +/-inf too
    except ValueError:
        in_range = False
    if not in_range:
        raise ValueError(f"{name} must be a number in [0, 1], got {raw!r}")
    return value


def _raise_note(name: str, requested: float, floor: float) -> str:
    """Describe one clamp so the caller sees its argument was overridden, not ignored."""
    return f"{name} raised from {float(requested)} to {float(floor)}."


def _gate_config(
    field_threshold: float, overall_threshold: float
) -> tuple[GateConfig, str] | dict[str, Any]:
    """Validate the caller's thresholds, then raise them to the operator floor.

    Validation runs first so an out-of-range or NaN argument stays an error
    instead of being silently "fixed" by the clamp (`max(nan, x)` is also
    order-dependent). Returns the effective config plus the notes suffix
    (empty when nothing was raised), or the error envelope; never builds a
    pipeline, so a bad value costs no LLM call.
    """
    try:
        requested = GateConfig(field_threshold=field_threshold, overall_threshold=overall_threshold)
        field_floor = _env_floor(FIELD_FLOOR_ENV, _PACKAGE_DEFAULTS.field_threshold)
        overall_floor = _env_floor(OVERALL_FLOOR_ENV, _PACKAGE_DEFAULTS.overall_threshold)
    except (ValidationError, ValueError) as exc:
        return _error(f"invalid thresholds: {exc}")
    raised = [
        _raise_note(name, req, floor)
        for name, req, floor in (
            ("field_threshold", requested.field_threshold, field_floor),
            ("overall_threshold", requested.overall_threshold, overall_floor),
        )
        if req < floor
    ]
    effective = GateConfig(
        field_threshold=max(requested.field_threshold, field_floor),
        overall_threshold=max(requested.overall_threshold, overall_floor),
    )
    return effective, " ".join(raised)


def _with_raise_note(result: dict[str, Any], suffix: str) -> dict[str, Any]:
    """Append the clamp sentences to the gate's notes; a no-op when nothing was raised."""
    if suffix:
        result["notes"] = f"{result['notes']} {suffix}"
    return result


def _build_pipeline(
    gate_config: GateConfig, heuristics_only: bool, *, validate_only: bool = False
) -> ReflectivePipeline:
    """Call the factory, adding only the kwargs the caller opted out of (Critic, Generator).

    Omitting the kwarg in the default case mirrors the CLI's `_build_pipeline_kwargs`.
    `validate_only` drops the Generator client, which `validate` never calls.
    """
    kwargs: dict[str, Any] = {"gate_config": gate_config}
    if heuristics_only:
        kwargs["critic_enabled"] = False
    if validate_only:
        kwargs["generator_enabled"] = False
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
    resolved = _gate_config(field_threshold, overall_threshold)
    if isinstance(resolved, dict):
        return resolved
    gate_config, raise_note = resolved
    try:
        result = _build_pipeline(gate_config, heuristics_only).extract(text)
    except Exception as exc:  # noqa: BLE001 - the envelope is the contract, not a traceback
        return _error(str(exc))
    return _with_raise_note(result.model_dump(), raise_note)


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
    resolved = _gate_config(field_threshold, overall_threshold)
    if isinstance(resolved, dict):
        return resolved
    gate_config, raise_note = resolved
    try:
        invoice = InvoiceData.model_validate(data)
    except ValidationError as exc:
        return _error(f"data failed InvoiceData validation: {exc}")
    try:
        result = _build_pipeline(gate_config, heuristics_only, validate_only=True).validate(
            text, invoice
        )
    except Exception as exc:  # noqa: BLE001 - the envelope is the contract, not a traceback
        return _error(str(exc))
    return _with_raise_note(result.model_dump(), raise_note)
