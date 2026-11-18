"""Robust JSON extraction from raw LLM text.

Models sometimes wrap JSON in ```json fences or add a stray sentence despite
instructions. We defensively strip fences and, as a last resort, slice the
first balanced `{...}` object out of the text. If that object is not valid JSON
(prose such as "I used the {field: value} format"), each later *top-level*
balanced `{...}` is tried in turn, so a temperature-0 model that repeats the
same preamble does not burn a paid retry; we never descend into an object, so a
truncated reply still fails loudly instead of yielding a nested fragment.

The shared retry loop (`complete_parsed`) appends the previous attempt's
rejection reason (at most 300 chars) to the retry's user prompt: a model that
made a systematic mistake repeats it when re-sent a byte-identical prompt.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from importlib import resources
from typing import Any, TypeVar

from pydantic import ValidationError

from reflect_mcp.logging_config import log_stage
from reflect_mcp.providers import LLMClient

T = TypeVar("T")

_RETRY_DELAY_SECONDS = 0.5
# Cap on the rejection reason fed back to the model: enough to name the failing
# fields, bounded so a long pydantic error cannot bloat a paid prompt.
_RETRY_REASON_MAX_CHARS = 300


class ParseError(ValueError):
    """Raised when no valid JSON object can be recovered from model output."""


def load_prompt(name: str) -> str:
    """Load a versioned prompt file bundled in the package."""
    return resources.files("reflect_mcp.prompts").joinpath(name).read_text(encoding="utf-8")


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line (``` or ```json) and the trailing fence
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _first_json_object(text: str, pos: int = 0) -> str | None:
    """Return the first balanced `{...}` starting at or after *pos*, else None."""
    start = text.find("{", pos)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from raw model output.

    A malformed first balanced object is skipped in favour of the first later
    top-level balanced object that decodes; an unbalanced segment ends the walk.
    """
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _first_json_object(cleaned)
        if candidate is None:
            raise ParseError(f"No JSON object found in model output: {text[:200]!r}")
        # Walk top-level segments by index: each next candidate starts at the
        # first "{" after the previous one ends. `cleaned.find(candidate)` would
        # re-find an earlier copy of a repeated stray segment and loop forever.
        end = cleaned.find("{") + len(candidate)
        first_exc: json.JSONDecodeError | None = None
        while True:
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                first_exc = first_exc or exc
            nxt = _first_json_object(cleaned, end)
            if nxt is None:
                # Report the first failure so the message matches pre-walk output.
                raise ParseError(f"Malformed JSON in model output: {first_exc}") from first_exc
            end = cleaned.find("{", end) + len(nxt)
            candidate = nxt
    if not isinstance(obj, dict):
        raise ParseError(f"Expected a JSON object, got {type(obj).__name__}")
    return obj


def _retry_suffix(exc: ParseError | ValidationError) -> str:
    """Tell the model why its last reply was rejected (reason cut to 300 chars)."""
    reason = str(exc)[:_RETRY_REASON_MAX_CHARS]
    return f"\n\nYour previous reply was rejected: {reason}\nReply with only the corrected JSON object."


def complete_parsed(
    client: LLMClient,
    *,
    system: str,
    user: str,
    validate: Callable[[dict[str, Any]], T],
    max_retries: int,
    event: str,
    trace_id: str,
) -> T:
    """Call `client`, parse its reply and validate it, retrying on bad output.

    This is the one retry policy for every stage, so a change to backoff or
    retry logging cannot land in the Generator but miss the Critic. It shares
    plumbing only: each stage still passes its own client, prompts and schema
    `validate`, which keeps the Critic a decorrelated grader. A provider error
    from `complete` propagates unretried. Once retries run out, the last
    `ParseError`/`ValidationError` is re-raised for the stage to wrap.

    Attempt 1 sends `user` unchanged; each retry sends `user` plus only the
    *immediately preceding* rejection reason (at most 300 chars), so the model
    learns what failed instead of re-rolling blind. The reason is the stage's
    own error, so the Critic still sees nothing beyond its attempt-1 inputs.
    """
    last_exc: ParseError | ValidationError | None = None
    for attempt in range(1 + max_retries):
        prompt = user if last_exc is None else user + _retry_suffix(last_exc)
        raw = client.complete(system=system, user=prompt)
        try:
            return validate(parse_json_object(raw))
        except (ParseError, ValidationError) as exc:
            last_exc = exc
            if attempt < max_retries:
                log_stage(
                    event,
                    trace_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(exc),
                )
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_exc  # type: ignore[misc]  # set: max_retries >= 0 runs the loop
