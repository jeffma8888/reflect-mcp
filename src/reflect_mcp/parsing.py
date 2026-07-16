"""Robust JSON extraction from raw LLM text.

Models sometimes wrap JSON in ```json fences or add a stray sentence despite
instructions. We defensively strip fences and, as a last resort, slice the
first balanced `{...}` object out of the text.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


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


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
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
    """Best-effort parse of a JSON object from raw model output."""
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _first_json_object(cleaned)
        if candidate is None:
            raise ParseError(f"No JSON object found in model output: {text[:200]!r}")
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Malformed JSON in model output: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError(f"Expected a JSON object, got {type(obj).__name__}")
    return obj
