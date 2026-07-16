"""A deterministic `LLMClient` for offline runs (tests + eval).

This is the mock at the LLM boundary. It returns pre-seeded responses in call
order, records every call for assertions, and raises loudly if it runs dry --
so a test can never accidentally pass by making a real network call.
"""

from __future__ import annotations

from typing import Any


class ScriptedLLMClient:
    """Return canned responses in order; record calls for inspection."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise RuntimeError(
                "ScriptedLLMClient ran out of scripted responses -- the pipeline "
                "made more LLM calls than expected (or a real call leaked through)."
            )
        return self._responses.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._responses)
