"""Critic stage: a SECOND model scores each field + overall confidence.

Using a different model (or at least a different role/config) is the point --
self-critique with the same model tends to rubber-stamp its own errors. The
Critic is fed the source text AND the candidate extraction and must ground each
field against the source.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from reflect_mcp.parsing import ParseError, load_prompt, parse_json_object
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import CritiqueResult, InvoiceData

CRITIC_SYSTEM = load_prompt("critic.txt")


class CriticError(RuntimeError):
    """Raised when the Critic's output cannot be parsed into CritiqueResult."""


class Critic:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def critique(self, text: str, extraction: InvoiceData) -> CritiqueResult:
        user = (
            "SOURCE DOCUMENT:\n"
            f"{text}\n\n"
            "CANDIDATE EXTRACTION (JSON):\n"
            f"{json.dumps(extraction.model_dump(), sort_keys=True)}\n\n"
            "Verify each field against the source and return your critique JSON."
        )
        raw = self.client.complete(system=CRITIC_SYSTEM, user=user)
        try:
            obj = parse_json_object(raw)
            return CritiqueResult.model_validate(obj)
        except (ParseError, ValidationError) as exc:
            raise CriticError(f"Critic produced invalid critique: {exc}") from exc
