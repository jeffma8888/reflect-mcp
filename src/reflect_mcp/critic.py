"""Critic stage: a SECOND model scores each field + overall confidence.

Using a different model (or at least a different role/config) is the point --
self-critique with the same model tends to rubber-stamp its own errors. The
Critic is fed the source text AND the candidate extraction and must ground each
field against the source.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from reflect_mcp.parsing import ParseError, complete_parsed, load_prompt
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import CritiqueResult, InvoiceData

CRITIC_SYSTEM = load_prompt("critic.txt")


class CriticError(RuntimeError):
    """Raised when the Critic's output cannot be parsed into CritiqueResult."""


class Critic:
    def __init__(self, client: LLMClient, *, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def critique(
        self, text: str, extraction: InvoiceData, trace_id: str = ""
    ) -> CritiqueResult:
        """Grade `extraction`; `trace_id` tags retry logs so one call stays greppable."""
        user = (
            "SOURCE DOCUMENT:\n"
            f"{text}\n\n"
            "CANDIDATE EXTRACTION (JSON):\n"
            f"{json.dumps(extraction.model_dump(), sort_keys=True)}\n\n"
            "Verify each field against the source and return your critique JSON."
        )
        try:
            return complete_parsed(
                self.client,
                system=CRITIC_SYSTEM,
                user=user,
                validate=CritiqueResult.model_validate,
                max_retries=self.max_retries,
                event="critique.retry",
                trace_id=trace_id,
            )
        except (ParseError, ValidationError) as exc:
            raise CriticError(f"Critic produced invalid critique: {exc}") from exc
