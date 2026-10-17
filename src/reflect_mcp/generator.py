"""Generator stage: LLM call (with configurable retry) -> typed `InvoiceData`."""

from __future__ import annotations

import time

from pydantic import ValidationError

from reflect_mcp.logging_config import log_stage
from reflect_mcp.parsing import ParseError, load_prompt, parse_json_object
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import InvoiceData

GENERATOR_SYSTEM = load_prompt("generator.txt")

_RETRY_DELAY_SECONDS = 0.5


class GeneratorError(RuntimeError):
    """Raised when the Generator's output cannot be coerced into InvoiceData."""


class Generator:
    """Extractor with retry. Validates the LLM output against the schema."""

    def __init__(self, client: LLMClient, *, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def generate(self, text: str) -> InvoiceData:
        user = f"Extract the invoice fields from this document:\n\n{text}"
        last_exc: ParseError | ValidationError | None = None
        for attempt in range(1 + self.max_retries):
            raw = self.client.complete(system=GENERATOR_SYSTEM, user=user)
            try:
                obj = parse_json_object(raw)
                return InvoiceData.model_validate(obj)
            except (ParseError, ValidationError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    log_stage(
                        "generate.retry",
                        "",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(exc),
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)
        raise GeneratorError(
            f"Generator produced invalid extraction: {last_exc}"
        ) from last_exc
