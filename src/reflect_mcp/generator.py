"""Generator stage: LLM call (with configurable retry) -> typed `InvoiceData`."""

from __future__ import annotations

from pydantic import ValidationError

from reflect_mcp.parsing import ParseError, complete_parsed, load_prompt
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import InvoiceData

GENERATOR_SYSTEM = load_prompt("generator.txt")


class GeneratorError(RuntimeError):
    """Raised when the Generator's output cannot be coerced into InvoiceData."""


class Generator:
    """Extractor with retry. Validates the LLM output against the schema."""

    def __init__(self, client: LLMClient, *, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def generate(self, text: str, trace_id: str = "") -> InvoiceData:
        """Extract `text`; `trace_id` tags retry logs so one call stays greppable."""
        user = f"Extract the invoice fields from this document:\n\n{text}"
        try:
            return complete_parsed(
                self.client,
                system=GENERATOR_SYSTEM,
                user=user,
                validate=InvoiceData.model_validate,
                max_retries=self.max_retries,
                event="generate.retry",
                trace_id=trace_id,
            )
        except (ParseError, ValidationError) as exc:
            raise GeneratorError(
                f"Generator produced invalid extraction: {exc}"
            ) from exc
