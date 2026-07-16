"""Generator stage: one LLM call -> typed `InvoiceData`."""

from __future__ import annotations

from pydantic import ValidationError

from reflect_mcp.parsing import ParseError, load_prompt, parse_json_object
from reflect_mcp.providers import LLMClient
from reflect_mcp.schema import InvoiceData

GENERATOR_SYSTEM = load_prompt("generator.txt")


class GeneratorError(RuntimeError):
    """Raised when the Generator's output cannot be coerced into InvoiceData."""


class Generator:
    """Single-call extractor. Validates the LLM output against the schema."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate(self, text: str) -> InvoiceData:
        user = f"Extract the invoice fields from this document:\n\n{text}"
        raw = self.client.complete(system=GENERATOR_SYSTEM, user=user)
        try:
            obj = parse_json_object(raw)
            return InvoiceData.model_validate(obj)
        except (ParseError, ValidationError) as exc:
            raise GeneratorError(f"Generator produced invalid extraction: {exc}") from exc
