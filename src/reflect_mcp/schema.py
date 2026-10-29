"""Typed extraction schema + verification/decision payloads.

The domain is *invoice / receipt extraction*: messy text in, one typed
`InvoiceData` object out. Pydantic gives us validation for free and a stable
contract the Critic can score field-by-field.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class InvoiceData(BaseModel):
    """The Generator's extraction target.

    All fields are optional: a missing field on the source document should come
    back as `null`, not be hallucinated. That is a deliberate design choice --
    the Critic can only ground a value that the model actually committed to.
    """

    vendor_name: Optional[str] = Field(default=None, description="Legal/trading name of the seller.")
    invoice_number: Optional[str] = Field(default=None, description="Invoice / receipt identifier.")
    invoice_date: Optional[str] = Field(default=None, description="Invoice date, ISO 8601 (YYYY-MM-DD).")
    subtotal: Optional[float] = Field(default=None, description="Pre-tax amount.")
    tax: Optional[float] = Field(default=None, description="Tax amount.")
    total: Optional[float] = Field(default=None, description="Grand total charged.")
    currency: Optional[str] = Field(default=None, description="ISO 4217 currency code, e.g. USD.")

    model_config = {"extra": "ignore"}


# Canonical field order -- used for logging, gate iteration, and eval scoring.
EXTRACTION_FIELDS: list[str] = list(InvoiceData.model_fields.keys())


class FieldVerdict(BaseModel):
    """The Critic's assessment of one extracted field."""

    field: str
    confidence: float = Field(ge=0.0, le=1.0, description="Critic's [0,1] confidence the value is correct.")
    grounded: bool = Field(default=True, description="Whether the value is supported by the source text.")
    rationale: str = Field(default="", description="Short human-readable justification.")


class CritiqueResult(BaseModel):
    """Full second-model critique of one extraction."""

    field_verdicts: list[FieldVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""

    def verdict_for(self, field: str) -> Optional[FieldVerdict]:
        """Return the field's verdict; on duplicates, the least confident one.

        A Critic that grades the same field twice is contradicting itself, and
        "first one wins" would let a confident duplicate mask a doubtful one.
        Taking the minimum keeps the gate's rule that evidence only lowers
        confidence. On an equal-confidence tie the ungrounded verdict wins
        (``False < True``), so the result does not depend on verdict order.
        """
        matches = [v for v in self.field_verdicts if v.field == field]
        if not matches:
            return None
        return min(matches, key=lambda v: (v.confidence, v.grounded))


class FlaggedField(BaseModel):
    """A field the gate is not confident enough to emit silently."""

    field: str
    value: Any = None
    confidence: float
    reason: str


class ProvenanceSpan(BaseModel):
    """Character-offset span locating an extracted value in the source text."""

    start: int
    end: int


class ExtractionResult(BaseModel):
    """What the `extract` tool returns.

    `status == "needs_review"` means: do NOT trust `data` blindly; a human should
    confirm the fields listed in `flagged_fields`. This is the trust contract --
    the pipeline surfaces uncertainty instead of a confident guess.
    """

    status: Literal["ok", "needs_review"]
    data: InvoiceData
    field_confidence: dict[str, float] = Field(default_factory=dict)
    overall_confidence: float
    flagged_fields: list[FlaggedField] = Field(default_factory=list)
    field_provenance: dict[str, list[ProvenanceSpan]] = Field(default_factory=dict)
    trace_id: str
    critic_summary: str = ""
    notes: str = ""
