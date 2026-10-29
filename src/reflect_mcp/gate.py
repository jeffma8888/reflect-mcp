"""The human-in-the-loop decision gate.

Combines the Critic's semantic confidence with the deterministic heuristic
factor per field, then decides whether the result can be emitted (`ok`) or must
be escalated (`needs_review`). The core trust rule:

    A field is flagged if its combined confidence < field_threshold.
    The whole result needs review if ANY field is flagged OR the overall
    confidence < overall_threshold.

The gate also enforces the Critic's own contract on committed (non-null)
fields: a verdict marked `grounded:false`, or no verdict at all, caps that
field's critic confidence at 0.5 (`UNGROUNDED_CONFIDENCE_CAP`). When the Critic
returns duplicate verdicts for a field, the minimum-confidence one is used.

Below threshold we return a `needs_review` payload naming the flagged fields --
never a silent guess.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reflect_mcp.heuristics import HeuristicReport
from reflect_mcp.schema import (
    EXTRACTION_FIELDS,
    CritiqueResult,
    ExtractionResult,
    FieldVerdict,
    FlaggedField,
    InvoiceData,
)


# Critic confidence ceiling for a committed field the Critic did not vouch for
# (grounded:false or no verdict). A value, not a hard flag, so an integrator who
# deliberately sets field_threshold <= 0.5 keeps the permissive gate they asked for.
UNGROUNDED_CONFIDENCE_CAP = 0.5

NOT_GROUNDED_REASON = "critic marked the value not grounded in the source"
NO_VERDICT_REASON = "critic returned no verdict for this field"


class GateConfig(BaseModel):
    model_config = {"frozen": True}

    field_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    overall_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


def _combine(critic_conf: float, heuristic_factor: float) -> float:
    # Multiplicative: a structural failure (low factor) caps the field no matter
    # how confident the Critic was. Confidence can only go DOWN from evidence.
    return round(critic_conf * heuristic_factor, 4)


def _critic_confidence(
    value: object, verdict: FieldVerdict | None, overall_confidence: float
) -> tuple[float, str | None]:
    """Return the field's critic confidence and any contract-violation reason.

    Null fields keep inheriting the overall confidence untouched: they cannot be
    wrong-and-emitted, so the Critic owes them no verdict. The cap is a `min`,
    so it can only lower a confidence, never raise one.
    """
    if verdict is None:
        if value is None:
            return overall_confidence, None
        return min(overall_confidence, UNGROUNDED_CONFIDENCE_CAP), NO_VERDICT_REASON
    if value is not None and not verdict.grounded:
        return min(verdict.confidence, UNGROUNDED_CONFIDENCE_CAP), NOT_GROUNDED_REASON
    return verdict.confidence, None


def decide(
    data: InvoiceData,
    critique: CritiqueResult,
    heuristics: HeuristicReport,
    trace_id: str,
    config: GateConfig | None = None,
) -> ExtractionResult:
    """Fuse critic + heuristics into a gated ExtractionResult."""
    config = config or GateConfig()

    field_confidence: dict[str, float] = {}
    flagged: list[FlaggedField] = []

    for field in EXTRACTION_FIELDS:
        value = getattr(data, field)
        verdict = critique.verdict_for(field)
        critic_conf, contract_reason = _critic_confidence(
            value, verdict, critique.overall_confidence
        )
        factor = heuristics.factor_for(field)
        combined = _combine(critic_conf, factor)
        field_confidence[field] = combined

        # Only fields the model committed to (non-null) can be "wrong" and flagged.
        if value is not None and combined < config.field_threshold:
            reasons = []
            if verdict is not None and verdict.rationale:
                reasons.append(verdict.rationale)
            if contract_reason:
                reasons.append(contract_reason)
            heur_reason = heuristics.reason_for(field)
            if heur_reason:
                reasons.append(heur_reason)
            flagged.append(
                FlaggedField(
                    field=field,
                    value=value,
                    confidence=combined,
                    reason="; ".join(reasons) or "below confidence threshold",
                )
            )

    needs_review = bool(flagged) or critique.overall_confidence < config.overall_threshold
    status = "needs_review" if needs_review else "ok"

    if status == "needs_review":
        if flagged:
            notes = (
                f"{len(flagged)} field(s) below the {config.field_threshold:.0%} "
                "confidence threshold require human review before use."
            )
        else:
            notes = (
                f"Overall confidence {critique.overall_confidence:.0%} is below the "
                f"{config.overall_threshold:.0%} threshold; escalating for review."
            )
    else:
        notes = "All fields cleared the confidence thresholds."

    return ExtractionResult(
        status=status,
        data=data,
        field_confidence=field_confidence,
        overall_confidence=critique.overall_confidence,
        flagged_fields=flagged,
        field_provenance=heuristics.provenance,
        trace_id=trace_id,
        critic_summary=critique.summary,
        notes=notes,
    )
