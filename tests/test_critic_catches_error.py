"""THE flagship test: a deliberately wrong Generator output is caught by the
second-model Critic and routed to human review -- not silently returned.

This isolates the Critic as the decisive signal: the injected wrong value is
present in the source text (so the deterministic grounding heuristic does NOT
penalize it -- factor stays 1.0), yet the pipeline still escalates because the
Critic assigns it low confidence. Flip only the Critic's verdict and the
decision flips too -- proving the verification step is doing real work.
"""

from __future__ import annotations

import json

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

from tests.conftest import SOURCE

# Generator hallucinates the vendor: it emits "Invoice" (a word that literally
# appears in the source as a heading), so grounding can't catch it -- only a
# semantic critic can.
WRONG_EXTRACTION = {
    "vendor_name": "Invoice",  # WRONG: that's a heading, not the vendor (ACME CORP)
    "invoice_number": "INV-1001",
    "invoice_date": "2026-03-14",
    "subtotal": 100.0,
    "tax": 8.5,
    "total": 108.5,
    "currency": "USD",
}


def _critic_json(vendor_confidence: float, overall: float) -> str:
    verdicts = []
    for f in WRONG_EXTRACTION:
        if f == "vendor_name":
            verdicts.append({
                "field": "vendor_name",
                "confidence": vendor_confidence,
                "grounded": vendor_confidence >= 0.7,
                "rationale": "'Invoice' is a document heading, not the seller; the vendor is 'ACME CORP'",
            })
        else:
            verdicts.append({"field": f, "confidence": 0.95, "grounded": True, "rationale": "ok"})
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": overall,
                       "summary": "vendor_name looks wrong"})


def _pipeline(critic_raw: str) -> ReflectivePipeline:
    gen = Generator(ScriptedLLMClient([json.dumps(WRONG_EXTRACTION)]))
    crit = Critic(ScriptedLLMClient([critic_raw]))
    return ReflectivePipeline(gen, crit)


def test_grounding_alone_does_not_catch_this_error():
    # Guard the premise: the wrong value IS in the source, so heuristics pass it.
    data = InvoiceData.model_validate(WRONG_EXTRACTION)
    report = run_heuristics(SOURCE, data)
    assert report.factor_for("vendor_name") == 1.0


def test_critic_catches_injected_error_and_routes_to_review():
    # Critic assigns LOW confidence to the hallucinated vendor; overall stays
    # ABOVE the overall-threshold, so the escalation is purely field-driven.
    result = _pipeline(_critic_json(vendor_confidence=0.15, overall=0.85)).extract(SOURCE)

    assert result.status == "needs_review"
    flagged = {f.field for f in result.flagged_fields}
    assert "vendor_name" in flagged
    # The critic's rationale must surface to the human in the flagged payload.
    vendor_flag = next(f for f in result.flagged_fields if f.field == "vendor_name")
    assert "heading" in vendor_flag.reason.lower()
    assert vendor_flag.value == "Invoice"
    # And the (wrong) value is NOT silently emitted as trustworthy.
    assert result.field_confidence["vendor_name"] < 0.7


def test_same_extraction_passes_when_critic_approves_it():
    # Flip ONLY the critic verdict (high confidence) -> decision flips to ok.
    # This proves the Critic is the decisive signal, not the plumbing.
    result = _pipeline(_critic_json(vendor_confidence=0.95, overall=0.95)).extract(SOURCE)
    assert result.status == "ok"
    assert result.flagged_fields == []
