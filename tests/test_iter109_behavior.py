"""Black-box behavioral tests for iter-109 (roadmap #116): the Critic reply
boundary rejects a JSON boolean ``confidence`` / ``overall_confidence``.

A Critic that answers yes/no (``true``/``false``) instead of a number must not
pose as a 1.0 (or 0.0) grade: the reply is rejected at the
``FieldVerdict``/``CritiqueResult`` boundary, retried once with the validation
message, and never grades a field ``ok``. Numbers and numeric strings are
unchanged.

Covers EB1-EB11 from pm.md. EB1, EB2, EB4-EB9 are the discriminators (red on
the pre-change tree); EB3, EB10, EB11 are preservation checks, green on both
trees by design, each paired with a discriminator.

Isolation contract: only public interfaces are driven (schema models, Critic,
Generator, ReflectivePipeline, ScriptedLLMClient, tools.validate_tool, the
golden dataset/replay files, ``make eval``); no network, no secrets, retry
delay set to 0.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

import reflect_mcp.parsing as parsing
import reflect_mcp.pipeline as pipeline_mod
from reflect_mcp import tools
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import CritiqueResult, ExtractionResult, FieldVerdict, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"
MSG = "not a JSON boolean"

S = (
    "Northwind Traders\nInvoice No: NW-3310\nDate: 2026-05-01\n"
    "Shipped via Contoso Freight\nSubtotal: 300.00\nTax: 24.00\n"
    "Total: 324.00\nCurrency: USD"
)
D = InvoiceData(
    vendor_name="Contoso Freight", invoice_number="NW-3310", invoice_date="2026-05-01",
    subtotal=300.0, tax=24.0, total=324.0, currency="USD",
)
FIELDS = list(InvoiceData.model_fields)


def BOOL(x: bool) -> str:
    return json.dumps({
        "field_verdicts": [
            {"field": f, "confidence": x, "grounded": True, "rationale": "ok"} for f in FIELDS
        ],
        "overall_confidence": x,
        "summary": "s",
    })


def VALID() -> str:
    verdicts = []
    for f in FIELDS:
        if f == "vendor_name":
            verdicts.append({"field": f, "confidence": 0.2, "grounded": False,
                             "rationale": "vendor is the carrier, not the issuer"})
        else:
            verdicts.append({"field": f, "confidence": 0.95, "grounded": True, "rationale": "ok"})
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.6, "summary": "s"})


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch):
    monkeypatch.setattr(parsing, "_RETRY_DELAY_SECONDS", 0)


def _flag_names(result) -> list[str]:
    out = []
    for f in result.flagged_fields:
        out.append(f["field"] if isinstance(f, dict) else f.field)
    return sorted(out)


def _locs(exc: ValidationError) -> list[tuple]:
    return [tuple(e["loc"]) for e in exc.errors()]


# ---- EB1 -------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, False])
def test_eb1_field_verdict_rejects_bool(value):
    with pytest.raises(ValidationError) as ei:
        FieldVerdict(field="x", confidence=value)
    assert MSG in str(ei.value)
    assert ("confidence",) in _locs(ei.value)


# ---- EB2 -------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, False])
def test_eb2_critique_result_rejects_bool(value):
    with pytest.raises(ValidationError) as ei:
        CritiqueResult(overall_confidence=value)
    assert MSG in str(ei.value)
    assert ("overall_confidence",) in _locs(ei.value)


# ---- EB3 (preservation) ----------------------------------------------------
@pytest.mark.parametrize("value,expected", [(1, 1.0), (0, 0.0), ("0.9", 0.9), (0.5, 0.5)])
def test_eb3_numbers_still_accepted(value, expected):
    fv = FieldVerdict(field="x", confidence=value)
    cr = CritiqueResult(overall_confidence=value)
    assert fv.confidence == pytest.approx(expected) and type(fv.confidence) is float
    assert cr.overall_confidence == pytest.approx(expected) and type(cr.overall_confidence) is float


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_eb3_out_of_range_still_raises(value):
    with pytest.raises(ValidationError) as e1:
        FieldVerdict(field="x", confidence=value)
    with pytest.raises(ValidationError) as e2:
        CritiqueResult(overall_confidence=value)
    assert MSG not in str(e1.value) and MSG not in str(e2.value)


# ---- EB4 / EB5 / EB7 -------------------------------------------------------
@pytest.mark.parametrize("value", [True, False])
def test_eb4_eb7_bool_reply_raises_after_two_calls(value):
    client = ScriptedLLMClient([BOOL(value), BOOL(value)])
    with pytest.raises(CriticError) as ei:
        ReflectivePipeline(None, Critic(client)).validate(S, D)
    assert MSG in str(ei.value)
    assert len(client.calls) == 2


def test_eb5_retry_prompt_carries_message():
    client = ScriptedLLMClient([BOOL(True), BOOL(True)])
    with pytest.raises(CriticError):
        ReflectivePipeline(None, Critic(client)).validate(S, D)
    assert MSG not in client.calls[0]["user"]
    assert MSG in client.calls[1]["user"]


# ---- EB6 -------------------------------------------------------------------
def test_eb6_recovery_after_bool_reply():
    client = ScriptedLLMClient([BOOL(True), VALID()])
    r = ReflectivePipeline(None, Critic(client)).validate(S, D)
    assert r.status == "needs_review"
    assert "vendor_name" in _flag_names(r)
    assert len(client.calls) == 2


def test_eb6_counterpart_valid_reply_alone_one_call():
    client = ScriptedLLMClient([VALID()])
    r = ReflectivePipeline(None, Critic(client)).validate(S, D)
    assert r.status == "needs_review" and "vendor_name" in _flag_names(r)
    assert len(client.calls) == 1


# ---- EB8 -------------------------------------------------------------------
def test_eb8_mcp_validate_tool_is_error(monkeypatch):
    client = ScriptedLLMClient([BOOL(True), BOOL(True)])
    monkeypatch.setattr(pipeline_mod, "build_default_pipeline",
                        lambda *a, **k: ReflectivePipeline(None, Critic(client)))
    out = tools.validate_tool(text=S, data=D.model_dump())
    assert isinstance(out, dict)
    assert out["status"] != "ok"
    assert out["status"] == "error"
    assert MSG in json.dumps(out)
    assert MSG in str(out.get("error_message", ""))


# ---- EB9: golden-derived invariant -----------------------------------------
def _booled(raw: str) -> str:
    d = json.loads(raw)
    for v in d["field_verdicts"]:
        v["confidence"] = True
    d["overall_confidence"] = True
    return json.dumps(d)


def _rows():
    gold = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = json.loads(REPLAY.read_text(encoding="utf-8"))
    return gold, rep


def _extract(gen_raw: str, text: str, critic_replies: list[str]):
    c = ScriptedLLMClient(critic_replies)
    p = ReflectivePipeline(Generator(ScriptedLLMClient([gen_raw])), Critic(c))
    r = p.extract(text)
    return (r.status, round(r.overall_confidence, 6), _flag_names(r)), len(c.calls)


def test_eb9_golden_bool_then_good_equals_plain_replay():
    gold, rep = _rows()
    assert len(gold) > 0
    for row in gold:
        entry = rep[row["id"]]
        cr = entry["critic_raw"]
        assert len(json.loads(cr)["field_verdicts"]) > 0, row["id"]
        plain, n0 = _extract(entry["generator_raw"], row["text"], [cr])
        assert n0 == 1
        got, n = _extract(entry["generator_raw"], row["text"], [_booled(cr), cr])
        assert got == plain, row["id"]
        assert n == 2, row["id"]


def test_eb9_golden_bool_twice_raises():
    gold, rep = _rows()
    assert len(gold) > 0
    for row in gold:
        entry = rep[row["id"]]
        b = _booled(entry["critic_raw"])
        with pytest.raises(CriticError):
            _extract(entry["generator_raw"], row["text"], [b, b])


# ---- EB10 (preservation) ---------------------------------------------------
@pytest.mark.parametrize("model,sha", [
    (FieldVerdict, "7fb4decff128"),
    (CritiqueResult, "bc64cbf79a7f"),
    (ExtractionResult, "174131d84766"),
    (InvoiceData, "e25030a2013c"),
])
def test_eb10_schema_unchanged(model, sha):
    blob = json.dumps(model.model_json_schema(), sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:12] == sha


# ---- EB11 (preservation) ---------------------------------------------------
def test_eb11_make_eval_unchanged():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    lines = r.stdout.splitlines()
    for line in (
        "examples          : 13",
        "field accuracy    : 91.2%  (83/91)",
        "silent-wrong      : 0",
        "catch rate        : 100.0%  (8/8)",
    ):
        assert line in lines, line


def test_eb11_golden_files_untouched_vs_head():
    probe = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=REPO,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout")
    for rel in ("eval/golden/dataset.jsonl", "eval/fixtures/replay.json"):
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True)
        assert head.returncode == 0, rel
        assert (REPO / rel).read_bytes() == head.stdout, rel
