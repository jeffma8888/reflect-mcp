"""Black-box behavioral tests for iter-54: the gate enforces the Critic contract.

Covers the 12 Expected Behaviors from pm.md:
  1. grounded:false caps a committed field's critic confidence at 0.5 and flags it.
  2. All fields grounded:false -> all flagged at 0.5.
  3. The cap never raises a confidence (min, not assignment).
  4. The cap respects permissive thresholds (strict `<`).
  5. A missing verdict caps and flags (empty list and partial list).
  6. Null fields are unchanged (inherit overall, never flagged/capped).
  7. Duplicate verdicts resolve to the minimum (order-independent).
  8. Clean path unchanged.
  9. Neutral-critique paths (critic=None) unaffected.
 10. Golden row inv-007 + replay fixture -> needs_review on vendor_name.
 11. Eval readout (CLI stdout + evaluate()).
 12. Docs: README Gate bullet, gate module docstring, eval block == stdout,
     gate-only README snippet executes to `ok`.

Isolation contract: tests drive the public package API, the golden data files,
the README and the eval CLI, asserting on observable output only.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp
from reflect_mcp import (
    EXTRACTION_FIELDS,
    CritiqueResult,
    FieldVerdict,
    GateConfig,
    InvoiceData,
    ReflectivePipeline,
    decide,
)
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"
README = REPO / "README.md"

SOURCE = (
    "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: $100.00\n"
    "Tax: $8.50\nTotal: $108.50\nPaid in USD"
)
CONTRACT_REASON = "critic marked the value not grounded in the source"
MISSING_REASON = "critic returned no verdict for this field"

INV007_TEXT = (
    "NORTHWIND TRADERS\nInvoice NW-2201\nDate: 2026-05-02\nBill to: Contoso Ltd\n"
    "Subtotal: 200.00\nTax: 16.00\nTotal: 216.00\nCurrency: USD"
)
INV007_EXPECTED = {
    "vendor_name": "NORTHWIND TRADERS", "invoice_number": "NW-2201",
    "invoice_date": "2026-05-02", "subtotal": 200.0, "tax": 16.0,
    "total": 216.0, "currency": "USD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data() -> InvoiceData:
    return InvoiceData(
        vendor_name="ACME CORP", invoice_number="INV-1001", invoice_date="2026-03-14",
        subtotal=100.0, tax=8.5, total=108.5, currency="USD",
    )


def _h(data: InvoiceData | None = None):
    return run_heuristics(SOURCE, data or _data())


def _verdicts(conf: float = 0.95, grounded: bool = True, skip=(), overrides=None):
    overrides = overrides or {}
    out = []
    for f in EXTRACTION_FIELDS:
        if f in skip:
            continue
        if f in overrides:
            out.append(overrides[f])
        else:
            out.append(FieldVerdict(field=f, confidence=conf, grounded=grounded,
                                    rationale="matches source"))
    return out


def _flags(result) -> dict:
    return {f.field: f for f in result.flagged_fields}


def _rows() -> dict[str, dict]:
    out = {}
    for line in GOLDEN.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _run_row(row_id: str, critic_raw: str | None = None):
    row = _rows()[row_id]
    fx = _replay()[row_id]
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([fx["generator_raw"]])),
        Critic(ScriptedLLMClient([critic_raw or fx["critic_raw"]])),
    )
    return pipe.extract(row["text"])


def _eval_stdout() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "eval" / "run_eval.py")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


def test_setup_heuristics_are_neutral():
    h = _h()
    for f in EXTRACTION_FIELDS:
        assert h.factor_for(f) == 1.0, (f, h.reason_for(f))


# ---------------------------------------------------------------------------
# EB-1: grounded:false caps and flags
# ---------------------------------------------------------------------------

def _eb1_critique() -> CritiqueResult:
    tax = FieldVerdict(field="tax", confidence=0.95, grounded=False,
                       rationale="tax line says 8.50")
    return CritiqueResult(field_verdicts=_verdicts(overrides={"tax": tax}),
                          overall_confidence=0.95)


def test_eb1_grounded_false_caps_and_flags():
    result = decide(_data(), _eb1_critique(), _h(), trace_id="t1")
    assert result.status == "needs_review"
    assert result.field_confidence["tax"] == 0.5
    flags = _flags(result)
    assert list(flags) == ["tax"]
    assert flags["tax"].confidence == 0.5
    assert flags["tax"].reason == f"tax line says 8.50; {CONTRACT_REASON}"


def test_eb1_other_fields_untouched():
    result = decide(_data(), _eb1_critique(), _h(), trace_id="t1")
    for f in EXTRACTION_FIELDS:
        if f != "tax":
            assert result.field_confidence[f] == 0.95, f


def test_eb1_reason_order_with_heuristic_reason():
    # tax wrong & ungrounded -> heuristic reason appended after contract reason.
    data = _data().model_copy(update={"tax": 9.99})
    h = run_heuristics(SOURCE, data)
    assert h.factor_for("tax") < 1.0
    tax = FieldVerdict(field="tax", confidence=0.95, grounded=False, rationale="r1")
    crit = CritiqueResult(field_verdicts=_verdicts(overrides={"tax": tax}),
                          overall_confidence=0.95)
    result = decide(data, crit, h, trace_id="t1b")
    reason = _flags(result)["tax"].reason
    assert reason.startswith(f"r1; {CONTRACT_REASON}; ")
    assert h.reason_for("tax") in reason


# ---------------------------------------------------------------------------
# EB-2: all grounded:false
# ---------------------------------------------------------------------------

def test_eb2_all_grounded_false_all_flagged():
    crit = CritiqueResult(field_verdicts=_verdicts(grounded=False), overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t2")
    assert result.status == "needs_review"
    flags = _flags(result)
    assert set(flags) == set(EXTRACTION_FIELDS)
    assert len(result.flagged_fields) == 7
    for f in EXTRACTION_FIELDS:
        assert flags[f].confidence == 0.5
        assert CONTRACT_REASON in flags[f].reason


# ---------------------------------------------------------------------------
# EB-3: cap never raises
# ---------------------------------------------------------------------------

def test_eb3_cap_never_raises_confidence():
    tax = FieldVerdict(field="tax", confidence=0.3, grounded=False, rationale="x")
    crit = CritiqueResult(field_verdicts=_verdicts(overrides={"tax": tax}),
                          overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t3")
    assert result.field_confidence["tax"] == pytest.approx(0.3)
    assert _flags(result)["tax"].confidence == pytest.approx(0.3)
    assert result.status == "needs_review"


# ---------------------------------------------------------------------------
# EB-4: permissive thresholds
# ---------------------------------------------------------------------------

def test_eb4_permissive_threshold_accepts_capped_value():
    cfg = GateConfig(field_threshold=0.5, overall_threshold=0.5)
    result = decide(_data(), _eb1_critique(), _h(), trace_id="t4", config=cfg)
    assert result.status == "ok"
    assert result.flagged_fields == []
    assert result.field_confidence["tax"] == 0.5


def test_eb4_threshold_just_above_cap_flags():
    cfg = GateConfig(field_threshold=0.51, overall_threshold=0.5)
    result = decide(_data(), _eb1_critique(), _h(), trace_id="t4b", config=cfg)
    assert result.status == "needs_review"
    assert list(_flags(result)) == ["tax"]


# ---------------------------------------------------------------------------
# EB-5: missing verdict
# ---------------------------------------------------------------------------

def test_eb5_empty_verdicts_caps_and_flags_all():
    crit = CritiqueResult(field_verdicts=[], overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t5")
    assert result.status == "needs_review"
    flags = _flags(result)
    assert set(flags) == set(EXTRACTION_FIELDS)
    for f in EXTRACTION_FIELDS:
        assert flags[f].confidence == 0.5
        assert flags[f].reason == MISSING_REASON


def test_eb5_partial_list_flags_only_missing():
    crit = CritiqueResult(field_verdicts=_verdicts(skip=("currency",)),
                          overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t5b")
    assert result.status == "needs_review"
    flags = _flags(result)
    assert list(flags) == ["currency"]
    assert flags["currency"].confidence == 0.5
    assert flags["currency"].reason == MISSING_REASON
    assert result.field_confidence["currency"] == 0.5


def test_eb5_missing_verdict_with_low_overall_does_not_raise():
    crit = CritiqueResult(field_verdicts=[], overall_confidence=0.3)
    result = decide(_data(), crit, _h(), trace_id="t5c")
    assert result.status == "needs_review"
    for f in EXTRACTION_FIELDS:
        assert result.field_confidence[f] <= 0.3 + 1e-9


# ---------------------------------------------------------------------------
# EB-6: null fields unchanged
# ---------------------------------------------------------------------------

def test_eb6_null_fields_inherit_overall_and_are_not_flagged():
    data = InvoiceData(vendor_name="ACME CORP")
    crit = CritiqueResult(
        field_verdicts=[FieldVerdict(field="vendor_name", confidence=0.9, grounded=True)],
        overall_confidence=0.9,
    )
    result = decide(data, crit, run_heuristics(SOURCE, data), trace_id="t6")
    assert result.status == "ok"
    assert result.flagged_fields == []
    for f in EXTRACTION_FIELDS:
        if f != "vendor_name":
            assert result.field_confidence[f] == 0.9, f


def test_eb6_null_field_with_grounded_false_not_capped_or_flagged():
    data = InvoiceData(vendor_name="ACME CORP")
    crit = CritiqueResult(
        field_verdicts=[
            FieldVerdict(field="vendor_name", confidence=0.9, grounded=True),
            FieldVerdict(field="tax", confidence=0.9, grounded=False, rationale="none"),
        ],
        overall_confidence=0.9,
    )
    result = decide(data, crit, run_heuristics(SOURCE, data), trace_id="t6b")
    assert result.status == "ok"
    assert "tax" not in _flags(result)
    assert result.field_confidence["tax"] != 0.5
    assert result.field_confidence["tax"] >= 0.9 - 1e-9


# ---------------------------------------------------------------------------
# EB-7: duplicates -> minimum
# ---------------------------------------------------------------------------

LOW_TOTAL = FieldVerdict(field="total", confidence=0.10, grounded=True, rationale="low")


@pytest.mark.parametrize("low_first", [False, True])
def test_eb7_duplicate_resolves_to_minimum(low_first):
    base = _verdicts()
    verdicts = [LOW_TOTAL] + base if low_first else base + [LOW_TOTAL]
    crit = CritiqueResult(field_verdicts=verdicts, overall_confidence=0.95)
    assert crit.verdict_for("total").confidence == pytest.approx(0.10)
    result = decide(_data(), crit, _h(), trace_id="t7")
    assert result.status == "needs_review"
    flags = _flags(result)
    assert list(flags) == ["total"]
    assert flags["total"].confidence == pytest.approx(0.1)
    assert flags["total"].reason == "low"


def test_eb7_tie_with_ungrounded_listed_first_caps():
    # Equal-confidence duplicates where one is grounded:false. Spec is silent on
    # ties; this pins only the order that must hold under any reading.
    pair = [FieldVerdict(field="tax", confidence=0.95, grounded=False, rationale="u"),
            FieldVerdict(field="tax", confidence=0.95, grounded=True, rationale="g")]
    crit = CritiqueResult(field_verdicts=_verdicts(skip=("tax",)) + pair,
                          overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t7t")
    assert result.status == "needs_review"
    assert _flags(result)["tax"].confidence == 0.5


def test_eb7_tie_with_ungrounded_listed_second_also_caps():
    # Ties resolve to the ungrounded verdict regardless of order, so a
    # confident grounded duplicate cannot mask a grounded:false one.
    pair = [FieldVerdict(field="tax", confidence=0.95, grounded=True, rationale="g"),
            FieldVerdict(field="tax", confidence=0.95, grounded=False, rationale="u")]
    crit = CritiqueResult(field_verdicts=_verdicts(skip=("tax",)) + pair,
                          overall_confidence=0.95)
    assert crit.verdict_for("tax").grounded is False
    result = decide(_data(), crit, _h(), trace_id="t7r")
    assert result.status == "needs_review"
    assert _flags(result)["tax"].confidence == 0.5


def test_eb7_verdict_for_missing_is_none():
    crit = CritiqueResult(field_verdicts=_verdicts(skip=("currency",)),
                          overall_confidence=0.95)
    assert crit.verdict_for("currency") is None
    assert crit.verdict_for("vendor_name").confidence == 0.95


# ---------------------------------------------------------------------------
# EB-8: clean path
# ---------------------------------------------------------------------------

def test_eb8_clean_path_unchanged():
    crit = CritiqueResult(field_verdicts=_verdicts(), overall_confidence=0.95)
    result = decide(_data(), crit, _h(), trace_id="t8")
    assert result.status == "ok"
    assert result.flagged_fields == []
    for f in EXTRACTION_FIELDS:
        assert result.field_confidence[f] == 0.95


# ---------------------------------------------------------------------------
# EB-9: neutral critique paths
# ---------------------------------------------------------------------------

def test_eb9_heuristics_only_golden_inv001_ok():
    row = _rows()["inv-001"]
    fx = _replay()["inv-001"]
    pipe = ReflectivePipeline(Generator(ScriptedLLMClient([fx["generator_raw"]])), critic=None)
    result = pipe.extract(row["text"])
    assert result.status == "ok"
    assert result.flagged_fields == []


def test_eb9_heuristics_only_shared_source_ok():
    gen = Generator(ScriptedLLMClient([json.dumps(_data().model_dump())]))
    result = ReflectivePipeline(gen, critic=None).extract(SOURCE)
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# EB-10: golden inv-007
# ---------------------------------------------------------------------------

def test_eb10_golden_row_inv007():
    row = _rows()["inv-007"]
    assert row["text"] == INV007_TEXT
    assert row["expected"] == INV007_EXPECTED
    assert list(_rows())[6] == "inv-007"


def test_eb10_fixture_shape():
    fx = _replay()["inv-007"]
    gen = json.loads(fx["generator_raw"])
    assert gen == {**INV007_EXPECTED, "vendor_name": "Contoso Ltd"}
    crit = json.loads(fx["critic_raw"])
    by_field = {v["field"]: v for v in crit["field_verdicts"]}
    assert set(by_field) == set(EXTRACTION_FIELDS)
    vn = by_field["vendor_name"]
    assert vn["confidence"] == 0.9 and vn["grounded"] is False
    assert vn["rationale"] == "Contoso Ltd is the bill-to customer, not the issuing vendor"
    for f, v in by_field.items():
        if f != "vendor_name":
            assert v["confidence"] == 0.95 and v["grounded"] is True, f
    assert crit["overall_confidence"] == 0.9


def test_eb10_pipeline_flags_vendor():
    result = _run_row("inv-007")
    assert result.status == "needs_review"
    flags = _flags(result)
    assert list(flags) == ["vendor_name"]
    assert flags["vendor_name"].confidence == 0.5
    assert CONTRACT_REASON in flags["vendor_name"].reason


def test_eb10_catch_comes_from_grounded_signal():
    # Heuristics alone pass the bill-to customer (it is in the source), and the
    # same critic with grounded:true would be ok -> the contract is what catches it.
    gen = InvoiceData(**json.loads(_replay()["inv-007"]["generator_raw"]))
    h = run_heuristics(INV007_TEXT, gen)
    assert h.factor_for("vendor_name") == 1.0
    crit = json.loads(_replay()["inv-007"]["critic_raw"])
    for v in crit["field_verdicts"]:
        v["grounded"] = True
    result = _run_row("inv-007", critic_raw=json.dumps(crit))
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# EB-11: eval readout
# ---------------------------------------------------------------------------

def test_eb11_eval_cli_readout():
    proc = _eval_stdout()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = re.sub(r"[ \t]+", " ", proc.stdout)
    assert re.search(r"examples : 8\b", out), out
    assert re.search(r"field accuracy : 92\.9% \(52/56\)", out), out
    assert re.search(r"exact-match acc : 50\.0% \(4/8\)", out), out
    assert re.search(r"needs-review rate : 50\.0% \(4/8\)", out), out
    assert "[inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'" in out
    assert "PASS" in out


def test_eb11_evaluate_metrics():
    sys.path.insert(0, str(REPO / "eval"))
    try:
        import run_eval
    finally:
        sys.path.remove(str(REPO / "eval"))
    m = run_eval.evaluate(live=False)
    assert (m.examples, m.total_fields, m.correct_fields, m.exact_matches, m.reviewed) == (
        8, 56, 52, 4, 4)
    assert ("inv-007", "vendor_name", "NORTHWIND TRADERS", "Contoso Ltd") in m.failures


# ---------------------------------------------------------------------------
# EB-12: docs
# ---------------------------------------------------------------------------

def _section(text: str, heading_prefix: str) -> str:
    start = text.index(heading_prefix)
    nxt = text.find("\n## ", start + 1)
    return text[start: nxt if nxt != -1 else len(text)]


def test_eb12_readme_gate_bullet_states_rule():
    arch = _section(README.read_text(), "## 7.")
    bullet = next(line for line in arch.splitlines() if line.startswith("- **Gate**"))
    assert "grounded:false" in bullet
    assert "0.5" in bullet
    assert "no verdict" in bullet or "missing verdict" in bullet
    assert "duplicate" in bullet.lower() and "minimum" in bullet.lower()


def test_eb12_gate_docstring_states_rule():
    import reflect_mcp.gate as gate_module

    doc = gate_module.__doc__
    assert doc
    assert "grounded" in doc and "0.5" in doc
    assert "verdict" in doc.lower() and "duplicate" in doc.lower()


def test_eb12_readme_eval_block_matches_stdout():
    sec = _section(README.read_text(), "## 8.")
    # Parse fences line-by-line so an opening ```bash is not mistaken for a close.
    blocks, cur = [], None
    for line in sec.splitlines(keepends=True):
        if line.startswith("```"):
            if cur is None:
                cur = []
            else:
                blocks.append("".join(cur))
                cur = None
        elif cur is not None:
            cur.append(line)
    block = next(b for b in blocks if "reflect-mcp eval" in b)
    proc = _eval_stdout()
    assert proc.returncode == 0
    assert block == proc.stdout


def test_eb12_gate_only_snippet_executes_ok():
    sec = README.read_text()
    start = sec.index("### Gate-only verification")
    m = re.search(r"```python\n(.*?)```", sec[start:], flags=re.S)
    code = m.group(1)
    assert '"ACME Corp, total $250.00 USD"' in code
    assert '# result.status == "ok"' in code
    for f in ("vendor_name", "total", "currency"):
        assert f'FieldVerdict(field="{f}", confidence=0.95, grounded=True)' in code
    ns: dict = {}
    exec(compile(code, "README-gate-only", "exec"), ns)
    assert ns["result"].status == "ok"
    assert ns["result"].flagged_fields == []


def test_eb12_all_unchanged_no_new_exports():
    assert "decide" in reflect_mcp.__all__
    assert "FieldVerdict" in reflect_mcp.__all__
