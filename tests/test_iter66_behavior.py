"""Black-box behavioral tests for iter-66: party-role grounding for ``vendor_name``.

A ``vendor_name`` that is grounded in the source but appears ONLY on counterparty
lines ("Bill to", "Ship to", "Sold to", "Deliver to" and variants) is penalized
with factor 0.3 and the exact reason R(v). Covers Expected Behaviors 1-10 of
pm.md:

  1. Basic bill-to penalty: factor, exact reason, provenance span kept.
  2. Label variants (BILL-TO, Billed to, indented Ship To, Shipped/Sold/Deliver(ed) to).
  3. Abstains when any line containing the value is not a counterparty line.
  4. Vendor-side / non-matching labels never count (Remit to, Sold by, Customer,
     Shipping to, Billing date, name on the line after the label).
  5. Ungrounded vendor keeps only the 0.25 not-found reason.
  6. vendor_name only.
  7. Golden heuristics-only (4, 0, 0); inv-007 through library, MCP tool, CLI.
  8. Golden Critic-on: inv-007 at 0.15 with the composed reason; the other 7 rows
     byte-identical to HEAD (digests measured on a ``git archive HEAD`` copy).
  9. ``make eval`` stdout byte-identical to HEAD; heuristics.done factors.
 10. iter-54 attribution: grounded:false alone still flags at 0.5; grounded:true
     plus real heuristics is now needs_review at 0.27.

Isolation contract: tests drive the public package API, the MCP tool functions,
the CLI and the eval script, asserting on observable output only. No
implementation source, engineer notes or diff were read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.gate import decide
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import HeuristicReport, run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"

FIELDS = ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency")
NR_NOTE = "1 field(s) below the 70% confidence threshold require human review before use."
CONTRACT_REASON = "critic marked the value not grounded in the source"
CRITIC_RATIONALE_007 = "Contoso Ltd is the bill-to customer, not the issuing vendor"


def R(v: str) -> str:
    return f"vendor {v!r} appears only on a bill-to/ship-to/sold-to line (the buyer, not the seller)"


# Per-row sha256 of model_dump(mode="json") minus trace_id (sorted-keys JSON),
# measured on a `git archive HEAD` copy before iter 66. inv-007 is excluded on
# purpose: it is the only row this iteration may move.
HEAD_ROW_SHA256 = {
    "off/extract/inv-001": "0261b9cd6dff6f4c84c9842a778e576c37b957befb519efac6ae544c7885d58e",
    "off/extract/inv-002": "d0f4361ab5a08cfa5571382634fc1d0514a55dcb46d8eabdc415a745a85be567",
    "off/extract/inv-003": "4fb8a97c59a68843a4e753de203ad412565f1dd5066892a625043858c4d53549",
    "off/extract/inv-004": "f4e92c1b010061eee6ddb4aa46c3d4547aeedbed2ed4caf3026cfd7fe124ec5b",
    "off/extract/inv-005": "24b0068d4227eeafb2ea881a0da38214833a8102e5f5dde781b8181fd701289f",
    "off/extract/inv-006": "691e43649ef2bcbc7cc99ad2fd53ddec34599ea1e68d0d94b3ad333bb006f045",
    "off/extract/inv-008": "bb63de8e4f187f987f21eca6be6609eab5eabeb0b9ef6c7726191c5fcaefe557",
    "off/validate/inv-001": "0261b9cd6dff6f4c84c9842a778e576c37b957befb519efac6ae544c7885d58e",
    "off/validate/inv-002": "d0f4361ab5a08cfa5571382634fc1d0514a55dcb46d8eabdc415a745a85be567",
    "off/validate/inv-003": "4fb8a97c59a68843a4e753de203ad412565f1dd5066892a625043858c4d53549",
    "off/validate/inv-004": "f4e92c1b010061eee6ddb4aa46c3d4547aeedbed2ed4caf3026cfd7fe124ec5b",
    "off/validate/inv-005": "24b0068d4227eeafb2ea881a0da38214833a8102e5f5dde781b8181fd701289f",
    "off/validate/inv-006": "691e43649ef2bcbc7cc99ad2fd53ddec34599ea1e68d0d94b3ad333bb006f045",
    "off/validate/inv-008": "bb63de8e4f187f987f21eca6be6609eab5eabeb0b9ef6c7726191c5fcaefe557",
    "on/extract/inv-001": "6490f6cd4ea758edd355504ddd25a006d48456c3e2c1036bf7f7b3ab551282d9",
    "on/extract/inv-002": "53f9cb88de47199aa47c9864597060caa08928bbf09176e4308216e6c9cf7def",
    "on/extract/inv-003": "98a8c38f760754480716d9cd04a9324e843af4f1ce02eb27f73a5f52fb2334db",
    "on/extract/inv-004": "97a4f4eb0657672e1c213175e091780d894edf06b142f451b56df3491fa365d7",
    "on/extract/inv-005": "eed0f2ae4e7322c7fada2d9c6c604c9a7fd27edffce74ae0d74d7d0c6d83f587",
    "on/extract/inv-006": "ab8c9472bdbc1730a7f52593562009972156e06e418a61e9ea161db4e18e2164",
    "on/extract/inv-008": "ccf3617b9186a44eef5451261db549556ec65218b988fd118d86d0afa610eb5e",
    "on/validate/inv-001": "6490f6cd4ea758edd355504ddd25a006d48456c3e2c1036bf7f7b3ab551282d9",
    "on/validate/inv-002": "53f9cb88de47199aa47c9864597060caa08928bbf09176e4308216e6c9cf7def",
    "on/validate/inv-003": "98a8c38f760754480716d9cd04a9324e843af4f1ce02eb27f73a5f52fb2334db",
    "on/validate/inv-004": "97a4f4eb0657672e1c213175e091780d894edf06b142f451b56df3491fa365d7",
    "on/validate/inv-005": "eed0f2ae4e7322c7fada2d9c6c604c9a7fd27edffce74ae0d74d7d0c6d83f587",
    "on/validate/inv-006": "ab8c9472bdbc1730a7f52593562009972156e06e418a61e9ea161db4e18e2164",
    "on/validate/inv-008": "ccf3617b9186a44eef5451261db549556ec65218b988fd118d86d0afa610eb5e",
}

HEAD_EVAL_STDOUT = """==============================================================
reflect-mcp eval  |  mode: OFFLINE (replay fixtures)
==============================================================
examples          : 8
field accuracy    : 92.9%  (52/56)
exact-match acc   : 50.0%  (4/8)
needs-review rate : 50.0%  (4/8)
catch rate        : 100.0%  (4/4)
silent-wrong      : 0
false alarms      : 0
--------------------------------------------------------------
field-level failures (where it breaks):
  [inv-004] total: expected 583.2 but got 683.2  (flagged)
  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'  (flagged)
  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'  (flagged)
  [inv-008] total: expected 75.0 but got 45.0  (flagged)
==============================================================
PASS: field accuracy 92.9% >= required 90.0%
PASS: silent-wrong fields 0 <= allowed 0
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _h(text: str, **fields):
    return run_heuristics(text, InvoiceData(**fields))


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _golden() -> list[dict]:
    rows = [json.loads(ln) for ln in GOLDEN.read_text().splitlines() if ln.strip()]
    assert len(rows) == 8
    return rows


def _row(rid: str) -> dict:
    return {r["id"]: r for r in _golden()}[rid]


def _pipe(fx: dict, critic_on: bool, critic_raw: str | None = None) -> ReflectivePipeline:
    crit = Critic(ScriptedLLMClient([critic_raw or fx["critic_raw"]])) if critic_on else None
    return ReflectivePipeline(Generator(ScriptedLLMClient([fx["generator_raw"]])), crit)


def _run(row: dict, critic_on: bool, mode: str, critic_raw: str | None = None):
    fx = _replay()[row["id"]]
    pipe = _pipe(fx, critic_on, critic_raw)
    if mode == "validate":
        return pipe.validate(row["text"], InvoiceData(**json.loads(fx["generator_raw"])))
    return pipe.extract(row["text"])


def _status(r) -> str:
    s = r["status"] if isinstance(r, dict) else r.status
    return getattr(s, "value", s)


def _flag_dicts(r) -> list[dict]:
    d = r if isinstance(r, dict) else r.model_dump(mode="json")
    return d["flagged_fields"]


def _flag_core(flags: list[dict]) -> list[dict]:
    # The spec names field/confidence/reason; the payload also carries the
    # flagged ``value``, asserted separately.
    return [{k: f[k] for k in ("field", "confidence", "reason")} for f in flags]


def _digest(result) -> str:
    d = result.model_dump(mode="json")
    d.pop("trace_id", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _score(results: dict) -> tuple[int, int, int, list]:
    caught = silent = fa = 0
    silent_ids = []
    for row in _golden():
        r = results[row["id"]]
        d = r.model_dump(mode="json")
        flagged = {f["field"] for f in d["flagged_fields"]}
        for f in FIELDS:
            wrong = d["data"][f] != row["expected"][f]
            if wrong and f in flagged:
                caught += 1
            elif wrong:
                silent += 1
                silent_ids.append((row["id"], f))
            elif f in flagged:
                fa += 1
    return caught, silent, fa, silent_ids


# ---------------------------------------------------------------------------
# EB-1: basic bill-to penalty
# ---------------------------------------------------------------------------

EB1_TEXT = "ACME\nBill to: Contoso Ltd\nTotal: 5.00"


def test_eb1_bill_to_vendor_penalized():
    h = _h(EB1_TEXT, vendor_name="Contoso Ltd")
    assert h.factor_for("vendor_name") == 0.3


def test_eb1_exact_reason():
    h = _h(EB1_TEXT, vendor_name="Contoso Ltd")
    assert h.reason_for("vendor_name") == R("Contoso Ltd")
    assert R("Contoso Ltd") == (
        "vendor 'Contoso Ltd' appears only on a bill-to/ship-to/sold-to line "
        "(the buyer, not the seller)"
    )


def test_eb1_provenance_span_kept():
    h = _h(EB1_TEXT, vendor_name="Contoso Ltd")
    spans = h.provenance["vendor_name"]
    assert len(spans) == 1
    assert EB1_TEXT[spans[0].start:spans[0].end] == "Contoso Ltd"


# ---------------------------------------------------------------------------
# EB-2: label variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "BILL-TO: Contoso Ltd",
    "Billed to Contoso Ltd",
    "   Ship To: Contoso Ltd",
    "Shipped to: Contoso Ltd",
    "Sold to: Contoso Ltd",
    "Deliver to: Contoso Ltd",
    "Delivered to: Contoso Ltd",
])
def test_eb2_counterparty_label_variants(line):
    h = _h("ACME\n" + line, vendor_name="Contoso Ltd")
    assert h.factor_for("vendor_name") == 0.3, line
    assert h.reason_for("vendor_name") == R("Contoso Ltd"), line


def test_eb2_value_on_two_counterparty_lines_still_penalized():
    # Every line containing the value is a counterparty line -> penalized.
    h = _h("ACME\nBill to: Contoso Ltd\nShip to: Contoso Ltd", vendor_name="Contoso Ltd")
    assert h.factor_for("vendor_name") == 0.3
    assert h.reason_for("vendor_name") == R("Contoso Ltd")


# ---------------------------------------------------------------------------
# EB-3: abstain when any containing line is not a counterparty line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "ACME CORP\nBill to: ACME CORP",
    "From: ACME CORP  Bill to: Contoso",
])
def test_eb3_abstains_when_value_also_elsewhere(text):
    h = _h(text, vendor_name="ACME CORP")
    assert h.factor_for("vendor_name") == 1.0, text
    assert not h.reason_for("vendor_name"), text


def test_eb3_containment_is_case_insensitive():
    # The header line holds the value in a different case: it still counts as a
    # non-counterparty line containing the value, so the check abstains.
    h = _h("acme corp\nBill to: ACME CORP", vendor_name="ACME CORP")
    assert R("ACME CORP") not in (h.reason_for("vendor_name") or "")
    assert h.factor_for("vendor_name") == 1.0


# ---------------------------------------------------------------------------
# EB-4: vendor-side / non-matching labels never count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,vendor", [
    ("Invoice 7\nRemit to: ACME CORP", "ACME CORP"),
    ("Invoice 7\nSold by: ACME CORP", "ACME CORP"),
    ("ACME\nCustomer: Contoso Ltd", "Contoso Ltd"),
    ("ACME\nShipping to: Contoso Ltd", "Contoso Ltd"),
    ("ACME\nBilling date 2026", "ACME"),
    ("ACME\nBill to:\nContoso Ltd", "Contoso Ltd"),
])
def test_eb4_non_counterparty_labels_not_penalized(text, vendor):
    h = _h(text, vendor_name=vendor)
    assert h.factor_for("vendor_name") == 1.0, text
    assert not h.reason_for("vendor_name"), text


def test_eb4_golden_header_vendor_not_penalized():
    # inv-007's real vendor sits on the header line: no false alarm.
    row = _row("inv-007")
    h = run_heuristics(row["text"], InvoiceData(**row["expected"]))
    assert h.factor_for("vendor_name") == 1.0
    assert not h.reason_for("vendor_name")


# ---------------------------------------------------------------------------
# EB-5: ungrounded vendor keeps only the not-found reason
# ---------------------------------------------------------------------------

def test_eb5_not_found_vendor_only_not_found_reason():
    h = _h("ACME\nBill to: Contoso Ltd", vendor_name="Initrode")
    assert h.factor_for("vendor_name") == 0.25
    reason = h.reason_for("vendor_name")
    assert "not found in source text" in reason
    assert "bill-to/ship-to/sold-to" not in reason
    assert reason == "value 'Initrode' not found in source text"


# ---------------------------------------------------------------------------
# EB-6: vendor_name only
# ---------------------------------------------------------------------------

def test_eb6_other_fields_on_bill_to_line_not_penalized():
    text = "ACME\nBill to: NW-2201 USD"
    h = _h(text, vendor_name="ACME", invoice_number="NW-2201", currency="USD")
    for f in ("vendor_name", "invoice_number", "currency"):
        assert h.factor_for(f) == 1.0, f
        assert "bill-to/ship-to/sold-to" not in (h.reason_for(f) or ""), f


# ---------------------------------------------------------------------------
# EB-7: golden heuristics-only, library / MCP tool / CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb7_golden_heuristics_only_caught4_silent0_fa0(mode):
    results = {row["id"]: _run(row, False, mode) for row in _golden()}
    assert len(results) == 8
    caught, silent, fa, silent_ids = _score(results)
    assert (caught, silent, fa) == (4, 0, 0), silent_ids
    assert silent_ids == []


def _assert_inv007_ho(d: dict):
    assert _status(d) == "needs_review"
    assert _flag_core(d["flagged_fields"]) == [
        {"field": "vendor_name", "confidence": 0.3, "reason": R("Contoso Ltd")}
    ]
    assert d["flagged_fields"][0]["value"] == "Contoso Ltd"
    assert d["overall_confidence"] == 1.0
    assert d["notes"] == NR_NOTE


@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb7_inv007_library_heuristics_only(mode):
    r = _run(_row("inv-007"), False, mode)
    _assert_inv007_ho(r.model_dump(mode="json"))


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv("REFLECT_FIELD_THRESHOLD", raising=False)
    monkeypatch.delenv("REFLECT_OVERALL_THRESHOLD", raising=False)


def test_eb7_inv007_validate_tool_heuristics_only(_clean_env):
    row = _row("inv-007")
    data = json.loads(_replay()["inv-007"]["generator_raw"])
    res = tools.validate_tool(row["text"], dict(data), heuristics_only=True)
    _assert_inv007_ho(res)


def test_eb7_inv007_extract_tool_heuristics_only(monkeypatch, _clean_env):
    fx = _replay()["inv-007"]

    def _factory(gate_config=None, **kwargs):
        gen = Generator(ScriptedLLMClient([fx["generator_raw"]]))
        critic = None
        if kwargs.get("critic_enabled", True):
            critic = Critic(ScriptedLLMClient([fx["critic_raw"]]))
        return ReflectivePipeline(gen, critic, gate_config=gate_config)

    monkeypatch.setattr(pipeline_module, "build_default_pipeline", _factory)
    res = tools.extract_tool(_row("inv-007")["text"], heuristics_only=True)
    _assert_inv007_ho(res)


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c",
           "import sys, runpy; sys.modules['anthropic'] = None; "
           "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
           "runpy.run_module('reflect_mcp', run_name='__main__')", *args]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(cmd, input="", capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=120)


def test_eb7_inv007_cli_validate_heuristics_only(tmp_path):
    p = tmp_path / "inv007.txt"
    p.write_text(_row("inv-007")["text"])
    proc = _cli(["validate", "--heuristics-only", "--file", str(p),
                 "--data", _replay()["inv-007"]["generator_raw"]])
    assert proc.returncode == 1, proc.stderr[-2000:]
    _assert_inv007_ho(json.loads(proc.stdout))


def test_eb7_cli_clean_row_still_exits_0(tmp_path):
    # Counterfactual so the exit-1 above is not vacuous: a clean golden row exits 0.
    p = tmp_path / "inv001.txt"
    p.write_text(_row("inv-001")["text"])
    proc = _cli(["validate", "--heuristics-only", "--file", str(p),
                 "--data", _replay()["inv-001"]["generator_raw"]])
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert json.loads(proc.stdout)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-8: golden Critic-on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb8_inv007_critic_on(mode):
    d = _run(_row("inv-007"), True, mode).model_dump(mode="json")
    assert _status(d) == "needs_review"
    assert d["flagged_fields"][0]["value"] == "Contoso Ltd"
    assert d["overall_confidence"] == 0.9
    assert _flag_core(d["flagged_fields"]) == [{
        "field": "vendor_name",
        "confidence": 0.15,
        "reason": CRITIC_RATIONALE_007 + "; " + CONTRACT_REASON + "; " + R("Contoso Ltd"),
    }]


@pytest.mark.parametrize("critic_on", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb8_other_seven_rows_byte_identical_to_head(critic_on, mode):
    rows = [r for r in _golden() if r["id"] != "inv-007"]
    assert len(rows) == 7
    for row in rows:
        key = f"{'on' if critic_on else 'off'}/{mode}/{row['id']}"
        assert _digest(_run(row, critic_on, mode)) == HEAD_ROW_SHA256[key], key


def test_eb8_head_digest_table_complete():
    assert len(HEAD_ROW_SHA256) == 28
    assert all(len(v) == 64 for v in HEAD_ROW_SHA256.values())


# ---------------------------------------------------------------------------
# EB-9: make eval stdout + heuristics.done log
# ---------------------------------------------------------------------------

def test_eb9_eval_stdout_byte_identical_to_head():
    proc = subprocess.run([sys.executable, str(REPO / "eval" / "run_eval.py")],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert proc.stdout == HEAD_EVAL_STDOUT


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def named(self, name: str) -> list[dict]:
        return [dict(getattr(r, "fields", {}) or {}) for r in self.records
                if r.getMessage() == name]


@pytest.fixture
def logcap66():
    lg = logging.getLogger("reflect_mcp")
    h = _Capture()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


@pytest.mark.parametrize("critic_on", [True, False], ids=["on", "off"])
def test_eb9_heuristics_done_factors_inv007(logcap66, critic_on):
    _run(_row("inv-007"), critic_on, "extract")
    done = logcap66.named("heuristics.done")
    assert len(done) == 1
    assert done[0]["factors"] == {"vendor_name": 0.3}


# ---------------------------------------------------------------------------
# EB-10: iter-54 attribution
# ---------------------------------------------------------------------------

def _inv007_gen() -> InvoiceData:
    return InvoiceData(**json.loads(_replay()["inv-007"]["generator_raw"]))


def test_eb10_contract_alone_still_flags_at_cap():
    text = _row("inv-007")["text"]
    gen = _inv007_gen()
    critique = Critic(ScriptedLLMClient([_replay()["inv-007"]["critic_raw"]])).critique(text, gen)
    res = decide(gen, critique, HeuristicReport(), trace_id="t66")
    flags = {f.field: f for f in res.flagged_fields}
    assert list(flags) == ["vendor_name"]
    assert flags["vendor_name"].confidence == 0.5
    assert CONTRACT_REASON in flags["vendor_name"].reason
    assert R("Contoso Ltd") not in flags["vendor_name"].reason


def test_eb10_grounded_true_with_real_heuristics_now_needs_review():
    crit = json.loads(_replay()["inv-007"]["critic_raw"])
    for v in crit["field_verdicts"]:
        v["grounded"] = True
    r = _run(_row("inv-007"), True, "extract", critic_raw=json.dumps(crit))
    assert _status(r) == "needs_review"
    flags = _flag_dicts(r)
    assert [f["field"] for f in flags] == ["vendor_name"]
    assert flags[0]["confidence"] == 0.27
    assert R("Contoso Ltd") in flags[0]["reason"]
    assert CONTRACT_REASON not in flags[0]["reason"]
