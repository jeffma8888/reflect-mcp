"""Black-box behavioral tests for iter-71: date-role grounding for ``invoice_date``.

A grounded ``invoice_date`` whose every occurrence in the source is labelled as a
due / payment / ship / delivery date is penalized with factor 0.3 and the exact
reason REASON(x). Covers Expected Behaviors 1-11 of pm.md:

  1. Due date copied as the invoice date is caught (HO + run_heuristics).
  2. Other role labels (ship, payment due, delivery, pay by, one-line, shipped)
     in every supported date form.
  3. Per-token attribution on a shared line.
  4. Abstains whenever any occurrence is not role-labelled (incl. ambiguity and
     labels embedded in longer words).
  5. Existing non-ISO / not-found date penalties unchanged, never stacked.
  6. Provenance span is kept.
  7. Critic-on replay of the deferred inv-009 row with inv-008's critic_raw.
  8. Golden set: no invoice_date factor on any of the 8 rows.
  9. ``run_eval.py`` stdout sha256 and exit 0 unchanged.
 10. CLI ``validate --heuristics-only`` exit codes and payload.
 11. Docs: heuristics module docstring and README mention date-role grounding.

Isolation contract: tests drive the public package API, the MCP tool functions,
the CLI and the eval script, asserting on observable output only. No
implementation source, engineer notes or diff were read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.heuristics as heuristics_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests._golden import frozen_eval, frozen_replay, frozen_rows

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

FIELDS = ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency")
HEAD_EVAL_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"

EB1_DATES = "Invoice date: 2026-07-01\nDue date: 2026-07-31"


def T(dates: str) -> str:
    return (
        "Umbrella Logistics\nInvoice UL-4410\n" + dates
        + "\nSubtotal: 300.00\nTax: 24.00\nTotal: 324.00\nCurrency: USD"
    )


def D(x: str) -> dict:
    return {
        "vendor_name": "Umbrella Logistics",
        "invoice_number": "UL-4410",
        "invoice_date": x,
        "subtotal": 300.0,
        "tax": 24.0,
        "total": 324.0,
        "currency": "USD",
    }


def REASON(x: str) -> str:
    return f"date {x!r} appears only as a due/ship/delivery date, not the invoice date"


def _h(dates: str, x: str):
    return run_heuristics(T(dates), InvoiceData(**D(x)))


def _ho(dates: str, x: str) -> dict:
    res = tools.validate_tool(T(dates), D(x), 0.70, 0.75, True)
    return res if isinstance(res, dict) else res.model_dump(mode="json")


def _status(r) -> str:
    s = r["status"] if isinstance(r, dict) else r.status
    return getattr(s, "value", s)


def _flag_fields(r) -> list[str]:
    d = r if isinstance(r, dict) else r.model_dump(mode="json")
    return [f["field"] for f in d["flagged_fields"]]


def _replay() -> dict:
    return frozen_replay()


def _golden() -> list[dict]:
    return frozen_rows()


# ---------------------------------------------------------------------------
# EB-1: due date as invoice date
# ---------------------------------------------------------------------------

def test_reason_literal_matches_spec_example():
    assert REASON("2026-07-31") == (
        "date '2026-07-31' appears only as a due/ship/delivery date, not the invoice date"
    )


def test_eb1_heuristics_factor_and_reason():
    h = _h(EB1_DATES, "2026-07-31")
    assert h.factors == {"invoice_date": 0.3}
    assert h.reason_for("invoice_date") == REASON("2026-07-31")


def test_eb1_heuristics_only_tool_needs_review():
    d = _ho(EB1_DATES, "2026-07-31")
    assert _status(d) == "needs_review"
    assert _flag_fields(d) == ["invoice_date"]
    assert d["flagged_fields"][0]["reason"] == REASON("2026-07-31")


# ---------------------------------------------------------------------------
# EB-2: other role labels, every date form
# ---------------------------------------------------------------------------

EB2_CASES = {
    "a_ship_date_us_form": ("Issued: March 14, 2026\nShip date: 03/20/2026", "2026-03-20"),
    "b_payment_due_by": ("Payment due by 2026-07-31", "2026-07-31"),
    "c_delivery_long_form": ("Invoice date: 01.07.2026\nDelivery date: 15 July 2026", "2026-07-15"),
    "d_pay_by": ("Invoice date: 2026-07-01\nPay by 2026-07-31", "2026-07-31"),
    "e_one_line": ("Invoice date: 2026-07-01   Due date: 2026-07-31", "2026-07-31"),
    "f_shipped": ("Date: 2026-07-01\nShipped: 2026-07-31", "2026-07-31"),
}


@pytest.mark.parametrize("case", sorted(EB2_CASES))
def test_eb2_role_labels_penalized(case):
    dates, x = EB2_CASES[case]
    h = _h(dates, x)
    assert h.factors == {"invoice_date": 0.3}, (case, h.factors)
    assert h.reason_for("invoice_date") == REASON(x)


@pytest.mark.parametrize("case", sorted(EB2_CASES))
def test_eb2_role_labels_heuristics_only_needs_review(case):
    dates, x = EB2_CASES[case]
    d = _ho(dates, x)
    assert _status(d) == "needs_review", case
    assert _flag_fields(d) == ["invoice_date"], case


@pytest.mark.parametrize("label", [
    "DUE DATE:", "due:", "Payment by", "PAY BY", "Shipping date:", "SHIP:", "Delivered:", "delivery:",
])
def test_eb2_labels_case_insensitive_and_full_label_set(label):
    dates = f"Invoice date: 2026-07-01\n{label} 2026-07-31"
    h = _h(dates, "2026-07-31")
    assert h.factors == {"invoice_date": 0.3}, (label, h.factors)
    assert h.reason_for("invoice_date") == REASON("2026-07-31")


# ---------------------------------------------------------------------------
# EB-3: per-token attribution on a shared line
# ---------------------------------------------------------------------------

def test_eb3_first_token_on_shared_line_not_attributed_to_later_label():
    dates = "Invoice date: 2026-07-01   Due date: 2026-07-31"
    h = _h(dates, "2026-07-01")
    assert h.factors == {}
    assert _status(_ho(dates, "2026-07-01")) == "ok"


# ---------------------------------------------------------------------------
# EB-4: abstain when any occurrence is not role-labelled
# ---------------------------------------------------------------------------

EB4_CASES = {
    "a_real_invoice_date": ("Invoice date: 2026-07-01\nDue date: 2026-07-31", "2026-07-01"),
    "b_same_date_both_roles": ("Date: 2026-07-01\nShipped: 2026-07-01", "2026-07-01"),
    "c_unlabelled": ("2026-07-31", "2026-07-31"),
    "d_ambiguous_day_first": ("Date: 07/03/2026\nDue: 03/07/2026", "2026-03-07"),
    "e_label_inside_word": ("Shipment ref 2026-07-31", "2026-07-31"),
}


@pytest.mark.parametrize("case", sorted(EB4_CASES))
def test_eb4_abstains(case):
    dates, x = EB4_CASES[case]
    h = _h(dates, x)
    assert h.factors == {}, (case, h.factors, h.reasons)
    assert _status(_ho(dates, x)) == "ok", case


@pytest.mark.parametrize("dates", [
    "Overdue 2026-07-31",
    "Shipper 2026-07-31",
    "Payday 2026-07-31",
])
def test_eb4_whole_word_only(dates):
    # Labels match only as whole words: "Overdue", "Shipper", "Payday" are not labels.
    h = _h(dates, "2026-07-31")
    assert h.factors == {}, (dates, h.factors, h.reasons)


# ---------------------------------------------------------------------------
# EB-5: existing date penalties unchanged, never stacked
# ---------------------------------------------------------------------------

def test_eb5_non_iso_single_reason():
    h = _h(EB1_DATES, "07/31/2026")
    assert h.factors == {"invoice_date": 0.2}
    assert h.reason_for("invoice_date") == "date '07/31/2026' is not ISO 8601 (YYYY-MM-DD)"


def test_eb5_absent_single_reason():
    h = _h(EB1_DATES, "2026-08-09")
    assert h.factors == {"invoice_date": 0.3}
    assert h.reason_for("invoice_date") == "date '2026-08-09' not found in source text"


def test_eb5_non_iso_on_due_line_not_stacked():
    # Non-ISO value whose calendar date is only on the due line: still the one ISO reason.
    h = _h(EB1_DATES, "07/31/2026")
    assert "due/ship/delivery" not in h.reason_for("invoice_date")


# ---------------------------------------------------------------------------
# EB-6: provenance unchanged
# ---------------------------------------------------------------------------

def test_eb6_provenance_span_kept():
    text = T(EB1_DATES)
    h = run_heuristics(text, InvoiceData(**D("2026-07-31")))
    spans = h.provenance["invoice_date"]
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == "2026-07-31"


# ---------------------------------------------------------------------------
# EB-7: Critic-on replay of the deferred inv-009 row
# ---------------------------------------------------------------------------

def _critic_on_result(x: str):
    crit = _replay()["inv-008"]["critic_raw"]
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([json.dumps(D(x))])),
        Critic(ScriptedLLMClient([crit])),
    )
    return pipe.extract(T(EB1_DATES))


def test_eb7_inv008_critic_raw_is_all_095_grounded():
    crit = json.loads(_replay()["inv-008"]["critic_raw"])
    verdicts = {v["field"]: v for v in crit["field_verdicts"]}
    assert set(verdicts) == set(FIELDS)
    assert all(v["confidence"] == 0.95 and v["grounded"] is True for v in verdicts.values())


def test_eb7_critic_on_catch():
    res = _critic_on_result("2026-07-31")
    assert _status(res) == "needs_review"
    flagged = _flag_fields(res)
    assert flagged == ["invoice_date"]
    assert res.field_confidence["invoice_date"] == pytest.approx(0.95 * 0.3)
    for f in FIELDS:
        if f != "invoice_date":
            assert res.field_confidence[f] == pytest.approx(0.95), f


def test_eb7_critic_on_counterfactual_correct_date_ok():
    res = _critic_on_result("2026-07-01")
    assert _status(res) == "ok"
    assert _flag_fields(res) == []


# ---------------------------------------------------------------------------
# EB-8: golden set, no false alarm
# ---------------------------------------------------------------------------

def test_eb8_golden_rows_have_no_invoice_date_role_factor():
    rows = _golden()
    assert len(rows) == 8
    for row in rows:
        h = run_heuristics(row["text"], InvoiceData(**row["expected"]))
        assert "invoice_date" not in h.factors, (row["id"], h.factors, h.reasons)
        assert "due/ship/delivery" not in (h.reason_for("invoice_date") or ""), row["id"]


# ---------------------------------------------------------------------------
# EB-9: eval contract unchanged
# ---------------------------------------------------------------------------

def test_eb9_eval_stdout_sha_and_exit():
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_EVAL_SHA256, proc.stdout
    assert "silent-wrong" in proc.stdout


# ---------------------------------------------------------------------------
# EB-10: CLI surface
# ---------------------------------------------------------------------------

def _cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c",
           "import sys, runpy; sys.modules['anthropic'] = None; "
           "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
           "runpy.run_module('reflect_mcp', run_name='__main__')", *args]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(cmd, input="", capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=120)


def test_eb10_cli_due_date_exits_1(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(T(EB1_DATES))
    proc = _cli(["validate", "--heuristics-only", "--file", str(p),
                 "--data", json.dumps(D("2026-07-31"))])
    assert proc.returncode == 1, proc.stdout + proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    assert out["status"] == "needs_review"
    assert out["flagged_fields"][0]["field"] == "invoice_date"


def test_eb10_cli_correct_date_exits_0(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(T(EB1_DATES))
    proc = _cli(["validate", "--heuristics-only", "--file", str(p),
                 "--data", json.dumps(D("2026-07-01"))])
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert json.loads(proc.stdout)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-11: docs
# ---------------------------------------------------------------------------

def test_eb11_module_docstring_lists_date_role_grounding():
    assert "date-role grounding" in (heuristics_module.__doc__ or "").lower()


def test_eb11_readme_names_date_role_grounding_three_places():
    text = README.read_text().lower()
    assert text.count("date-role grounding") >= 3, text.count("date-role grounding")


def test_eb11_readme_architecture_bullet_names_label_families():
    lines = [ln.lower() for ln in README.read_text().splitlines() if "date-role" in ln.lower()]
    assert any(all(w in ln for w in ("due", "pay", "ship", "deliver")) for ln in lines), lines
