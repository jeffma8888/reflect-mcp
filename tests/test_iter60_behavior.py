"""Black-box behavioral tests for iter-60: amount-role grounding.

A committed subtotal/tax/total that is present in the source but differs from
every number the source *labels* for that role is penalized to 0.3 with a
reason naming the labelled value(s). Covers Expected Behaviors 1-10 of pm.md:

  1. inv-001 subtotal<->tax swap is penalized, exact reasons.
  2. The swap reaches the user as needs_review under a fooled Critic.
  3. inv-005 stray "STORE #12" number as total.
  4. Label grammar: Sub-total / VAT / Grand Total, currency symbols, thousands
     separators, case-insensitivity, Net / GST / Amount Due.
  5. Abstains: no label, no two decimals, number on next line, label in a word.
  6. Any labelled match clears; reason lists distinct values in order; 0.005.
  7. No stacking on ungrounded values; inv-004 reasons byte-identical.
  8. Nulls untouched; only lowers scores (vs role check disabled); exactly 0.3.
  9. Zero false positives on golden expected rows and replay model outputs.
 10. Golden row inv-008 + replay entry; eval stdout; genuine catch when the
     labelled-number helper is disabled.

Isolation contract: tests drive the public package API, the golden data files
and the eval CLI, asserting on observable output. No implementation source was
read; the only private name used is `_labelled_amounts`, which the spec names
as the seam to disable for the counterfactual.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.heuristics as heuristics
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"
README = REPO / "README.md"

sys.path.insert(0, str(REPO / "eval"))
import run_eval  # noqa: E402

FIELDS = ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency")
AMOUNTS = ("subtotal", "tax", "total")

INV008_TEXT = (
    "Hooli Cafe\nReceipt HC-0912\n2026-06-15\n1 x Catering tray 45.00\n"
    "2 x Coffee urn 30.00\nTotal: 75.00\nUSD"
)
INV008_EXPECTED = {
    "vendor_name": "Hooli Cafe", "invoice_number": "HC-0912", "invoice_date": "2026-06-15",
    "subtotal": None, "tax": None, "total": 75.0, "currency": "USD",
}

EXPECTED_STDOUT = """\
==============================================================
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
# Helpers
# ---------------------------------------------------------------------------

def _rows() -> dict[str, dict]:
    out = {}
    for line in GOLDEN.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _data(**kw) -> InvoiceData:
    base = {f: None for f in FIELDS}
    base.update(kw)
    return InvoiceData(**base)


def _role_reasons(report) -> dict[str, str]:
    out = {}
    for f in FIELDS:
        r = report.reason_for(f) or ""
        if "source labels" in r:
            out[f] = r
    return out


def _fooled_critic_raw() -> str:
    verdicts = [
        {"field": f, "confidence": 0.95, "grounded": True, "rationale": "matches source"}
        for f in FIELDS
    ]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.95,
                       "summary": "All fields grounded in source."})


def _swap_inv001() -> tuple[str, dict]:
    row = _rows()["inv-001"]
    return row["text"], row["expected"] | {"subtotal": 8.5, "tax": 100.0}


def _disable_role_check(monkeypatch) -> None:
    assert hasattr(heuristics, "_labelled_amounts"), "spec-named helper missing"
    monkeypatch.setattr(heuristics, "_labelled_amounts", lambda *a, **k: [])


# ---------------------------------------------------------------------------
# EB-1: swap penalized
# ---------------------------------------------------------------------------

def test_eb1_swap_penalized_with_exact_reasons():
    text, data = _swap_inv001()
    report = run_heuristics(text, InvoiceData(**data))
    assert report.factor_for("subtotal") == 0.3
    assert report.factor_for("tax") == 0.3
    assert report.reason_for("subtotal") == "source labels subtotal as 100.00, extraction says 8.5"
    assert report.reason_for("tax") == "source labels tax as 8.50, extraction says 100.0"
    assert report.factor_for("total") == 1.0
    assert not report.reason_for("total")


def test_eb1_unswapped_golden_is_clean():
    row = _rows()["inv-001"]
    report = run_heuristics(row["text"], InvoiceData(**row["expected"]))
    for f in AMOUNTS:
        assert report.factor_for(f) == 1.0, report.reason_for(f)


def test_eb1_swap_silent_with_role_check_disabled(monkeypatch):
    # Counterfactual: the role check (not another heuristic) does the catching.
    _disable_role_check(monkeypatch)
    text, data = _swap_inv001()
    report = run_heuristics(text, InvoiceData(**data))
    assert report.factor_for("subtotal") == 1.0
    assert report.factor_for("tax") == 1.0


# ---------------------------------------------------------------------------
# EB-2: pipeline needs_review
# ---------------------------------------------------------------------------

def test_eb2_swap_reaches_user_as_needs_review():
    text, data = _swap_inv001()
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([json.dumps(data)])),
        Critic(ScriptedLLMClient([_fooled_critic_raw()])),
    )
    result = pipe.extract(text)
    assert result.status == "needs_review"
    assert {f.field for f in result.flagged_fields} == {"subtotal", "tax"}


# ---------------------------------------------------------------------------
# EB-3: stray number as total
# ---------------------------------------------------------------------------

def test_eb3_stray_store_number_as_total():
    row = _rows()["inv-005"]
    assert "STORE #12" in row["text"]
    report = run_heuristics(row["text"], InvoiceData(**(row["expected"] | {"total": 12.0})))
    assert report.factor_for("total") == 0.3
    assert report.reason_for("total") == "source labels total as 15.00, extraction says 12.0"


# ---------------------------------------------------------------------------
# EB-4: label grammar and formats
# ---------------------------------------------------------------------------

EURO_TEXT = "Sub-total: €1,200.00\nVAT: €240.00\nGrand Total: €1,440.00"


def test_eb4_euro_thousands_subtotal_vat_grand_total():
    report = run_heuristics(EURO_TEXT, _data(subtotal=240.0, tax=1200.0, total=1440.0))
    assert report.factor_for("subtotal") == 0.3
    assert report.factor_for("tax") == 0.3
    assert report.reason_for("subtotal") == "source labels subtotal as 1200.00, extraction says 240.0"
    assert report.reason_for("tax") == "source labels tax as 240.00, extraction says 1200.0"
    assert not report.reason_for("total")
    assert report.factor_for("total") == 1.0


def test_eb4_sub_total_is_not_a_total_label():
    # 1200.0 is present but labelled only by "Sub-total", which is not a total
    # label; the reason lists Grand Total's 1440.00 alone. subtotal/tax null so
    # arithmetic stays out of the reason.
    report = run_heuristics(EURO_TEXT, _data(total=1200.0))
    assert report.reason_for("total") == "source labels total as 1440.00, extraction says 1200.0"


@pytest.mark.parametrize("text, field, good, bad, label", [
    ("SUBTOTAL: 100.00\nItem 99.00", "subtotal", 100.0, 99.0, "100.00"),
    ("Net: 100.00\nItem 99.00", "subtotal", 100.0, 99.0, "100.00"),
    ("GST: 10.00\nItem 9.00", "tax", 10.0, 9.0, "10.00"),
    ("Amount due: 110.00\nItem 109.00", "total", 110.0, 109.0, "110.00"),
    ("AMOUNT DUE: 110.00\nItem 109.00", "total", 110.0, 109.0, "110.00"),
    ("vat: £5.00\nItem 4.00", "tax", 5.0, 4.0, "5.00"),
    ("Total: $ 20.00\nItem 19.00", "total", 20.0, 19.0, "20.00"),
])
def test_eb4_labels_case_insensitive(text, field, good, bad, label):
    ok = run_heuristics(text, _data(**{field: good}))
    assert field not in _role_reasons(ok)
    wrong = run_heuristics(text, _data(**{field: bad}))
    assert wrong.factor_for(field) == 0.3
    assert wrong.reason_for(field) == f"source labels {field} as {label}, extraction says {bad}"


# ---------------------------------------------------------------------------
# EB-5: abstains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, kw", [
    ("Paid 50.00 and 7.00", {"total": 7.0}),
    ("Order 12\nTotal: 15", {"total": 12.0}),
    ("Order 12.00\nTotal:\n15.00", {"total": 12.0}),
    ("Internet 9.00\nTotal: 9.00", {"subtotal": 9.0}),
])
def test_eb5_abstains(text, kw):
    report = run_heuristics(text, _data(**kw))
    assert _role_reasons(report) == {}


def test_eb5_label_too_far_or_digit_between_abstains():
    # More than 25 chars between label and number -> not labelled.
    far = "Total" + " " * 30 + "15.00\nItem 12.00"
    assert _role_reasons(run_heuristics(far, _data(total=12.0))) == {}
    # A digit between label and number -> not labelled.
    digit = "Total 3 items: 15.00\nItem 12.00"
    assert "total" not in _role_reasons(run_heuristics(digit, _data(total=12.0)))


def test_eb5_number_before_label_abstains():
    text = "15.00 Total\nItem 12.00"
    assert _role_reasons(run_heuristics(text, _data(total=12.0))) == {}


# ---------------------------------------------------------------------------
# EB-6: any match clears; distinct values listed in order
# ---------------------------------------------------------------------------

def test_eb6_any_labelled_match_clears():
    report = run_heuristics("Total: 10.00\nTotal: 12.00", _data(total=12.0))
    assert "total" not in _role_reasons(report)
    assert report.factor_for("total") == 1.0


def test_eb6_reason_lists_distinct_values_in_order():
    text = "Item 11.00\nTotal: 10.00\nTotal: 12.00\nTotal: 10.00"
    report = run_heuristics(text, _data(total=11.0))
    assert report.reason_for("total") == "source labels total as 10.00, 12.00, extraction says 11.0"
    assert report.factor_for("total") == 0.3


def test_eb6_tolerance_within_half_cent():
    # 12.004 is within 0.005 of the labelled 12.00 -> match; but it must be
    # "present" for the check to even run, so presence decides the rest.
    report = run_heuristics("Total: 12.00", _data(total=12.004))
    assert "total" not in _role_reasons(report)


# ---------------------------------------------------------------------------
# EB-7: no stacking; inv-004 byte-identical
# ---------------------------------------------------------------------------

def test_eb7_absent_amount_gets_only_presence_reason():
    report = run_heuristics("Total: 15.00\nItem 12.00", _data(total=99.0))
    assert report.reason_for("total") == "amount 99.0 not found in source text"
    assert report.factor_for("total") == 0.35


def test_eb7_inv004_reasons_unchanged():
    row = _rows()["inv-004"]
    raw = json.loads(_replay()["inv-004"]["generator_raw"])
    report = run_heuristics(row["text"], InvoiceData(**raw))
    assert report.reasons == {"total": [
        "amount 683.2 not found in source text",
        "subtotal(540.0) + tax(43.2) != total(683.2)",
    ]}


# ---------------------------------------------------------------------------
# EB-8: nulls untouched; only lowers
# ---------------------------------------------------------------------------

def test_eb8_null_subtotal_not_flagged():
    report = run_heuristics("Subtotal: 10.00\nTotal: 10.00", _data(subtotal=None, total=10.0))
    assert report.factor_for("subtotal") == 1.0
    assert not report.reason_for("subtotal")
    assert "subtotal" not in report.reasons


def test_eb8_inv005_expected_no_reasons():
    row = _rows()["inv-005"]
    report = run_heuristics(row["text"], InvoiceData(**row["expected"]))
    assert report.reasons == {}


_LOWER_CASES = [
    ("inv-001 swap", *_swap_inv001()),
    ("euro", EURO_TEXT, {"subtotal": 240.0, "tax": 1200.0, "total": 1440.0}),
    ("dup", "Item 11.00\nTotal: 10.00\nTotal: 12.00", {"total": 11.0}),
    ("absent", "Total: 15.00\nItem 12.00", {"total": 99.0}),
    ("clean", "Subtotal: 10.00\nTax: 1.00\nTotal: 11.00", {"subtotal": 10.0, "tax": 1.0, "total": 11.0}),
    ("arith", "Subtotal: 10.00\nTax: 1.00\nTotal: 11.00\nX 12.00", {"subtotal": 10.0, "tax": 1.0, "total": 12.0}),
]


@pytest.mark.parametrize("name, text, kw", _LOWER_CASES, ids=[c[0] for c in _LOWER_CASES])
def test_eb8_only_lowers_vs_role_check_disabled(monkeypatch, name, text, kw):
    data = {f: None for f in FIELDS} | kw
    new = run_heuristics(text, InvoiceData(**data))
    with monkeypatch.context() as m:
        _disable_role_check(m)
        old = run_heuristics(text, InvoiceData(**data))
    for f in FIELDS:
        assert new.factor_for(f) <= old.factor_for(f), f
        if new.factor_for(f) != old.factor_for(f):
            # the only new penalty value is exactly 0.3
            assert "source labels" in (new.reason_for(f) or "")
    # provenance spans unchanged
    assert new.provenance == old.provenance


# ---------------------------------------------------------------------------
# EB-9: zero false positives on shipped data
# ---------------------------------------------------------------------------

def test_eb9_no_fp_on_golden_expected():
    rows = _rows()
    assert set(rows) >= {f"inv-00{i}" for i in range(1, 9)}
    for rid, row in rows.items():
        report = run_heuristics(row["text"], InvoiceData(**row["expected"]))
        assert _role_reasons(report) == {}, rid


def test_eb9_no_fp_on_replay_model_outputs():
    rows, rep = _rows(), _replay()
    for i in range(1, 8):
        rid = f"inv-00{i}"
        raw = json.loads(rep[rid]["generator_raw"])
        report = run_heuristics(rows[rid]["text"], InvoiceData(**raw))
        assert _role_reasons(report) == {}, rid


# ---------------------------------------------------------------------------
# EB-10: golden row inv-008 + replay; eval output; genuine catch
# ---------------------------------------------------------------------------

def test_eb10_golden_row_exact():
    row = _rows()["inv-008"]
    assert row["text"] == INV008_TEXT
    assert row["expected"] == INV008_EXPECTED


def test_eb10_replay_entry_shape():
    fx = _replay()["inv-008"]
    gen = json.loads(fx["generator_raw"])
    assert gen == {k: v for k, v in INV008_EXPECTED.items() if v is not None} | {"total": 45.0}
    assert "subtotal" not in gen and "tax" not in gen
    crit = json.loads(fx["critic_raw"])
    assert crit["overall_confidence"] == 0.95
    assert {v["field"] for v in crit["field_verdicts"]} == set(FIELDS)
    for v in crit["field_verdicts"]:
        assert v["confidence"] == 0.95 and v["grounded"] is True


def test_eb10_inv008_flagged_by_role_check():
    raw = json.loads(_replay()["inv-008"]["generator_raw"])
    report = run_heuristics(INV008_TEXT, InvoiceData(**raw))
    assert report.reason_for("total") == "source labels total as 75.00, extraction says 45.0"
    assert report.factor_for("total") == 0.3


def test_eb10_eval_stdout_exact():
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "run_eval.py")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == EXPECTED_STDOUT


def test_eb10_readme_block_matches_stdout():
    text = README.read_text()
    inner = EXPECTED_STDOUT.rstrip("\n")
    assert f"```\n{inner}\n```" in text


def test_eb10_genuine_catch_when_role_check_disabled(monkeypatch, capsys):
    _disable_role_check(monkeypatch)
    m = run_eval.evaluate(live=False)
    assert m.silent_wrong == 1
    assert m.silent_failures == [("inv-008", "total", 75.0, 45.0)]
    assert run_eval.run(live=False, min_accuracy=0.90) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "  [inv-008] total: expected 75.0 but got 45.0  (SILENT)" in lines
    assert "FAIL: silent-wrong fields 1 > allowed 0" in lines


def test_eb10_role_check_enabled_eval_passes():
    m = run_eval.evaluate(live=False)
    assert m.silent_wrong == 0
    assert run_eval.run(live=False, min_accuracy=0.90) == 0
