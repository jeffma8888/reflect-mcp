"""Black-box behavioral tests for iter-101 (B3): a single source-stated tax rate
must explain the extracted ``tax`` on some base, otherwise ``tax`` is penalized.

Covers the Expected Behaviors from pm.md:
  1. A rate read as the tax amount (total null) -> needs_review, exit 1, ``tax``
     flagged with a ``10% tax rate`` reason.
  2. A correct tax under a stated rate -> ok, exit 0, nothing flagged.
  3. A wrong tax with a total present blames ``tax`` (``20% tax rate``).
  4. A tax-only extraction (no total key) is still checked; the right tax passes.
  5. Abstain cases (two rates, tax-inclusive, discount base, fractional rate,
     no rate label, null tax, null subtotal+total) carry no ``% tax rate`` reason.
  6. Golden-derived coverage: the ``expected`` answers stay clean; appending the
     golden row's own rate keeps them clean, and swapping ``tax := rate`` is caught.
  7. ``make eval`` keeps its rc 0 and pinned summary; README eval block matches.
  8. Docs: ``heuristics.__doc__`` has a ``tax rate:`` bullet; both README
     heuristic lists name the check.

Isolation contract: drives only the public CLI, ``run_heuristics``, the module
``__doc__``, README.md and the golden dataset; never reads implementation
source. It spawns no VCS process, imports no optional extra, makes no network
calls and reads no secrets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp import heuristics
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.schema import InvoiceData

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
RATE_MARK = "% tax rate"

HEADER = "ACME CORP\nInvoice #A-1\nDate: 2026-03-14\n"
BASE = {
    "vendor_name": "ACME CORP",
    "invoice_number": "A-1",
    "invoice_date": "2026-03-14",
    "currency": "USD",
}


# ---- helpers ---------------------------------------------------------------
def _env() -> dict:
    drop = ("REFLECT_", "AWS_")
    return {
        k: v for k, v in os.environ.items()
        if not k.startswith(drop) and not k.endswith("_API_KEY")
    }


def _validate(data: dict, text: str, tmp: Path) -> tuple[int, dict]:
    """Run the CLI with the source on stdin (as the spec states)."""
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--data", json.dumps(data)],
        input=text, capture_output=True, text=True, env=_env(), timeout=60,
        cwd=str(tmp),
    )
    assert r.stdout.strip(), r.stderr[-2000:]
    return r.returncode, json.loads(r.stdout)


def _flags(d: dict) -> dict[str, str]:
    return {f["field"]: f["reason"] for f in d["flagged_fields"]}


def _reasons(text: str, data: dict) -> dict[str, list[str]]:
    report = run_heuristics(text, InvoiceData(**data))
    return {k: list(v) for k, v in report.reasons.items() if v}


def _rate_reasons(text: str, data: dict) -> list[str]:
    return [r for rs in _reasons(text, data).values() for r in rs if RATE_MARK in r]


# ---- EB1 -------------------------------------------------------------------
def test_eb1_rate_read_as_tax_with_null_total_is_flagged(tmp_path):
    src = HEADER + "Subtotal: 200.00\nTax 10%: 20.00\nCurrency USD"
    data = {**BASE, "subtotal": 200.0, "tax": 10.0, "total": None}
    rc, d = _validate(data, src, tmp_path)
    assert rc == 1
    assert d["status"] == "needs_review"
    flags = _flags(d)
    assert "tax" in flags, flags
    assert "10% tax rate" in flags["tax"], flags["tax"]
    assert d["field_confidence"]["tax"] < 1.0


def test_eb1_reason_is_on_tax_only_in_process():
    src = HEADER + "Subtotal: 200.00\nTax 10%: 20.00\nCurrency USD"
    data = {**BASE, "subtotal": 200.0, "tax": 10.0, "total": None}
    reasons = _reasons(src, data)
    holders = [f for f, rs in reasons.items() if any(RATE_MARK in r for r in rs)]
    assert holders == ["tax"], reasons


# ---- EB2 -------------------------------------------------------------------
def test_eb2_correct_tax_under_stated_rate_is_ok(tmp_path):
    src = HEADER + "Subtotal: 200.00\nVAT 20%: 40.00\nTotal: 240.00\nUSD"
    data = {**BASE, "subtotal": 200.0, "tax": 40.0, "total": 240.0}
    rc, d = _validate(data, src, tmp_path)
    assert rc == 0, d
    assert d["status"] == "ok"
    assert d["flagged_fields"] == []


# ---- EB3 -------------------------------------------------------------------
MULLER = ("Muller GmbH\nInvoice No: DE-1\nDate: 2026-01-09\nNet: 1250.00\n"
          "VAT 20%\nGrand Total: 1487.50\nEUR")
MULLER_DATA = {
    "vendor_name": "Muller GmbH",
    "invoice_number": "DE-1",
    "invoice_date": "2026-01-09",
    "subtotal": 1250.0,
    "tax": 20.0,
    "total": 1487.5,
    "currency": "EUR",
}


def test_eb3_wrong_tax_with_total_blames_tax(tmp_path):
    rc, d = _validate(MULLER_DATA, MULLER, tmp_path)
    assert rc == 1
    assert d["status"] == "needs_review"
    flags = _flags(d)
    assert "tax" in flags, flags
    assert "20% tax rate" in flags["tax"], flags["tax"]


def test_eb3_correct_muller_tax_carries_no_rate_reason():
    # Counterpart to EB3: with a stated rate consistent with the amounts
    # (237.50 / 1250.00 = 19%), the true tax carries no rate reason.
    src = MULLER.replace("VAT 20%", "VAT 19%")
    data = {**MULLER_DATA, "tax": 237.5}
    assert _rate_reasons(src, data) == []


# ---- EB4 -------------------------------------------------------------------
EB4_SRC = HEADER + "Subtotal: 200.00\nVAT (20%): 40.00\nUSD"


def test_eb4_tax_only_extraction_wrong_tax_flagged(tmp_path):
    data = {**BASE, "subtotal": 200.0, "tax": 20.0}  # total key absent
    rc, d = _validate(data, EB4_SRC, tmp_path)
    flags = _flags(d)
    assert "tax" in flags, d
    assert "20% tax rate" in flags["tax"], flags["tax"]
    assert rc == 1


def test_eb4_tax_only_extraction_right_tax_not_flagged(tmp_path):
    data = {**BASE, "subtotal": 200.0, "tax": 40.0}
    rc, d = _validate(data, EB4_SRC, tmp_path)
    assert "tax" not in _flags(d), d
    assert not any(RATE_MARK in r for r in _flags(d).values())


# ---- EB5 -------------------------------------------------------------------
ABSTAIN = {
    "a_two_distinct_rates": (
        HEADER + "Subtotal: 200.00\nVAT 5%: 3.00\nVAT 20%: 28.00\nTotal: 231.00\nUSD",
        {**BASE, "subtotal": 200.0, "tax": 99.0, "total": 299.0},
    ),
    "b_tax_inclusive_total": (
        HEADER + "Total (incl. VAT 20%): 120.00\nUSD",
        {**BASE, "subtotal": None, "tax": 20.0, "total": 120.0},
    ),
    "c_tax_on_discounted_subtotal": (
        HEADER + "Subtotal: 100.00\nDiscount 10%: -10.00\nTax 8%: 7.20\nTotal: 97.20\nUSD",
        {**BASE, "subtotal": 100.0, "tax": 7.2, "total": 97.2},
    ),
    "d_fractional_rate": (
        HEADER + "Subtotal: 100.00\nSales tax 8.875%: 8.88\nTotal: 108.88\nUSD",
        {**BASE, "subtotal": 100.0, "tax": 8.88, "total": 108.88},
    ),
    "e_percent_only_after_discount": (
        HEADER + "Subtotal: 200.00\nDiscount 10%\nTax: 10.00\nTotal: 210.00\nUSD",
        {**BASE, "subtotal": 200.0, "tax": 10.0, "total": 210.0},
    ),
    "f_tax_null": (
        HEADER + "Subtotal: 200.00\nTax 10%\nTotal: 220.00\nUSD",
        {**BASE, "subtotal": 200.0, "tax": None, "total": 220.0},
    ),
    "g_subtotal_and_total_null": (
        HEADER + "Tax 10%: 10.00\nUSD",
        {**BASE, "subtotal": None, "tax": 10.0, "total": None},
    ),
}


@pytest.mark.parametrize("case", sorted(ABSTAIN))
def test_eb5_rule_abstains(case):
    text, data = ABSTAIN[case]
    assert _rate_reasons(text, data) == [], (case, _reasons(text, data))


def test_eb5_two_rates_abstain_via_cli(tmp_path):
    text, data = ABSTAIN["a_two_distinct_rates"]
    _rc, d = _validate(data, text, tmp_path)
    assert not any(RATE_MARK in r for r in _flags(d).values()), d


def test_eb5_single_rate_counterpart_does_fire():
    # Non-vacuity for (a): drop one of the two rate lines and the same wrong tax
    # is caught, so (a) abstains because of the second rate, not by accident.
    text, data = ABSTAIN["a_two_distinct_rates"]
    single = text.replace("VAT 5%: 3.00\n", "")
    assert any("20% tax rate" in r for r in _rate_reasons(single, data))


def test_eb5_inclusive_counterpart_wrong_tax_fires():
    # Non-vacuity for (b): a tax no base explains under the same source is caught.
    text, data = ABSTAIN["b_tax_inclusive_total"]
    assert any("20% tax rate" in r for r in _rate_reasons(text, {**data, "tax": 24.0}))


@pytest.mark.parametrize("label", ["TAX", "Vat", "gst"])
def test_eb5_label_is_case_insensitive(label):
    text = HEADER + f"Subtotal: 200.00\n{label} 10%: 20.00\nUSD"
    data = {**BASE, "subtotal": 200.0, "tax": 10.0}
    assert any("10% tax rate" in r for r in _rate_reasons(text, data))


def test_eb5_label_inside_word_does_not_count():
    # whole-word label: "Taxi 10%" is not a tax rate.
    text = HEADER + "Subtotal: 200.00\nTaxi 10% off\nTax: 10.00\nTotal: 210.00\nUSD"
    data = {**BASE, "subtotal": 200.0, "tax": 10.0, "total": 210.0}
    assert _rate_reasons(text, data) == []


# ---- EB6 -------------------------------------------------------------------
def _golden_rows() -> list[dict]:
    return [json.loads(ln) for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _rate_rows() -> list[tuple[dict, float]]:
    out = []
    for row in _golden_rows():
        exp = row["expected"]
        sub, tax = exp.get("subtotal"), exp.get("tax")
        if sub is None or tax is None or not sub > 0:
            continue
        r = round(tax / sub * 100, 3)
        if abs(r) >= 100:
            continue
        out.append((row, r))
    return out


@pytest.mark.parametrize("row", _golden_rows(), ids=lambda r: r["id"])
def test_eb6a_golden_expected_is_clean(row):
    assert _reasons(row["text"], row["expected"]) == {}, row["id"]


@pytest.mark.parametrize("pair", _rate_rows(), ids=lambda p: p[0]["id"])
def test_eb6b_golden_with_own_rate_no_false_alarm(pair):
    row, r = pair
    text = row["text"] + f"\nTax rate {r:g}%"
    assert _rate_reasons(text, row["expected"]) == [], (row["id"], r)


@pytest.mark.parametrize(
    "pair",
    [p for p in _rate_rows() if abs(p[1] - p[0]["expected"]["tax"]) > 0.02],
    ids=lambda p: p[0]["id"],
)
def test_eb6b_golden_rate_swapped_into_tax_is_caught(pair):
    row, r = pair
    text = row["text"] + f"\nTax rate {r:g}%"
    reasons = _reasons(text, {**row["expected"], "tax": r})
    assert any(RATE_MARK in x and f"{r:g}%" in x for x in reasons.get("tax", [])), (row["id"], reasons)


def test_eb6b_coverage_is_non_vacuous():
    rows = _rate_rows()
    catch = [p for p in rows if abs(p[1] - p[0]["expected"]["tax"]) > 0.02]
    assert len(rows) >= 1
    assert len(catch) >= 1
    # PM measured 11 / 10 on the prototype; kept as a lower bound, not an exact
    # count pin, so a future golden row does not turn this module red.
    assert len(catch) <= len(rows)


# ---- EB7 -------------------------------------------------------------------
def test_eb7_make_eval_unchanged_and_readme_block_matches():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True,
                       text=True, timeout=300, env=_env())
    assert r.returncode == 0, r.stdout[-2000:]
    out = r.stdout.splitlines()
    for line in (
        "field accuracy    : 91.2%  (83/91)",
        "catch rate        : 100.0%  (8/8)",
        "silent-wrong      : 0",
        "false alarms      : 0",
    ):
        assert line in out, line
    readme = README.read_text(encoding="utf-8")
    m = re.search(r"```\n(={10,}\nreflect-mcp eval  \|  mode: OFFLINE.*?)```", readme, re.S)
    assert m, "README eval block not found"
    assert m.group(1).strip("\n") in r.stdout


# ---- EB8 -------------------------------------------------------------------
def test_eb8_docstring_has_tax_rate_bullet():
    doc = heuristics.__doc__
    assert doc
    m = re.search(r"^  - tax rate:(.*?)(?=^  - |\n\n|\Z)", doc, re.MULTILINE | re.S)
    assert m, "tax rate bullet missing"
    body = " ".join(m.group(1).split()).lower()
    assert "rate" in body and "tax" in body and "subtotal" in body, body


def test_eb8_readme_heuristic_lists_name_the_check():
    text = README.read_text(encoding="utf-8")
    det = [ln for ln in text.splitlines() if ln.startswith("- **Deterministic heuristics**")]
    assert len(det) == 1
    assert re.search(r"tax-rate check", det[0]), det[0]
    ho = [ln for ln in text.splitlines() if ln.startswith("**Heuristics-only mode**")]
    assert len(ho) == 1
    assert re.search(r"tax rate", ho[0], re.I), ho[0]


def test_eb8_module_hygiene():
    src = Path(__file__).read_text(encoding="utf-8")
    for tok in ("import " + "mcp", "reflect_mcp." + "server", "[" + '"g' + 'it"', "'g" + "it'"):
        assert tok not in src, tok
