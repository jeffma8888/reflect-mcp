"""Black-box behavioral tests for iter-95 (#71): a currency the source writes by
an unambiguous name grounds its ISO code.

Covers the Expected Behaviors from pm.md:
  1. The correct inv-011 answer passes ``validate --heuristics-only`` (status ok,
     currency confidence 1.0, provenance over ``US Dollars``).
  2. Every golden ``expected`` answer is clean under ``run_heuristics``;
     non-vacuity: a bare ``dollars`` keeps the HEAD penalty.
  3. Each listed name grounds its own code (case-insensitive, any whitespace run).
  4. Ambiguous / foreign / bare names do NOT ground (factor 0.25, one reason).
  5. A literal code still wins (span over the literal ``USD``).
  6. The ISO-shape check is unchanged (``US Dollars`` as a value is penalized).
  7. ``make eval`` exits 0 and keeps the pinned summary + inv-011 row; the README
     eval block matches its stdout (HEAD byte-identity is a one-off report check).
  8. Docs: heuristics ``__doc__`` keeps the 10 labels and its grounding bullet
     mentions written names; README heuristics bullet names the rule.

Isolation contract: drives only the public CLI, ``run_heuristics``, the module
``__doc__``, README.md and the golden dataset; never reads implementation
source. It spawns no VCS process and makes no network calls.
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

S11 = (
    "Fabrikam Supplies\nInvoice No: FS-3307\nDate: 2026-09-15\nSubtotal: 80.00\n"
    "Tax: 6.40\nTotal: 86.40\nAmounts in US Dollars"
)
D11 = {
    "vendor_name": "Fabrikam Supplies",
    "invoice_number": "FS-3307",
    "invoice_date": "2026-09-15",
    "subtotal": 80.0,
    "tax": 6.4,
    "total": 86.4,
    "currency": "USD",
}

EXPECTED_LABELS = {
    "grounding",
    "arithmetic",
    "date format",
    "date grounding",
    "amount-role grounding",
    "party-role grounding",
    "date-role grounding",
    "identifier-role grounding",
    "currency-code shape",
    "omitted total",
}


# ---- helpers ---------------------------------------------------------------
def _validate(data: dict, text: str, tmp: Path) -> tuple[int, dict]:
    src = tmp / "src.txt"
    src.write_text(text, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--data", json.dumps(data), "--file", str(src)],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(tmp),
    )
    return r.returncode, json.loads(r.stdout)


def _spans(report, field: str) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in report.provenance.get(field, [])]


def _all_reasons(report) -> dict:
    return {k: v for k, v in report.reasons.items() if v}


# ---- EB1 -------------------------------------------------------------------
def test_eb1_correct_inv011_answer_passes_cli(tmp_path):
    assert S11[105:115] == "US Dollars"
    rc, d = _validate(D11, S11, tmp_path)
    assert rc == 0
    assert d["status"] == "ok"
    assert d["flagged_fields"] == []
    assert d["field_confidence"]["currency"] == 1.0
    assert d["field_provenance"]["currency"] == [{"start": 105, "end": 115}]


# ---- EB2 -------------------------------------------------------------------
def _golden_rows() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_eb2_golden_has_13_rows_with_inv011_source():
    rows = _golden_rows()
    assert len(rows) == 13
    inv011 = [r for r in rows if r["id"] == "inv-011"]
    assert len(inv011) == 1
    assert inv011[0]["text"] == S11
    assert inv011[0]["expected"] == D11


@pytest.mark.parametrize("row", _golden_rows(), ids=lambda r: r["id"])
def test_eb2_every_golden_expected_answer_is_clean(row):
    report = run_heuristics(row["text"], InvoiceData(**row["expected"]))
    assert _all_reasons(report) == {}, (row["id"], report.reasons)
    for field, factor in report.factors.items():
        assert factor is None or factor >= 1.0, (row["id"], field, factor)


def test_eb2_non_vacuity_bare_dollars_keeps_penalty():
    text = S11.replace("US Dollars", "dollars")
    assert text != S11
    report = run_heuristics(text, InvoiceData(**D11))
    assert report.reasons.get("currency") == ["value 'USD' not found in source text"]
    assert report.factors.get("currency") == 0.25


# ---- EB3 -------------------------------------------------------------------
GROUNDS = [
    ("USD", "US Dollars"),
    ("USD", "U.S. dollars"),
    ("USD", "United States Dollar"),
    ("USD", "us  dollars"),
    ("GBP", "pounds sterling"),
    ("GBP", "Pound Sterling"),
    ("GBP", "British pounds"),
    ("JPY", "Japanese yen"),
    ("CAD", "Canadian dollars"),
    ("AUD", "Australian Dollar"),
]


@pytest.mark.parametrize("code,phrase", GROUNDS)
def test_eb3_name_grounds_its_code(code, phrase):
    text = "X Co\n" + phrase
    report = run_heuristics(text, InvoiceData(vendor_name="X Co", currency=code))
    assert not report.reasons.get("currency"), report.reasons
    assert "currency" not in report.factors or report.factors["currency"] >= 1.0
    spans = _spans(report, "currency")
    assert len(spans) == 1, spans
    s, e = spans[0]
    assert text[s:e] == phrase


def test_eb3_extra_whitespace_kinds_ground():
    # "any whitespace run between words counts": tab and newline too.
    for phrase in ("US\tDollars", "pounds \t sterling"):
        code = "USD" if "US" in phrase else "GBP"
        text = "X Co\n" + phrase
        report = run_heuristics(text, InvoiceData(vendor_name="X Co", currency=code))
        assert not report.reasons.get("currency"), (phrase, report.reasons)
        (s, e), = _spans(report, "currency")
        assert text[s:e] == phrase


# ---- EB4 -------------------------------------------------------------------
NOT_GROUNDS = [
    ("USD", "Paid in dollars"),
    ("USD", "Australian dollars"),
    ("USD", "Canadian dollars"),
    ("USD", "AUS Dollars"),
    ("GBP", "US Dollars"),
    ("CHF", "US Dollars"),
    ("GBP", "pounds"),
    ("JPY", "yen"),
]


@pytest.mark.parametrize("code,phrase", NOT_GROUNDS)
def test_eb4_name_does_not_ground(code, phrase):
    text = "X Co\n" + phrase
    report = run_heuristics(text, InvoiceData(vendor_name="X Co", currency=code))
    assert report.factors.get("currency") == 0.25
    assert report.reasons.get("currency") == [f"value '{code}' not found in source text"]
    assert report.provenance.get("currency", []) == []


# ---- EB5 -------------------------------------------------------------------
def test_eb5_literal_code_still_wins():
    text = "X Co\nCurrency: USD\nAmounts in US Dollars"
    report = run_heuristics(text, InvoiceData(vendor_name="X Co", currency="USD"))
    assert not report.reasons.get("currency")
    spans = _spans(report, "currency")
    assert spans, spans
    s, e = spans[0]
    assert (s, text[s:e]) == (15, "USD")


# ---- EB6 -------------------------------------------------------------------
def test_eb6_shape_check_unchanged_in_process():
    data = dict(D11, currency="US Dollars")
    report = run_heuristics(S11, InvoiceData(**data))
    assert report.factors.get("currency") == 0.2
    assert report.reasons.get("currency") == [
        "currency 'US Dollars' is not an ISO 4217 code (3 uppercase letters)"
    ]
    assert report.provenance.get("currency", []) == []


def test_eb6_shape_check_unchanged_on_cli(tmp_path):
    rc, d = _validate(dict(D11, currency="US Dollars"), S11, tmp_path)
    assert d["status"] == "needs_review"
    assert [f["field"] for f in d["flagged_fields"]] == ["currency"]


# ---- EB7 -------------------------------------------------------------------
def test_eb7_make_eval_unchanged_and_readme_block_matches():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    out = r.stdout
    for line in (
        "field accuracy    : 91.2%  (83/91)",
        "catch rate        : 100.0%  (8/8)",
        "silent-wrong      : 0",
        "false alarms      : 0",
        "  [inv-011] currency: expected 'USD' but got 'US Dollars'  (flagged)",
    ):
        assert line in out.splitlines(), line
    readme = README.read_text(encoding="utf-8")
    m = re.search(r"```\n(={10,}\nreflect-mcp eval  \|  mode: OFFLINE.*?)```", readme, re.S)
    assert m, "README eval block not found"
    block = m.group(1).strip("\n")
    assert block in out, "README eval block drifted from make eval stdout"


# ---- EB8 -------------------------------------------------------------------
_BULLET = re.compile(r"^  - ([^:\n]+):", re.MULTILINE)


def test_eb8_docstring_labels_unchanged_and_grounding_mentions_names():
    doc = heuristics.__doc__
    assert doc
    assert {m.group(1).strip() for m in _BULLET.finditer(doc)} == EXPECTED_LABELS
    m = re.search(r"^  - grounding:(.*?)(?=^  - |\Z)", doc, re.MULTILINE | re.S)
    assert m, "grounding bullet missing"
    body = " ".join(m.group(1).split()).lower()
    assert "currency" in body and "name" in body, body


def test_eb8_readme_heuristics_bullet_names_rule():
    lines = [ln for ln in README.read_text(encoding="utf-8").splitlines()
             if ln.startswith("- **Deterministic heuristics**")]
    assert len(lines) == 1
    bullet = lines[0]
    assert "`US Dollars`" in bullet and "`USD`" in bullet
    assert re.search(r"bare `dollars` grounds nothing", bullet), bullet


def test_eb8_no_vcs_in_this_module():
    src = Path(__file__).read_text(encoding="utf-8")
    forbidden = ["[" + '"g' + 'it"', "'g" + "it'"]
    for tok in forbidden:
        assert tok not in src, tok
