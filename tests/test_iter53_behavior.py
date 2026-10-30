"""Black-box behavioral tests for iter-53: date-grounding heuristic.

Covers the 12 Expected Behaviors from pm.md:
  1. ISO date found in source -> grounded, span slices to the ISO token.
  2. Slash month-first (golden inv-002) -> grounded.
  3. Slash day-first / dotted / ambiguous (either reading grounds).
  4. One-digit day or month in slash/dot formats.
  5. Month-name forms (full / abbreviated, case-insensitive, both orders).
  6. Made-up ISO date -> penalized (0.3), reason names the value, provenance [].
  7. Near-miss and digit-run embedded dates are not matches.
  8. Non-ISO committed value keeps the 0.2 ISO penalty only.
  9. Null date -> no invoice_date key anywhere in the report.
 10. Gate effect on the heuristics-only path (critic=None).
 11. Golden-set coverage: inv-006 + offline eval readout.
 12. No regression on golden rows inv-001..inv-005.

Isolation contract: tests drive the public package API, the golden data files
and the eval CLI, asserting on observable output. No implementation source was read.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(date: str | None) -> InvoiceData:
    return InvoiceData(
        vendor_name="ACME CORP", invoice_number="INV-1001", invoice_date=date,
        subtotal=100.0, tax=8.5, total=108.5, currency="USD",
    )


def _assert_grounded(text: str, date: str, token: str | None = None) -> None:
    report = run_heuristics(text, _data(date))
    assert report.factor_for("invoice_date") == 1.0, report.reason_for("invoice_date")
    assert not report.reason_for("invoice_date")
    spans = report.provenance["invoice_date"]
    assert len(spans) == 1
    sliced = text[spans[0].start:spans[0].end]
    if token is not None:
        assert sliced == token


def _assert_penalized(text: str, date: str) -> None:
    report = run_heuristics(text, _data(date))
    assert report.factor_for("invoice_date") == 0.3
    assert "not found in source text" in report.reason_for("invoice_date")


def _rows() -> dict[str, dict]:
    out = {}
    for line in GOLDEN.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _run_golden(row_id: str):
    row = _rows()[row_id]
    fx = _replay()[row_id]
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([fx["generator_raw"]])),
        Critic(ScriptedLLMClient([fx["critic_raw"]])),
    )
    return pipe.extract(row["text"])


# ---------------------------------------------------------------------------
# EB-1: ISO in source
# ---------------------------------------------------------------------------

def test_eb1_iso_in_source_grounded():
    text = "ACME CORP\nDate: 2026-03-14\nTotal: 108.50"
    _assert_grounded(text, "2026-03-14", "2026-03-14")


def test_eb1_shared_source_grounded():
    _assert_grounded(SOURCE, "2026-03-14", "2026-03-14")


# ---------------------------------------------------------------------------
# EB-2: Slash month-first
# ---------------------------------------------------------------------------

def test_eb2_slash_month_first_grounded():
    text = "Blue Bottle Coffee\nTOTAL 9.81\n03/22/2026\nUSD"
    _assert_grounded(text, "2026-03-22", "03/22/2026")


def test_eb2_golden_inv002_not_penalized():
    row = _rows()["inv-002"]
    data = InvoiceData(**row["expected"])
    report = run_heuristics(row["text"], data)
    assert report.factor_for("invoice_date") == 1.0
    span = report.provenance["invoice_date"][0]
    assert row["text"][span.start:span.end] == "03/22/2026"


# ---------------------------------------------------------------------------
# EB-3: day-first, dotted, ambiguous
# ---------------------------------------------------------------------------

def test_eb3_slash_day_first():
    _assert_grounded("Invoice date 22/03/2026 ok", "2026-03-22", "22/03/2026")


def test_eb3_dotted():
    _assert_grounded("Datum: 22.03.2026", "2026-03-22", "22.03.2026")


@pytest.mark.parametrize("date", ["2026-04-03", "2026-03-04"])
def test_eb3_ambiguous_either_reading(date):
    _assert_grounded("Date 04/03/2026", date, "04/03/2026")


def test_eb3_ambiguous_does_not_ground_unrelated():
    _assert_penalized("Date 04/03/2026", "2026-04-04")


# ---------------------------------------------------------------------------
# EB-4: one-digit day/month
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date", ["2026-03-04", "2026-04-03"])
def test_eb4_single_digit_slash(date):
    _assert_grounded("Date 3/4/2026 thanks", date, "3/4/2026")


def test_eb4_single_digit_dotted():
    _assert_grounded("Date 3.4.2026", "2026-04-03", "3.4.2026")


# ---------------------------------------------------------------------------
# EB-5: month-name forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "token",
    ["March 14, 2026", "Mar 14, 2026", "14 March 2026", "14 Mar 2026", "14 MAR 2026"],
)
def test_eb5_month_name_forms(token):
    text = f"Initech\nIssued {token}\nTotal 216.00"
    _assert_grounded(text, "2026-03-14", token)


def test_eb5_lowercase_month_name():
    _assert_grounded("issued march 14, 2026", "2026-03-14", "march 14, 2026")


def test_eb5_month_name_wrong_day_penalized():
    _assert_penalized("Issued 14 March 2026", "2026-03-15")


# ---------------------------------------------------------------------------
# EB-6: made-up date
# ---------------------------------------------------------------------------

def test_eb6_made_up_date_penalized():
    _assert_penalized(SOURCE, "2019-07-04")


def test_eb6_reason_names_value_and_no_provenance():
    report = run_heuristics(SOURCE, _data("2019-07-04"))
    assert "2019-07-04" in report.reason_for("invoice_date")
    assert report.provenance["invoice_date"] == []


def test_eb6_penalty_only_lowers_other_fields_untouched():
    report = run_heuristics(SOURCE, _data("2019-07-04"))
    for f in ("vendor_name", "invoice_number", "total", "currency"):
        assert report.factor_for(f) == 1.0


# ---------------------------------------------------------------------------
# EB-7: near-miss / digit boundary
# ---------------------------------------------------------------------------

def test_eb7_near_miss_penalized():
    _assert_penalized("Date: 2026-03-14", "2026-03-15")


def test_eb7_embedded_in_digit_run_penalized():
    _assert_penalized("Ref 12026-03-145", "2026-03-14")


def test_eb7_embedded_slash_digit_run_penalized():
    _assert_penalized("Ref 103/22/20261", "2026-03-22")


# ---------------------------------------------------------------------------
# EB-8: non-ISO committed value unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["March 14", "2026-13-40"])
def test_eb8_non_iso_keeps_iso_penalty(bad):
    report = run_heuristics(SOURCE, _data(bad))
    assert report.factor_for("invoice_date") == 0.2
    reason = report.reason_for("invoice_date")
    assert "ISO 8601" in reason
    assert "not found in source text" not in reason
    assert report.provenance["invoice_date"] == []


# ---------------------------------------------------------------------------
# EB-9: null date
# ---------------------------------------------------------------------------

def test_eb9_null_date_absent_everywhere():
    report = run_heuristics(SOURCE, _data(None))
    assert "invoice_date" not in report.factors
    assert "invoice_date" not in report.reasons
    assert "invoice_date" not in report.provenance
    assert report.factor_for("invoice_date") == 1.0


# ---------------------------------------------------------------------------
# EB-10: gate effect on heuristics-only path
# ---------------------------------------------------------------------------

def _heuristics_only(date: str):
    payload = dict(CORRECT_EXTRACTION, invoice_date=date)
    gen = Generator(ScriptedLLMClient([json.dumps(payload)]))
    return ReflectivePipeline(gen, critic=None).extract(_rows()["inv-001"]["text"])


def test_eb10_made_up_date_needs_review():
    result = _heuristics_only("2019-07-04")
    assert result.status == "needs_review"
    assert "invoice_date" in {f.field for f in result.flagged_fields}


def test_eb10_correct_date_ok():
    result = _heuristics_only("2026-03-14")
    assert result.status == "ok"
    assert result.flagged_fields == []


# ---------------------------------------------------------------------------
# EB-11: golden-set coverage
# ---------------------------------------------------------------------------

def test_eb11_inv006_row_uses_month_name_date():
    rows = _rows()
    assert "inv-006" in rows
    text = rows["inv-006"]["text"]
    months = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    assert any(m in text.lower() for m in months)
    expected_date = rows["inv-006"]["expected"]["invoice_date"]
    # the expected date grounds against the source
    report = run_heuristics(text, InvoiceData(**rows["inv-006"]["expected"]))
    assert report.factor_for("invoice_date") == 1.0
    assert expected_date


def test_eb11_inv006_fixture_shape():
    rows = _rows()
    fx = _replay()["inv-006"]
    gen = json.loads(fx["generator_raw"])
    expected = rows["inv-006"]["expected"]
    assert gen["invoice_date"] != expected["invoice_date"]
    for k, v in expected.items():
        if k != "invoice_date":
            assert gen[k] == v, k
    crit = json.loads(fx["critic_raw"])
    assert all(v["confidence"] == 0.95 for v in crit["field_verdicts"])
    assert len(crit["field_verdicts"]) == 7
    # wrong date is ISO-valid but absent from the source
    report = run_heuristics(rows["inv-006"]["text"], InvoiceData(**gen))
    assert report.factor_for("invoice_date") == 0.3


def test_eb11_inv006_pipeline_needs_review():
    result = _run_golden("inv-006")
    assert result.status == "needs_review"
    assert "invoice_date" in {f.field for f in result.flagged_fields}


def test_eb11_eval_cli_readout():
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "run_eval.py")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = re.sub(r"[ \t]+", " ", proc.stdout)
    assert re.search(r"examples : 8\b", out), out
    assert re.search(r"field accuracy : 92\.9% \(52/56\)", out), out
    assert re.search(r"exact-match acc : 50\.0% \(4/8\)", out), out
    assert re.search(r"needs-review rate : 50\.0% \(4/8\)", out), out
    assert "[inv-006] invoice_date" in out
    assert "PASS" in out


# ---------------------------------------------------------------------------
# EB-12: no regression on inv-001..inv-005
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "row_id,status",
    [("inv-001", "ok"), ("inv-002", "ok"), ("inv-003", "ok"),
     ("inv-004", "needs_review"), ("inv-005", "ok")],
)
def test_eb12_golden_statuses_unchanged(row_id, status):
    result = _run_golden(row_id)
    assert result.status == status
    assert "invoice_date" not in {f.field for f in result.flagged_fields}


@pytest.mark.parametrize("row_id", ["inv-001", "inv-002", "inv-003", "inv-004", "inv-005"])
def test_eb12_no_date_penalty_on_golden_rows(row_id):
    row = _rows()[row_id]
    for data in (row["expected"], json.loads(_replay()[row_id]["generator_raw"])):
        report = run_heuristics(row["text"], InvoiceData(**data))
        assert report.factor_for("invoice_date") == 1.0
        assert not report.reason_for("invoice_date")
        assert len(report.provenance["invoice_date"]) == 1
