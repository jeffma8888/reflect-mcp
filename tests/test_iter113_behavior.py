"""Black-box behavioral tests for iter-113 (roadmap #107): heuristics read EU
decimal-comma amounts ("1.234,03", "197,03") as their true values.

The correct EU reading grounds, and the classic dot-fragment misread (1.234 out
of "1.234,03") is flagged instead of returning a silent ``ok``. US formats and
non-two-decimal comma forms keep their HEAD behaviour.

Covers EB1-EB10 from pm.md. EB1, EB2, EB4 (clean half), EB5 (clean half), EB6,
EB9 (EU rewrite) and EB10 (provenance span) are the discriminators (red on the
pre-change tree); EB3, EB7, EB8 and the eval/dataset checks are preservation
checks, each paired with a discriminator.

Isolation contract: only public interfaces are driven
(``build_default_pipeline``, ``InvoiceData``, the golden dataset, ``make eval``);
heuristics-only pipeline, so no network, no secrets, no model calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

import pytest

from reflect_mcp.pipeline import build_default_pipeline
from reflect_mcp.schema import InvoiceData

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
# sha256 of eval/golden/dataset.jsonl as shipped before iter 113 (fixed literal,
# not a git baseline, so it survives the ship commit).
GOLDEN_SHA256 = "43843904de624a1677542329761a3c47190f17da2d957484eb5f9602db10fe30"

EU_TOTAL = "Total: 1.234,03 EUR"


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _run(text: str, data: dict):
    pipe = build_default_pipeline(generator_enabled=False, critic_enabled=False)
    return pipe.validate(text, InvoiceData.model_validate(data))


def _flagged(result) -> set[str]:
    return {f.field for f in result.flagged_fields}


def _assert_clean(text: str, data: dict) -> None:
    r = _run(text, data)
    assert r.status == "ok", (text, data, r.status, [(f.field, f.reason) for f in r.flagged_fields])
    assert _flagged(r) == set(), (text, data, _flagged(r))


def _assert_flags(text: str, data: dict, field: str) -> None:
    r = _run(text, data)
    assert r.status == "needs_review", (text, data, r.status)
    assert field in _flagged(r), (text, data, _flagged(r))


# ---------------------------------------------------------------------------
# EB1: correct EU total grounds
# ---------------------------------------------------------------------------

def test_eb1_correct_eu_total_grounds():
    _assert_clean(EU_TOTAL, {"total": 1234.03, "currency": "EUR"})


# ---------------------------------------------------------------------------
# EB2: the dot-fragment misread is flagged (was a silent ok)
# ---------------------------------------------------------------------------

def test_eb2_fragment_misread_flagged():
    _assert_flags(EU_TOTAL, {"total": 1.234, "currency": "EUR"}, "total")


# ---------------------------------------------------------------------------
# EB3: integer misreads stay flagged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wrong", [1234.0, 1.0])
def test_eb3_integer_misreads_flagged(wrong):
    _assert_flags(EU_TOTAL, {"total": wrong, "currency": "EUR"}, "total")


# ---------------------------------------------------------------------------
# EB4: full EU invoice grounds every amount
# ---------------------------------------------------------------------------

DE_TEXT = "Zwischensumme: 1.037,00\nMwSt: 197,03\nGesamt: 1.234,03\nCurrency: EUR"
EU_FULL = {"subtotal": 1037.0, "tax": 197.03, "total": 1234.03, "currency": "EUR"}


def test_eb4_full_de_invoice_grounds():
    _assert_clean(DE_TEXT, EU_FULL)


def test_eb4_full_de_invoice_wrong_tax_flagged():
    _assert_flags(DE_TEXT, {**EU_FULL, "tax": 19703.0}, "tax")


# ---------------------------------------------------------------------------
# EB5: labelled-role check reads the comma form
# ---------------------------------------------------------------------------

EN_EU_TEXT = "Subtotal: 1.037,00\nTax: 197,03\nTotal: 1.234,03\nCurrency: EUR"


def test_eb5_labelled_eu_amounts_ground():
    _assert_clean(EN_EU_TEXT, EU_FULL)


def test_eb5_total_under_subtotal_label_flagged():
    # 1037.0 is present in the source, but under the Subtotal label.
    _assert_flags(EN_EU_TEXT, {**EU_FULL, "total": 1037.0}, "total")


# ---------------------------------------------------------------------------
# EB6: prose comma after an amount still grounds
# ---------------------------------------------------------------------------

PROSE = ("Items came to 540,00 before tax, tax was 43,20, and the amount due is "
         "583,20. Pay in GBP.")


def test_eb6_prose_trailing_comma_grounds():
    _assert_clean(PROSE, {"subtotal": 540.0, "tax": 43.2, "total": 583.2, "currency": "GBP"})


def test_eb6_prose_wrong_tax_still_flagged():
    # Pairs the clean case: a wrong tax in the same prose is not waved through.
    _assert_flags(PROSE, {"subtotal": 540.0, "tax": 4320.0, "total": 583.2,
                          "currency": "GBP"}, "tax")


# ---------------------------------------------------------------------------
# EB7: US format unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,total", [
    ("Total: 1,234.03 USD", 1234.03),
    ("Total: 1,500 USD", 1500.0),
    ("Total: 1,234,567.00 USD", 1234567.0),
])
def test_eb7_us_format_unchanged(text, total):
    _assert_clean(text, {"total": total, "currency": "USD"})


# ---------------------------------------------------------------------------
# EB8: exactly two decimals, no thousands-comma chains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,total", [
    ("Total: 1,5 EUR", 1.5),
    ("Total: 1,500 EUR", 1.5),
    ("Total: 1,234,567 EUR", 1.23),
])
def test_eb8_non_two_decimal_comma_not_eu(text, total):
    _assert_flags(text, {"total": total, "currency": "EUR"}, "total")


# ---------------------------------------------------------------------------
# EB9: derived golden coverage (no dataset edit)
# ---------------------------------------------------------------------------

_US_AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\.(\d{2})(?!\d)")


def _eu(m: re.Match) -> str:
    digits = m.group(1).replace(",", "")
    groups = []
    while len(digits) > 3:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    groups.insert(0, digits)
    return ".".join(groups) + "," + m.group(2)


def _to_eu(text: str) -> str:
    return _US_AMOUNT.sub(_eu, text)


def _rows() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_eb9_rewriter_examples():
    # Guard the test helper itself against a vacuous rewrite.
    assert _to_eu("Total: 1487.50") == "Total: 1.487,50"
    assert _to_eu("tax 43.20 x") == "tax 43,20 x"
    assert _to_eu("1,234,567.00") == "1.234.567,00"
    assert _to_eu("2026-03-14") == "2026-03-14"


def test_eb9_golden_rows_clean_on_original_text():
    rows = _rows()
    assert len(rows) == 13
    for row in rows:
        r = _run(row["text"], row["expected"])
        assert r.status == "ok" and not r.flagged_fields, (row["id"], _flagged(r))


@pytest.mark.parametrize("idx", range(13))
def test_eb9_golden_rows_clean_in_eu_form(idx):
    row = _rows()[idx]
    eu_text = _to_eu(row["text"])
    assert eu_text != row["text"], row["id"]  # every row carries a rewritable amount
    r = _run(eu_text, row["expected"])
    assert r.status == "ok", (row["id"], eu_text, [(f.field, f.reason) for f in r.flagged_fields])
    assert not r.flagged_fields, (row["id"], _flagged(r))


def test_eb9_inv003_dot_fragment_flagged():
    row = next(r for r in _rows() if r["id"] == "inv-003")
    assert row["expected"]["total"] == 1487.5
    eu_text = _to_eu(row["text"])
    assert "1.487,50" in eu_text
    _assert_flags(eu_text, {**row["expected"], "total": 1.487}, "total")


# ---------------------------------------------------------------------------
# EB10: provenance covers the whole token; eval and dataset unchanged
# ---------------------------------------------------------------------------

def test_eb10_provenance_whole_eu_token():
    r = _run(EU_TOTAL, {"total": 1234.03, "currency": "EUR"})
    spans = r.field_provenance["total"]
    assert len(spans) == 1, spans
    assert EU_TOTAL[spans[0].start:spans[0].end] == "1.234,03"


def test_eb10_golden_dataset_byte_identical():
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_eb10_make_eval_unchanged():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = r.stdout.splitlines()
    for line in (
        "examples          : 13",
        "field accuracy    : 91.2%  (83/91)",
        "silent-wrong      : 0",
    ):
        assert line in lines, (line, r.stdout[-2000:])


# ---------------------------------------------------------------------------
# AC: README Deterministic heuristics bullet documents the EU reading
# ---------------------------------------------------------------------------

def test_readme_heuristics_bullet_mentions_decimal_comma():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    bullets = [ln for ln in readme.splitlines() if ln.startswith("- **Deterministic heuristics**")]
    assert len(bullets) == 1
    bullet = bullets[0]
    assert "1.234,03" in bullet
    assert "currency-code shape" in bullet
    assert bullet.index("1.234,03") < bullet.index("currency-code shape")
