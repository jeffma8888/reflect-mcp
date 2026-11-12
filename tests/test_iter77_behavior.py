"""Black-box behavioral tests for iter-77: identifier-role grounding + golden row inv-010.

Covers Expected Behaviors of pm.md:

  1. PO number extracted as invoice_number -> factor 0.3, reason names it, span kept.
  2. All 8 other-id labels (PO, P.O., Purchase order, Order, Customer, Client,
     Account, Acct) penalize at 0.3 when an ``Invoice No:`` line exists.
  3. Abstain cases (a)-(e): no invoice label, value also on invoice line,
     ``Customer invoice ...`` line, non-whole-word ``Orderly``, null value.
  4. Ungrounded invoice_number keeps the single 0.25 "not found" penalty.
  5. No other field's factor/reason changes (vs HEAD 0b7a53a heuristics), and the
     9 pre-existing golden rows' reports are identical to HEAD.
  6. inv-010 in dataset.jsonl and replay.json (PO number copied, Critic fooled).
  7. Live ``run_eval.py`` stdout numbers + failure line, exit 0.
  8. HEAD heuristics + inv-010 -> (SILENT), silent-wrong 1, non-zero exit.
  9. ``reflect-mcp validate --heuristics-only`` -> needs_review / ok.
 10. README sections + heuristics module docstring name identifier-role grounding;
     roadmap #53 DONE + ledger row.

Isolation contract: tests drive the public ``run_heuristics`` API, the CLI, the
eval script, the dataset/fixture files, README/PRODUCT.md and the module
docstring. HEAD's heuristics.py is only *executed* (as a counterfactual), never
read. No implementation source, engineer notes or diff were read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.schema import EXTRACTION_FIELDS, InvoiceData

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
DATASET = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"
RUN_EVAL = EVAL_DIR / "run_eval.py"
README = REPO / "README.md"
PRODUCT = REPO / "PRODUCT.md"
HEAD_COMMIT = "0b7a53a"

EB1_TEXT = (
    "Tailspin Hardware\nInvoice No: TH-5120\nPO Number: PO-88213\nDate: 2026-08-03\n"
    "Subtotal: 120.00\nTax: 9.60\nTotal: 129.60\nCurrency: USD"
)
EB1_EXPECTED = {
    "vendor_name": "Tailspin Hardware",
    "invoice_number": "TH-5120",
    "invoice_date": "2026-08-03",
    "subtotal": 120.0,
    "tax": 9.6,
    "total": 129.6,
    "currency": "USD",
}

# iter 83: re-pinned for the 11th golden row (inv-011); iter 84: 12th (inv-012, clean).
# iter 89: 13th (inv-013, a caught null total).
EVAL_LINES = (
    "examples          : 13",
    "field accuracy    : 91.2%  (83/91)",
    "exact-match acc   : 38.5%  (5/13)",
    "needs-review rate : 61.5%  (8/13)",
    "catch rate        : 100.0%  (8/8)",
    "silent-wrong      : 0",
    "false alarms      : 0",
)
INV10_FAIL_LINE = "[inv-010] invoice_number: expected 'TH-5120' but got 'PO-88213'  (flagged)"


def _model(**kw) -> InvoiceData:
    return InvoiceData(**kw)


def _doc(*middle: str) -> str:
    return "\n".join(["Acme Corp", *middle, "Total: 10.00", "USD"])


def _git_show(spec: str) -> str | None:
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        return None
    r = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


_HEAD_MOD = None


def _head_heuristics():
    """HEAD's heuristics module, executed as a sibling module in the package."""
    global _HEAD_MOD
    if _HEAD_MOD is not None:
        return _HEAD_MOD
    src = _git_show(f"{HEAD_COMMIT}:src/reflect_mcp/heuristics.py")
    if src is None:
        pytest.skip("HEAD commit unavailable (archive copy)")
    name = "reflect_mcp._iter77_head_heuristics"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "reflect_mcp"
    sys.modules[name] = mod
    try:
        exec(compile(src, f"<{HEAD_COMMIT}:heuristics.py>", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    _HEAD_MOD = mod
    return mod


def _report_tuple(h):
    return (
        {f: h.factor_for(f) for f in EXTRACTION_FIELDS},
        {f: h.reason_for(f) for f in EXTRACTION_FIELDS},
        {f: [(s.start, s.end) for s in spans] for f, spans in sorted(h.provenance.items())},
    )


def _rows() -> list[dict]:
    return [json.loads(ln) for ln in DATASET.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------- EB1
def test_eb1_po_number_as_invoice_number_is_penalized():
    m = _model(**{**EB1_EXPECTED, "invoice_number": "PO-88213"})
    h = run_heuristics(EB1_TEXT, m)
    assert h.factor_for("invoice_number") == 0.3
    reason = h.reason_for("invoice_number")
    assert reason and "PO-88213" in reason


def test_eb1_provenance_span_still_points_at_po_line():
    m = _model(**{**EB1_EXPECTED, "invoice_number": "PO-88213"})
    h = run_heuristics(EB1_TEXT, m)
    spans = h.provenance["invoice_number"]
    assert len(spans) == 1
    assert EB1_TEXT[spans[0].start:spans[0].end] == "PO-88213"
    line_start = EB1_TEXT.rfind("\n", 0, spans[0].start) + 1
    assert EB1_TEXT[line_start:].startswith("PO Number:")


def test_eb1_correct_invoice_number_clean():
    h = run_heuristics(EB1_TEXT, _model(**EB1_EXPECTED))
    for f in EXTRACTION_FIELDS:
        assert h.factor_for(f) == 1.0, (f, h.reason_for(f))


# ---------------------------------------------------------------- EB2
EB2_CASES = [
    ("PO Number: PO-1", "PO-1"),
    ("P.O. 77881", "77881"),
    ("Purchase order: 4410", "4410"),
    ("Order no: ORD-9", "ORD-9"),
    ("Customer ID: C-552", "C-552"),
    ("Client: CL-7", "CL-7"),
    ("Account: AC-31", "AC-31"),
    ("Acct # 9001", "9001"),
]


@pytest.mark.parametrize("line,value", EB2_CASES, ids=[c[0] for c in EB2_CASES])
def test_eb2_every_other_id_label_penalizes(line, value):
    text = _doc("Invoice No: A-100", line)
    h = run_heuristics(text, _model(vendor_name="Acme Corp", invoice_number=value))
    assert h.factor_for("invoice_number") == 0.3, (line, h.reason_for("invoice_number"))
    assert value in (h.reason_for("invoice_number") or "")


@pytest.mark.parametrize("line,value", EB2_CASES, ids=[c[0] for c in EB2_CASES])
def test_eb2_label_case_insensitive_and_leading_whitespace(line, value):
    text = _doc("Invoice No: A-100", "   " + line.lower())
    h = run_heuristics(text, _model(invoice_number=value.lower()))
    assert h.factor_for("invoice_number") == 0.3, (line, h.reason_for("invoice_number"))


@pytest.mark.parametrize("line,value", EB2_CASES, ids=[c[0] for c in EB2_CASES])
def test_eb2_same_label_abstains_without_invoice_label(line, value):
    # Same other-id line, but no invoice-number label anywhere -> abstain (EB3a).
    text = _doc("INVOICE", line)
    h = run_heuristics(text, _model(invoice_number=value))
    assert h.factor_for("invoice_number") == 1.0, (line, h.reason_for("invoice_number"))


# ---------------------------------------------------------------- EB3
def test_eb3a_golden_inv002_order_number_stays_clean():
    row = next(r for r in _rows() if r["id"] == "inv-002")
    assert "Order 5567" in row["text"]
    h = run_heuristics(row["text"], _model(invoice_number="5567"))
    assert h.factor_for("invoice_number") == 1.0
    assert not h.reason_for("invoice_number")


@pytest.mark.parametrize("invoice_line", ["INVOICE", "Invoice date: 2026-07-01"])
def test_eb3a_non_label_invoice_lines_do_not_count(invoice_line):
    text = _doc(invoice_line, "PO Number: PO-1")
    h = run_heuristics(text, _model(invoice_number="PO-1"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


@pytest.mark.parametrize("label", ["Invoice #INV-1001", "Invoice INI-3307", "Invoice Number: Z-1",
                                   "Invoice ID: Q-77", "Invoice No. N-5"])
def test_eb3a_counterpart_invoice_labels_do_count(label):
    # Each label form in the definition switches the check on.
    text = _doc(label, "PO Number: PO-1")
    h = run_heuristics(text, _model(invoice_number="PO-1"))
    assert h.factor_for("invoice_number") == 0.3, (label, h.reason_for("invoice_number"))


def test_eb3b_value_on_invoice_line_and_po_line_abstains():
    text = _doc("Invoice No: X-9", "PO ref X-9")
    h = run_heuristics(text, _model(invoice_number="X-9"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


def test_eb3c_customer_invoice_line_is_not_other_id_literal():
    # Literal spec text: value on "Customer invoice no:" plus an "Invoice #CI-42" line.
    text = _doc("Customer invoice no: CI-42", "Invoice #CI-42")
    h = run_heuristics(text, _model(invoice_number="CI-42"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


def test_eb3c_customer_invoice_only_line_with_other_invoice_label():
    # Strict reading: the value's ONLY line is "Customer invoice no:", while a
    # different invoice-number label exists. Still abstains.
    text = _doc("Customer invoice no: CI-42", "Invoice #INV-77")
    h = run_heuristics(text, _model(invoice_number="CI-42"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


def test_eb3d_orderly_is_not_whole_word_label():
    text = _doc("Invoice No: A-100", "Orderly Supplies Ltd OS-12")
    h = run_heuristics(text, _model(invoice_number="OS-12"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


def test_eb3d_orderly_literal_value():
    text = _doc("Invoice No: A-100", "Orderly Supplies Ltd")
    h = run_heuristics(text, _model(invoice_number="Orderly Supplies Ltd"))
    assert h.factor_for("invoice_number") == 1.0, h.reason_for("invoice_number")


def test_eb3e_null_invoice_number_abstains():
    h = run_heuristics(EB1_TEXT, _model(**{**EB1_EXPECTED, "invoice_number": None}))
    assert h.factor_for("invoice_number") == 1.0
    assert not h.reason_for("invoice_number")


# ---------------------------------------------------------------- EB4
def test_eb4_ungrounded_value_single_not_found_penalty():
    h = run_heuristics(EB1_TEXT, _model(**{**EB1_EXPECTED, "invoice_number": "ZZ-404"}))
    assert h.factor_for("invoice_number") == 0.25
    reason = h.reason_for("invoice_number")
    assert "not found in source text" in reason
    assert "PO" not in reason.replace("ZZ-404", "")


def test_eb4_ungrounded_value_identical_to_head():
    head = _head_heuristics()
    m = _model(**{**EB1_EXPECTED, "invoice_number": "ZZ-404"})
    assert _report_tuple(run_heuristics(EB1_TEXT, m)) == _report_tuple(head.run_heuristics(EB1_TEXT, m))


# ---------------------------------------------------------------- EB5
def _eb5_inputs():
    cases = [(EB1_TEXT, {**EB1_EXPECTED, "invoice_number": "PO-88213"}),
             (EB1_TEXT, EB1_EXPECTED)]
    for line, value in EB2_CASES:
        cases.append((_doc("Invoice No: A-100", line),
                      {"vendor_name": "Acme Corp", "invoice_number": value, "total": 10.0,
                       "currency": "USD"}))
    return cases


def test_eb5_other_fields_unchanged_vs_head():
    head = _head_heuristics()
    for text, data in _eb5_inputs():
        m = _model(**data)
        new_f, new_r, new_p = _report_tuple(run_heuristics(text, m))
        old_f, old_r, old_p = _report_tuple(head.run_heuristics(text, m))
        for f in EXTRACTION_FIELDS:
            if f == "invoice_number":
                continue
            assert new_f[f] == old_f[f], (text, f)
            assert new_r[f] == old_r[f], (text, f)
        assert new_p == old_p, text  # grounding/provenance unchanged


def test_eb5_existing_golden_rows_identical_to_head():
    head = _head_heuristics()
    replay = json.loads(REPLAY.read_text())
    # iter 83: inv-011 postdates this HEAD and is caught by a newer check, so it is excluded too;
    # iter 84: so does inv-012, whose "APRİL" date crashes this HEAD's month reader.
    # iter 89: inv-013's null total is caught by the newer omitted-total check, so it is excluded too.
    rows = [r for r in _rows() if r["id"] not in ("inv-010", "inv-011", "inv-012", "inv-013")]
    assert len(rows) == 9
    for row in rows:
        for data in (row["expected"], json.loads(replay[row["id"]]["generator_raw"])):
            m = _model(**data)
            assert _report_tuple(run_heuristics(row["text"], m)) == \
                _report_tuple(head.run_heuristics(row["text"], m)), row["id"]


def test_eb5_heuristics_only_lower():
    for text, data in _eb5_inputs():
        h = run_heuristics(text, _model(**data))
        assert all(0.0 <= h.factor_for(f) <= 1.0 for f in EXTRACTION_FIELDS)


# ---------------------------------------------------------------- EB6
def test_eb6_inv010_golden_row():
    rows = _rows()
    assert len(rows) == 13  # iter 83: inv-011 appended after inv-010; iter 84: inv-012; iter 89: inv-013
    row = rows[9]
    assert row["id"] == "inv-010"
    assert row["text"] == EB1_TEXT
    assert row["expected"] == EB1_EXPECTED


def test_eb6_inv010_replay_fixture():
    replay = json.loads(REPLAY.read_text())
    gen = json.loads(replay["inv-010"]["generator_raw"])
    assert gen == {**EB1_EXPECTED, "invoice_number": "PO-88213"}
    critic = json.loads(replay["inv-010"]["critic_raw"])
    verdicts = {v["field"]: v for v in critic["field_verdicts"]}
    assert set(verdicts) == set(EXTRACTION_FIELDS)
    for f, v in verdicts.items():
        assert v["grounded"] is True, f
        assert v["confidence"] >= 0.95, f


# ---------------------------------------------------------------- EB7
def _run_eval(cwd: Path, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(cwd / "eval" / "run_eval.py")], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=180,
    )


def test_eb7_live_eval_numbers_and_exit():
    r = _run_eval(REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    out = [ln.strip() for ln in r.stdout.splitlines()]
    for line in EVAL_LINES:
        assert line in out, line
    assert INV10_FAIL_LINE in out


# ---------------------------------------------------------------- EB8
def test_eb8_head_heuristics_make_inv010_silent(tmp_path):
    old = _git_show(f"{HEAD_COMMIT}:src/reflect_mcp/heuristics.py")
    if old is None:
        pytest.skip("HEAD commit unavailable (archive copy)")
    copy = tmp_path / "copy"
    shutil.copytree(
        REPO, copy,
        ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    (copy / "src" / "reflect_mcp" / "heuristics.py").write_text(old)
    # iter 84: this HEAD crashes on inv-012's "APRİL", so run the 11 rows it was measured on.
    dataset = copy / "eval" / "golden" / "dataset.jsonl"
    dataset.write_text("".join(dataset.read_text().splitlines(keepends=True)[:11]))
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env["PYTHONPATH"] = str(copy / "src")
    probe = subprocess.run(
        [sys.executable, "-c", "import reflect_mcp; print(reflect_mcp.__file__)"],
        cwd=copy, env=env, capture_output=True, text=True,
    )
    assert probe.stdout.strip().startswith(str(copy)), probe.stdout + probe.stderr
    r = _run_eval(copy, env)
    assert r.returncode != 0, r.stdout + r.stderr
    out = [ln.strip() for ln in r.stdout.splitlines()]
    inv10 = [ln for ln in out if ln.startswith("[inv-010]")]
    assert inv10 == ["[inv-010] invoice_number: expected 'TH-5120' but got 'PO-88213'  (SILENT)"]
    # iter 83: that HEAD also predates the currency-code shape check, so inv-011 is silent too.
    assert "silent-wrong      : 2" in out
    assert "FAIL: silent-wrong fields 2 > allowed 0" in out


# ---------------------------------------------------------------- EB9
def _cli_validate(tmp_path, data: dict) -> tuple[int, dict]:
    src = tmp_path / "inv.txt"
    src.write_text(EB1_TEXT)
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--file", str(src),
         "--data", json.dumps(data), "--heuristics-only"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    return r.returncode, json.loads(r.stdout)


def test_eb9_cli_validate_po_number_needs_review(tmp_path):
    code, out = _cli_validate(tmp_path, {"invoice_number": "PO-88213"})
    assert out["status"] == "needs_review", out
    flagged = [f["field"] for f in out["flagged_fields"]]
    assert "invoice_number" in flagged
    inv = next(f for f in out["flagged_fields"] if f["field"] == "invoice_number")
    assert inv["confidence"] <= 0.3 and "PO-88213" in inv["reason"]
    assert code == 1


def test_eb9_cli_validate_correct_number_ok(tmp_path):
    code, out = _cli_validate(tmp_path, {"invoice_number": "TH-5120"})
    assert out["status"] == "ok", out
    assert "invoice_number" not in [f["field"] for f in (out.get("flagged_fields") or [])]
    assert code == 0


# ---------------------------------------------------------------- EB10
def test_eb10_readme_names_identifier_role_in_all_sections():
    readme = README.read_text()
    assert readme.count("identifier-role grounding") >= 3
    hs = next(p for p in readme.split("\n\n") if p.startswith("**Heuristics-only mode**"))
    assert "identifier-role grounding" in hs
    h7 = next(ln for ln in readme.splitlines() if ln.startswith("- **Deterministic heuristics**"))
    assert "identifier-role grounding" in h7 and "invoice_number" in h7


def test_eb10_readme_eval_block_byte_equal_and_inv010_sentence():
    readme = README.read_text()
    out = _run_eval(REPO).stdout.strip("\n")
    assert out in readme
    idx = readme.index("examples          : 13")  # iter 83, iter 84, iter 89
    intro = readme[:idx].rsplit("Offline run", 1)[1]
    assert "inv-010" in intro and "identifier-role grounding" in intro
    assert "PO number" in intro


def test_eb10_module_docstring_names_identifier_role():
    import reflect_mcp.heuristics as h

    assert h.__doc__ and "identifier-role" in h.__doc__.lower()


def test_eb10_roadmap_53_done_and_ledger_row():
    text = PRODUCT.read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| 53 |"))
    assert "DONE (iter 77)" in row
    ledger = [ln for ln in text.splitlines() if ln.startswith("- iter 77 — ")]
    assert len(ledger) == 1 and len(ledger[0]) <= 120
