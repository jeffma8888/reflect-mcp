"""Black-box behavioral tests for iter-83: currency-code shape check + golden row inv-011.

Covers Expected Behaviors of pm.md:

  1. Malformed currency values -> factor 0.2, exactly one shape reason, no provenance.
  2. Well-shaped ``[A-Z]{3}`` codes behave exactly as on HEAD (grounded / ungrounded).
  3. ``currency=None`` unaffected (no factor, reason or provenance key).
  4. No other field's factor/reason/provenance changes vs HEAD; factors stay in [0, 1].
  5. ``reflect-mcp validate --heuristics-only`` -> needs_review (exit 1) / ok (exit 0).
  6. inv-011 golden row + replay entry (Critic fooled); rows 1..10 byte-identical.
  7. Live ``make eval`` stdout lines, in order, exit 0.
  8. HEAD heuristics + new dataset -> inv-011 (SILENT), silent-wrong 1, non-zero exit.
  9. inv-001..inv-010 reports (expected + generator_raw) identical to HEAD.
 10. README names "currency-code shape" in all three heuristics lists, eval block
     equals stdout, intro clause for inv-011, module docstring lists the check.

Isolation contract: tests drive the public ``run_heuristics`` API, the CLI, the
eval script, the dataset/fixture files, README and the module docstring. The
pre-change heuristics.py (ship commit of iter 82) is only *executed* as a
counterfactual, never read. No implementation source, engineer/reviewer notes or
diff were read.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp.schema import EXTRACTION_FIELDS, InvoiceData

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
DATASET = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"
RUN_EVAL = EVAL_DIR / "run_eval.py"
README = REPO / "README.md"
# The commit this iteration builds on (iter 82 ship). Pinned so the
# counterfactual stays meaningful after this iteration ships.
BASE_COMMIT = "f3de092"

MALFORMED = ["$", "US Dollars", "usd", "US", "EURO", "EUR1", " USD", "USD "]
WELL_SHAPED = ["USD", "EUR", "GBP", "XYZ"]

INV11_TEXT = (
    "Fabrikam Supplies\nInvoice No: FS-3307\nDate: 2026-09-15\nSubtotal: 80.00\n"
    "Tax: 6.40\nTotal: 86.40\nAmounts in US Dollars"
)
INV11_EXPECTED = {
    "vendor_name": "Fabrikam Supplies",
    "invoice_number": "FS-3307",
    "invoice_date": "2026-09-15",
    "subtotal": 80.0,
    "tax": 6.4,
    "total": 86.4,
    "currency": "USD",
}

# iter 84: live counts re-pinned for the clean 12th golden row (inv-012).
# iter 89: 13th (inv-013, a caught null total).
EVAL_LINES = [
    "examples          : 13",
    "field accuracy    : 91.2%  (83/91)",
    "exact-match acc   : 38.5%  (5/13)",
    "needs-review rate : 61.5%  (8/13)",
    "catch rate        : 100.0%  (8/8)",
    "silent-wrong      : 0",
    "false alarms      : 0",
    "  [inv-011] currency: expected 'USD' but got 'US Dollars'  (flagged)",
    "PASS: field accuracy 91.2% >= required 90.0%",
]


def _h():
    # Resolve at call time: other modules re-import the package.
    return importlib.import_module("reflect_mcp.heuristics")


def _run(text: str, data: dict):
    return _h().run_heuristics(text, InvoiceData(**data))


def _reason_text(v: str) -> str:
    return f"currency {v!r} is not an ISO 4217 code (3 uppercase letters)"


def _texts_for(v: str) -> list[str]:
    return [
        "Acme Corp\nInvoice No: A-1\nTotal: 12.00\n",          # value absent
        f"Acme Corp\nInvoice No: A-1\nTotal: 12.00\n{v}\n",    # value verbatim
        f"Acme Corp\nTotal: $ 12.00 {v} USD US Dollars EURO\n",
    ]


def _git(*args: str) -> subprocess.CompletedProcess | None:
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        return None
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r if r.returncode == 0 else None


def _git_show(spec: str) -> str | None:
    r = _git("show", spec)
    return r.stdout if r is not None else None


_BASE_MOD = None


def _base_heuristics():
    """The pre-change heuristics module, executed as a sibling in the package."""
    global _BASE_MOD
    if _BASE_MOD is not None:
        return _BASE_MOD
    src = _git_show(f"{BASE_COMMIT}:src/reflect_mcp/heuristics.py")
    if src is None:
        pytest.skip("base commit unavailable (archive copy)")
    name = "reflect_mcp._iter83_base_heuristics"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "reflect_mcp"
    sys.modules[name] = mod
    try:
        exec(compile(src, f"<{BASE_COMMIT}:heuristics.py>", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    _BASE_MOD = mod
    return mod


def _report(h, skip: str | None = None):
    fields = [f for f in EXTRACTION_FIELDS if f != skip]
    return (
        {f: h.factor_for(f) for f in fields},
        {f: h.reasons.get(f) for f in fields},
        {f: [(s.start, s.end) for s in h.provenance[f]] if f in h.provenance else None
         for f in fields},
    )


def _full_data(currency):
    return {
        "vendor_name": "Acme Corp", "invoice_number": "A-1", "invoice_date": "2026-01-05",
        "subtotal": 10.0, "tax": 2.0, "total": 12.0, "currency": currency,
    }


def _eb_inputs():
    """Every (text, data) pair used for EB1-EB3, for the HEAD differential."""
    out = []
    for v in MALFORMED + WELL_SHAPED:
        for t in _texts_for(v):
            out.append((t + "Date: 2026-01-05\nSubtotal: 10.00\nTax: 2.00\n", _full_data(v)))
            out.append((t, {"currency": v}))
    out.append(("Acme Corp\nTotal: 12.00\nUSD\n", _full_data(None)))
    out.append(("Acme Corp\nTotal: 12.00\nUSD\n", {}))
    return out


def _rows() -> list[dict]:
    return [json.loads(ln) for ln in DATASET.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------- EB1
@pytest.mark.parametrize("v", MALFORMED, ids=[repr(v) for v in MALFORMED])
def test_eb1_malformed_currency_flagged(v):
    for text in _texts_for(v):
        h = _run(text, {"currency": v})
        assert h.factor_for("currency") == 0.2, (v, text, h.reasons)
        assert h.reasons["currency"] == [_reason_text(v)], (v, text)
        assert h.provenance["currency"] == [], (v, text)


@pytest.mark.parametrize("v", MALFORMED, ids=[repr(v) for v in MALFORMED])
def test_eb1_no_not_found_reason_stacked(v):
    h = _run("nothing relevant here", {"currency": v})
    assert len(h.reasons["currency"]) == 1
    assert "not found in source text" not in h.reasons["currency"][0]
    assert h.factor_for("currency") == 0.2


def test_eb1_dollar_sign_present_still_no_span():
    text = "Acme Corp\nInvoice No: A-1\nTotal: $ 12.00\n"
    h = _run(text, {"currency": "$"})
    assert "$" in text
    assert h.provenance["currency"] == []


# ---------------------------------------------------------------- EB2
def test_eb2_grounded_usd_clean_with_span():
    text = "Acme Corp\nTotal: 12.00\nUSD\n"
    h = _run(text, {"currency": "USD"})
    assert h.factor_for("currency") == 1.0
    assert not h.reasons.get("currency")
    spans = h.provenance["currency"]
    assert spans and text[spans[0].start:spans[0].end] == "USD"


def test_eb2_ungrounded_xyz_keeps_not_found():
    h = _run("Acme Corp\nTotal: 12.00\nUSD\n", {"currency": "XYZ"})
    assert h.factor_for("currency") == 0.25
    assert h.reasons["currency"] == ["value 'XYZ' not found in source text"]
    assert h.provenance["currency"] == []


@pytest.mark.parametrize("v", WELL_SHAPED)
def test_eb2_well_shaped_identical_to_base(v):
    base = _base_heuristics()
    for text in _texts_for(v):
        m = InvoiceData(currency=v)
        new = _h().run_heuristics(text, m)
        old = base.run_heuristics(text, m)
        assert new.factor_for("currency") == old.factor_for("currency"), (v, text)
        assert new.reasons.get("currency") == old.reasons.get("currency"), (v, text)
        assert [(s.start, s.end) for s in new.provenance.get("currency", [])] == \
            [(s.start, s.end) for s in old.provenance.get("currency", [])], (v, text)


def test_eb2_malformed_differs_from_base():
    # Discriminator: the base commit let "$" through as grounded ok.
    base = _base_heuristics()
    text = "Acme Corp\nInvoice No: A-1\nTotal: $ 12.00\n"
    m = InvoiceData(currency="$")
    assert base.run_heuristics(text, m).factor_for("currency") == 1.0
    assert _h().run_heuristics(text, m).factor_for("currency") == 0.2


# ---------------------------------------------------------------- EB3
def test_eb3_none_currency_unaffected():
    for data in ({}, _full_data(None)):
        h = _run("Acme Corp\nTotal: 12.00\nUSD\n", data)
        assert h.factor_for("currency") == 1.0
        assert "currency" not in h.reasons
        assert "currency" not in h.provenance


def test_eb3_none_currency_identical_to_base():
    base = _base_heuristics()
    text = "Acme Corp\nTotal: 12.00\nUSD\n"
    m = InvoiceData(**_full_data(None))
    assert _report(_h().run_heuristics(text, m)) == _report(base.run_heuristics(text, m))


# ---------------------------------------------------------------- EB4
def test_eb4_other_fields_identical_to_base():
    base = _base_heuristics()
    for text, data in _eb_inputs():
        m = InvoiceData(**data)
        new = _report(_h().run_heuristics(text, m), skip="currency")
        old = _report(base.run_heuristics(text, m), skip="currency")
        assert new == old, (text, data)


def test_eb4_factors_within_unit_interval():
    for text, data in _eb_inputs():
        h = _run(text, data)
        for f in EXTRACTION_FIELDS:
            assert 0.0 <= h.factor_for(f) <= 1.0, (f, text, data)


# ---------------------------------------------------------------- EB5
def _cli_validate(tmp_path, text: str, data: dict) -> tuple[int, dict]:
    src = tmp_path / "inv.txt"
    src.write_text(text)
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--file", str(src), "--data", json.dumps(data)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    return r.returncode, json.loads(r.stdout)


def test_eb5_cli_dollar_sign_needs_review(tmp_path):
    code, out = _cli_validate(
        tmp_path, "Acme Corp\nInvoice No: A-1\nTotal: $ 12.00\n",
        {"vendor_name": "Acme Corp", "total": 12.0, "currency": "$"},
    )
    assert out["status"] == "needs_review", out
    assert out["field_confidence"]["currency"] == 0.2
    assert {"field": "currency", "value": "$", "confidence": 0.2,
            "reason": "currency '$' is not an ISO 4217 code (3 uppercase letters)"} \
        in out["flagged_fields"]
    assert out["field_provenance"]["currency"] == []
    assert code == 1


def test_eb5_cli_usd_ok(tmp_path):
    code, out = _cli_validate(
        tmp_path, "Acme Corp\nInvoice No: A-1\nTotal: $ 12.00\nUSD",
        {"vendor_name": "Acme Corp", "total": 12.0, "currency": "USD"},
    )
    assert out["status"] == "ok", out
    assert code == 0


# ---------------------------------------------------------------- EB6
def test_eb6_inv011_golden_row_is_11th_line():
    rows = _rows()
    assert len(rows) == 13  # iter 84: inv-012 appended after inv-011; iter 89: inv-013
    row = rows[10]
    assert row["id"] == "inv-011"
    assert row["text"] == INV11_TEXT
    assert row["expected"] == INV11_EXPECTED


def test_eb6_inv011_replay_entry_critic_fooled():
    replay = json.loads(REPLAY.read_text())
    gen = json.loads(replay["inv-011"]["generator_raw"])
    assert gen == {**INV11_EXPECTED, "currency": "US Dollars"}
    critic = json.loads(replay["inv-011"]["critic_raw"])
    verdicts = {v["field"]: v for v in critic["field_verdicts"]}
    assert set(verdicts) == set(EXTRACTION_FIELDS)
    for f, v in verdicts.items():
        assert v["grounded"] is True, f
        assert v["confidence"] == 0.95, f
    # Same shape as inv-010's critic entry.
    c10 = json.loads(replay["inv-010"]["critic_raw"])
    assert set(critic) == set(c10)


def test_eb6_rows_1_to_10_byte_identical_to_base():
    old_ds = _git_show(f"{BASE_COMMIT}:eval/golden/dataset.jsonl")
    old_rp = _git_show(f"{BASE_COMMIT}:eval/fixtures/replay.json")
    if old_ds is None or old_rp is None:
        pytest.skip("base commit unavailable (archive copy)")
    old_lines = old_ds.splitlines()
    new_lines = DATASET.read_text().splitlines()
    assert len(old_lines) == 10
    assert new_lines[:10] == old_lines
    old_r, new_r = json.loads(old_rp), json.loads(REPLAY.read_text())
    assert set(old_r) | {"inv-011"} <= set(new_r)  # iter 84: later rows (inv-012) may follow
    for k, v in old_r.items():
        assert new_r[k] == v, k


# ---------------------------------------------------------------- EB7
def _run_eval(cwd: Path, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(cwd / "eval" / "run_eval.py")], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=180,
    )


def _in_order(needles: list[str], lines: list[str]) -> bool:
    it = iter(lines)
    return all(any(ln == n for ln in it) for n in needles)


def test_eb7_live_eval_lines_in_order_and_exit_zero():
    r = _run_eval(REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    for n in EVAL_LINES:
        assert n in lines, n
    assert _in_order(EVAL_LINES, lines), r.stdout


def test_eb7_make_eval_exit_zero():
    if shutil.which("make") is None:
        pytest.skip("make unavailable")
    r = subprocess.run(["make", "eval"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS: field accuracy 91.2% >= required 90.0%" in r.stdout  # iter 84: inv-012; iter 89: inv-013


# ---------------------------------------------------------------- EB8
def test_eb8_base_heuristics_leave_inv011_silent_in_process():
    base = _base_heuristics()
    replay = json.loads(REPLAY.read_text())
    m = InvoiceData(**json.loads(replay["inv-011"]["generator_raw"]))
    assert base.run_heuristics(INV11_TEXT, m).factor_for("currency") == 1.0
    assert _h().run_heuristics(INV11_TEXT, m).factor_for("currency") == 0.2


def test_eb8_base_heuristics_eval_is_silent_and_fails(tmp_path):
    old = _git_show(f"{BASE_COMMIT}:src/reflect_mcp/heuristics.py")
    if old is None:
        pytest.skip("base commit unavailable (archive copy)")
    copy = tmp_path / "copy"
    shutil.copytree(
        REPO, copy,
        ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    (copy / "src" / "reflect_mcp" / "heuristics.py").write_text(old)
    # iter 84: base heuristics crash on inv-012's "APRİL", so run the 11 rows measured here.
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
    assert "[inv-011] currency: expected 'USD' but got 'US Dollars'  (SILENT)" in out
    assert "silent-wrong      : 1" in out


# ---------------------------------------------------------------- EB9
def test_eb9_existing_rows_reports_identical_to_base():
    base = _base_heuristics()
    replay = json.loads(REPLAY.read_text())
    # iter 84: inv-012 postdates base, whose month reader crashes on its "APRİL".
    # iter 89: inv-013's null total is caught by the newer omitted-total check, so it is excluded too.
    rows = [r for r in _rows() if r["id"] not in ("inv-011", "inv-012", "inv-013")]
    assert [r["id"] for r in rows] == [f"inv-{i:03d}" for i in range(1, 11)]
    for row in rows:
        for data in (row["expected"], json.loads(replay[row["id"]]["generator_raw"])):
            m = InvoiceData(**data)
            assert _report(_h().run_heuristics(row["text"], m)) == \
                _report(base.run_heuristics(row["text"], m)), row["id"]


# ---------------------------------------------------------------- EB10
def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start:] if nxt == -1 else text[start:nxt]


def _readme_eval_block() -> str:
    sec = _section(README.read_text(), "## 8.")
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
    return next(b for b in blocks if "reflect-mcp eval" in b)


def test_eb10_readme_three_lists_name_currency_code_shape():
    readme = README.read_text()
    harness = next(p for p in readme.split("\n\n") if "evaluation harness" in p
                   and "programmatic pass criterion" in p)
    hs = next(p for p in readme.split("\n\n") if p.startswith("**Heuristics-only mode**"))
    arch = next(ln for ln in readme.splitlines() if ln.startswith("- **Deterministic heuristics**"))
    for name, part in (("harness", harness), ("heuristics-only", hs), ("architecture", arch)):
        assert "currency-code shape" in part, name
        tail = part[part.index("currency-code shape"):][:200]
        assert "3 uppercase letters" in tail and "penalized" in tail, (name, tail)


def test_eb10_readme_eval_block_equals_stdout():
    r = _run_eval(REPO)
    assert r.returncode == 0
    assert _readme_eval_block() == r.stdout


def test_eb10_readme_intro_has_inv011_clause():
    clause = ("in `inv-011` a currency written out as `US Dollars` instead of its ISO code, "
              "which the currency-code shape check catches even though the Critic is fooled")
    assert clause in README.read_text()


def test_eb10_module_docstring_lists_check():
    doc = _h().__doc__ or ""
    assert "currency" in doc.lower()
    assert "iso 4217" in doc.lower() or "uppercase" in doc.lower() or "shape" in doc.lower()
