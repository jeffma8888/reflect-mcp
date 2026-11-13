"""Black-box behavioral tests for iter-84: non-ASCII month tokens + provenance spans.

Covers Expected Behaviors of pm.md:

  1. CLI ``validate --heuristics-only`` on fixture T: invoice_number span is 34..45.
  2. CLI with the full T extraction: exit 0, ok, all confidences 1.0, date span text.
  3. ``run_heuristics`` grounds case-folding-variant month tokens and never raises.
  4. A wrong date on a non-ASCII month line is still penalized (0.3, same as ASCII).
  5. Span property for text-grounded fields (case-insensitive match of the value).
  6. Seeded offline property fuzz (>=5,000 cases, <5 s); discriminates vs base.
  7. All-ASCII behavior unchanged vs pinned base 40b82c2 (golden rows + ASCII fuzz).
  8. Golden row inv-012 is the 12th line, lines 1..11 byte-identical, replay correct.
  9. ``make eval`` 91.7% (77/84), silent-wrong 0; base heuristics crash on it.
 10. README eval block equals ``make eval`` stdout.

EB5 ambiguity: the spec states the property as ``span.casefold() == value.casefold()``
but its own example (value ``istanbul ihracat a.ş.`` -> span ``İSTANBUL İHRACAT A.Ş.``)
violates it, because ``'İ'.casefold()`` is ``'i̇'`` (two code points). These tests use the
reading that is consistent with the example: the span text matches the value
case-insensitively, ``re.fullmatch(re.escape(value), span, re.IGNORECASE)``.

Isolation contract: tests drive the public ``run_heuristics`` API, the CLI, the eval
script, the dataset/fixture files and README. The pre-change heuristics.py (ship commit
of iter 83, 40b82c2) is only *executed* as a counterfactual, never read. No
implementation source, engineer/reviewer notes or diff were read.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reflect_mcp.schema import EXTRACTION_FIELDS, InvoiceData

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
DATASET = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"
RUN_EVAL = EVAL_DIR / "run_eval.py"
README = REPO / "README.md"
# The commit this iteration builds on (iter 83 ship). Pinned, never HEAD.
BASE_COMMIT = "40b82c2"

T = "İSTANBUL İHRACAT A.Ş.\nInvoice No: TR-2026-114\nDate: APRİL 3, 2026\nTotal: 236.00\n"
T_FULL = {
    "vendor_name": "İSTANBUL İHRACAT A.Ş.",
    "invoice_number": "TR-2026-114",
    "invoice_date": "2026-04-03",
    "total": 236.0,
}
TEXT_FIELDS = ("vendor_name", "invoice_number", "currency")

EVAL_LINES = [  # iter 89: re-pinned for the 13th golden row (inv-013, a caught null total)
    "examples          : 13",
    "field accuracy    : 91.2%  (83/91)",
    "silent-wrong      : 0",
    "PASS: field accuracy 91.2% >= required 90.0%",
]


def _h():
    # Resolve at call time: other modules re-import the package.
    return importlib.import_module("reflect_mcp.heuristics")


def _run(text: str, data: dict, mod=None):
    return (mod or _h()).run_heuristics(text, InvoiceData(**data))


def _git_show(spec: str) -> str | None:
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        return None
    r = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


_BASE_MOD = None


def _base_heuristics():
    """The pre-change heuristics module, executed as a sibling in the package."""
    global _BASE_MOD
    if _BASE_MOD is not None:
        return _BASE_MOD
    src = _git_show(f"{BASE_COMMIT}:src/reflect_mcp/heuristics.py")
    if src is None:
        pytest.skip("base commit unavailable (archive copy)")
    name = "reflect_mcp._iter84_base_heuristics"
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


def _spans(h, field: str) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in h.provenance.get(field, [])]


def _report(h):
    return (
        {f: h.factor_for(f) for f in EXTRACTION_FIELDS},
        {f: h.reasons.get(f) for f in EXTRACTION_FIELDS},
        {f: _spans(h, f) if f in h.provenance else None for f in EXTRACTION_FIELDS},
    )


def _cli_validate(tmp_path, text: str, data: dict) -> tuple[int, dict]:
    src = tmp_path / "inv.txt"
    src.write_bytes(text.encode("utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--file", str(src), "--data", json.dumps(data)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
        encoding="utf-8",
    )
    return r.returncode, json.loads(r.stdout)


def _span_ok(text: str, start: int, end: int, value: str) -> bool:
    return 0 <= start <= end <= len(text) and \
        re.fullmatch(re.escape(value), text[start:end], re.IGNORECASE) is not None


# ---------------------------------------------------------------- EB1
def test_eb1_fixture_bytes_put_invoice_number_at_34():
    assert T.index("TR-2026-114") == 34
    assert T[34:45] == "TR-2026-114"


def test_eb1_cli_invoice_number_span_is_34_45(tmp_path):
    code, out = _cli_validate(tmp_path, T, {"invoice_number": "TR-2026-114", "total": 236.0})
    assert code == 0, out
    assert out["status"] == "ok", out
    assert out["field_provenance"]["invoice_number"] == [{"start": 34, "end": 45}]
    s = out["field_provenance"]["invoice_number"][0]
    assert T[s["start"]:s["end"]] == "TR-2026-114"


def test_eb1_base_reports_shifted_span():
    base = _base_heuristics()
    h = _run(T, {"invoice_number": "TR-2026-114", "total": 236.0}, base)
    assert _spans(h, "invoice_number") == [(36, 47)]
    assert _spans(_run(T, {"invoice_number": "TR-2026-114", "total": 236.0}),
                  "invoice_number") == [(34, 45)]


# ---------------------------------------------------------------- EB2
def test_eb2_cli_full_extraction_ok_with_date_span(tmp_path):
    code, out = _cli_validate(tmp_path, T, T_FULL)
    assert code == 0, out
    assert out["status"] == "ok", out
    conf = out["field_confidence"]
    for f in T_FULL:
        assert conf[f] == 1.0, (f, conf)
    assert all(v == 1.0 for v in conf.values()), conf
    assert out["flagged_fields"] == []
    spans = out["field_provenance"]["invoice_date"]
    assert spans, out["field_provenance"]
    assert [T[s["start"]:s["end"]] for s in spans] == ["APRİL 3, 2026"]
    for f in ("vendor_name", "invoice_number"):
        assert [T[s["start"]:s["end"]] for s in out["field_provenance"][f]] == [T_FULL[f]], f


def test_eb2_base_crashes_on_full_extraction():
    base = _base_heuristics()
    with pytest.raises(KeyError):
        _run(T, T_FULL, base)


# ---------------------------------------------------------------- EB3
MONTH_CASES = [
    ("Date: ſep 1, 2026", "2026-09-01"),
    ("Date: 3 APRİL 2026", "2026-04-03"),
    ("Date: APRİL 3, 2026", "2026-04-03"),
]


@pytest.mark.parametrize(("text", "iso"), MONTH_CASES)
def test_eb3_variant_month_token_grounds(text, iso):
    h = _run(text, {"invoice_date": iso})
    assert "invoice_date" not in h.factors, (h.factors, h.reasons)
    assert h.factor_for("invoice_date") == 1.0
    assert h.reasons.get("invoice_date") is None
    spans = _spans(h, "invoice_date")
    assert spans, h.provenance
    for s, e in spans:
        assert text[s:e] == text[len("Date: "):], (s, e)


@pytest.mark.parametrize(("text", "iso"), MONTH_CASES)
def test_eb3_variant_month_token_never_raises_any_date(text, iso):
    for d in ("2026-01-01", "2026-04-03", "2026-09-01", "2025-12-31", iso):
        h = _run(text, {"invoice_date": d})
        assert 0.0 <= h.factor_for("invoice_date") <= 1.0


@pytest.mark.parametrize(("text", "iso"), MONTH_CASES)
def test_eb3_base_raises_on_variant_month(text, iso):
    base = _base_heuristics()
    with pytest.raises(KeyError):
        _run(text, {"invoice_date": iso}, base)


# ---------------------------------------------------------------- EB4
def test_eb4_wrong_date_on_variant_month_line_penalized_like_ascii():
    bad = _run("Date: APRİL 3, 2026", {"invoice_date": "2026-05-03"})
    ascii_bad = _run("Date: APRIL 3, 2026", {"invoice_date": "2026-05-03"})
    assert bad.factor_for("invoice_date") == pytest.approx(0.3)
    assert ascii_bad.factor_for("invoice_date") == pytest.approx(0.3)
    assert bad.reasons.get("invoice_date")
    assert _spans(bad, "invoice_date") == []


@pytest.mark.parametrize("text", ["Date: ſep 1, 2026", "Date: 3 APRİL 2026"])
def test_eb4_other_variant_tokens_penalize_wrong_date(text):
    h = _run(text, {"invoice_date": "2026-11-11"})
    assert h.factor_for("invoice_date") == pytest.approx(0.3)


# ---------------------------------------------------------------- EB5
def test_eb5_case_variant_vendor_grounds_on_original_span():
    value = "istanbul ihracat a.ş."
    h = _run(T, {"vendor_name": value})
    assert "vendor_name" not in h.factors, (h.factors, h.reasons)
    spans = _spans(h, "vendor_name")
    assert [T[s:e] for s, e in spans] == ["İSTANBUL İHRACAT A.Ş."]
    for s, e in spans:
        assert _span_ok(T, s, e, value)


def test_eb5_every_text_span_matches_value_on_fixture():
    data = {**T_FULL, "currency": "EUR"}
    text = T + "Currency: EUR\n"
    h = _run(text, data)
    for f in TEXT_FIELDS:
        for s, e in _spans(h, f):
            assert _span_ok(text, s, e, str(data[f])), (f, text[s:e])
            assert text[s:e].casefold() == str(data[f]).casefold(), (f, text[s:e])


# ---------------------------------------------------------------- EB6
POOL = [
    "ſ", "İ", "ı", "K", "ß", "ﬁ", "\u00a0", "Σ", "σ", "ς", "Ş", "ş", "é",
    "Jan", "January", "Feb", "Mar", "March", "Apr", "APRİL", "april", "May", "Jun",
    "July", "Aug", "ſep", "Sept", "September", "Oct", "Nov", "NOVEMBER", "Dec",
    "1", "3", "14", "2026", "2025", "12", "31", "-", ",", ".", ":", " ", " ", "\n",
    "Date:", "Invoice No:", "Total:", "Bill to:", "PO:", "Acme", "EUR", "USD", "ACME",
]
FUZZ_CASES = 6000
VARIANT_MONTHS = ["ſep", "ſept", "APRİL", "apr", "MARCH", "ſeptember", "JULY", "Aug", "DEC"]
FUZZ_SEED = 84


def _fuzz_cases(seed: int = FUZZ_SEED, n: int = FUZZ_CASES):
    rng = random.Random(seed)
    for _ in range(n):
        toks = [rng.choice(POOL) for _ in range(rng.randint(3, 18))]
        if rng.random() < 0.3:  # a well-formed month-name date built from variant tokens
            mon = rng.choice(VARIANT_MONTHS)
            d, y = rng.randint(1, 28), rng.choice([2025, 2026])
            toks.insert(rng.randrange(len(toks) + 1),
                        rng.choice([f" {mon} {d}, {y} ", f" {d} {mon} {y} "]))
        text = "".join(t if rng.random() < 0.5 else t + " " for t in toks)
        data: dict = {}
        for f in TEXT_FIELDS:
            if rng.random() < 0.7:
                i = rng.randrange(len(text))
                j = rng.randint(i + 1, min(len(text), i + 14))
                v = text[i:j]
                if v.strip():
                    data[f] = v
        if rng.random() < 0.7:
            data["invoice_date"] = (
                f"{rng.choice([2025, 2026])}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            )
        yield text, data


def _check_case(mod, text: str, data: dict) -> str | None:
    """Return a failure description, or None if the case satisfies EB6."""
    try:
        h = mod.run_heuristics(text, InvoiceData(**data))
    except Exception as exc:  # noqa: BLE001 - the property is "never raises"
        return f"raised {type(exc).__name__}: {exc!r}"
    for f in TEXT_FIELDS:
        if f not in data:
            continue
        v = str(data[f])
        for s in h.provenance.get(f, []):
            if not _span_ok(text, s.start, s.end, v):
                return f"{f} span {text[s.start:s.end]!r} != {v!r}"
            first = text.find(v)
            if first != -1 and s.start > first:
                return f"{f} span starts at {s.start} after verbatim match at {first}"
    return None


def _fuzz_failures(mod) -> list[tuple[str, dict, str]]:
    out = []
    for text, data in _fuzz_cases():
        msg = _check_case(mod, text, data)
        if msg is not None:
            out.append((text, data, msg))
    return out


def test_eb6_fuzz_generator_is_seeded_and_large_enough():
    a = list(_fuzz_cases())
    assert len(a) >= 5000
    assert a == list(_fuzz_cases())
    assert any("İ" in t for t, _ in a) and any("ſ" in t for t, _ in a)
    assert sum(1 for _, d in a if set(d) & set(TEXT_FIELDS)) > 3000


def test_eb6_fuzz_never_raises_and_spans_match_within_5s():
    t0 = time.perf_counter()
    failures = _fuzz_failures(_h())
    elapsed = time.perf_counter() - t0
    assert failures == [], failures[:5]
    assert elapsed < 5.0, elapsed


def test_eb6_fuzz_fails_on_base():
    failures = _fuzz_failures(_base_heuristics())
    kinds = {msg.split(" ")[0] for _, _, msg in failures}
    assert failures, "fuzz does not discriminate against base"
    # Base exhibits the crash class (KeyError) and/or the span shift.
    assert any(m.startswith("raised KeyError") for _, _, m in failures), kinds
    assert any(" span " in m for _, _, m in failures), kinds


# ---------------------------------------------------------------- EB7
def _rows() -> list[dict]:
    return [json.loads(ln) for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_eb7_existing_golden_rows_identical_to_base():
    base = _base_heuristics()
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    rows = _rows()[:11]
    assert [r["id"] for r in rows] == [f"inv-{i:03d}" for i in range(1, 12)]
    for row in rows:
        for data in (row["expected"], json.loads(replay[row["id"]]["generator_raw"])):
            m = InvoiceData(**data)
            got = _report(_h().run_heuristics(row["text"], m))
            want = _report(base.run_heuristics(row["text"], m))
            if row["id"] == "inv-011" and data is row["expected"]:
                # iter 95 (#71): "Amounts in US Dollars" now grounds the correct USD,
                # so only this answer's currency may move; every other field stays pinned.
                got, want = (tuple({k: v for k, v in d.items() if k != "currency"} for d in r)
                             for r in (got, want))
            assert got == want, row["id"]


def test_eb7_ascii_fuzz_identical_to_base():
    base = _base_heuristics()
    rng = random.Random(7)
    ascii_pool = [t for t in POOL if t.isascii()]
    for _ in range(1500):
        text = " ".join(rng.choice(ascii_pool) for _ in range(rng.randint(3, 16)))
        i = rng.randrange(len(text))
        v = text[i:rng.randint(i + 1, min(len(text), i + 12))]
        data = {"invoice_date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"}
        if v.strip():
            data[rng.choice(TEXT_FIELDS)] = rng.choice([v, v.upper(), v.lower()])
        m = InvoiceData(**data)
        assert _report(_h().run_heuristics(text, m)) == _report(base.run_heuristics(text, m)), \
            (text, data)


# ---------------------------------------------------------------- EB8
def test_eb8_inv012_is_12th_line_with_t_style_text():
    rows = _rows()
    assert len(rows) == 13  # iter 89: inv-013 appended after inv-012
    row = rows[11]
    assert row["id"] == "inv-012"
    text = row["text"]
    assert text.startswith("İSTANBUL İHRACAT A.Ş.\nInvoice No: TR-2026-114\nDate: APRİL 3, 2026\n")
    for line in ("Subtotal: 200.00", "Tax: 36.00", "Total: 236.00", "Currency: EUR"):
        assert line in text.splitlines(), line
    assert row["expected"] == {
        "vendor_name": "İSTANBUL İHRACAT A.Ş.", "invoice_number": "TR-2026-114",
        "invoice_date": "2026-04-03", "subtotal": 200.0, "tax": 36.0, "total": 236.0,
        "currency": "EUR",
    }


def test_eb8_inv012_replay_entry_all_correct():
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    row = _rows()[11]
    gen = json.loads(replay["inv-012"]["generator_raw"])
    assert gen == row["expected"]
    critic = json.loads(replay["inv-012"]["critic_raw"])
    verdicts = {v["field"]: v for v in critic["field_verdicts"]}
    assert set(verdicts) == set(EXTRACTION_FIELDS)
    assert all(v["grounded"] is True for v in verdicts.values())


def test_eb8_lines_1_to_11_and_replay_byte_identical_to_base():
    old_ds = _git_show(f"{BASE_COMMIT}:eval/golden/dataset.jsonl")
    old_rp = _git_show(f"{BASE_COMMIT}:eval/fixtures/replay.json")
    if old_ds is None or old_rp is None:
        pytest.skip("base commit unavailable (archive copy)")
    old_lines = old_ds.splitlines()
    new_lines = DATASET.read_text(encoding="utf-8").splitlines()
    assert len(old_lines) == 11
    assert len(new_lines) == 13  # iter 89: inv-013
    assert new_lines[:11] == old_lines
    old_r, new_r = json.loads(old_rp), json.loads(REPLAY.read_text(encoding="utf-8"))
    assert set(new_r) == set(old_r) | {"inv-012", "inv-013"}  # iter 89: inv-013
    for k, v in old_r.items():
        assert new_r[k] == v, k


# ---------------------------------------------------------------- EB9
def _run_eval(cwd: Path, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(cwd / "eval" / "run_eval.py")], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=180,
    )


def _in_order(needles: list[str], lines: list[str]) -> bool:
    it = iter(lines)
    return all(any(ln == n for ln in it) for n in needles)


def test_eb9_make_eval_exit_zero_917():
    if shutil.which("make") is None:
        pytest.skip("make unavailable")
    r = subprocess.run(["make", "eval"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert _in_order(EVAL_LINES, lines), r.stdout
    assert not any("[inv-012]" in ln for ln in lines), r.stdout


def test_eb9_base_heuristics_crash_on_new_dataset(tmp_path):
    old = _git_show(f"{BASE_COMMIT}:src/reflect_mcp/heuristics.py")
    if old is None:
        pytest.skip("base commit unavailable (archive copy)")
    copy = tmp_path / "copy"
    shutil.copytree(
        REPO, copy,
        ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    (copy / "src" / "reflect_mcp" / "heuristics.py").write_text(old)
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env["PYTHONPATH"] = str(copy / "src")
    probe = subprocess.run(
        [sys.executable, "-c", "import reflect_mcp; print(reflect_mcp.__file__)"],
        cwd=copy, env=env, capture_output=True, text=True,
    )
    assert probe.stdout.strip().startswith(str(copy)), probe.stdout + probe.stderr
    r = _run_eval(copy, env)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "KeyError" in r.stderr, r.stderr[-2000:]
    # Same tree with the working-tree heuristics passes: the row is the discriminator.
    (copy / "src" / "reflect_mcp" / "heuristics.py").write_text(
        (REPO / "src" / "reflect_mcp" / "heuristics.py").read_text())
    r2 = _run_eval(copy, env)
    assert r2.returncode == 0, r2.stdout + r2.stderr


# ---------------------------------------------------------------- EB10
def _readme_eval_block() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## 8.")
    nxt = text.find("\n## ", start + 5)
    sec = text[start:] if nxt == -1 else text[start:nxt]
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


def test_eb10_readme_eval_block_equals_stdout():
    r = _run_eval(REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _readme_eval_block() == r.stdout
    assert "field accuracy    : 91.2%  (83/91)" in _readme_eval_block()  # iter 89: inv-013
