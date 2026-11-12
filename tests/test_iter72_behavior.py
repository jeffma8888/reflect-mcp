"""Black-box behavioral tests for iter-72: frozen golden subset + golden row inv-009.

Covers Expected Behaviors of pm.md:

  1. dataset.jsonl gains inv-009; rows 1-8 byte-identical (sha pin).
  2. replay.json gains inv-009 (due date copied, Critic fooled); 1-8 unchanged.
  3. Live ``run_eval.py`` / ``make eval`` stdout lines and exit 0 (9 rows).
  4. inv-009 is real coverage: pre-date-role heuristics -> silent-wrong 1, exit 1.
  5. ``tests._golden`` constants; pytest collects nothing from it.
  6. ``frozen_rows`` order, exclusion of inv-009, deleted-row AssertionError.
  7. ``frozen_replay`` restricted to the frozen ids.
  8. ``frozen_eval`` sha / exit / arg pass-through, no committed file touched.
  9. ``use_frozen_golden`` in-process 8-row metrics.
 10. Migrated pin modules import the shared helper.
 12. README eval block == live stdout; intro names five examples and inv-009.

EB13 (growth proof) runs the whole suite on a 10-row copy, so it is measured
by the tester outside the suite (recursion hazard) and reported in tester.md.
EB14 (no src/ diff vs HEAD) is a one-time pre-commit property, measured in
tester.md rather than pinned (it would go red on every later src change).

Isolation contract: tests drive the eval script, the dataset/fixture files, the
README and the public ``tests._golden`` helper. No implementation source,
engineer notes or diff were read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tests._golden as golden
from tests._golden import (
    FROZEN_DATASET_SHA256,
    FROZEN_EVAL_SHA256,
    FROZEN_IDS,
    frozen_eval,
    frozen_replay,
    frozen_rows,
    use_frozen_golden,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
DATASET = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"
RUN_EVAL = EVAL_DIR / "run_eval.py"
README = REPO / "README.md"

HEAD_DATASET_SHA = "ffd8faeb8935e65d15e2edf8f62e6cd310a9d3261497f5ed17b1e8a01c2e6dfc"
HEAD_EVAL_SHA = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
PRE_DATE_ROLE_COMMIT = "1a09a74"
# The commit iter 72 was built on (its 8-row identity is what is frozen).
BASE_COMMIT = "823cdfb"

INV9_TEXT = (
    "Fabrikam Supplies\nInvoice FB-3307\nInvoice date: 2026-07-01\nDue date: 2026-07-31\n"
    "Subtotal: 400.00\nTax: 32.00\nTotal: 432.00\nCurrency: USD"
)
INV9_EXPECTED = {
    "vendor_name": "Fabrikam Supplies",
    "invoice_number": "FB-3307",
    "invoice_date": "2026-07-01",
    "subtotal": 400.0,
    "tax": 32.0,
    "total": 432.0,
    "currency": "USD",
}
# iter 77: live counts re-pinned for the 10th golden row (inv-010); iter 83: 11th (inv-011);
# iter 84: 12th (inv-012, clean).
# iter 89: 13th (inv-013, a caught null total).
LIVE_LINES = (
    "examples          : 13",
    "field accuracy    : 91.2%  (83/91)",
    "exact-match acc   : 38.5%  (5/13)",
    "needs-review rate : 61.5%  (8/13)",
    "catch rate        : 100.0%  (8/8)",
    "silent-wrong      : 0",
    "false alarms      : 0",
    "  [inv-009] invoice_date: expected '2026-07-01' but got '2026-07-31'  (flagged)",
    "PASS: field accuracy 91.2% >= required 90.0%",
)
OLD_FAILURE_LINES = (
    "  [inv-004] total: expected 583.2 but got 683.2  (flagged)",
    "  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'  (flagged)",
    "  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'  (flagged)",
    "  [inv-008] total: expected 75.0 but got 45.0  (flagged)",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _has_git_head() -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=REPO, capture_output=True, text=True
        )
    except FileNotFoundError:
        return False
    return r.returncode == 0 and (REPO / ".git").exists()


def _git_show(spec: str) -> str | None:
    if not _has_git_head():
        return None
    r = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _live_eval(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_EVAL), *args], cwd=REPO, capture_output=True, text=True,
        timeout=180,
    )


# ---------------------------------------------------------------- EB1 dataset
def test_eb1_dataset_has_nine_rows_first_eight_unchanged():
    lines = _lines(DATASET)
    assert len(lines) == 13  # iter 77: inv-010 appended after inv-009; iter 83: inv-011; iter 84: inv-012; iter 89: inv-013
    assert _sha("\n".join(lines[:8]) + "\n") == HEAD_DATASET_SHA


def test_eb1_row_nine_is_inv009_due_date_case():
    row = json.loads(_lines(DATASET)[8])
    assert row["id"] == "inv-009"
    assert row["text"] == INV9_TEXT
    assert row["expected"] == INV9_EXPECTED


# ---------------------------------------------------------------- EB2 replay
def test_eb2_replay_keys_and_inv009_entry():
    replay = json.loads(REPLAY.read_text())
    assert isinstance(replay, dict)
    assert sorted(replay) == [f"inv-{i:03d}" for i in range(1, 14)]  # iter 77: + inv-010; iter 83: + inv-011; iter 84: + inv-012; iter 89: + inv-013
    gen = json.loads(replay["inv-009"]["generator_raw"])
    assert gen == {**INV9_EXPECTED, "invoice_date": "2026-07-31"}
    assert replay["inv-009"]["critic_raw"] == replay["inv-008"]["critic_raw"]


def test_eb2_replay_first_eight_equal_head():
    head = _git_show(f"{BASE_COMMIT}:eval/fixtures/replay.json")
    if head is None:
        pytest.skip("base commit unavailable (archive copy)")
    head_replay = json.loads(head)
    replay = json.loads(REPLAY.read_text())
    for rid in FROZEN_IDS:
        assert replay[rid] == head_replay[rid], rid


def test_eb2_critic_raw_is_fooled_all_high_and_grounded():
    critic = json.loads(json.loads(REPLAY.read_text())["inv-009"]["critic_raw"])
    text = json.dumps(critic)
    assert "0.95" in text
    assert "false" not in text.lower()


# ---------------------------------------------------------------- EB3 live eval
def test_eb3_live_eval_stdout_and_exit():
    r = _live_eval()
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout.splitlines()
    for line in LIVE_LINES + OLD_FAILURE_LINES:
        assert line in out, line


def test_eb3_make_eval_exits_zero_with_nine_examples():
    if shutil.which("make") is None or not (REPO / ".venv" / "bin" / "python").exists():
        pytest.skip("make or .venv unavailable")
    r = subprocess.run(["make", "eval"], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "examples          : 13" in r.stdout.splitlines()  # iter 77, iter 83, iter 84, iter 89
    assert "catch rate        : 100.0%  (8/8)" in r.stdout.splitlines()  # iter 89: inv-013


# ---------------------------------------------------------------- EB4 counterfactual
def test_eb4_inv009_is_silent_wrong_on_pre_date_role_heuristics(tmp_path):
    old = _git_show(f"{PRE_DATE_ROLE_COMMIT}:src/reflect_mcp/heuristics.py")
    if old is None:
        pytest.skip("pre-date-role commit unavailable")
    copy = tmp_path / "copy"
    shutil.copytree(
        REPO, copy,
        ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    (copy / "src" / "reflect_mcp" / "heuristics.py").write_text(old)
    # iter 84: pre-iter-84 heuristics crash on inv-012's "APRİL", so the
    # counterfactual runs on the 11 rows its counts were measured on.
    dataset = copy / "eval" / "golden" / "dataset.jsonl"
    dataset.write_text("".join(dataset.read_text().splitlines(keepends=True)[:11]))
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env["PYTHONPATH"] = str(copy / "src")
    probe = subprocess.run(
        [sys.executable, "-c", "import reflect_mcp; print(reflect_mcp.__file__)"],
        cwd=copy, env=env, capture_output=True, text=True,
    )
    assert probe.stdout.strip().startswith(str(copy)), probe.stdout + probe.stderr
    r = subprocess.run(
        [sys.executable, str(copy / "eval" / "run_eval.py")],
        cwd=copy, env=env, capture_output=True, text=True, timeout=180,
    )
    out = r.stdout.splitlines()
    assert r.returncode == 1, r.stdout + r.stderr
    # iter 77: that tree also predates identifier-role grounding, so inv-010 is
    # silent there too; iter 83: and the currency-code shape check, so inv-011 is
    # silent as well -> 3 silent-wrong, 4/7 caught.
    assert "silent-wrong      : 3" in out
    assert "catch rate        : 57.1%  (4/7)" in out
    inv9 = [ln for ln in out if ln.startswith("  [inv-009]")]
    assert len(inv9) == 1 and inv9[0].endswith("(SILENT)"), inv9
    assert "FAIL: silent-wrong fields 3 > allowed 0" in out


# ---------------------------------------------------------------- EB5 constants
def test_eb5_helper_constants():
    assert FROZEN_IDS == tuple(f"inv-00{i}" for i in range(1, 9))
    assert isinstance(FROZEN_IDS, tuple) and len(FROZEN_IDS) == 8
    assert FROZEN_EVAL_SHA256 == HEAD_EVAL_SHA
    assert FROZEN_DATASET_SHA256 == HEAD_DATASET_SHA


def test_eb5_pytest_collects_no_tests_from_helper():
    assert not [n for n in dir(golden) if n.startswith("test")]
    assert not [n for n in dir(golden) if n.startswith("Test")]
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         str(REPO / "tests" / "_golden.py")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert "::" not in r.stdout, r.stdout


# ---------------------------------------------------------------- EB6 frozen_rows
def test_eb6_frozen_rows_order_and_exclusion():
    rows = frozen_rows()
    assert [r["id"] for r in rows] == list(FROZEN_IDS)
    assert all(isinstance(r, dict) for r in rows)
    assert "inv-009" not in {r["id"] for r in rows}


def test_eb6_frozen_rows_match_original_lines_and_sha():
    by_id = {json.loads(ln)["id"]: ln for ln in _lines(DATASET)}
    raw = [by_id[rid] for rid in FROZEN_IDS]
    assert _sha("\n".join(raw) + "\n") == FROZEN_DATASET_SHA256
    assert frozen_rows() == [json.loads(ln) for ln in raw]


def test_eb6_frozen_rows_explicit_path(tmp_path):
    p = tmp_path / "dataset.jsonl"
    p.write_text(DATASET.read_text())
    assert [r["id"] for r in frozen_rows(p)] == list(FROZEN_IDS)
    assert [r["id"] for r in frozen_rows(str(p))] == list(FROZEN_IDS)


def test_eb6_missing_frozen_row_raises_naming_it(tmp_path):
    kept = [ln for ln in _lines(DATASET) if json.loads(ln)["id"] != "inv-003"]
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(kept) + "\n")
    with pytest.raises(AssertionError) as exc:
        frozen_rows(p)
    assert "inv-003" in str(exc.value)


def test_eb6_edited_frozen_row_changes_parsed_rows(tmp_path):
    lines = _lines(DATASET)
    row = json.loads(lines[1])
    row["text"] = row["text"] + "\nEDITED"
    lines[1] = json.dumps(row)
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(lines) + "\n")
    assert frozen_rows(p) != frozen_rows()


# ---------------------------------------------------------------- EB7 frozen_replay
def test_eb7_frozen_replay_restricted():
    fr = frozen_replay()
    assert list(fr) == list(FROZEN_IDS) or sorted(fr) == sorted(FROZEN_IDS)
    assert len(fr) == 8 and "inv-009" not in fr
    replay = json.loads(REPLAY.read_text())
    assert fr == {rid: replay[rid] for rid in FROZEN_IDS}


# ---------------------------------------------------------------- EB8 frozen_eval
def test_eb8_frozen_eval_reproduces_head_sha_without_touching_files():
    before = (DATASET.read_bytes(), RUN_EVAL.read_bytes(), REPLAY.read_bytes())
    r = frozen_eval()
    after = (DATASET.read_bytes(), RUN_EVAL.read_bytes(), REPLAY.read_bytes())
    assert isinstance(r, subprocess.CompletedProcess)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _sha(r.stdout) == FROZEN_EVAL_SHA256 == HEAD_EVAL_SHA
    assert "examples          : 8" in r.stdout.splitlines()
    assert "[inv-009]" not in r.stdout
    assert before == after


def test_eb8_frozen_eval_differs_from_live():
    assert _sha(_live_eval().stdout) != FROZEN_EVAL_SHA256


def test_eb8_frozen_eval_passes_args_through():
    r = frozen_eval("--min-accuracy", "0.99")
    assert r.returncode == 1, r.stdout + r.stderr


# ---------------------------------------------------------------- EB9 in-process
def test_eb9_use_frozen_golden_metrics(monkeypatch):
    use_frozen_golden(monkeypatch)
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import run_eval

    m = run_eval.evaluate(live=False)
    assert m.examples == 8
    assert (m.correct_fields, m.total_fields) == (52, 56)
    assert m.reviewed == 4
    assert (m.caught, m.silent_wrong, m.false_alarms) == (4, 0, 0)
    assert not any(f[0] == "inv-009" for f in m.failures)


def test_eb9_patch_is_undone_after_test():
    # Runs after test_eb9_use_frozen_golden_metrics in file order: live again.
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import run_eval

    assert run_eval.evaluate(live=False).examples == 13  # iter 77, iter 83, iter 84, iter 89


# ---------------------------------------------------------------- EB10 migration
MIGRATED = (53, 54, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71)


@pytest.mark.parametrize("n", MIGRATED)
def test_eb10_failing_modules_use_shared_helper(n):
    src = (REPO / "tests" / f"test_iter{n}_behavior.py").read_text()
    assert "tests._golden" in src or "from ._golden" in src, n


# ---------------------------------------------------------------- EB12 README
def test_eb12_readme_contains_live_stdout_block():
    readme = README.read_text()
    out = _live_eval().stdout
    body = out.strip("\n")
    assert body in readme
    for line in LIVE_LINES:
        assert line in readme.splitlines(), line


def test_eb12_readme_intro_sentence_five_and_inv009():
    readme = README.read_text()
    idx = readme.index("examples          : 13")  # iter 77, iter 83, iter 84, iter 89
    intro = readme[:idx].rsplit("Offline run", 1)
    assert len(intro) == 2, "intro sentence above eval block not found"
    sentence = intro[1]
    assert "eight examples" in sentence  # iter 77: inv-010 is the sixth; iter 83: inv-011 the seventh; iter 89: inv-013 the eighth
    assert "inv-009" in sentence
    assert "due date" in sentence and "invoice date" in sentence
    assert "date-role" in sentence
