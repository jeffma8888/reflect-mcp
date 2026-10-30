"""Iter 59: `make eval` grades the trust contract, not just accuracy.

Black-box over `run_eval.evaluate` / `run` / `main` / stdout. It reports catch
rate, silent-wrong and false-alarm counts, tags each failure `(flagged)` or
`(SILENT)`, and exits non-zero when silent-wrong > `--max-silent-wrong`.
Mutations live only in tmp copies of the replay file (monkeypatched
`run_eval.REPLAY`); shipped fixtures are never edited. Offline, no secrets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

import run_eval  # noqa: E402

README = REPO / "README.md"
SHIPPED_REPLAY = REPO / "eval" / "fixtures" / "replay.json"
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"

BASE_FAILURES = [
    ("inv-004", "total", 583.2, 683.2),
    ("inv-006", "invoice_date", "2026-03-14", "2026-04-14"),
    ("inv-007", "vendor_name", "NORTHWIND TRADERS", "Contoso Ltd"),
]

# Default offline stdout as produced by the pre-iteration HEAD (captured by
# running HEAD's eval in a detached worktree). Every line here must survive
# byte-identically, except the failure lines gain a tag.
HEAD_STDOUT = (
    "==============================================================\n"
    "reflect-mcp eval  |  mode: OFFLINE (replay fixtures)\n"
    "==============================================================\n"
    "examples          : 7\n"
    "field accuracy    : 93.9%  (46/49)\n"
    "exact-match acc   : 57.1%  (4/7)\n"
    "needs-review rate : 42.9%  (3/7)\n"
    "--------------------------------------------------------------\n"
    "field-level failures (where it breaks):\n"
    "  [inv-004] total: expected 583.2 but got 683.2\n"
    "  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'\n"
    "  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'\n"
    "==============================================================\n"
    "PASS: field accuracy 93.9% >= required 90.0%\n"
)

NEW_LINES = [
    "catch rate        : 100.0%  (3/3)",
    "silent-wrong      : 0",
    "false alarms      : 0",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _replay() -> dict:
    return json.loads(SHIPPED_REPLAY.read_text())


def _golden() -> dict:
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def _install(monkeypatch, tmp_path, data: dict) -> Path:
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(run_eval, "REPLAY", p)
    return p


def _null_total(monkeypatch, tmp_path) -> None:
    data = _replay()
    gen = json.loads(data["inv-001"]["generator_raw"])
    gen["total"] = None
    data["inv-001"]["generator_raw"] = json.dumps(gen)
    _install(monkeypatch, tmp_path, data)


def _low_vendor_confidence(monkeypatch, tmp_path) -> None:
    data = _replay()
    crit = json.loads(data["inv-001"]["critic_raw"])
    for v in crit["field_verdicts"]:
        if v["field"] == "vendor_name":
            v["confidence"] = 0.2
            assert v["grounded"] is True
    data["inv-001"]["critic_raw"] = json.dumps(crit)
    _install(monkeypatch, tmp_path, data)


def _all_correct(monkeypatch, tmp_path) -> None:
    """Make every generator output equal the golden expected values."""
    data = _replay()
    gold = _golden()
    for rid in data:
        data[rid]["generator_raw"] = json.dumps(gold[rid]["expected"])
    _install(monkeypatch, tmp_path, data)


def _stdout_lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


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


def _eval_proc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "eval" / "run_eval.py"), *args],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


# ---------------------------------------------------------------------------
# EB-1: metrics
# ---------------------------------------------------------------------------

def test_eb1_existing_metrics_unchanged():
    m = run_eval.evaluate(live=False)
    assert (m.examples, m.total_fields, m.correct_fields, m.exact_matches, m.reviewed) == (
        7, 49, 46, 4, 3)
    assert sorted(m.failures) == sorted(BASE_FAILURES)
    assert len(m.failures) == 3


def test_eb1_new_metrics_on_shipped_fixtures():
    m = run_eval.evaluate(live=False)
    assert m.caught == 3 and isinstance(m.caught, int)
    assert m.silent_wrong == 0 and isinstance(m.silent_wrong, int)
    assert m.false_alarms == 0 and isinstance(m.false_alarms, int)
    assert m.silent_failures == []
    assert m.catch_rate == 1.0


def test_eb1_invariants_hold():
    m = run_eval.evaluate(live=False)
    assert m.caught + m.silent_wrong == len(m.failures)
    assert m.correct_fields + m.caught + m.silent_wrong == m.total_fields


def test_eb1_catch_rate_none_when_nothing_wrong(monkeypatch, tmp_path):
    _all_correct(monkeypatch, tmp_path)
    m = run_eval.evaluate(live=False)
    assert m.failures == []
    assert m.correct_fields == m.total_fields == 49
    assert m.caught == 0 and m.silent_wrong == 0
    assert m.catch_rate is None


# ---------------------------------------------------------------------------
# EB-2: readout lines
# ---------------------------------------------------------------------------

def test_eb2_new_lines_placed_after_review_rate(capsys):
    assert run_eval.run(live=False, min_accuracy=0.90) == 0
    lines = _stdout_lines(capsys)
    i = lines.index("needs-review rate : 42.9%  (3/7)")
    assert lines[i + 1: i + 4] == NEW_LINES
    assert lines[i + 4] == "-" * 62


def test_eb2_other_lines_byte_identical_to_head():
    proc = _eval_proc()
    assert proc.returncode == 0
    out = proc.stdout
    for line in NEW_LINES:
        out = out.replace(line + "\n", "", 1)
    out = out.replace("  (flagged)\n", "\n")
    out = out.replace("PASS: silent-wrong fields 0 <= allowed 0\n", "", 1)
    assert out == HEAD_STDOUT


def test_eb2_label_padding_matches_existing_style(capsys):
    run_eval.run(live=False, min_accuracy=0.90)
    lines = _stdout_lines(capsys)
    i = lines.index("needs-review rate : 42.9%  (3/7)")
    for line in lines[i - 3: i + 4]:
        assert line[18:20] == ": ", line
    labels = [ln[:18].rstrip() for ln in lines[i + 1: i + 4]]
    assert labels == ["catch rate", "silent-wrong", "false alarms"]


def test_eb2_catch_rate_na_when_nothing_wrong(monkeypatch, tmp_path, capsys):
    _all_correct(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.90) == 0
    lines = _stdout_lines(capsys)
    assert "catch rate        : n/a  (0/0)" in lines
    assert "silent-wrong      : 0" in lines
    assert "PASS: silent-wrong fields 0 <= allowed 0" in lines


# ---------------------------------------------------------------------------
# EB-3: failure tags
# ---------------------------------------------------------------------------

def test_eb3_shipped_failure_lines_all_flagged(capsys):
    run_eval.run(live=False, min_accuracy=0.90)
    lines = _stdout_lines(capsys)
    expected = [
        "  [inv-004] total: expected 583.2 but got 683.2  (flagged)",
        "  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'  (flagged)",
        "  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'  (flagged)",
    ]
    i = lines.index("field-level failures (where it breaks):")
    assert lines[i + 1: i + 4] == expected
    assert not any("(SILENT)" in line for line in lines)


# ---------------------------------------------------------------------------
# EB-4: second verdict line
# ---------------------------------------------------------------------------

def test_eb4_process_exit_and_stdout_tail():
    proc = _eval_proc()
    assert proc.returncode == 0
    assert proc.stdout.endswith(
        "PASS: field accuracy 93.9% >= required 90.0%\n"
        "PASS: silent-wrong fields 0 <= allowed 0\n"
    )
    verdicts = [ln for ln in proc.stdout.splitlines() if "silent-wrong fields" in ln]
    assert verdicts == ["PASS: silent-wrong fields 0 <= allowed 0"]


# ---------------------------------------------------------------------------
# EB-5: silent-wrong mutation fails the gate
# ---------------------------------------------------------------------------

def test_eb5_mutation_metrics(monkeypatch, tmp_path):
    _null_total(monkeypatch, tmp_path)
    m = run_eval.evaluate(live=False)
    assert m.silent_wrong == 1
    assert m.silent_failures == [("inv-001", "total", 108.5, None)]
    assert m.caught == 3
    assert m.correct_fields == 45
    assert m.catch_rate == pytest.approx(0.75)
    assert m.caught + m.silent_wrong == len(m.failures) == 4
    assert m.correct_fields + m.caught + m.silent_wrong == m.total_fields


def test_eb5_mutation_run_returns_1_and_reports(monkeypatch, tmp_path, capsys):
    _null_total(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.90) == 1
    lines = _stdout_lines(capsys)
    for want in (
        "PASS: field accuracy 91.8% >= required 90.0%",
        "FAIL: silent-wrong fields 1 > allowed 0",
        "silent-wrong      : 1",
        "catch rate        : 75.0%  (3/4)",
        "  [inv-001] total: expected 108.5 but got None  (SILENT)",
    ):
        assert want in lines, want
    assert lines[-2:] == [
        "PASS: field accuracy 91.8% >= required 90.0%",
        "FAIL: silent-wrong fields 1 > allowed 0",
    ]


def test_eb5_replay_read_at_call_time(monkeypatch, tmp_path):
    # Same process: base, then mutated, then base again -> no caching.
    assert run_eval.evaluate(live=False).silent_wrong == 0
    _null_total(monkeypatch, tmp_path)
    assert run_eval.evaluate(live=False).silent_wrong == 1
    monkeypatch.setattr(run_eval, "REPLAY", SHIPPED_REPLAY)
    assert run_eval.evaluate(live=False).silent_wrong == 0


def test_eb5_shipped_fixture_untouched_by_mutation(monkeypatch, tmp_path):
    before = SHIPPED_REPLAY.read_bytes()
    _null_total(monkeypatch, tmp_path)
    run_eval.evaluate(live=False)
    assert SHIPPED_REPLAY.read_bytes() == before


# ---------------------------------------------------------------------------
# EB-6: allowance
# ---------------------------------------------------------------------------

def test_eb6_allowance_passes(monkeypatch, tmp_path, capsys):
    _null_total(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.90, max_silent_wrong=1) == 0
    lines = _stdout_lines(capsys)
    assert "PASS: silent-wrong fields 1 <= allowed 1" in lines
    assert not any(ln.startswith("FAIL:") for ln in lines)


def test_eb6_default_is_zero(monkeypatch, tmp_path):
    _null_total(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.90) == 1
    assert run_eval.run(live=False, min_accuracy=0.90, max_silent_wrong=0) == 1


# ---------------------------------------------------------------------------
# EB-7: CLI flag
# ---------------------------------------------------------------------------

def test_eb7_main_forwards_flag(monkeypatch, tmp_path, capsys):
    _null_total(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--max-silent-wrong", "1"])
    assert run_eval.main() == 0
    assert "PASS: silent-wrong fields 1 <= allowed 1" in _stdout_lines(capsys)


def test_eb7_main_default_fails_mutation(monkeypatch, tmp_path, capsys):
    _null_total(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    assert run_eval.main() == 1
    assert "FAIL: silent-wrong fields 1 > allowed 0" in _stdout_lines(capsys)


@pytest.mark.parametrize("bad", ["-1", "abc"])
def test_eb7_bad_values_exit_2(monkeypatch, capsys, bad):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--max-silent-wrong", bad])
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2
    assert "max-silent-wrong" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["-1", "abc"])
def test_eb7_bad_values_exit_2_subprocess(bad):
    proc = _eval_proc("--max-silent-wrong", bad)
    assert proc.returncode == 2
    assert "max-silent-wrong" in proc.stderr


def test_eb7_help_lists_flag():
    proc = _eval_proc("--help")
    assert proc.returncode == 0
    assert "--max-silent-wrong" in proc.stdout


# ---------------------------------------------------------------------------
# EB-8: false alarms are reported, never gate
# ---------------------------------------------------------------------------

def test_eb8_false_alarm_metrics(monkeypatch, tmp_path):
    _low_vendor_confidence(monkeypatch, tmp_path)
    m = run_eval.evaluate(live=False)
    assert m.false_alarms == 1
    assert m.silent_wrong == 0
    assert m.correct_fields == 46
    assert m.caught == 3


def test_eb8_false_alarm_does_not_gate(monkeypatch, tmp_path, capsys):
    _low_vendor_confidence(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.90) == 0
    lines = _stdout_lines(capsys)
    assert "false alarms      : 1" in lines
    assert "PASS: silent-wrong fields 0 <= allowed 0" in lines
    # A correct-but-flagged field is not a failure line.
    assert not any("[inv-001]" in ln for ln in lines)


# ---------------------------------------------------------------------------
# EB-9: accuracy gate preserved
# ---------------------------------------------------------------------------

def test_eb9_accuracy_gate_still_fails(capsys):
    assert run_eval.run(live=False, min_accuracy=0.99) == 1
    lines = _stdout_lines(capsys)
    assert "FAIL: field accuracy 93.9% < required 99.0%" in lines
    assert "PASS: silent-wrong fields 0 <= allowed 0" in lines
    assert lines[-2:] == [
        "FAIL: field accuracy 93.9% < required 99.0%",
        "PASS: silent-wrong fields 0 <= allowed 0",
    ]


def test_eb9_both_failing_prints_both_reasons(monkeypatch, tmp_path, capsys):
    _null_total(monkeypatch, tmp_path)
    assert run_eval.run(live=False, min_accuracy=0.99) == 1
    lines = _stdout_lines(capsys)
    assert "FAIL: field accuracy 91.8% < required 99.0%" in lines
    assert "FAIL: silent-wrong fields 1 > allowed 0" in lines


def test_eb9_cli_min_accuracy_exit_1():
    proc = _eval_proc("--min-accuracy", "0.99")
    assert proc.returncode == 1
    assert "FAIL: field accuracy 93.9% < required 99.0%" in proc.stdout
    assert "PASS: silent-wrong fields 0 <= allowed 0" in proc.stdout


# ---------------------------------------------------------------------------
# EB-10: docs truth
# ---------------------------------------------------------------------------

def test_eb10_readme_block_equals_stdout():
    proc = _eval_proc()
    assert proc.returncode == 0
    block = _readme_eval_block()
    assert block == proc.stdout
    for line in NEW_LINES:
        assert line + "\n" in block
    assert "(flagged)" in block


def test_eb10_readme_prose_names_metrics_and_exit_conditions():
    sec = _section(README.read_text(), "## 8.")
    prose = sec.replace(_readme_eval_block(), "")
    low = prose.lower()
    assert "catch rate" in low
    assert "silent-wrong" in low
    assert "false alarm" in low
    assert "flagged_fields" in prose
    assert "--min-accuracy" in prose and "0.90" in prose
    assert "--max-silent-wrong" in prose
    assert "(default 0)" in prose


def test_eb10_module_docstring_states_both_exit_conditions():
    doc = " ".join((run_eval.__doc__ or "").split())
    assert "--min-accuracy (default 0.90)" in doc
    assert "--max-silent-wrong (default 0)" in doc
    assert "non-zero" in doc.lower()
    for metric in ("catch rate", "silent-wrong", "false-alarm"):
        assert metric in doc.lower(), metric


def test_final_failure_flagged_is_parallel_to_failures(monkeypatch, tmp_path):
    """The readout tag is recorded where it is classified, not re-derived by set membership."""
    _null_total(monkeypatch, tmp_path)
    m = run_eval.evaluate(live=False)
    assert len(m.failure_flagged) == len(m.failures)
    silent = [f for f, flagged in zip(m.failures, m.failure_flagged) if not flagged]
    assert silent == m.silent_failures
    assert sum(m.failure_flagged) == m.caught
