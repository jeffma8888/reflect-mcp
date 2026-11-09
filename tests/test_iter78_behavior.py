"""Black-box behavioral tests for iter-78: when stdout's reader closes early
(e.g. ``| head -1``), ``reflect-mcp validate`` stops reading input and exits
with its documented 0/1/2/3 code -- never 120, never a traceback.

Covers the 11 Expected Behaviors from pm.md:
  1. --batch --file, 2000 ok rows, closed after one line -> exit 0, clean stderr.
  2. Same with 2000 needs_review rows -> exit 1, clean stderr.
  3. EB1/EB2 also hold with --batch-concurrency 4.
  4. EB2 with rows on stdin and --batch-concurrency 4 -> exit 1, no Traceback.
  5. The one line read is a complete JSON object with the expected status.
  6. In-process: run_validate returns 1 (not raises) on EPIPE; stdin pulls
     <= 2 serial, <= 2N+1 = 9 at N=4.
  7. Same with ok rows returns 0 (serial and N=4).
  8. Only EPIPE is swallowed: OSError(28) from stdout propagates.
  9. Single document with stdout closed before the write -> exit 0, no Traceback.
 10. Normal (fully read) runs unchanged: 3 rows -> 3 lines, exit 0 / 1.
 11. README exit-code / batch paragraph documents the early-close rule.

Isolation contract: tests run the product as a subprocess
(``python -m reflect_mcp``) or call the public ``cli.run_validate`` entry
point with stubbed stdin/stdout, and assert observable output only. No
implementation source was read.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

OK_ROW = {"text": "Invoice No: INV-1\n", "data": {"invoice_number": "INV-1"}}
NR_ROW = {
    "text": "Invoice No: INV-1\nTotal: 10.00 USD",
    "data": {"invoice_number": "INV-1", "total": 99.0},
}
ROWS = {"ok": OK_ROW, "needs_review": NR_ROW}
EXPECTED_RC = {"ok": 0, "needs_review": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env() -> dict:
    """Subprocess env with the threshold-floor vars removed (they change status)."""
    env = dict(os.environ)
    for k in ("REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD"):
        env.pop(k, None)
    return env


def _write_rows(path: Path, kind: str, n: int) -> Path:
    line = json.dumps(ROWS[kind])
    path.write_text("".join(line + "\n" for _ in range(n)), encoding="utf-8")
    return path


def _run_closed_early(tmp_path: Path, kind: str, concurrency: int | None, via_stdin: bool):
    """Run validate --batch on 2000 rows, read one line, close stdout, wait."""
    rows = _write_rows(tmp_path / f"{kind}.jsonl", kind, 2000)
    cmd = [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only"]
    if concurrency is not None:
        cmd += ["--batch-concurrency", str(concurrency)]
    if not via_stdin:
        cmd += ["--file", str(rows)]
    err_path = tmp_path / "stderr.txt"
    with open(err_path, "wb") as err, open(rows, "rb") as stdin_fh:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin_fh if via_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err,
            cwd=str(REPO_ROOT),
            env=_env(),
        )
        first = proc.stdout.readline()
        proc.stdout.close()
        rc = proc.wait(timeout=60)
    return rc, first.decode("utf-8"), err_path.read_text(encoding="utf-8", errors="replace")


class _CountingStdin:
    """Iterator over N identical lines that counts how many it handed out."""

    def __init__(self, kind: str, n: int):
        self._line = json.dumps(ROWS[kind]) + "\n"
        self._remaining = n
        self.pulled = 0

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def _next_line(self) -> str:
        if self._remaining <= 0:
            return ""
        self._remaining -= 1
        self.pulled += 1
        return self._line

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self._next_line()
        if not line:
            raise StopIteration
        return line

    def readline(self, *a) -> str:
        return self._next_line()

    def read(self, *a) -> str:
        out = []
        while True:
            line = self._next_line()
            if not line:
                break
            out.append(line)
        return "".join(out)


class _BreakingStdout:
    """Accepts writes until one newline has been written, then raises ``exc``."""

    def __init__(self, exc: BaseException):
        self._exc = exc
        self.buf: list[str] = []
        self.newlines = 0

    def write(self, s: str) -> int:
        if self.newlines >= 1:
            raise self._exc
        self.buf.append(s)
        self.newlines += s.count("\n")
        return len(s)

    def flush(self) -> None:
        pass  # only write() breaks; the CLI must handle EPIPE at write time

    def isatty(self) -> bool:
        return False


def _cli():
    # Bind the sys.modules entry (other modules re-import reflect_mcp.* in-suite).
    return importlib.import_module("reflect_mcp.cli")


def _run_in_process(monkeypatch, kind: str, concurrency: int | None, exc: BaseException):
    for k in ("REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)
    cli = _cli()
    stdin = _CountingStdin(kind, 1000)
    stdout = _BreakingStdout(exc)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    argv = ["--batch", "--heuristics-only"]
    if concurrency is not None:
        argv += ["--batch-concurrency", str(concurrency)]
    try:
        rc = cli.run_validate(argv)
    finally:
        monkeypatch.undo()
    return rc, stdin, stdout


# ---------------------------------------------------------------------------
# EB1-EB5: subprocess, closed early after one line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("concurrency", [None, 4], ids=["serial", "N4"])
@pytest.mark.parametrize("kind", ["ok", "needs_review"])
def test_eb1_eb2_eb3_file_closed_early_keeps_exit_contract(tmp_path, kind, concurrency):
    rc, first, err = _run_closed_early(tmp_path, kind, concurrency, via_stdin=False)
    assert rc == EXPECTED_RC[kind], (rc, err[-2000:])
    assert rc != 120
    assert "Traceback" not in err, err[-2000:]
    assert "BrokenPipeError" not in err, err[-2000:]
    # EB5: the line read is a complete JSON object with the expected status.
    assert first.endswith("\n")
    obj = json.loads(first)
    assert isinstance(obj, dict)
    assert obj["status"] == kind


def test_eb4_stdin_closed_early_needs_review_n4(tmp_path):
    rc, first, err = _run_closed_early(tmp_path, "needs_review", 4, via_stdin=True)
    assert rc == 1, (rc, err[-2000:])
    assert "Traceback" not in err, err[-2000:]
    assert "BrokenPipeError" not in err, err[-2000:]
    obj = json.loads(first)
    assert obj["status"] == "needs_review"


def test_eb5_first_line_identical_to_normal_run(tmp_path):
    """Output before the close is unchanged: line 1 equals a full run's line 1."""
    rows = _write_rows(tmp_path / "nr3.jsonl", "needs_review", 3)
    full = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only",
         "--file", str(rows)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=_env(), timeout=60,
    )
    _, first, _ = _run_closed_early(tmp_path, "needs_review", None, via_stdin=False)
    a, b = json.loads(full.stdout.splitlines()[0]), json.loads(first)
    for d in (a, b):
        d.pop("trace_id", None)
    assert a == b


# ---------------------------------------------------------------------------
# EB6-EB8: in-process
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,concurrency,max_pulls",
    [("needs_review", None, 2), ("needs_review", 4, 9), ("ok", None, 2), ("ok", 4, 9)],
    ids=["nr-serial", "nr-N4", "ok-serial", "ok-N4"],
)
def test_eb6_eb7_in_process_epipe_returns_code_and_stops_pulling(
    monkeypatch, kind, concurrency, max_pulls
):
    rc, stdin, stdout = _run_in_process(
        monkeypatch, kind, concurrency, BrokenPipeError(32, "Broken pipe")
    )
    assert rc == EXPECTED_RC[kind]
    assert 1 <= stdin.pulled <= max_pulls, stdin.pulled
    # The one row that made it out is a complete JSON line with the right status.
    written = "".join(stdout.buf)
    assert json.loads(written.splitlines()[0])["status"] == kind


@pytest.mark.parametrize("concurrency", [None, 4], ids=["serial", "N4"])
def test_eb8_non_epipe_oserror_propagates(monkeypatch, concurrency):
    with pytest.raises(OSError) as ei:
        _run_in_process(
            monkeypatch, "ok", concurrency, OSError(28, "No space left on device")
        )
    assert not isinstance(ei.value, BrokenPipeError)
    assert ei.value.errno == 28


# ---------------------------------------------------------------------------
# EB9: single document, stdout closed before the write
# ---------------------------------------------------------------------------

def test_eb9_single_document_stdout_closed_before_write(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Invoice No: INV-1\n", encoding="utf-8")
    err_path = tmp_path / "stderr.txt"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
             "--data", '{"invoice_number": "INV-1"}', "--file", str(doc)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
            cwd=str(REPO_ROOT), env=_env(),
        )
        proc.stdout.close()
        rc = proc.wait(timeout=60)
    text = err_path.read_text(encoding="utf-8", errors="replace")
    assert rc == 0, (rc, text[-2000:])
    assert "Traceback" not in text, text[-2000:]


# ---------------------------------------------------------------------------
# EB10: normal runs unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ok", "needs_review"])
def test_eb10_fully_read_runs_unchanged(tmp_path, kind):
    rows = _write_rows(tmp_path / f"{kind}3.jsonl", kind, 3)
    proc = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only",
         "--file", str(rows)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=_env(), timeout=60,
    )
    assert proc.returncode == EXPECTED_RC[kind], proc.stderr[-2000:]
    lines = proc.stdout.splitlines()
    assert len(lines) == 3
    assert [json.loads(x)["status"] for x in lines] == [kind] * 3
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# EB11: README documents the rule; AC line caps
# ---------------------------------------------------------------------------

def test_eb11_readme_documents_early_close_rule():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    paras = [p for p in readme.split("\n\n")
             if p.lstrip().startswith("Exit codes (both verbs):") or "--batch" in p]
    hits = [
        p for p in paras
        if ("closes" in p or "closed" in p) and "head" in p
        and "stops reading" in p and "worst" in p and "traceback" in p.lower()
    ]
    assert hits, "no exit-code/batch paragraph documents the early-close rule"


def test_ac_cli_line_cap_and_pin():
    n = len((REPO_ROOT / "src" / "reflect_mcp" / "cli.py").read_text(encoding="utf-8").splitlines())
    assert n < 380
    import re
    pin_src = (REPO_ROOT / "tests" / "test_iter68_behavior.py").read_text(encoding="utf-8")
    m = re.search(r"^HEAD_CLI_LINES\s*=\s*(\d+)", pin_src, re.M)
    assert m, "HEAD_CLI_LINES pin missing from test_iter68"
    assert int(m.group(1)) == n


# ---------------------------------------------------------------------------
# Extra (beyond the named EBs): mixed-order files, fail-safe direction
# ---------------------------------------------------------------------------

def _run_mixed(tmp_path: Path, kinds: list[str], concurrency: int | None):
    rows = tmp_path / "mixed.jsonl"
    rows.write_text("".join(json.dumps(ROWS[k]) + "\n" for k in kinds), encoding="utf-8")
    cmd = [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only",
           "--file", str(rows)]
    if concurrency is not None:
        cmd += ["--batch-concurrency", str(concurrency)]
    err_path = tmp_path / "stderr.txt"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=err, cwd=str(REPO_ROOT), env=_env())
        first = proc.stdout.readline()
        proc.stdout.close()
        rc = proc.wait(timeout=60)
    return rc, json.loads(first), err_path.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("concurrency", [None, 4], ids=["serial", "N4"])
def test_extra_trailing_needs_review_is_never_read_after_close(tmp_path, concurrency):
    """1999 ok rows then 1 needs_review: input stops long before the last row, so 0."""
    rc, first, err = _run_mixed(tmp_path, ["ok"] * 1999 + ["needs_review"], concurrency)
    assert first["status"] == "ok"
    assert rc == 0, (rc, err[-2000:])
    assert "Traceback" not in err


def test_extra_ok_then_needs_review_is_fail_safe(tmp_path):
    """1 ok row then 1999 needs_review: rows buffered before EPIPE surfaced may
    count (code 1); the code is never 120 and there is no traceback."""
    rc, first, err = _run_mixed(tmp_path, ["ok"] + ["needs_review"] * 1999, None)
    assert first["status"] == "ok"
    assert rc in (0, 1), (rc, err[-2000:])
    assert "Traceback" not in err
