"""Black-box behavioral tests for iter-57: one batch path + one envelope rule.

Covers the 12 Expected Behaviors from pm.md:
  1. Factory raising on first call -> one error envelope, exit 3, every c.
  2. Same raising factory over empty / blank-only input -> exit 3, one envelope.
  3. Working factory over empty / blank-only input -> exit 0, empty stdout.
  4. Real provider seam (REFLECT_PROVIDER=nonexistent) at c=2 -> exit 3, envelope.
  5. Factory succeeds first then fails -> per-line envelopes, no escape.
  6. Mixed input: order, messages, exit code, c=1 vs c=3 byte-identical.
  7. Serial path streams lazily (line 1 printed before line 2 pulled).
  8. Input readers keep their messages (file not found / no input provided).
  9. Single (non-batch) extract unchanged.
 10. Structural deletion checks on reflect_mcp.cli.
 11. cli.py has strictly fewer than 343 lines.
 12. Bad threshold flags in batch mode -> exit 3 + one envelope, never exit 1.

Isolation contract: tests drive the public CLI interface (run_extract,
run_validate, ``python -m reflect_mcp``) and inspect observable output. No
implementation source was read; behavior 10/11 use only the structural checks
the spec itself prescribes.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.schema import ExtractionResult

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCURRENCIES = [None, 1, 2, 4]  # None == flag omitted (c=default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c_flags(c):
    return [] if c is None else ["--batch-concurrency", str(c)]


def _result(status: str, text: str) -> ExtractionResult:
    return ExtractionResult(
        status=status,
        data={},
        field_confidence={},
        overall_confidence=0.9 if status == "ok" else 0.3,
        flagged_fields=[],
        critic_summary=f"summary:{text}",
        trace_id=f"trace:{text}",
        notes="",
    )


class FakePipeline:
    """Deterministic pipeline: result depends only on ``text``."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def extract(self, text, **kwargs):
        with self._lock:
            self.calls.append(text)
        if self.delay:
            time.sleep(self.delay)
        status = "needs_review" if "review" in text else "ok"
        return _result(status, text)


class TTYStringIO(io.StringIO):
    def isatty(self):  # noqa: D401
        return True


def _run(func_name: str, args: list[str], stdin=None):
    """Run a cli entry point in-process; return (code, stdout, stderr)."""
    import reflect_mcp.cli as cli

    func = getattr(cli, func_name)
    out, err = io.StringIO(), io.StringIO()
    stdin = io.StringIO("") if stdin is None else stdin
    with patch("sys.stdin", stdin), patch("sys.stdout", out), patch("sys.stderr", err):
        code = func(args)
    return code, out.getvalue(), err.getvalue()


def _write(tmp_path: Path, content: str, name: str = "in.jsonl") -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _jsonl(*texts: str) -> str:
    return "".join(json.dumps({"text": t}) + "\n" for t in texts)


def _lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.split("\n") if ln != ""]


def _raising_factory(exc):
    calls = []

    def factory(*a, **k):
        calls.append((a, k))
        raise exc

    factory.calls = calls
    return factory


# ---------------------------------------------------------------------------
# EB-1: factory fails on first call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCURRENCIES)
def test_eb1_first_build_failure_one_envelope_exit3(tmp_path, c):
    path = _write(tmp_path, _jsonl("doc one", "doc two"))
    factory = _raising_factory(ValueError("boom-config"))
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run("run_extract", ["--batch", "--file", path, *_c_flags(c)])
    lines = _lines(out)
    assert len(lines) == 1, out
    assert json.loads(lines[0]) == {"status": "error", "error_message": "boom-config"}
    assert code == 3
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# EB-2: raising factory, empty input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["", "\n\n", "\r\n\n"], ids=["zero-bytes", "lf-blank", "crlf-blank"])
@pytest.mark.parametrize("c", CONCURRENCIES)
def test_eb2_build_failure_empty_input_exit3(tmp_path, c, content):
    path = _write(tmp_path, content)
    factory = _raising_factory(ValueError("boom-config"))
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run("run_extract", ["--batch", "--file", path, *_c_flags(c)])
    assert code == 3
    lines = _lines(out)
    assert len(lines) == 1, out
    assert json.loads(lines[0]) == {"status": "error", "error_message": "boom-config"}
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# EB-3: working factory, empty input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["", "\n\n", "\r\n\n"], ids=["zero-bytes", "lf-blank", "crlf-blank"])
@pytest.mark.parametrize("c", CONCURRENCIES)
def test_eb3_working_factory_empty_input_exit0(tmp_path, c, content):
    path = _write(tmp_path, content)
    fake = FakePipeline()
    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=fake):
        code, out, err = _run("run_extract", ["--batch", "--file", path, *_c_flags(c)])
    assert code == 0
    assert out == ""
    assert fake.calls == []


# ---------------------------------------------------------------------------
# EB-4: real seam, no stub
# ---------------------------------------------------------------------------

def _subprocess(args: list[str], provider: str = "nonexistent"):
    env = dict(os.environ)
    env["REFLECT_PROVIDER"] = provider
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60,
    )


def test_eb4_unknown_provider_real_seam_c2(tmp_path):
    path = _write(tmp_path, _jsonl("doc one", "doc two"))
    proc = _subprocess(["extract", "--batch", "--batch-concurrency", "2", "--file", path])
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    lines = _lines(proc.stdout)
    assert len(lines) == 1, proc.stdout
    env = json.loads(lines[0])
    assert env["status"] == "error"
    assert "Unknown REFLECT_PROVIDER" in env["error_message"]
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


@pytest.mark.parametrize("c", [None, 1, 4])
def test_eb4_unknown_provider_real_seam_other_concurrencies(tmp_path, c):
    """Extra: the same envelope rule holds at the other concurrencies."""
    path = _write(tmp_path, _jsonl("doc one", "doc two"))
    proc = _subprocess(["extract", "--batch", *_c_flags(c), "--file", path])
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    lines = _lines(proc.stdout)
    assert len(lines) == 1
    assert "Unknown REFLECT_PROVIDER" in json.loads(lines[0])["error_message"]
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# EB-5: factory succeeds first, then fails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [0.0, 0.05], ids=["instant", "slow"])
def test_eb5_late_build_failure_is_per_line(tmp_path, delay):
    path = _write(tmp_path, _jsonl("a", "b", "c", "d"))
    fake = FakePipeline(delay=delay)
    n_calls = []
    lock = threading.Lock()

    def factory(*a, **k):
        with lock:
            n_calls.append(1)
            first = len(n_calls) == 1
        if first:
            return fake
        raise RuntimeError("late")

    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run("run_extract", ["--batch", "--batch-concurrency", "2", "--file", path])

    lines = _lines(out)
    assert len(lines) == 4, out
    parsed = [json.loads(ln) for ln in lines]
    any_error = False
    for p in parsed:
        if p["status"] == "error":
            assert p == {"status": "error", "error_message": "late"}
            any_error = True
        else:
            assert p["status"] in ("ok", "needs_review")
    expected = 3 if any_error else max(1 if p["status"] == "needs_review" else 0 for p in parsed)
    assert code == expected
    assert "Traceback" not in err
    # every non-error line was served by the working pipeline
    assert len(fake.calls) == sum(1 for p in parsed if p["status"] != "error")


# ---------------------------------------------------------------------------
# EB-6: output order + exit code unchanged
# ---------------------------------------------------------------------------

MIXED = (
    json.dumps({"text": "good invoice"}) + "\n"
    + "\n"
    + "{not json\n"
    + json.dumps({"id": 1}) + "\n"
    + json.dumps({"text": "please review me"}) + "\n"
)


def _run_mixed(tmp_path, c):
    path = _write(tmp_path, MIXED, name=f"mixed{c}.jsonl")
    fake = FakePipeline()
    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=fake):
        return _run("run_extract", ["--batch", "--batch-concurrency", str(c), "--file", path])


@pytest.mark.parametrize("c", [1, 3])
def test_eb6_mixed_order_messages_exit(tmp_path, c):
    code, out, err = _run_mixed(tmp_path, c)
    lines = _lines(out)
    assert len(lines) == 4, out
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["status"] == "ok"
    assert parsed[0]["trace_id"] == "trace:good invoice"
    assert parsed[1]["status"] == "error"
    assert parsed[1]["error_message"].startswith("invalid JSON:")
    assert parsed[2] == {"status": "error", "error_message": 'missing "text" key'}
    assert parsed[3]["status"] == "needs_review"
    assert parsed[3]["trace_id"] == "trace:please review me"
    assert code == 3


def test_eb6_c1_and_c3_byte_identical(tmp_path):
    code1, out1, _ = _run_mixed(tmp_path, 1)
    code3, out3, _ = _run_mixed(tmp_path, 3)
    assert out1 == out3
    assert code1 == code3 == 3


# ---------------------------------------------------------------------------
# EB-7: serial streaming preserved
# ---------------------------------------------------------------------------

class CountingStdin:
    """Lazy line source that counts pulls; refuses bulk reads to stay lazy."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pulled >= len(self._lines):
            raise StopIteration
        line = self._lines[self.pulled]
        self.pulled += 1
        return line

    def readline(self):
        try:
            return next(self)
        except StopIteration:
            return ""

    def isatty(self):
        return False

    def close(self):
        pass


class RecordingStdout(io.StringIO):
    def __init__(self, stdin):
        super().__init__()
        self._stdin = stdin
        self.pulls_at_first_newline = None

    def write(self, s):
        n = super().write(s)
        if self.pulls_at_first_newline is None and "\n" in self.getvalue():
            self.pulls_at_first_newline = self._stdin.pulled
        return n


def test_eb7_serial_batch_streams_lazily():
    import reflect_mcp.cli as cli

    stdin = CountingStdin([json.dumps({"text": t}) + "\n" for t in ("one", "two", "three")])
    out = RecordingStdout(stdin)
    fake = FakePipeline()
    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=fake), \
            patch("sys.stdin", stdin), patch("sys.stdout", out), patch("sys.stderr", io.StringIO()):
        code = cli.run_extract(["--batch"])
    assert code == 0
    assert len(_lines(out.getvalue())) == 3
    assert out.pulls_at_first_newline == 1, (
        f"line 1's result was printed after {out.pulls_at_first_newline} lines were pulled"
    )


# ---------------------------------------------------------------------------
# EB-8: input readers keep their messages
# ---------------------------------------------------------------------------

READER_CASES = [
    ("run_extract", []),
    ("run_extract", ["--batch"]),
    ("run_validate", ["--data", "{}"]),
]


@pytest.mark.parametrize("func,extra", READER_CASES, ids=["extract", "batch", "validate"])
def test_eb8_file_not_found(func, extra):
    factory = _raising_factory(AssertionError("factory must not be called"))
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run(func, [*extra, "--file", "/nonexistent/x.txt"])
    assert code == 2
    assert "Error: file not found: /nonexistent/x.txt" in err


@pytest.mark.parametrize("func,extra", READER_CASES, ids=["extract", "batch", "validate"])
def test_eb8_no_input_tty(func, extra):
    factory = _raising_factory(AssertionError("factory must not be called"))
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run(func, list(extra), stdin=TTYStringIO(""))
    assert code == 2
    assert "Error: no input provided" in err
    assert factory.calls == []


# ---------------------------------------------------------------------------
# EB-9: single mode unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   \n\t  \n"], ids=["empty", "whitespace"])
def test_eb9_single_empty_input_exit2(text):
    fake = FakePipeline()
    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=fake):
        code, out, err = _run("run_extract", [], stdin=io.StringIO(text))
    assert code == 2
    assert "Error: input is empty." in err
    assert fake.calls == []


def test_eb9_single_factory_raises_envelope_exit3():
    factory = _raising_factory(ValueError("boom-single"))
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory):
        code, out, err = _run("run_extract", [], stdin=io.StringIO("ACME invoice text"))
    assert code == 3
    lines = _lines(out)
    assert len(lines) >= 1
    env = json.loads(out)
    assert env["status"] == "error"
    assert env["error_message"] == "boom-single"
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# EB-10: structural deletion
# ---------------------------------------------------------------------------

def test_eb10_deleted_helpers_gone():
    import reflect_mcp.cli as cli

    assert not hasattr(cli, "_run_batch_concurrent")
    assert not hasattr(cli, "_thread_id")


def test_eb10_read_text_delegates():
    import reflect_mcp.cli as cli

    if not hasattr(cli, "_read_text"):
        return
    src = inspect.getsource(cli._read_text)
    assert "open(" not in src
    assert "_read_input_stream" in src


def test_eb10_single_code3_fold():
    import reflect_mcp.cli as cli

    assert inspect.getsource(cli).count("if code == 3") <= 1


# ---------------------------------------------------------------------------
# EB-11: line count
# ---------------------------------------------------------------------------

def test_eb11_cli_shorter_than_343_lines():
    path = REPO_ROOT / "src" / "reflect_mcp" / "cli.py"
    n = len(path.read_text(encoding="utf-8").splitlines())
    assert n < 343, f"cli.py has {n} lines"


# ---------------------------------------------------------------------------
# EB-12: bad threshold in batch mode
# ---------------------------------------------------------------------------

BAD_THRESHOLDS = [
    ["--field-threshold", "1.5"],
    ["--field-threshold", "nan"],
    ["--overall-threshold", "-0.1"],
]


@pytest.mark.parametrize("bad", BAD_THRESHOLDS, ids=["field-1.5", "field-nan", "overall-neg"])
@pytest.mark.parametrize("c", [None, 1, 2])
def test_eb12_bad_threshold_batch_exit3_envelope(tmp_path, c, bad):
    path = _write(tmp_path, _jsonl("doc one"))
    fake = FakePipeline()
    with patch("reflect_mcp.pipeline.build_default_pipeline", return_value=fake):
        code, out, err = _run("run_extract", ["--batch", "--file", path, *bad, *_c_flags(c)])
    assert code == 3, (code, out, err)
    lines = _lines(out)
    assert len(lines) == 1, out
    env = json.loads(lines[0])
    assert env["status"] == "error"
    assert isinstance(env["error_message"], str) and env["error_message"].strip()
    assert "Traceback" not in err
    assert fake.calls == []


@pytest.mark.parametrize("c", [None, 2])
def test_eb12_bad_threshold_real_seam_subprocess(tmp_path, c):
    """The scout's original repro, end to end: never exit 1, never a traceback."""
    path = _write(tmp_path, _jsonl("doc one"))
    proc = _subprocess(["extract", "--batch", "--field-threshold", "1.5", *_c_flags(c), "--file", path])
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    lines = _lines(proc.stdout)
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "error"
    assert "Traceback" not in proc.stderr
