"""Black-box behavioral tests for iter-75: concurrent --batch streams through a
bounded, order-preserving window of at most 2N pending lines.

Covers the 8 Expected Behaviors from pm.md:
  1. extract --batch --batch-concurrency N (N in {2,4}), 100 lines: when row 1
     is written at most 2N lines have been pulled.
  2. At every output row k (1-based) at most k - 1 + 2N lines have been pulled.
  3. validate --batch obeys the same bound.
  4. Order equals input order; a stub whose later lines finish first still
     yields N=4 output byte-identical to N=1.
  5. Mixed input: stdout bytes + exit code identical across N=1,2,4,8; exit
     code is the worst severity (3 > 1 > 0).
  6. Serial (flag omitted or N=1): line 1's row is written after exactly 1 pull.
  7. Fewer lines than the window; empty / blank-only input; failing factory.
  8. Blank lines are skipped and never count toward the window.
Plus acceptance-criteria checks observable at runtime: executor.map is not
used for concurrent batches, the _run_batch docstring and README state the
bounded-window contract, cli.py stays <= 375 lines.

Isolation contract: tests drive the public CLI entry points (run_extract,
run_validate) in-process with a stubbed ``build_default_pipeline`` and patched
stdin/stdout, and assert observable output. No implementation source was read.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.schema import ExtractionResult

REPO_ROOT = Path(__file__).resolve().parent.parent

GOOD_DATA = {
    "invoice_number": "INV-1",
    "vendor_name": "Acme",
    "invoice_date": "2024-01-05",
    "total": 10.0,
    "currency": "USD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Deterministic: result depends only on ``text``.

    ``delay_for`` maps text -> seconds slept, so tests can make later lines
    finish before earlier ones.
    """

    def __init__(self, delay_for=None):
        self.delay_for = delay_for or (lambda text: 0.0)
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _go(self, text):
        with self._lock:
            self.calls.append(text)
        d = self.delay_for(text)
        if d:
            time.sleep(d)
        return _result("needs_review" if "review" in text else "ok", text)

    def extract(self, text, *a, **k):
        return self._go(text)

    def validate(self, text, *a, **k):
        return self._go(text)


class CountingStdin:
    """Lazy line source that counts every pull (blank lines included) and,
    separately, pulls of non-blank lines."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.pulled = 0
        self.nonblank_pulled = 0
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self.pulled >= len(self._lines):
                raise StopIteration
            line = self._lines[self.pulled]
            self.pulled += 1
            if line.strip():
                self.nonblank_pulled += 1
            return line

    def readline(self):
        try:
            return next(self)
        except StopIteration:
            return ""

    def read(self, *a):
        return "".join(list(self))

    def isatty(self):
        return False

    def close(self):
        pass


class RecordingStdout(io.StringIO):
    """Records (pulled, nonblank_pulled) at every newline written."""

    def __init__(self, stdin: CountingStdin):
        super().__init__()
        self._stdin = stdin
        self.marks: list[tuple[int, int]] = []

    def write(self, s):
        n = super().write(s)
        for _ in range(s.count("\n")):
            self.marks.append((self._stdin.pulled, self._stdin.nonblank_pulled))
        return n


def _c_flags(c):
    return [] if c is None else ["--batch-concurrency", str(c)]


def _run_counting(verb: str, lines: list[str], c, fake=None, factory=None):
    """Run ``verb --batch`` over a counting stdin. Returns (code, stdout, marks, stdin)."""
    import reflect_mcp.cli as cli

    stdin = CountingStdin(lines)
    out = RecordingStdout(stdin)
    fake = fake if fake is not None else FakePipeline()
    side_effect = factory if factory is not None else (lambda *a, **k: fake)
    func = getattr(cli, "run_extract" if verb == "extract" else "run_validate")
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=side_effect), \
            patch("sys.stdin", stdin), patch("sys.stdout", out), patch("sys.stderr", io.StringIO()):
        code = func(["--batch", *_c_flags(c)])
    return code, out.getvalue(), out.marks, stdin


def _run_text(verb: str, body: str, c, fake=None, factory=None):
    return _run_counting(verb, body.splitlines(keepends=True), c, fake=fake, factory=factory)


def _extract_lines(n: int) -> list[str]:
    return [json.dumps({"text": f"doc-{i:03d}"}) + "\n" for i in range(n)]


def _validate_lines(n: int) -> list[str]:
    return [json.dumps({"text": f"doc-{i:03d}", "data": GOOD_DATA}) + "\n" for i in range(n)]


def _rows(stdout: str) -> list[str]:
    return [ln for ln in stdout.split("\n") if ln != ""]


def _raising_factory(exc):
    def factory(*a, **k):
        raise exc

    return factory


# ---------------------------------------------------------------------------
# EB-1: first row is written after at most 2N pulls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [0.0, 0.003], ids=["instant", "slow"])
@pytest.mark.parametrize("n", [2, 4])
def test_eb1_extract_first_row_after_at_most_2n_pulls(n, delay):
    fake = FakePipeline(delay_for=lambda text: delay)
    code, out, marks, stdin = _run_counting("extract", _extract_lines(100), n, fake=fake)
    assert code == 0
    assert len(_rows(out)) == 100
    assert stdin.pulled == 100
    first_pulled = marks[0][0]
    assert first_pulled <= 2 * n, (
        f"N={n}: row 1 written after {first_pulled} of 100 lines were pulled (bound {2 * n})"
    )


# ---------------------------------------------------------------------------
# EB-2: at row k at most k - 1 + 2N pulls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [0.0, 0.003], ids=["instant", "slow"])
@pytest.mark.parametrize("n", [2, 4])
def test_eb2_extract_every_row_bounded(n, delay):
    fake = FakePipeline(delay_for=lambda text: delay)
    code, out, marks, _ = _run_counting("extract", _extract_lines(100), n, fake=fake)
    assert code == 0
    assert len(marks) == 100
    violations = [(k, p) for k, (p, _nb) in enumerate(marks, start=1) if p > k - 1 + 2 * n]
    assert violations == [], f"N={n}: (row, pulled) over bound k-1+2N: {violations[:5]}"
    # unprinted pulled lines never exceed 2N
    assert max(p - (k - 1) for k, (p, _nb) in enumerate(marks, start=1)) <= 2 * n


# ---------------------------------------------------------------------------
# EB-3: validate --batch follows the same bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4])
def test_eb3_validate_first_row_and_every_row_bounded(n):
    code, out, marks, stdin = _run_counting("validate", _validate_lines(100), n)
    assert code == 0
    rows = _rows(out)
    assert len(rows) == 100
    assert all(json.loads(r)["status"] == "ok" for r in rows)
    assert stdin.pulled == 100
    assert marks[0][0] <= 2 * n, f"N={n}: validate row 1 after {marks[0][0]} pulls"
    violations = [(k, p) for k, (p, _nb) in enumerate(marks, start=1) if p > k - 1 + 2 * n]
    assert violations == [], f"N={n}: validate rows over bound: {violations[:5]}"


@pytest.mark.parametrize("n", [2, 4])
def test_eb3_validate_order_matches_serial(n):
    body = "".join(_validate_lines(30))
    code1, out1, _, _ = _run_text("validate", body, 1)
    coden, outn, _, _ = _run_text("validate", body, n)
    assert outn == out1
    assert coden == code1 == 0
    assert [json.loads(r)["trace_id"] for r in _rows(outn)] == [f"trace:doc-{i:03d}" for i in range(30)]


# ---------------------------------------------------------------------------
# EB-4: order preserved even when later lines finish first
# ---------------------------------------------------------------------------

def _reverse_delay(total: int, step: float = 0.002):
    def delay_for(text):
        idx = int(text.split("-")[1])
        return (total - idx) * step

    return delay_for


def test_eb4_later_lines_finish_first_order_preserved():
    total = 24
    lines = _extract_lines(total)
    code1, out1, _, _ = _run_counting("extract", lines, 1, fake=FakePipeline(_reverse_delay(total)))
    fake4 = FakePipeline(_reverse_delay(total))
    code4, out4, marks4, _ = _run_counting("extract", lines, 4, fake=fake4)
    assert out4 == out1
    assert code4 == code1 == 0
    traces = [json.loads(r)["trace_id"] for r in _rows(out4)]
    assert traces == [f"trace:doc-{i:03d}" for i in range(total)]
    # still bounded while out-of-order completions are buffered
    assert all(p <= k - 1 + 8 for k, (p, _nb) in enumerate(marks4, start=1))


@pytest.mark.parametrize("n", [2, 4, 8])
def test_eb4_order_equals_input_order_every_concurrency(n):
    lines = _extract_lines(40)
    code, out, _, _ = _run_counting("extract", lines, n, fake=FakePipeline(_reverse_delay(40, 0.001)))
    assert code == 0
    assert [json.loads(r)["trace_id"] for r in _rows(out)] == [f"trace:doc-{i:03d}" for i in range(40)]


# ---------------------------------------------------------------------------
# EB-5: mixed input, byte-identical across N, worst-severity exit
# ---------------------------------------------------------------------------

MIXED = (
    json.dumps({"text": "good invoice"}) + "\n"
    + json.dumps({"text": "please review me"}) + "\n"
    + "{not json\n"
    + json.dumps({"id": 7}) + "\n"
    + json.dumps({"id": "abc", "text": "id carrying good"}) + "\n"
    + json.dumps({"id": [1, 2], "text": "id carrying review"}) + "\n"
    + json.dumps({"text": "tail ok"}) + "\n"
)

REVIEW_ONLY = (
    json.dumps({"text": "good invoice"}) + "\n"
    + json.dumps({"id": 1, "text": "please review me"}) + "\n"
    + json.dumps({"text": "tail ok"}) + "\n"
)

OK_ONLY = (
    json.dumps({"text": "good invoice"}) + "\n"
    + json.dumps({"id": 2, "text": "another good"}) + "\n"
)


@pytest.mark.parametrize(
    "body,expected_code",
    [(MIXED, 3), (REVIEW_ONLY, 1), (OK_ONLY, 0)],
    ids=["mixed-error", "review", "ok"],
)
def test_eb5_identical_across_concurrencies_and_worst_exit(body, expected_code):
    results = {c: _run_text("extract", body, c)[:2] for c in (1, 2, 4, 8)}
    codes = {c: r[0] for c, r in results.items()}
    outs = {c: r[1] for c, r in results.items()}
    assert set(codes.values()) == {expected_code}, codes
    assert outs[2] == outs[1]
    assert outs[4] == outs[1]
    assert outs[8] == outs[1]


def test_eb5_mixed_rows_content():
    code, out, _, _ = _run_text("extract", MIXED, 4)
    rows = [json.loads(r) for r in _rows(out)]
    assert len(rows) == 7
    assert rows[0]["status"] == "ok" and rows[0]["trace_id"] == "trace:good invoice"
    assert rows[1]["status"] == "needs_review"
    assert rows[2]["status"] == "error" and rows[2]["error_message"].startswith("invalid JSON:")
    assert rows[3] == {"id": 7, "status": "error", "error_message": 'missing "text" key'}
    assert list(rows[4])[0] == "id" and rows[4]["id"] == "abc" and rows[4]["status"] == "ok"
    assert list(rows[5])[0] == "id" and rows[5]["id"] == [1, 2] and rows[5]["status"] == "needs_review"
    assert rows[6]["trace_id"] == "trace:tail ok"
    assert code == 3


# ---------------------------------------------------------------------------
# EB-6: serial unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [None, 1], ids=["omitted", "n1"])
@pytest.mark.parametrize("verb", ["extract", "validate"])
def test_eb6_serial_first_row_after_exactly_one_pull(verb, c):
    lines = (_extract_lines if verb == "extract" else _validate_lines)(10)
    code, out, marks, _ = _run_counting(verb, lines, c)
    assert code == 0
    assert len(_rows(out)) == 10
    assert marks[0][0] == 1, f"serial row 1 written after {marks[0][0]} pulls"
    # every serial row k is written after exactly k pulls
    assert [p for p, _nb in marks] == list(range(1, 11))


# ---------------------------------------------------------------------------
# EB-7: short input, empty input, failing factory
# ---------------------------------------------------------------------------

def test_eb7_fewer_lines_than_window():
    body = (
        json.dumps({"text": "a ok"}) + "\n"
        + json.dumps({"text": "b review"}) + "\n"
        + json.dumps({"text": "c ok"}) + "\n"
    )
    code, out, _, _ = _run_text("extract", body, 4, fake=FakePipeline(_short_reverse))
    rows = [json.loads(r) for r in _rows(out)]
    assert [r["trace_id"] for r in rows] == ["trace:a ok", "trace:b review", "trace:c ok"]
    assert code == 1


def _short_reverse(text):
    return {"a": 0.03, "b": 0.015, "c": 0.0}[text[0]]


@pytest.mark.parametrize("n", [2, 4, 8])
def test_eb7_single_line_below_window(n):
    code, out, _, _ = _run_text("extract", json.dumps({"text": "only one"}) + "\n", n)
    rows = _rows(out)
    assert len(rows) == 1 and json.loads(rows[0])["trace_id"] == "trace:only one"
    assert code == 0


@pytest.mark.parametrize("body", ["", "\n\n\n", "\r\n\n\r\n"], ids=["empty", "lf-blank", "crlf-blank"])
@pytest.mark.parametrize("verb", ["extract", "validate"])
def test_eb7_empty_input_working_factory(verb, body):
    fake = FakePipeline()
    code, out, _, _ = _run_text(verb, body, 4, fake=fake)
    assert code == 0
    assert out == ""
    assert fake.calls == []


@pytest.mark.parametrize("body", ["", "\n\n", json.dumps({"text": "x"}) + "\n" + json.dumps({"text": "y"}) + "\n"],
                         ids=["empty", "blank", "two-lines"])
@pytest.mark.parametrize("verb", ["extract", "validate"])
def test_eb7_failing_factory_one_envelope_exit3(verb, body):
    code, out, _, _ = _run_text(verb, body, 4, factory=_raising_factory(ValueError("boom-config")))
    rows = _rows(out)
    assert len(rows) == 1, out
    assert json.loads(rows[0]) == {"status": "error", "error_message": "boom-config"}
    assert code == 3


# ---------------------------------------------------------------------------
# EB-8: blank lines skipped, never count toward the window
# ---------------------------------------------------------------------------

def _interleaved(n_docs: int, blanks_between: int) -> list[str]:
    out: list[str] = []
    for i in range(n_docs):
        out.append(json.dumps({"text": f"doc-{i:03d}"}) + "\n")
        # "blank" = empty line (LF or CRLF). Whitespace-only lines are a
        # pre-existing error row at HEAD (tolerance is roadmap #46, out of scope).
        out.extend((["\n", "\r\n"] * blanks_between)[:blanks_between])
    return out


@pytest.mark.parametrize("n", [2, 4])
def test_eb8_blank_lines_skipped_and_output_matches_serial(n):
    lines = _interleaved(30, 3)
    code_s, out_s, _, _ = _run_counting("extract", lines, None)
    code_n, out_n, _, _ = _run_counting("extract", lines, n)
    assert out_n == out_s
    assert code_n == code_s == 0
    assert [json.loads(r)["trace_id"] for r in _rows(out_n)] == [f"trace:doc-{i:03d}" for i in range(30)]


@pytest.mark.parametrize("n", [2, 4])
def test_eb8_blank_lines_do_not_count_toward_window(n):
    # 10 blank lines after every document and a slow pipeline: the window fills
    # with real documents long before the head finishes. If blanks counted
    # toward the 2N window, at most 2N raw lines could be pulled before row 1;
    # since they do not, row 1 is written after more than 2N raw pulls while
    # still at most 2N non-blank ones.
    lines = _interleaved(16, 10)
    fake = FakePipeline(delay_for=lambda text: 0.03)
    code, out, marks, _ = _run_counting("extract", lines, n, fake=fake)
    assert code == 0
    assert len(_rows(out)) == 16
    raw_first, nb_first = marks[0]
    assert nb_first <= 2 * n, f"row 1 after {nb_first} non-blank pulls (bound {2 * n})"
    assert raw_first > 2 * n, f"row 1 after only {raw_first} raw pulls: blanks seem to count toward the window"
    nb_marks = [nb for _p, nb in marks]
    assert all(nb <= k - 1 + 2 * n for k, nb in enumerate(nb_marks, start=1)), nb_marks[:8]


def test_eb8_blank_lines_never_reach_pipeline():
    fake = FakePipeline()
    code, out, _, _ = _run_counting("extract", _interleaved(12, 3), 4, fake=fake)
    assert code == 0
    assert sorted(fake.calls) == [f"doc-{i:03d}" for i in range(12)]


# ---------------------------------------------------------------------------
# Acceptance criteria observable at runtime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verb", ["extract", "validate"])
def test_ac_concurrent_batch_does_not_use_executor_map(verb):
    def _boom(self, *a, **k):
        raise AssertionError("executor.map must not be used by concurrent --batch")

    lines = (_extract_lines if verb == "extract" else _validate_lines)(12)
    with patch.object(concurrent.futures.ThreadPoolExecutor, "map", _boom):
        code, out, _, _ = _run_counting(verb, lines, 4)
    assert code == 0
    assert len(_rows(out)) == 12


def test_ac_run_batch_docstring_states_bounded_window():
    import reflect_mcp.cli as cli

    doc = (getattr(cli, "_run_batch").__doc__ or "")
    assert "executor.map" not in doc
    low = doc.lower()
    assert "2" in doc and ("window" in low or "in flight" in low or "pending" in low), doc


def test_ac_readme_states_bounded_window():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "at most 2N lines in flight" in readme
    assert "memory stays bounded" in readme


def test_ac_cli_line_budget():
    n = len((REPO_ROOT / "src" / "reflect_mcp" / "cli.py").read_text(encoding="utf-8").splitlines())
    assert n <= 375, f"cli.py has {n} lines (spec AC: <= 375)"
