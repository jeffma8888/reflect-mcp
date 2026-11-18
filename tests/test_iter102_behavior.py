"""Black-box behavioral tests for iter-102: the suite is hermetic against exported gate floors.

Covers the Expected Behaviors of the iter-102 spec (roadmap #59):

  1. With both MCP floor vars exported at 0.99, the four formerly-red files pass.
  3. Inside a test body every ``REFLECT_*_THRESHOLD`` name is absent, including a
     not-yet-existing ``REFLECT_FUTURE_THRESHOLD`` (generic prefix/suffix rule).
  4. Narrow: a non-floor ``REFLECT_`` var and a probe marker survive with their values.
  5. A floor a test sets on purpose with ``monkeypatch.setenv`` still applies.

EB2 (whole suite with the vars exported) and EB6 (no src/eval change, ``make eval``
byte-identical) are checked by the tester outside the suite: a whole-suite run from inside
the suite would recurse.

Subprocess runs use ``sys.executable`` with an explicit ``env=`` built from ``os.environ``
plus the vars, and ``-p no:cacheprovider -o addopts=""``. Per the iter-96 base-differential
lesson, nothing here needs network, a git remote, or a pinned commit.

EB5 ambiguity: the spec says "no thresholds passed" yields a ``raised from 0.7 to 0.99``
note, but README states an omitted threshold "adds no note" and only an explicit argument
below the floor is noted. The test follows README: it passes ``field_threshold=0.7``
explicitly for the note, and proves an omitted-threshold call still reads the deliberately
set env by setting a malformed floor (README: a malformed value returns ``status: error``).

Isolation: only the spec, README, PRODUCT.md and ``tests/`` were read; no implementation
source, engineer/reviewer notes or diff.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp import tools

REPO = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()

PROBE_MARKER = "REFLECT_ITER102_PROBE"
FLOOR_VARS = {
    "REFLECT_FIELD_THRESHOLD": "0.99",
    "REFLECT_OVERALL_THRESHOLD": "0.99",
    "REFLECT_FUTURE_THRESHOLD": "0.99",
}
NARROW_VARS = {"REFLECT_LOG_LEVEL": "DEBUG", PROBE_MARKER: "on-7f3a"}

FORMERLY_RED = [
    "tests/test_iter55_behavior.py",
    "tests/test_iter69_behavior.py",
    "tests/test_iter73_behavior.py",
    "tests/test_iter89_behavior.py",
]

SOURCE = (
    "ACME CORP\n"
    "Invoice #INV-1001\n"
    "Date: 2026-03-14\n"
    "Subtotal: $100.00\n"
    "Tax: $8.50\n"
    "Total: $108.50\n"
    "Paid in USD"
)
EXTRACTION = {
    "vendor_name": "ACME CORP",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-03-14",
    "subtotal": 100.0,
    "tax": 8.5,
    "total": 108.5,
    "currency": "USD",
}


def _env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != PROBE_MARKER}
    env.pop("PYTEST_ADDOPTS", None)
    env.update(extra)
    return env


def _pytest(args: list[str], extra: dict[str, str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", *args]
    return subprocess.run(
        cmd, cwd=str(REPO), env=_env(extra), capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=600,
    )


# ------------------------------------------------------------------ EB3 / EB4 probe
@pytest.mark.skipif(os.environ.get(PROBE_MARKER) is None, reason="probe runs only in a child pytest")
def test_probe_floor_vars_absent_and_others_kept() -> None:
    leaked = sorted(k for k in os.environ if k.startswith("REFLECT_") and k.endswith("_THRESHOLD"))
    assert leaked == [], f"floor vars leaked into the test body: {leaked}"
    for name, value in NARROW_VARS.items():
        assert os.environ.get(name) == value, f"{name} was touched by the fixture"


def _probe(extra_args: list[str]) -> subprocess.CompletedProcess:
    return _pytest(
        [*extra_args, f"{THIS}::test_probe_floor_vars_absent_and_others_kept", "-rA"],
        {**FLOOR_VARS, **NARROW_VARS},
    )


def test_eb3_eb4_probe_sees_no_floor_vars_but_keeps_other_vars() -> None:
    proc = _probe([])
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "1 passed" in out and "skipped" not in out, out


def test_eb3_probe_is_not_vacuous_without_conftest() -> None:
    """The same probe goes red when conftest (and so the fixture) is not loaded."""
    proc = _probe(["--noconftest"])
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "1 failed" in out, out
    assert "floor vars leaked" in out, out
    assert "REFLECT_FUTURE_THRESHOLD" in out, out


def test_eb3_probe_skipped_in_a_normal_run() -> None:
    proc = _pytest([f"{THIS}::test_probe_floor_vars_absent_and_others_kept", "-rA"], {})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "1 skipped" in out and "passed" not in out, out


# ------------------------------------------------------------------ EB1
def test_eb1_formerly_red_files_green_with_floors_exported() -> None:
    proc = _pytest(FORMERLY_RED, {
        "REFLECT_FIELD_THRESHOLD": "0.99", "REFLECT_OVERALL_THRESHOLD": "0.99",
    })
    tail = (proc.stdout + proc.stderr)[-4000:]
    assert proc.returncode == 0, tail
    assert " failed" not in tail and " error" not in tail, tail
    assert " passed" in tail, tail


# ------------------------------------------------------------------ EB5
def test_eb5_deliberate_floor_raises_explicit_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLECT_FIELD_THRESHOLD", "0.99")
    result = tools.validate_tool(SOURCE, dict(EXTRACTION), heuristics_only=True, field_threshold=0.7)
    assert "field_threshold raised from 0.7 to 0.99" in result["notes"]


def test_eb5_counterpart_no_setenv_no_raise() -> None:
    result = tools.validate_tool(SOURCE, dict(EXTRACTION), heuristics_only=True, field_threshold=0.7)
    assert result["status"] == "ok"
    assert "raised from" not in result["notes"]


def test_eb5_deliberate_floor_read_with_thresholds_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberately set (malformed) floor is still read when no threshold is passed."""
    monkeypatch.setenv("REFLECT_FIELD_THRESHOLD", "1.5")
    result = tools.validate_tool(SOURCE, dict(EXTRACTION), heuristics_only=True)
    assert result["status"] == "error"
    assert result.get("error_message")


def test_eb5_omitted_threshold_adds_no_note_under_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLECT_FIELD_THRESHOLD", "0.99")
    result = tools.validate_tool(SOURCE, dict(EXTRACTION), heuristics_only=True)
    assert result["status"] == "ok"
    assert "raised from" not in result["notes"]
