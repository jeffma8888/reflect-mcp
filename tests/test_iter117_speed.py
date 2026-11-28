"""iter117 tester sibling: black-box checks of the zero-retry-delay harness.

Spec (B1): `tests/conftest.py` holds an autouse fixture that sets
`reflect_mcp.parsing._RETRY_DELAY_SECONDS` to 0.0 through `monkeypatch`.
Three things opt a test out of it: a `sleeps` fixture anywhere in its fixture
closure, the registered `real_retry_delay` marker, and iter86 EB5a by its
exact name. The product's own 0.5 s back-off stays unchanged.

These tests count sleeps through a recorder. None of them asserts wall-clock
(ship hazard H2), and none pins a sha or a line number (H1).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reflect_mcp import parsing
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.schema import CritiqueResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
PRODUCT_DELAY = 0.5
BAD = "not json {"
VALID_INV = json.dumps(CORRECT_EXTRACTION)


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A recorder that is NOT a fixture named `sleeps`, so zeroing stays on."""
    rec: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda *a, **_k: rec.append(a[0] if a else None))
    return rec


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    rec: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda *a, **_k: rec.append(a[0] if a else None))
    return rec


@pytest.fixture
def wraps_sleeps(sleeps: list[float]) -> list[float]:
    """Pulls `sleeps` in transitively: the test never names it directly."""
    return sleeps


@pytest.fixture
def unrelated_recorder(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A recorder with a different name must NOT opt out."""
    return _record_sleeps(monkeypatch)


def _env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.startswith("AWS_") or k.endswith("_API_KEY"))
    }
    return env


# ---------------------------------------------------------------------------
# EB1: the autouse fixture zeroes the delay and monkeypatch restores it
# ---------------------------------------------------------------------------

def test_eb1_ordinary_test_sees_zero_delay() -> None:
    assert parsing._RETRY_DELAY_SECONDS == 0.0


@pytest.mark.real_retry_delay
def test_eb1_marker_test_after_zeroed_tests_sees_product_value() -> None:
    # Follows a zeroed test in file order: the 0.0 did not leak.
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY


@pytest.mark.parametrize("case", ["a", "b", "c"])
def test_eb1_parametrized_ordinary_test_sees_zero(case: str) -> None:
    assert case in {"a", "b", "c"}
    assert parsing._RETRY_DELAY_SECONDS == 0.0


class TestEb1InClass:
    def test_method_sees_zero(self) -> None:
        assert parsing._RETRY_DELAY_SECONDS == 0.0


# ---------------------------------------------------------------------------
# EB2: the retry still happens, with one 0.0 back-off
# ---------------------------------------------------------------------------

def test_eb2_generator_retry_sleeps_zero_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _record_sleeps(monkeypatch)
    client = ScriptedLLMClient([BAD, VALID_INV])

    out = Generator(client, max_retries=1).generate(SOURCE)

    assert out == InvoiceData.model_validate(CORRECT_EXTRACTION)
    assert len(client.calls) == 2
    assert rec == [0.0]


def test_eb2_critic_retry_sleeps_zero_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _record_sleeps(monkeypatch)
    crit_raw = critique_all(0.9, 0.9)
    client = ScriptedLLMClient([BAD, crit_raw])
    ext = InvoiceData.model_validate(CORRECT_EXTRACTION)

    out = Critic(client, max_retries=1).critique(SOURCE, ext)

    assert out == CritiqueResult.model_validate(json.loads(crit_raw))
    assert len(client.calls) == 2
    assert rec == [0.0]


def test_eb2_exhaustion_still_raises_with_zero_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _record_sleeps(monkeypatch)
    client = ScriptedLLMClient([BAD, BAD, BAD])

    with pytest.raises(GeneratorError):
        Generator(client, max_retries=2).generate(SOURCE)

    assert len(client.calls) == 3
    assert rec == [0.0, 0.0]


def test_eb2_differently_named_recorder_does_not_opt_out(unrelated_recorder: list[float]) -> None:
    assert parsing._RETRY_DELAY_SECONDS == 0.0
    client = ScriptedLLMClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE)
    assert unrelated_recorder == [0.0]


# ---------------------------------------------------------------------------
# EB3: a `sleeps` fixture in the closure keeps the real delay
# ---------------------------------------------------------------------------

def test_eb3_sleeps_fixture_keeps_product_delay(sleeps: list[float]) -> None:
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY
    client = ScriptedLLMClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE)
    assert len(client.calls) == 2
    assert sleeps == [PRODUCT_DELAY]


def test_eb3_transitive_sleeps_fixture_keeps_product_delay(wraps_sleeps: list[float]) -> None:
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY
    client = ScriptedLLMClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE)
    assert wraps_sleeps == [PRODUCT_DELAY]


@pytest.mark.parametrize("name", ["test_iter86_behavior.py", "test_iter106_behavior.py"])
def test_eb3_existing_sleep_pins_still_present(name: str) -> None:
    text = (TESTS / name).read_text(encoding="utf-8")
    assert "def sleeps(" in text
    assert "sleeps == [0.5]" in text
    assert "real_retry_delay" not in text


# ---------------------------------------------------------------------------
# EB4: the `real_retry_delay` marker is registered and opts out
# ---------------------------------------------------------------------------

def test_eb4_marker_listed_by_pytest_markers_with_description() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@pytest.mark.real_retry_delay")]
    assert len(lines) == 1, proc.stdout
    desc = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else ""
    assert desc


def test_eb4_marker_not_registered_via_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "real_retry_delay" not in text


def test_eb4_marker_registered_via_pytest_configure_in_conftest() -> None:
    text = (TESTS / "conftest.py").read_text(encoding="utf-8")
    assert "def pytest_configure(" in text
    assert "real_retry_delay" in text


@pytest.mark.real_retry_delay
def test_eb4_marked_test_keeps_product_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY
    rec = _record_sleeps(monkeypatch)
    client = ScriptedLLMClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE)
    assert rec == [PRODUCT_DELAY]


@pytest.mark.real_retry_delay
class TestEb4MarkedClass:
    def test_class_marker_reaches_method(self) -> None:
        assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY


# ---------------------------------------------------------------------------
# EB5: iter86 EB5a opts out by exact name, and only that name
# ---------------------------------------------------------------------------

def test_eb5a_iter86_constant_test_still_pins_half_second() -> None:
    text = (TESTS / "test_iter86_behavior.py").read_text(encoding="utf-8")
    assert "def test_eb5a_complete_parsed_exists_and_owns_delay(" in text
    assert '_RETRY_DELAY_SECONDS", None) == 0.5' in text


def test_eb5a_complete_parsed_exists_and_owns_delay() -> None:
    # Same function name as iter86 EB5a: the name opt-out applies here too,
    # which proves the opt-out is by test name and keeps the product value.
    assert callable(getattr(parsing, "complete_parsed", None))
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY


def test_eb5a_complete_parsed_exists_and_owns_delay_near_miss() -> None:
    # A longer name is not the exact name: zeroing applies.
    assert parsing._RETRY_DELAY_SECONDS == 0.0


# ---------------------------------------------------------------------------
# EB6: the product delay is unchanged outside the harness
# ---------------------------------------------------------------------------

def test_eb6_fresh_interpreter_sees_product_delay() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import reflect_mcp.parsing as p; print(repr(p._RETRY_DELAY_SECONDS))"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0.5"


def test_eb6_source_still_assigns_half_second() -> None:
    text = (ROOT / "src" / "reflect_mcp" / "parsing.py").read_text(encoding="utf-8")
    assert "\n_RETRY_DELAY_SECONDS = 0.5\n" in text
