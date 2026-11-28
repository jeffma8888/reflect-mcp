"""iter117 B1: the suite stops paying parsing's real retry back-off sleep.

`tests/conftest.py` zeroes `parsing._RETRY_DELAY_SECONDS` for every in-process
test, except tests that observe the real delay: a `sleeps` recorder fixture, the
`real_retry_delay` marker, or iter86 EB5a (which reads the constant itself).
These tests pin each branch so a later conftest edit cannot silently re-add the
wall-clock cost or, worse, zero the delay under a test that pins `[0.5]`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from reflect_mcp import parsing
from reflect_mcp.generator import Generator
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE

ROOT = Path(__file__).resolve().parent.parent
PRODUCT_DELAY = 0.5


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record `time.sleep` args; its name alone opts the test out of zeroing."""
    rec: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: rec.append(seconds))
    return rec


def test_eb1_ordinary_test_sees_zero_delay() -> None:
    assert parsing._RETRY_DELAY_SECONDS == 0.0


def test_eb2_retry_runs_once_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    rec: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: rec.append(seconds))
    client = ScriptedLLMClient(["not json {", json.dumps(CORRECT_EXTRACTION)])

    out = Generator(client, max_retries=1).generate(SOURCE)

    assert out.invoice_number == CORRECT_EXTRACTION["invoice_number"]
    assert len(client.calls) == 2
    assert rec == [0.0]


def test_eb3_sleeps_fixture_keeps_real_delay(sleeps: list[float]) -> None:
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY
    client = ScriptedLLMClient(["not json {", json.dumps(CORRECT_EXTRACTION)])
    Generator(client, max_retries=1).generate(SOURCE)
    assert sleeps == [PRODUCT_DELAY]


def test_eb4_marker_is_registered_with_description(pytestconfig: pytest.Config) -> None:
    lines = pytestconfig.getini("markers")
    hits = [line for line in lines if line.startswith("real_retry_delay:")]
    assert len(hits) == 1
    assert hits[0].split(":", 1)[1].strip()


def test_eb4_marker_not_registered_in_pyproject() -> None:
    assert "real_retry_delay" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.real_retry_delay
def test_eb4_marker_keeps_real_delay() -> None:
    assert parsing._RETRY_DELAY_SECONDS == PRODUCT_DELAY


def test_eb6_product_constant_unchanged_in_source() -> None:
    src = (ROOT / "src" / "reflect_mcp" / "parsing.py").read_text(encoding="utf-8")
    assert "\n_RETRY_DELAY_SECONDS = 0.5\n" in src
