"""iter107 black-box: README §6 library error contract (docs-only change).

EB1-EB7 pin the README paragraph's claims and execute each one against the
public API; EB8 (zero src diff, eval unchanged) is checked by the ship gate and
by the frozen eval digest pins elsewhere. Offline, no keys, ``time.sleep``
patched so retries add no wall-clock time; no sha/line pins of shipped files.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import reflect_mcp
from reflect_mcp import ReflectivePipeline, build_default_pipeline
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
TEXT = "ACME total 5.00"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def _section6() -> str:
    body = README.read_text(encoding="utf-8")
    m = re.search(r"^## 6\..*?(?=^## 7\.)", body, re.S | re.M)
    assert m, "README has no '## 6.' ... '## 7.' span"
    return m.group(0)


def _paragraph() -> str:
    sec = _section6()
    paras = [p for p in re.split(r"\n\s*\n", sec) if "Error contract" in p]
    assert len(paras) == 1, f"expected exactly one Error contract paragraph, got {len(paras)}"
    return paras[0]


class _Boom:
    """Provider stub whose SDK call fails with a non-library exception."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise ConnectionError("down")


# EB1 ---------------------------------------------------------------------
def test_eb1_paragraph_after_errors_sentence_before_section7():
    sec = _section6()
    para = _paragraph()
    anchor = sec.find("Errors (`GeneratorError`, `CriticError`) are raised")
    assert anchor != -1, "existing Errors sentence missing from §6"
    assert sec.find(para) > anchor
    assert "Error contract" not in sec[:anchor]


# EB2 ---------------------------------------------------------------------
def test_eb2_default_max_retries_matches_signature():
    para = _paragraph()
    assert "max_retries" in para
    m = re.search(r"max_retries=(\d+)", para)
    assert m, "paragraph does not state max_retries=<int>"
    default = inspect.signature(build_default_pipeline).parameters["max_retries"].default
    assert int(m.group(1)) == default == 1


# EB3 ---------------------------------------------------------------------
def test_eb3_readme_states_call_budget():
    para = _paragraph()
    assert "max_retries + 1" in para
    assert "at most 2 model calls per stage" in para


def test_eb3_generator_default_budget_two_calls():
    client = ScriptedLLMClient(["bad", "bad"])
    with pytest.raises(GeneratorError):
        ReflectivePipeline(Generator(client), None, None).extract(TEXT)
    assert len(client.calls) == 2


def test_eb3_generator_max_retries_3_four_calls():
    client = ScriptedLLMClient(["bad"] * 4)
    with pytest.raises(GeneratorError):
        ReflectivePipeline(Generator(client, max_retries=3), None, None).extract(TEXT)
    assert len(client.calls) == 4


# EB4 ---------------------------------------------------------------------
def test_eb4_critic_default_budget_two_calls():
    client = ScriptedLLMClient(["bad", "bad"])
    with pytest.raises(CriticError):
        ReflectivePipeline(None, Critic(client), None).validate(
            TEXT, InvoiceData(vendor_name="ACME", total=5.0)
        )
    assert len(client.calls) == 2


# EB5 ---------------------------------------------------------------------
def test_eb5_readme_states_provider_error_unwrapped():
    para = _paragraph()
    assert "not retried" in para
    assert "unwrapped" in para
    assert "except (GeneratorError, CriticError)" in para


def test_eb5_generator_provider_error_propagates_bare_after_one_call():
    boom = _Boom()
    with pytest.raises(ConnectionError) as ei:
        ReflectivePipeline(Generator(boom), None, None).extract("x")
    assert not isinstance(ei.value, GeneratorError)
    assert boom.calls == 1


def test_eb5_critic_provider_error_propagates_bare_after_one_call():
    boom = _Boom()
    with pytest.raises(ConnectionError) as ei:
        ReflectivePipeline(None, Critic(boom), None).validate(
            TEXT, InvoiceData(vendor_name="ACME", total=5.0)
        )
    assert not isinstance(ei.value, CriticError)
    assert boom.calls == 1


# EB6 ---------------------------------------------------------------------
def test_eb6_readme_contrasts_tools_and_cli():
    para = _paragraph()
    assert "Unlike the MCP tools and the CLI" in para
    assert "raises" in para


def test_eb6_cli_prints_error_not_traceback():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only", "--data", '{"totl": 1}'],
        input=(TEXT + "\n").encode(),
        capture_output=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    err = proc.stderr.decode("utf-8")
    assert proc.returncode == 2, err
    assert err.startswith("Error:"), err
    assert "unknown field(s): totl" in err
    assert "Traceback" not in err


# EB7 ---------------------------------------------------------------------
def test_eb7_extract_without_generator_raises_runtimeerror():
    assert "RuntimeError" in _paragraph()
    with pytest.raises(RuntimeError):
        ReflectivePipeline(None, None, None).extract("x")


def test_public_names_exported():
    assert hasattr(reflect_mcp, "ReflectivePipeline")
    assert hasattr(reflect_mcp, "build_default_pipeline")
