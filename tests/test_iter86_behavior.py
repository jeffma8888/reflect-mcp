"""Black-box behavioral tests for iter-86: Generator and Critic share one
parse-and-retry loop (`reflect_mcp.parsing.complete_parsed`), with retry
behavior, logs and errors unchanged.

Covers the 6 Expected Behaviors from pm.md:
  1. Generator retry after one bad reply: result, calls, prompts, log, sleep.
  2. Critic retry after one bad reply: result, calls, byte-exact user prompt,
     log, sleep, and no generate.retry record.
  3. Exhaustion keeps the error contract (message prefix, __cause__ identity),
     including schema-invalid JSON (ValidationError cause).
  4. max_retries=0 -> one call, no retry; provider exception propagates
     unwrapped after one call.
  5. One policy, one place (structural; red on base dcda81c by design).
  6. Nothing else moves: frozen eval digest, error import paths, __all__,
     no new dependency, mcp imported only by server.py, scoped test edits.

EB1-EB4 are behavior-preservation checks and are green on BOTH the base tree
and the working tree by design: sleeps are recorded by patching the stdlib
``time.sleep`` (one module object shared by every importer), so the same
probe runs on a tree whose ``parsing`` module has no ``time`` attribute.
EB5 is the discriminator.

Isolation contract: only public interfaces are driven (classes, module
attributes, `ast` structure checks the spec names, git metadata, the eval
script).
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import reflect_mcp
import reflect_mcp.parsing as parsing
from reflect_mcp import critic as critic_mod
from reflect_mcp import generator as generator_mod
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.parsing import ParseError, parse_json_object
from reflect_mcp.schema import CritiqueResult, InvoiceData
from tests._golden import FROZEN_EVAL_SHA256, frozen_eval_sha256
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

REPO = Path(__file__).resolve().parent.parent
SRC = Path(reflect_mcp.__file__).resolve().parent
BASE_SHA = "dcda81c"

BAD = "not json"
VALID_INV = json.dumps(CORRECT_EXTRACTION)
VALID_CRIT = critique_all(0.95, 0.9, summary="all good")
BAD_SCHEMA_CRIT = json.dumps({"field_verdicts": "nope", "overall_confidence": 0.9})
BAD_SCHEMA_INV = json.dumps({"vendor_name": "ACME", "total": "not-a-number"})

GEN_USER = "Extract the invoice fields from this document:\n\n" + SOURCE

EXPECTED_ALL = [
    "InvoiceData", "CritiqueResult", "FieldVerdict", "FlaggedField",
    "ProvenanceSpan", "ExtractionResult", "EXTRACTION_FIELDS",
    "ReflectivePipeline", "build_default_pipeline", "Generator",
    "GeneratorError", "Critic", "CriticError", "GateConfig", "decide",
    "__version__",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RecordingClient:
    """List-backed LLMClient that records every complete() call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def stage(self, name: str) -> list[dict]:
        out = []
        for r in self.records:
            f = dict(getattr(r, "fields", {}) or {})
            if f.get("stage") == name:
                out.append(f)
        return out


@pytest.fixture
def logcap():
    lg = logging.getLogger("reflect_mcp")
    h = _Capture()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


@pytest.fixture
def sleeps(monkeypatch):
    rec: list = []
    monkeypatch.setattr(time, "sleep", lambda *a, **_k: rec.append(a[0] if a else None))
    return rec


def _extraction() -> InvoiceData:
    return InvoiceData.model_validate(CORRECT_EXTRACTION)


def _crit_user(text: str, extraction: InvoiceData) -> str:
    return (
        "SOURCE DOCUMENT:\n" + text
        + "\n\nCANDIDATE EXTRACTION (JSON):\n"
        + json.dumps(extraction.model_dump(), sort_keys=True)
        + "\n\nVerify each field against the source and return your critique JSON."
    )


def _parse_error_str(raw: str) -> str:
    with pytest.raises(ParseError) as ei:
        parse_json_object(raw)
    return str(ei.value)


def _src_tree(name: str) -> ast.Module:
    return ast.parse((SRC / name).read_text(encoding="utf-8"))


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _require_base_commit() -> None:
    r = _git("cat-file", "-e", BASE_SHA + "^{commit}")
    if r.returncode != 0:
        pytest.skip("base commit unavailable (archive copy)")


# ---------------------------------------------------------------------------
# EB1: Generator retry, one bad reply
# ---------------------------------------------------------------------------

def test_eb1_generator_retry_one_bad_reply(logcap, sleeps):
    client = _RecordingClient([BAD, VALID_INV])
    out = Generator(client, max_retries=1).generate(SOURCE, trace_id="t1")
    assert out == InvoiceData.model_validate(json.loads(VALID_INV))
    assert len(client.calls) == 2
    for call in client.calls:
        assert call["system"] == generator_mod.GENERATOR_SYSTEM
        assert call["user"] == GEN_USER
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    r = retries[0]
    assert r["trace_id"] == "t1"
    assert r["attempt"] == 1
    assert r["max_retries"] == 1
    assert r["error"] == _parse_error_str(BAD)
    assert sleeps == [0.5]
    assert logcap.stage("critique.retry") == []


# ---------------------------------------------------------------------------
# EB2: Critic retry, one bad reply
# ---------------------------------------------------------------------------

def test_eb2_critic_retry_one_bad_reply(logcap, sleeps):
    ext = _extraction()
    client = _RecordingClient([BAD, VALID_CRIT])
    out = Critic(client, max_retries=1).critique(SOURCE, ext, trace_id="t2")
    assert isinstance(out, CritiqueResult)
    assert out == CritiqueResult.model_validate(json.loads(VALID_CRIT))
    assert len(client.calls) == 2
    for call in client.calls:
        assert call["system"] == critic_mod.CRITIC_SYSTEM
        assert call["user"] == _crit_user(SOURCE, ext)
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    r = retries[0]
    assert r["trace_id"] == "t2"
    assert r["attempt"] == 1
    assert r["max_retries"] == 1
    assert r["error"] == _parse_error_str(BAD)
    assert sleeps == [0.5]
    assert logcap.stage("generate.retry") == []


def test_eb2_stages_keep_distinct_system_prompts():
    # Decorrelation: the shared helper must not merge the two stages' prompts.
    assert generator_mod.GENERATOR_SYSTEM != critic_mod.CRITIC_SYSTEM


# ---------------------------------------------------------------------------
# EB3: Exhaustion keeps the error contract
# ---------------------------------------------------------------------------

def test_eb3_generator_exhaustion(logcap, sleeps):
    client = _RecordingClient([BAD, BAD, BAD])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=2).generate(SOURCE, trace_id="t3")
    e = ei.value
    assert len(client.calls) == 3
    retries = logcap.stage("generate.retry")
    assert [r["attempt"] for r in retries] == [1, 2]
    assert all(r["max_retries"] == 2 and r["trace_id"] == "t3" for r in retries)
    assert sleeps == [0.5, 0.5]
    assert isinstance(e.__cause__, ParseError)
    assert str(e) == "Generator produced invalid extraction: " + str(e.__cause__)
    assert str(e.__cause__) == _parse_error_str(BAD)


def test_eb3_critic_exhaustion(logcap, sleeps):
    client = _RecordingClient([BAD, BAD, BAD])
    with pytest.raises(CriticError) as ei:
        Critic(client, max_retries=2).critique(SOURCE, _extraction(), trace_id="t4")
    e = ei.value
    assert len(client.calls) == 3
    retries = logcap.stage("critique.retry")
    assert [r["attempt"] for r in retries] == [1, 2]
    assert all(r["max_retries"] == 2 and r["trace_id"] == "t4" for r in retries)
    assert sleeps == [0.5, 0.5]
    assert isinstance(e.__cause__, ParseError)
    assert str(e) == "Critic produced invalid critique: " + str(e.__cause__)
    assert str(e.__cause__) == _parse_error_str(BAD)


def test_eb3_cause_is_the_last_error_instance(sleeps):
    # Each parse failure gets a distinct message; the stage error must chain
    # the LAST one, not the first.
    replies = ["bad-one", "bad-two", "bad-three"]
    client = _RecordingClient(replies)
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=2).generate(SOURCE)
    assert isinstance(ei.value.__cause__, ParseError)
    assert str(ei.value.__cause__) == _parse_error_str("bad-three")
    assert "bad-three" in str(ei.value)


def test_eb3_critic_schema_invalid_retried_validation_cause(logcap, sleeps):
    client = _RecordingClient([BAD_SCHEMA_CRIT, BAD_SCHEMA_CRIT])
    with pytest.raises(CriticError) as ei:
        Critic(client, max_retries=1).critique(SOURCE, _extraction(), trace_id="t5")
    assert len(client.calls) == 2
    assert len(logcap.stage("critique.retry")) == 1
    assert sleeps == [0.5]
    assert isinstance(ei.value.__cause__, ValidationError)
    assert str(ei.value) == "Critic produced invalid critique: " + str(ei.value.__cause__)


def test_eb3_critic_schema_invalid_then_valid_recovers(logcap, sleeps):
    client = _RecordingClient([BAD_SCHEMA_CRIT, VALID_CRIT])
    out = Critic(client, max_retries=1).critique(SOURCE, _extraction(), trace_id="t6")
    assert out == CritiqueResult.model_validate(json.loads(VALID_CRIT))
    assert len(client.calls) == 2
    assert len(logcap.stage("critique.retry")) == 1


def test_eb3_generator_schema_invalid_validation_cause(logcap, sleeps):
    client = _RecordingClient([BAD_SCHEMA_INV, BAD_SCHEMA_INV])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=1).generate(SOURCE, trace_id="t7")
    assert len(client.calls) == 2
    assert len(logcap.stage("generate.retry")) == 1
    assert sleeps == [0.5]
    assert isinstance(ei.value.__cause__, ValidationError)
    assert str(ei.value) == "Generator produced invalid extraction: " + str(ei.value.__cause__)


# ---------------------------------------------------------------------------
# EB4: No retry when retries are off, or when the provider fails
# ---------------------------------------------------------------------------

def test_eb4_generator_max_retries_zero(logcap, sleeps):
    client = _RecordingClient([BAD])
    with pytest.raises(GeneratorError) as ei:
        Generator(client, max_retries=0).generate(SOURCE)
    assert len(client.calls) == 1
    assert logcap.stage("generate.retry") == []
    assert sleeps == []
    assert isinstance(ei.value.__cause__, ParseError)


def test_eb4_critic_max_retries_zero(logcap, sleeps):
    client = _RecordingClient([BAD])
    with pytest.raises(CriticError) as ei:
        Critic(client, max_retries=0).critique(SOURCE, _extraction())
    assert len(client.calls) == 1
    assert logcap.stage("critique.retry") == []
    assert sleeps == []
    assert isinstance(ei.value.__cause__, ParseError)


def test_eb4_generator_provider_error_propagates_unwrapped(logcap, sleeps):
    boom = RuntimeError("boom")
    client = _RecordingClient([boom, VALID_INV])
    with pytest.raises(RuntimeError) as ei:
        Generator(client, max_retries=2).generate(SOURCE)
    assert ei.value is boom
    assert not isinstance(ei.value, GeneratorError)
    assert len(client.calls) == 1
    assert logcap.stage("generate.retry") == []
    assert sleeps == []


def test_eb4_critic_provider_error_propagates_unwrapped(logcap, sleeps):
    boom = RuntimeError("boom")
    client = _RecordingClient([boom, VALID_CRIT])
    with pytest.raises(RuntimeError) as ei:
        Critic(client, max_retries=2).critique(SOURCE, _extraction())
    assert ei.value is boom
    assert not isinstance(ei.value, CriticError)
    assert len(client.calls) == 1
    assert logcap.stage("critique.retry") == []
    assert sleeps == []


def test_eb4_provider_error_after_a_retry_propagates(logcap, sleeps):
    # A provider failure on the retry call also propagates unwrapped.
    boom = RuntimeError("boom")
    client = _RecordingClient([BAD, boom])
    with pytest.raises(RuntimeError) as ei:
        Generator(client, max_retries=2).generate(SOURCE)
    assert ei.value is boom
    assert len(client.calls) == 2
    assert len(logcap.stage("generate.retry")) == 1
    assert sleeps == [0.5]


# ---------------------------------------------------------------------------
# EB5: One policy, one place (HEAD-red)
# ---------------------------------------------------------------------------

def _assigned_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def test_eb5a_complete_parsed_exists_and_owns_delay():
    assert callable(getattr(parsing, "complete_parsed", None))
    assert getattr(parsing, "_RETRY_DELAY_SECONDS", None) == 0.5


def test_eb5b_delay_constant_assigned_only_in_parsing():
    owners = sorted(
        p.name for p in SRC.glob("*.py")
        if "_RETRY_DELAY_SECONDS" in _assigned_names(ast.parse(p.read_text(encoding="utf-8")))
    )
    assert owners == ["parsing.py"]


@pytest.mark.parametrize("module", ["generator.py", "critic.py"])
def test_eb5c_stage_modules_have_no_loop_and_no_time_import(module):
    tree = _src_tree(module)
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.AsyncFor, ast.While))]
    assert loops == [], f"{module} still has a loop at lines {[n.lineno for n in loops]}"
    time_imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name == "time" for a in n.names):
            time_imports.append(n.lineno)
        if isinstance(n, ast.ImportFrom) and n.module == "time":
            time_imports.append(n.lineno)
    assert time_imports == [], f"{module} imports time at lines {time_imports}"


def test_eb5d_generator_uses_shared_delay(monkeypatch, sleeps):
    assert hasattr(parsing, "_RETRY_DELAY_SECONDS")
    monkeypatch.setattr(parsing, "_RETRY_DELAY_SECONDS", 0.01)
    client = _RecordingClient([BAD, VALID_INV])
    Generator(client, max_retries=1).generate(SOURCE, trace_id="t1")
    assert sleeps == [0.01]


def test_eb5d_critic_uses_shared_delay(monkeypatch, sleeps):
    assert hasattr(parsing, "_RETRY_DELAY_SECONDS")
    monkeypatch.setattr(parsing, "_RETRY_DELAY_SECONDS", 0.01)
    client = _RecordingClient([BAD, VALID_CRIT])
    Critic(client, max_retries=1).critique(SOURCE, _extraction(), trace_id="t2")
    assert sleeps == [0.01]


# ---------------------------------------------------------------------------
# EB6: Nothing else moves
# ---------------------------------------------------------------------------

def test_eb6_frozen_eval_digest_unchanged():
    assert frozen_eval_sha256() == FROZEN_EVAL_SHA256


def test_eb6_error_import_paths_and_all_unchanged():
    assert generator_mod.GeneratorError is GeneratorError
    assert critic_mod.CriticError is CriticError
    assert issubclass(GeneratorError, Exception)
    assert issubclass(CriticError, Exception)
    assert list(reflect_mcp.__all__) == EXPECTED_ALL


def test_eb6_only_server_imports_mcp():
    importers = []
    for p in sorted(SRC.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import) and any(
                a.name == "mcp" or a.name.startswith("mcp.") for a in n.names
            ):
                importers.append(p.name)
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and (
                n.module == "mcp" or n.module.startswith("mcp.")
            ):
                importers.append(p.name)
    assert sorted(set(importers)) == ["server.py"]


def test_eb6_no_new_dependency():
    _require_base_commit()
    r = _git("diff", "--name-only", BASE_SHA, "--", "pyproject.toml", "uv.lock")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_eb6_existing_test_edits_scoped():
    _require_base_commit()
    r = _git("diff", "--name-only", BASE_SHA, "--", "tests/")
    assert r.returncode == 0, r.stderr
    changed = {line for line in r.stdout.splitlines() if line}
    assert changed <= {"tests/test_iter70_behavior.py", "tests/test_iter86_behavior.py"}, changed
    # The nosleep migration also swaps the two now-unused stage-module imports
    # for one `parsing` import (the spec counted only the fixture lines), so
    # pin the exact changed-line sets rather than a raw line count.
    d = _git("diff", "-U0", BASE_SHA, "--", "tests/test_iter70_behavior.py")
    assert d.returncode == 0, d.stderr
    added, removed = [], []
    for line in d.stdout.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            removed.append(line[1:].strip())
    assert added == [] or added == [
        "import reflect_mcp.parsing as parsing_mod",
        'monkeypatch.setattr(parsing_mod.time, "sleep", lambda *_a, **_k: None)',
    ], added
    assert set(removed) <= {
        "import reflect_mcp.critic as critic_mod",
        "import reflect_mcp.generator as generator_mod",
        'monkeypatch.setattr(generator_mod.time, "sleep", lambda *_a, **_k: None)',
        'monkeypatch.setattr(critic_mod.time, "sleep", lambda *_a, **_k: None)',
    }, removed
