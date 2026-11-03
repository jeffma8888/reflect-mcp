"""Black-box behavioral tests for iter-70: trace_id is unique per call and the
Generator/Critic retry log events carry the call's trace_id.

Covers the 11 Expected Behaviors from pm.md:
  1. new_trace_id() is unique in a tight loop and matches trace-<hex>-<hex>-<hex>.
  2. new_trace_id() is unique across 8 threads x 250 calls.
  3. 50 concurrent server.validate(heuristics_only=True) calls -> 50 distinct ids.
  4. 20 threads through the heuristics-only library pipeline -> 20 distinct ids.
  5. generate.retry record carries the trace_id passed to Generator.generate.
  6. critique.retry record carries the trace_id passed to Critic.critique.
  7. trace_id is optional (keyword, default "") on generate/critique.
  8. One extract() run: every record carries result.trace_id, all stages present.
  9. One validate() run: every record (incl. critique.retry) carries result.trace_id.
 10. Payloads unchanged: keysets, eval stdout digest vs HEAD, README demo prefix.
 11. Prose: README Observability bullet and logging_config docstring say
     "unique per call".

Isolation contract: only public interfaces are driven (module functions,
classes, the stub-MCP server import, the eval script, README text and the
module's runtime ``__doc__``). The eval stdout digest was measured on a
`git archive HEAD` (9bd522c, iter-69) copy so the test outlives the commit.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import re
import sys
import threading
import types
from pathlib import Path

import pytest

import reflect_mcp.critic as critic_mod
import reflect_mcp.generator as generator_mod
from reflect_mcp import logging_config
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline, build_default_pipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all
from tests._golden import frozen_eval

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# Measured on a `git archive HEAD` copy of 9bd522c (iter-69), stdout only.
HEAD_EVAL_STDOUT_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"

ID_RE = re.compile(r"^trace-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+$")

GEN_GOOD = json.dumps(CORRECT_EXTRACTION)
CRIT_GOOD = critique_all(0.95, 0.95)
BAD = "not json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def fields(self) -> list[dict]:
        return [dict(getattr(r, "fields", {}) or {}) for r in self.records]

    def stage(self, name: str) -> list[dict]:
        return [f for f in self.fields() if f.get("stage") == name]


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
def nosleep(monkeypatch):
    monkeypatch.setattr(generator_mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(critic_mod.time, "sleep", lambda *_a, **_k: None)


class _StubFastMCP:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def tool(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    def run(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("stub FastMCP.run must not be called")


@pytest.fixture
def server(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    mcp_server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _StubFastMCP
    mcp_mod.server = mcp_server_mod
    mcp_server_mod.fastmcp = fastmcp_mod
    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    sys.modules.pop("reflect_mcp.server", None)
    mod = importlib.import_module("reflect_mcp.server")
    try:
        yield mod
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _run_threads(n: int, fn) -> list:
    out: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(n)
    errors: list = []

    def worker():
        try:
            barrier.wait()
            vals = fn()
            with lock:
                out.extend(vals)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert errors == []
    return out


# ---------------------------------------------------------------------------
# EB1 / EB2: new_trace_id uniqueness and format
# ---------------------------------------------------------------------------

def test_eb1_tight_loop_unique_and_format():
    ids = [logging_config.new_trace_id() for _ in range(1000)]
    assert len(set(ids)) == 1000
    bad = [i for i in ids if not (i.startswith("trace-") and ID_RE.fullmatch(i))]
    assert bad == []


def test_eb1_returns_str():
    assert isinstance(logging_config.new_trace_id(), str)


def test_eb2_unique_across_threads():
    ids = _run_threads(8, lambda: [logging_config.new_trace_id() for _ in range(250)])
    assert len(ids) == 2000
    assert len(set(ids)) == 2000
    assert all(ID_RE.fullmatch(i) for i in ids)


# ---------------------------------------------------------------------------
# EB3: 50 parallel MCP validate calls -> 50 distinct trace ids
# ---------------------------------------------------------------------------

def test_eb3_parallel_mcp_validate_distinct_ids(server):
    async def main():
        return await asyncio.gather(
            *[server.validate(SOURCE, dict(CORRECT_EXTRACTION), heuristics_only=True) for _ in range(50)]
        )

    results = asyncio.run(main())
    assert len(results) == 50
    assert all(isinstance(r, dict) for r in results)
    ids = [r.get("trace_id") for r in results]
    assert all(isinstance(i, str) and i for i in ids), ids[:3]
    assert len(set(ids)) == 50


# ---------------------------------------------------------------------------
# EB4: library pipeline under threads
# ---------------------------------------------------------------------------

def test_eb4_library_pipeline_threads_distinct_ids():
    pipe = build_default_pipeline(generator_enabled=False, critic_enabled=False)

    def one():
        r = pipe.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION))
        return [r.trace_id]

    ids = _run_threads(20, one)
    assert len(ids) == 20
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 20


def test_eb4_bare_pipeline_threads_distinct_ids():
    pipe = ReflectivePipeline(None, None)
    ids = _run_threads(20, lambda: [pipe.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION)).trace_id])
    assert len(set(ids)) == 20


# ---------------------------------------------------------------------------
# EB5 / EB6 / EB7: retry events carry the id; signatures backward compatible
# ---------------------------------------------------------------------------

def test_eb5_generate_retry_carries_trace_id(logcap, nosleep):
    gen = Generator(ScriptedLLMClient([BAD, GEN_GOOD]), max_retries=1)
    out = gen.generate(SOURCE, trace_id="T-gen")
    assert isinstance(out, InvoiceData)
    assert out.total == 108.5
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    assert retries[0]["trace_id"] == "T-gen"


def test_eb6_critique_retry_carries_trace_id(logcap, nosleep):
    crit = Critic(ScriptedLLMClient([BAD, CRIT_GOOD]), max_retries=1)
    out = crit.critique(SOURCE, InvoiceData(total=108.5), trace_id="T-crit")
    assert type(out).__name__ == "CritiqueResult"
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    assert retries[0]["trace_id"] == "T-crit"


def test_eb7_signatures_keyword_default_empty():
    for fn in (Generator.generate, Critic.critique):
        p = inspect.signature(fn).parameters["trace_id"]
        assert p.default == ""
        assert p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def test_eb7_generate_without_trace_id_logs_empty(logcap, nosleep):
    gen = Generator(ScriptedLLMClient([BAD, GEN_GOOD]), max_retries=1)
    out = gen.generate(SOURCE)
    assert isinstance(out, InvoiceData)
    retries = logcap.stage("generate.retry")
    assert len(retries) == 1
    assert retries[0]["trace_id"] == ""


def test_eb7_critique_without_trace_id_logs_empty(logcap, nosleep):
    crit = Critic(ScriptedLLMClient([BAD, CRIT_GOOD]), max_retries=1)
    out = crit.critique(SOURCE, InvoiceData(total=108.5))
    assert type(out).__name__ == "CritiqueResult"
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    assert retries[0]["trace_id"] == ""


def test_eb7_happy_path_without_trace_id():
    assert isinstance(Generator(ScriptedLLMClient([GEN_GOOD])).generate(SOURCE), InvoiceData)
    res = Critic(ScriptedLLMClient([CRIT_GOOD])).critique(SOURCE, InvoiceData(**CORRECT_EXTRACTION))
    assert type(res).__name__ == "CritiqueResult"


# ---------------------------------------------------------------------------
# EB8 / EB9: one call is greppable end to end
# ---------------------------------------------------------------------------

EXTRACT_STAGES = {
    "generate.start", "generate.retry", "generate.done", "heuristics.done",
    "critique.retry", "critique.done", "decide.done",
}


def test_eb8_extract_greppable_end_to_end(logcap, nosleep):
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([BAD, GEN_GOOD])),
        Critic(ScriptedLLMClient([BAD, CRIT_GOOD])),
    )
    result = pipe.extract(SOURCE)
    assert isinstance(result.trace_id, str) and result.trace_id
    recs = logcap.fields()
    assert recs, "no log records captured"
    ids = {f.get("trace_id") for f in recs}
    assert ids == {result.trace_id}
    stages = {f.get("stage") for f in recs}
    assert EXTRACT_STAGES <= stages, sorted(EXTRACT_STAGES - stages)


def test_eb9_validate_greppable_end_to_end(logcap, nosleep):
    pipe = ReflectivePipeline(None, Critic(ScriptedLLMClient([BAD, CRIT_GOOD])))
    result = pipe.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION))
    assert isinstance(result.trace_id, str) and result.trace_id
    recs = logcap.fields()
    assert recs
    assert {f.get("trace_id") for f in recs} == {result.trace_id}
    retries = logcap.stage("critique.retry")
    assert len(retries) == 1
    assert retries[0]["trace_id"] == result.trace_id


def test_eb8_two_extracts_get_different_ids(logcap):
    pipe = ReflectivePipeline(
        Generator(ScriptedLLMClient([GEN_GOOD, GEN_GOOD])),
        Critic(ScriptedLLMClient([CRIT_GOOD, CRIT_GOOD])),
    )
    a = pipe.extract(SOURCE)
    b = pipe.extract(SOURCE)
    assert a.trace_id != b.trace_id


# ---------------------------------------------------------------------------
# EB10: payloads unchanged
# ---------------------------------------------------------------------------

def test_eb10_eval_stdout_byte_identical_to_head():
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_EVAL_STDOUT_SHA256


def test_eb10_result_trace_id_still_prefixed():
    r = ReflectivePipeline(None, None).validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION))
    assert r.trace_id.startswith("trace-")
    assert ID_RE.fullmatch(r.trace_id)


# ---------------------------------------------------------------------------
# EB11: prose pinned
# ---------------------------------------------------------------------------

def _observability_bullet() -> str:
    lines = [ln for ln in README.read_text().splitlines() if ln.startswith("- **Observability.**")]
    assert len(lines) == 1
    return lines[0]


def test_eb11_readme_observability_bullet():
    b = _observability_bullet()
    assert "shared `trace_id`" in b
    assert "greppable end to end" in b
    assert "unique per call" in b


def test_eb11_readme_bullet_in_design_decisions_section():
    text = README.read_text()
    sec10 = text.split("## 10. Design decisions", 1)[1].split("\n## ", 1)[0]
    assert "- **Observability.**" in sec10
    assert "unique per call" in sec10


def test_eb11_readme_section7_wording_unchanged():
    assert "Three roles, one shared trace id" in README.read_text()


def test_eb11_logging_config_docstring():
    doc = logging_config.__doc__ or ""
    assert "unique per call" in doc
