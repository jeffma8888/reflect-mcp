"""Black-box behavioral tests for iter-69: MCP tools run on a worker thread.

Covers the 9 Expected Behaviors from pm.md:
  1. server.extract / server.validate are coroutine functions; exactly two tools
     registered; parameter names and defaults unchanged.
  2. Error envelopes equal the tools.* result (empty input).
  3. heuristics_only validate result equals tools.validate_tool (minus trace_id).
  4. tools.* looked up at call time; args passed through positionally, return
     value passed back unchanged.
  5. The tool body runs off the event-loop thread.
  6. Parallel calls overlap (< 0.6 s for 4 x 0.2 s), keep input order, and the
     loop stays responsive (>= 5 heartbeat ticks).
  7. validate docstring names the envelope form; extract docstring unchanged.
  8. README section 4 has the **Parallel calls.** paragraph in the right place.
  9. No new dependency; server.py does not import anyio; eval stdout unchanged.

Isolation contract: server.py is imported with a stub MCP SDK installed in
sys.modules (no real SDK, no network); docstrings are read via AST from the
source file; HEAD-unchanged invariants are pinned by sha256 constants measured
on a `git archive HEAD` (iter-68) copy, so the tests outlive the commit.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from reflect_mcp import tools

REPO = Path(__file__).resolve().parent.parent
SERVER_PY = REPO / "src" / "reflect_mcp" / "server.py"
README = REPO / "README.md"

# Measured on a `git archive HEAD` copy of iter-68 (commit before this change).
HEAD_EXTRACT_DOC_SHA256 = "1c4b6a7ce36fd548e7cafd2477cb3adb66a6581c62cf02ad2ea81793d5208c62"
HEAD_PYPROJECT_SHA256 = "a477e3111ef82955d91d76f2024fd1f668beb8cba41591205b127bcc629ef9a6"
HEAD_EVAL_STDOUT_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"

TEXT = "Invoice INV-1\nTotal: $10.00"
DATA = {"invoice_number": "INV-1", "total": 10.0}


# ---------------------------------------------------------------------------
# Stub SDK fixture
# ---------------------------------------------------------------------------

class _StubFastMCP:
    instances: list["_StubFastMCP"] = []

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.tools: list = []
        _StubFastMCP.instances.append(self)

    def tool(self, *args, **kwargs):
        # Support both @mcp.tool() and bare @mcp.tool.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            fn = args[0]
            self.tools.append(fn)
            return fn

        def deco(fn):
            self.tools.append(fn)
            return fn

        return deco

    def run(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("stub FastMCP.run must not be called in tests")


@pytest.fixture
def server(monkeypatch):
    _StubFastMCP.instances = []
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
    import importlib

    mod = importlib.import_module("reflect_mcp.server")
    try:
        yield mod
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _recorded_tools():
    names = []
    for inst in _StubFastMCP.instances:
        names.extend(fn.__name__ for fn in inst.tools)
    return names


def _params(fn):
    return [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]


# ---------------------------------------------------------------------------
# EB1: coroutine functions, exactly two tools, signatures unchanged
# ---------------------------------------------------------------------------

class TestEB1Coroutines:
    def test_extract_is_coroutine_function(self, server):
        assert inspect.iscoroutinefunction(server.extract)

    def test_validate_is_coroutine_function(self, server):
        assert inspect.iscoroutinefunction(server.validate)

    def test_exactly_two_tools_recorded(self, server):
        assert sorted(_recorded_tools()) == ["extract", "validate"]

    def test_registered_tools_are_the_coroutines(self, server):
        recorded = [fn for inst in _StubFastMCP.instances for fn in inst.tools]
        assert len(recorded) == 2
        assert all(inspect.iscoroutinefunction(fn) for fn in recorded)

    def test_extract_signature(self, server):
        assert _params(server.extract) == [
            ("text", inspect.Parameter.empty),
            ("field_threshold", 0.70),
            ("overall_threshold", 0.75),
            ("heuristics_only", False),
        ]

    def test_validate_signature(self, server):
        assert _params(server.validate) == [
            ("text", inspect.Parameter.empty),
            ("data", inspect.Parameter.empty),
            ("field_threshold", 0.70),
            ("overall_threshold", 0.75),
            ("heuristics_only", False),
        ]


# ---------------------------------------------------------------------------
# EB2: error envelopes match tools.*
# ---------------------------------------------------------------------------

def _strip_trace(d):
    d = dict(d)
    d.pop("trace_id", None)
    return d


class TestEB2ErrorParity:
    def test_extract_empty_matches_tools(self, server):
        got = asyncio.run(server.extract(""))
        want = tools.extract_tool("")
        assert isinstance(got, dict)
        assert got["status"] == "error"
        assert got["error_message"] == want["error_message"]
        assert _strip_trace(got) == _strip_trace(want)

    def test_validate_empty_matches_tools(self, server):
        got = asyncio.run(server.validate("", {}))
        want = tools.validate_tool("", {})
        assert isinstance(got, dict)
        assert got["status"] == "error"
        assert got["error_message"] == want["error_message"]
        assert _strip_trace(got) == _strip_trace(want)

    def test_calling_without_await_returns_coroutine(self, server):
        coro = server.extract("")
        try:
            assert inspect.iscoroutine(coro)
        finally:
            coro.close()


# ---------------------------------------------------------------------------
# EB3: heuristics-only validate parity
# ---------------------------------------------------------------------------

class TestEB3HeuristicsOnlyParity:
    def test_validate_heuristics_only_matches_tools(self, server):
        got = asyncio.run(server.validate(TEXT, dict(DATA), heuristics_only=True))
        want = tools.validate_tool(TEXT, dict(DATA), 0.70, 0.75, True)
        assert isinstance(got, dict) and "status" in got
        assert got["status"] != "error", got
        assert got.pop("trace_id", None) is not None
        want.pop("trace_id", None)
        assert got == want


# ---------------------------------------------------------------------------
# EB4: call-time lookup, positional pass-through
# ---------------------------------------------------------------------------

class TestEB4CallTimeLookup:
    def test_extract_spy(self, server, monkeypatch):
        calls = []
        sentinel = {"status": "ok", "marker": object()}

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        monkeypatch.setattr(tools, "extract_tool", spy)
        got = asyncio.run(server.extract("doc", 0.9, 0.95, True))
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("doc", 0.9, 0.95, True)
        assert kwargs == {}
        assert got is sentinel

    def test_validate_spy(self, server, monkeypatch):
        calls = []
        sentinel = {"status": "ok", "marker": object()}

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        monkeypatch.setattr(tools, "validate_tool", spy)
        got = asyncio.run(server.validate("doc", {"total": 1.0}, 0.8, 0.85, True))
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("doc", {"total": 1.0}, 0.8, 0.85, True)
        assert kwargs == {}
        assert got is sentinel

    def test_extract_defaults_passed_positionally(self, server, monkeypatch):
        calls = []
        monkeypatch.setattr(tools, "extract_tool",
                            lambda *a, **k: calls.append((a, k)) or {"status": "ok"})
        asyncio.run(server.extract("doc"))
        assert calls == [(("doc", 0.70, 0.75, False), {})]


# ---------------------------------------------------------------------------
# EB5: off the event-loop thread
# ---------------------------------------------------------------------------

class TestEB5OffLoop:
    def test_extract_runs_on_worker_thread(self, server, monkeypatch):
        seen = []
        monkeypatch.setattr(tools, "extract_tool",
                            lambda *a, **k: seen.append(threading.get_ident()) or {"status": "ok"})

        async def main():
            loop_ident = threading.get_ident()
            await server.extract("doc")
            return loop_ident

        loop_ident = asyncio.run(main())
        assert len(seen) == 1
        assert seen[0] != loop_ident

    def test_validate_runs_on_worker_thread(self, server, monkeypatch):
        seen = []
        monkeypatch.setattr(tools, "validate_tool",
                            lambda *a, **k: seen.append(threading.get_ident()) or {"status": "ok"})

        async def main():
            loop_ident = threading.get_ident()
            await server.validate("doc", {})
            return loop_ident

        loop_ident = asyncio.run(main())
        assert len(seen) == 1
        assert seen[0] != loop_ident


# ---------------------------------------------------------------------------
# EB6: overlap + responsiveness
# ---------------------------------------------------------------------------

def _sleepy(text, *args, **kwargs):
    time.sleep(0.2)
    return {"status": "ok", "text": text}


def _run_overlap(server, fn_name):
    async def main():
        ticks = 0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.01)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # let the heartbeat start
        t0 = time.perf_counter()
        if fn_name == "extract":
            results = await asyncio.gather(*(server.extract(f"doc{i}") for i in range(4)))
        else:
            results = await asyncio.gather(*(server.validate(f"doc{i}", {}) for i in range(4)))
        elapsed = time.perf_counter() - t0
        ticks_during = ticks
        stop.set()
        await hb
        return results, elapsed, ticks_during

    return asyncio.run(main())


class TestEB6Overlap:
    def test_extract_gather_overlaps(self, server, monkeypatch):
        monkeypatch.setattr(tools, "extract_tool", _sleepy)
        results, elapsed, ticks = _run_overlap(server, "extract")
        assert elapsed < 0.6, f"4 x 0.2 s calls took {elapsed:.3f}s (serial >= 0.8 s)"

    def test_extract_gather_preserves_order(self, server, monkeypatch):
        monkeypatch.setattr(tools, "extract_tool", _sleepy)
        results, _, _ = _run_overlap(server, "extract")
        assert len(results) == 4
        assert [r["text"] for r in results] == ["doc0", "doc1", "doc2", "doc3"]
        assert all(r["status"] == "ok" for r in results)

    def test_extract_gather_keeps_loop_responsive(self, server, monkeypatch):
        monkeypatch.setattr(tools, "extract_tool", _sleepy)
        _, _, ticks = _run_overlap(server, "extract")
        assert ticks >= 5, f"heartbeat ran only {ticks} ticks during the gather"

    def test_validate_gather_overlaps(self, server, monkeypatch):
        monkeypatch.setattr(tools, "validate_tool", _sleepy)
        results, elapsed, ticks = _run_overlap(server, "validate")
        assert len(results) == 4
        assert [r["text"] for r in results] == ["doc0", "doc1", "doc2", "doc3"]
        assert elapsed < 0.6, f"4 x 0.2 s calls took {elapsed:.3f}s"
        assert ticks >= 5, f"heartbeat ran only {ticks} ticks"


# ---------------------------------------------------------------------------
# EB7: docstrings via AST
# ---------------------------------------------------------------------------

def _docstring(name):
    tree = ast.parse(SERVER_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            doc = ast.get_docstring(node)
            assert doc is not None, f"{name}() must have a docstring"
            return doc
    raise AssertionError(f"{name}() not found in server.py")


def _data_param_entry(doc):
    """Return the `- data:` bullet plus its continuation lines."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"(\s*)-\s+data\s*:", line)
        if m:
            indent = len(m.group(1))
            out = [line]
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    break
                if re.match(r"\s*-\s+\w+\s*:", nxt) and len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                out.append(nxt)
            return " ".join(s.strip() for s in out)
    raise AssertionError("validate docstring has no `- data:` entry")


class TestEB7Docstrings:
    def test_validate_doc_drops_invoicedata_shape(self):
        assert "InvoiceData shape" not in _docstring("validate")

    def test_validate_data_entry_names_envelope(self):
        entry = _data_param_entry(_docstring("validate"))
        assert "stored reflect-mcp" in entry
        assert "result envelope" in entry

    def test_validate_data_entry_says_data_object_verified(self):
        entry = _data_param_entry(_docstring("validate"))
        assert re.search(r"`?data`?\s+object is what gets verified", entry), entry

    def test_extract_doc_byte_identical_to_head(self):
        doc = _docstring("extract")
        assert hashlib.sha256(doc.encode()).hexdigest() == HEAD_EXTRACT_DOC_SHA256


# ---------------------------------------------------------------------------
# EB8: README section 4 paragraph
# ---------------------------------------------------------------------------

def _section4():
    text = README.read_text()
    m = re.search(r"^## 4\..*?(?=^## 5\.)", text, re.S | re.M)
    assert m, "README section 4 not found"
    return m.group(0)


def _paragraphs(section):
    return [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]


class TestEB8Readme:
    def test_parallel_calls_paragraph_present(self):
        paras = [p for p in _paragraphs(_section4()) if p.startswith("**Parallel calls.**")]
        assert len(paras) == 1

    def test_parallel_calls_paragraph_content(self):
        para = next(p for p in _paragraphs(_section4()) if p.startswith("**Parallel calls.**"))
        low = para.lower()
        assert "both tools" in low
        assert "`async`" in para or "async" in low
        assert "worker thread" in low
        assert "parallel" in low and "overlap" in low

    def test_parallel_calls_paragraph_position(self):
        paras = _paragraphs(_section4())
        idx = [i for i, p in enumerate(paras) if p.startswith("**Parallel calls.**")]
        assert len(idx) == 1
        i = idx[0]
        assert paras[i - 1].startswith("The tools never raise")
        assert paras[i + 1].startswith("**Operator threshold floor.**")


# ---------------------------------------------------------------------------
# EB9: no new dependency, eval unchanged
# ---------------------------------------------------------------------------

class TestEB9NoNewDependency:
    def test_pyproject_unchanged(self):
        digest = hashlib.sha256((REPO / "pyproject.toml").read_bytes()).hexdigest()
        assert digest == HEAD_PYPROJECT_SHA256

    def test_server_does_not_import_anyio(self):
        tree = ast.parse(SERVER_PY.read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert "anyio" not in mods
        assert "asyncio" in mods

    def test_only_server_imports_mcp(self):
        offenders = []
        for py in (REPO / "src" / "reflect_mcp").rglob("*.py"):
            if py.name == "server.py":
                continue
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "mcp" or n.startswith("mcp.") for n in names):
                    offenders.append(py.name)
        assert offenders == []

    def test_eval_stdout_byte_identical_to_head(self):
        proc = subprocess.run(
            [sys.executable, "eval/run_eval.py"],
            capture_output=True, cwd=REPO, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        assert proc.stdout, "eval printed nothing"
        assert hashlib.sha256(proc.stdout).hexdigest() == HEAD_EVAL_STDOUT_SHA256
