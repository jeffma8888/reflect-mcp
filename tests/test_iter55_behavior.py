"""Black-box behavioral tests for iter-55: MCP tool bodies reach CLI parity.

Covers the 12 Expected Behaviors from pm.md:
  1. ``reflect_mcp.tools`` imports without the MCP SDK.
  2. Successful extract returns ``pipeline.extract(text).model_dump()``.
  3. Thresholds forwarded to the factory as a GateConfig (both tools).
  4. ``heuristics_only`` on extract -> ``critic_enabled=False``.
  5. ``heuristics_only`` on validate -> ``critic_enabled=False``, Generator unused.
  6. Pipeline / factory failure -> error envelope (extract).
  7. Pipeline / factory failure -> error envelope (validate).
  8. Invalid ``data`` -> error envelope, factory never called.
  9. Empty input -> error envelope, factory never called.
 10. Invalid thresholds -> error envelope, factory never called.
 11. server.py (read via AST) declares identical params/defaults and delegates.
 12. README §4 + extract docstring document heuristics_only and the error status.

Isolation contract: tests drive ``reflect_mcp.tools`` through its public
functions with a monkeypatched ``reflect_mcp.pipeline.build_default_pipeline``
(pipelines built over ``ScriptedLLMClient``). server.py is only parsed with
``ast`` as the spec's behavior 11 mandates; it is never imported.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "reflect_mcp"
SERVER = PKG / "server.py"
README = REPO / "README.md"

RESULT_KEYS = {
    "status",
    "data",
    "field_confidence",
    "overall_confidence",
    "flagged_fields",
    "field_provenance",
    "critic_summary",
    "trace_id",
    "notes",
}
GARBAGE = "this is definitely not json <<<"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RaisingClient:
    """LLM client whose every call raises (simulates a provider failure)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list = []

    def complete(self, *args, **kwargs):  # noqa: D401
        self.calls.append((args, kwargs))
        raise self.exc


class Recorder:
    """Recording pipeline factory; patches reflect_mcp.pipeline at call time."""

    def __init__(self, gen_responses=None, critic_responses=None,
                 critic_client=None, max_retries=0):
        self.gen_responses = gen_responses
        self.critic_responses = critic_responses
        self.critic_client_override = critic_client
        self.max_retries = max_retries
        self.calls: list[tuple] = []
        self.gen_clients: list[ScriptedLLMClient] = []
        self.critic_clients: list = []

    def build(self):
        """Build a fresh pipeline identical to what the factory returns."""
        gen_client = ScriptedLLMClient(list(self.gen_responses or []))
        self.gen_clients.append(gen_client)
        gen = Generator(gen_client, max_retries=self.max_retries)
        return gen, gen_client

    def __call__(self, gate_config=None, **kwargs):
        self.calls.append((gate_config, kwargs))
        gen, _ = self.build()
        critic = None
        if kwargs.get("critic_enabled", True):
            if self.critic_client_override is not None:
                cclient = self.critic_client_override
            else:
                cclient = ScriptedLLMClient(list(self.critic_responses or []))
            self.critic_clients.append(cclient)
            critic = Critic(cclient)
        return ReflectivePipeline(gen, critic, gate_config=gate_config)


def _install(monkeypatch, recorder):
    monkeypatch.setattr(pipeline_module, "build_default_pipeline", recorder)
    return recorder


def _happy(monkeypatch):
    return _install(
        monkeypatch,
        Recorder(
            gen_responses=[json.dumps(CORRECT_EXTRACTION)],
            critic_responses=[critique_all(0.95, 0.92, summary="looks good")],
        ),
    )


def _assert_envelope(result):
    assert isinstance(result, dict)
    assert set(result) == {"status", "error_message"}, result
    assert result["status"] == "error"
    assert isinstance(result["error_message"], str) and result["error_message"]


def _critic_enabled(call_kwargs) -> bool:
    return call_kwargs.get("critic_enabled", True)


def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# EB-1: import without MCP SDK
# ---------------------------------------------------------------------------

class TestEB1ImportWithoutMcp:
    def test_fresh_interpreter_import_does_not_load_mcp(self):
        code = (
            "import sys, reflect_mcp.tools as t;"
            "print(callable(t.extract_tool), callable(t.validate_tool),"
            " 'mcp' in sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["True", "True", "False"]

    def test_import_survives_mcp_being_unimportable(self):
        # Even if mcp were present, blocking it must not break the import.
        code = (
            "import sys; sys.modules['mcp'] = None;"
            "sys.modules['mcp.server'] = None; sys.modules['mcp.server.fastmcp'] = None;"
            "import reflect_mcp.tools as t; print('ok', t.extract_tool.__name__)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "ok extract_tool"

    def test_signatures_match_spec(self):
        e = inspect.signature(tools.extract_tool).parameters
        v = inspect.signature(tools.validate_tool).parameters
        assert list(e) == ["text", "field_threshold", "overall_threshold", "heuristics_only"]
        assert list(v) == ["text", "data", "field_threshold", "overall_threshold", "heuristics_only"]
        for params in (e, v):
            assert params["field_threshold"].default == 0.70
            assert params["overall_threshold"].default == 0.75
            assert params["heuristics_only"].default is False


# ---------------------------------------------------------------------------
# EB-2: successful extract
# ---------------------------------------------------------------------------

class TestEB2SuccessfulExtract:
    def test_returns_model_dump_of_pipeline_extract(self, monkeypatch):
        rec = _happy(monkeypatch)
        result = tools.extract_tool(SOURCE)
        # Reference: same scripted pipeline driven directly.
        ref = Recorder(
            gen_responses=[json.dumps(CORRECT_EXTRACTION)],
            critic_responses=[critique_all(0.95, 0.92, summary="looks good")],
        )()
        expected = ref.extract(SOURCE).model_dump()
        assert set(result) == RESULT_KEYS
        assert result["status"] in ("ok", "needs_review")
        result_cmp = {k: v for k, v in result.items() if k != "trace_id"}
        expected_cmp = {k: v for k, v in expected.items() if k != "trace_id"}
        assert result_cmp == expected_cmp
        assert isinstance(result["trace_id"], str) and result["trace_id"]
        assert len(rec.calls) == 1

    def test_happy_path_is_ok_with_correct_data(self, monkeypatch):
        _happy(monkeypatch)
        result = tools.extract_tool(SOURCE)
        assert result["status"] == "ok"
        assert result["data"] == CORRECT_EXTRACTION
        assert result["critic_summary"] == "looks good"
        assert result["flagged_fields"] == []

    def test_result_is_json_serializable(self, monkeypatch):
        _happy(monkeypatch)
        result = tools.extract_tool(SOURCE)
        assert json.loads(json.dumps(result)) == result

    def test_factory_resolved_at_call_time(self, monkeypatch):
        rec1 = _happy(monkeypatch)
        tools.extract_tool(SOURCE)
        rec2 = _happy(monkeypatch)
        tools.extract_tool(SOURCE)
        assert len(rec1.calls) == 1 and len(rec2.calls) == 1


# ---------------------------------------------------------------------------
# EB-3: thresholds forwarded
# ---------------------------------------------------------------------------

class TestEB3ThresholdsForwarded:
    def test_extract_forwards_gate_config(self, monkeypatch):
        rec = _happy(monkeypatch)
        tools.extract_tool(SOURCE, field_threshold=0.9, overall_threshold=0.8)
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == GateConfig(field_threshold=0.9, overall_threshold=0.8)

    def test_validate_forwards_gate_config(self, monkeypatch):
        rec = _happy(monkeypatch)
        tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION),
                            field_threshold=0.9, overall_threshold=0.8)
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == GateConfig(field_threshold=0.9, overall_threshold=0.8)

    def test_default_thresholds_forwarded(self, monkeypatch):
        rec = _happy(monkeypatch)
        tools.extract_tool(SOURCE)
        gc = rec.calls[0][0]
        assert gc == GateConfig(field_threshold=0.70, overall_threshold=0.75)

    def test_forwarded_threshold_changes_outcome(self, monkeypatch):
        # Critic 0.95 per field passes 0.70 but fails a 0.99 field threshold.
        _happy(monkeypatch)
        result = tools.extract_tool(SOURCE, field_threshold=0.99)
        assert result["status"] == "needs_review"
        assert result["flagged_fields"]


# ---------------------------------------------------------------------------
# EB-4: heuristics_only on extract
# ---------------------------------------------------------------------------

class TestEB4HeuristicsOnlyExtract:
    def test_true_passes_critic_disabled(self, monkeypatch):
        rec = _happy(monkeypatch)
        tools.extract_tool(SOURCE, heuristics_only=True)
        assert rec.calls[0][1].get("critic_enabled") is False

    def test_default_keeps_critic_enabled(self, monkeypatch):
        rec = _happy(monkeypatch)
        tools.extract_tool(SOURCE)
        assert _critic_enabled(rec.calls[0][1]) is True

    def test_generator_only_script_gives_normal_result(self, monkeypatch):
        rec = _install(monkeypatch, Recorder(gen_responses=[json.dumps(CORRECT_EXTRACTION)]))
        result = tools.extract_tool(SOURCE, heuristics_only=True)
        assert result.get("status") in ("ok", "needs_review"), result
        assert set(result) == RESULT_KEYS
        assert result["critic_summary"] == ""
        assert result["data"] == CORRECT_EXTRACTION
        assert rec.critic_clients == []


# ---------------------------------------------------------------------------
# EB-5: heuristics_only on validate
# ---------------------------------------------------------------------------

class TestEB5HeuristicsOnlyValidate:
    def test_true_passes_critic_disabled_and_skips_generator(self, monkeypatch):
        rec = _install(monkeypatch, Recorder(gen_responses=[]))
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), heuristics_only=True)
        assert rec.calls[0][1].get("critic_enabled") is False
        assert set(result) == RESULT_KEYS
        assert result["status"] in ("ok", "needs_review")
        assert result["data"] == CORRECT_EXTRACTION
        assert all(len(c.calls) == 0 for c in rec.gen_clients)

    def test_default_validate_uses_critic_not_generator(self, monkeypatch):
        rec = _install(monkeypatch, Recorder(
            gen_responses=[], critic_responses=[critique_all(0.95, 0.92)]))
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION))
        assert _critic_enabled(rec.calls[0][1]) is True
        assert result["status"] == "ok"
        assert all(len(c.calls) == 0 for c in rec.gen_clients)
        assert len(rec.critic_clients[0].calls) == 1


# ---------------------------------------------------------------------------
# EB-6: pipeline failure -> envelope (extract)
# ---------------------------------------------------------------------------

class TestEB6ExtractFailureEnvelope:
    def test_generator_error_becomes_envelope(self, monkeypatch):
        _no_sleep(monkeypatch)
        _install(monkeypatch, Recorder(gen_responses=[GARBAGE] * 4,
                                       critic_responses=[critique_all(0.9, 0.9)]))
        result = tools.extract_tool(SOURCE)
        _assert_envelope(result)
        # Reference: drive the same Generator directly to capture str(exc).
        gen = Generator(ScriptedLLMClient([GARBAGE] * 4), max_retries=0)
        with pytest.raises(GeneratorError) as exc_info:
            ReflectivePipeline(gen, Critic(ScriptedLLMClient([critique_all(0.9, 0.9)]))).extract(SOURCE)
        assert result["error_message"] == str(exc_info.value)

    def test_factory_raising_becomes_envelope(self, monkeypatch):
        def bad_factory(*_a, **_k):
            raise ValueError("Unknown REFLECT_PROVIDER='bogus'; expected one of ...")

        monkeypatch.setattr(pipeline_module, "build_default_pipeline", bad_factory)
        result = tools.extract_tool(SOURCE)
        _assert_envelope(result)
        assert "REFLECT_PROVIDER" in result["error_message"]

    def test_real_factory_with_bogus_provider_env(self, monkeypatch):
        monkeypatch.setenv("REFLECT_PROVIDER", "bogus")
        result = tools.extract_tool(SOURCE)
        _assert_envelope(result)
        assert "REFLECT_PROVIDER" in result["error_message"]


# ---------------------------------------------------------------------------
# EB-7: pipeline failure -> envelope (validate)
# ---------------------------------------------------------------------------

class TestEB7ValidateFailureEnvelope:
    def test_factory_raising_becomes_envelope(self, monkeypatch):
        exc = ValueError("Unknown REFLECT_PROVIDER='bogus'")

        def bad_factory(*_a, **_k):
            raise exc

        monkeypatch.setattr(pipeline_module, "build_default_pipeline", bad_factory)
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION))
        _assert_envelope(result)
        assert result["error_message"] == str(exc)

    def test_critic_raising_becomes_envelope(self, monkeypatch):
        _no_sleep(monkeypatch)
        _install(monkeypatch, Recorder(
            critic_client=_RaisingClient(RuntimeError("provider exploded"))))
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION))
        _assert_envelope(result)
        # Reference: same pipeline driven directly.
        ref = ReflectivePipeline(
            Generator(ScriptedLLMClient([])),
            Critic(_RaisingClient(RuntimeError("provider exploded"))),
        )
        with pytest.raises(Exception) as exc_info:
            ref.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION))
        assert result["error_message"] == str(exc_info.value)

    def test_critic_garbage_does_not_raise(self, monkeypatch):
        _no_sleep(monkeypatch)
        _install(monkeypatch, Recorder(critic_responses=[GARBAGE] * 4))
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION))
        assert isinstance(result, dict)
        assert result["status"] in ("ok", "needs_review", "error")
        if result["status"] == "error":
            _assert_envelope(result)


# ---------------------------------------------------------------------------
# EB-8: invalid data -> envelope, no factory call
# ---------------------------------------------------------------------------

class TestEB8InvalidData:
    def test_bad_total(self, monkeypatch):
        rec = _happy(monkeypatch)
        result = tools.validate_tool(SOURCE, {"total": "abc"})
        _assert_envelope(result)
        assert result["error_message"].startswith("data failed InvoiceData validation")
        assert "total" in result["error_message"]
        assert rec.calls == []

    @pytest.mark.parametrize("bad", [None, [1, 2], "a string", 42])
    def test_non_dict_data(self, monkeypatch, bad):
        rec = _happy(monkeypatch)
        result = tools.validate_tool(SOURCE, bad)
        _assert_envelope(result)
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-9: empty input -> envelope, no factory call
# ---------------------------------------------------------------------------

class TestEB9EmptyInput:
    @pytest.mark.parametrize("text", ["", "   \n", "\t"])
    def test_extract_empty(self, monkeypatch, text):
        rec = _happy(monkeypatch)
        result = tools.extract_tool(text)
        _assert_envelope(result)
        assert result["error_message"] == "input is empty"
        assert rec.calls == []

    def test_validate_empty(self, monkeypatch):
        rec = _happy(monkeypatch)
        result = tools.validate_tool("  ", dict(CORRECT_EXTRACTION))
        _assert_envelope(result)
        assert result["error_message"] == "input is empty"
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-10: invalid thresholds -> envelope, no factory call
# ---------------------------------------------------------------------------

class TestEB10InvalidThresholds:
    @pytest.mark.parametrize("kw,name", [
        ({"field_threshold": 1.5}, "field_threshold"),
        ({"field_threshold": -0.1}, "field_threshold"),
        ({"overall_threshold": 2.0}, "overall_threshold"),
    ])
    def test_extract_bad_threshold(self, monkeypatch, kw, name):
        rec = _happy(monkeypatch)
        result = tools.extract_tool(SOURCE, **kw)
        _assert_envelope(result)
        assert name in result["error_message"]
        assert rec.calls == []

    @pytest.mark.parametrize("kw,name", [
        ({"field_threshold": 1.5}, "field_threshold"),
        ({"overall_threshold": -1.0}, "overall_threshold"),
    ])
    def test_validate_bad_threshold(self, monkeypatch, kw, name):
        rec = _happy(monkeypatch)
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), **kw)
        _assert_envelope(result)
        assert name in result["error_message"]
        assert rec.calls == []

    def test_nan_threshold_rejected(self, monkeypatch):
        rec = _happy(monkeypatch)
        result = tools.extract_tool(SOURCE, field_threshold=float("nan"))
        _assert_envelope(result)
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-11: server.py params parity + delegation (AST only)
# ---------------------------------------------------------------------------

def _server_tree():
    return ast.parse(SERVER.read_text(encoding="utf-8"))


def _server_func(name):
    for node in ast.walk(_server_tree()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"server.py has no function {name!r}")


def _params_and_defaults(fn):
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    pos = fn.args.posonlyargs + fn.args.args
    defaults = {}
    for a, d in zip(pos[len(pos) - len(fn.args.defaults):], fn.args.defaults):
        defaults[a.arg] = ast.literal_eval(d)
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if d is not None:
            defaults[a.arg] = ast.literal_eval(d)
    return names, defaults


def _sig_params_and_defaults(func):
    params = inspect.signature(func).parameters
    names = list(params)
    defaults = {n: p.default for n, p in params.items() if p.default is not inspect.Parameter.empty}
    return names, defaults


class TestEB11ServerParity:
    @pytest.mark.parametrize("server_name,tool_func", [
        ("extract", tools.extract_tool),
        ("validate", tools.validate_tool),
    ])
    def test_params_and_defaults_match(self, server_name, tool_func):
        s_names, s_defaults = _params_and_defaults(_server_func(server_name))
        t_names, t_defaults = _sig_params_and_defaults(tool_func)
        assert s_names == t_names
        assert s_defaults == t_defaults
        assert "heuristics_only" in s_names

    @pytest.mark.parametrize("server_name,target", [
        ("extract", "extract_tool"),
        ("validate", "validate_tool"),
    ])
    def test_body_delegates(self, server_name, target):
        fn = _server_func(server_name)
        called = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    called.add(f.attr)
                elif isinstance(f, ast.Name):
                    called.add(f.id)
        assert target in called

    def test_only_server_imports_mcp(self):
        importers = []
        for path in sorted(PKG.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module]
                if any(m == "mcp" or m.startswith("mcp.") for m in mods):
                    importers.append(path.name)
        assert sorted(set(importers)) == ["server.py"]


# ---------------------------------------------------------------------------
# EB-12: docs
# ---------------------------------------------------------------------------

class TestEB12Docs:
    def _tools_section(self):
        text = README.read_text(encoding="utf-8")
        start = text.index("**Tools exposed**")
        end = text.index("\n## ", start)
        return text[start:end]

    def test_table_rows_list_heuristics_only(self):
        sec = self._tools_section()
        rows = [ln for ln in sec.splitlines() if ln.startswith("| `extract`") or ln.startswith("| `validate`")]
        assert len(rows) == 2
        for row in rows:
            assert "heuristics_only=False" in row

    def test_error_envelope_documented(self):
        sec = self._tools_section()
        assert 'status: "error"' in sec
        assert "error_message" in sec
        assert "instead of raising" in sec

    def test_extract_docstring_status_line(self):
        doc = ast.get_docstring(_server_func("extract")) or ""
        status_lines = [ln for ln in doc.splitlines() if "status" in ln and "needs_review" in ln]
        assert status_lines, doc
        line = status_lines[0]
        for token in ('"ok"', '"needs_review"', '"error"'):
            assert token in line
