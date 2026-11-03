"""Black-box behavioral tests for iter-74: MCP threshold args default to null.

Covers the Expected Behaviors from pm.md:
  1. Signatures of tools.extract_tool / tools.validate_tool / server.extract /
     server.validate: names+order unchanged; thresholds default None,
     heuristics_only defaults False.
  2. Real SDK list_tools(): both threshold properties publish default null,
     accept number and null, are not required (importorskip("mcp")).
  3. Effective gate unchanged: omitted -> max(package default, floor);
     explicit -> max(arg, floor).
  4. Omitted thresholds never add a "raised from" sentence (env 0.95/0.95).
  5. Explicit arg below the floor still noted, byte-identical suffix.
  6. Mixed case: only the explicit, raised threshold is noted.
  7. Explicit invalid values / malformed env still return the error envelope
     and build no pipeline.
  8. server.extract / server.validate forward None defaults positionally.
  9. Real SDK low-level CallToolRequest under env 0.95/0.95: no error, no
     raise note; explicit JSON null behaves identically (importorskip("mcp")).
 10. CLI help byte-identical to HEAD; CLI ignores the env floors;
     frozen ``make eval`` stdout byte-identical to HEAD.

Isolation contract: ``reflect_mcp.tools`` is driven through its public
functions with a monkeypatched ``reflect_mcp.pipeline.build_default_pipeline``
(pipelines over ``ScriptedLLMClient``). server.py is imported with a stub MCP
SDK (as in iter 73) or, in the RealSDK class, with the real SDK when present.
HEAD digests were measured on a ``git archive HEAD`` copy (c466b8c).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all
from tests._golden import frozen_eval

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

FIELD_ENV = "REFLECT_FIELD_THRESHOLD"
OVERALL_ENV = "REFLECT_OVERALL_THRESHOLD"

# Measured on a `git archive HEAD` copy (c466b8c), COLUMNS=80.
HEAD_EXTRACT_HELP_SHA256 = "446711f8a2c1ee58a6eb4301fae50c35ddf1cc73928ef68a12d3d9267eff3bc9"
HEAD_VALIDATE_HELP_SHA256 = "6c318aec864ab2a63454240613d679b5c4222c5d8011163a3508eabc64bcc7a6"
# Frozen-subset eval stdout (re-measured on HEAD via tests._golden.frozen_eval).
HEAD_FROZEN_EVAL_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"

TOOLS = ["extract", "validate"]
RAISE = "raised from"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(FIELD_ENV, raising=False)
    monkeypatch.delenv(OVERALL_ENV, raising=False)


class Recorder:
    """Recording factory: stores every gate_config and builds a scripted pipeline."""

    def __init__(self, critic_raw: str | None = None):
        self.gen_raw = json.dumps(CORRECT_EXTRACTION)
        self.critic_raw = critic_raw or critique_all(0.95, 0.95)
        self.calls: list[tuple] = []

    def __call__(self, gate_config=None, **kwargs):
        self.calls.append((gate_config, kwargs))
        gen = Generator(ScriptedLLMClient([self.gen_raw]))
        critic = None
        if kwargs.get("critic_enabled", True):
            critic = Critic(ScriptedLLMClient([self.critic_raw]))
        return ReflectivePipeline(gen, critic, gate_config=gate_config)


def _install(monkeypatch, rec=None):
    rec = rec or Recorder()
    monkeypatch.setattr(pipeline_module, "build_default_pipeline", rec)
    return rec


def _built(rec):
    gc = rec.calls[-1][0]
    return gc.field_threshold, gc.overall_threshold


def _call(tool, *args, **kw):
    if tool == "extract":
        return tools.extract_tool(SOURCE, *args, **kw)
    return tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), *args, **kw)


def _direct_notes(tool, field, overall, heuristics_only=False):
    """Notes the pipeline itself returns for the given effective gate."""
    gen = Generator(ScriptedLLMClient([json.dumps(CORRECT_EXTRACTION)]))
    crit = None if heuristics_only else Critic(ScriptedLLMClient([critique_all(0.95, 0.95)]))
    p = ReflectivePipeline(gen, crit, gate_config=GateConfig(
        field_threshold=field, overall_threshold=overall))
    if tool == "extract":
        return p.extract(SOURCE).notes
    return p.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION)).notes


def _assert_envelope(result):
    assert isinstance(result, dict)
    assert set(result) == {"status", "error_message"}, result
    assert result["status"] == "error"
    assert result["error_message"].startswith("invalid thresholds:"), result


# Stub MCP SDK (FastMCP.tool() returns the function unchanged), as in iter 73.

class _StubCallToolResult:
    def __init__(self, content=None, structuredContent=None, isError=False, **kwargs):
        self.content = content
        self.structuredContent = structuredContent
        self.isError = isError


class _StubTextContent:
    def __init__(self, type, text, **kwargs):
        self.type = type
        self.text = text


class _StubFastMCP:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def tool(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    def run(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("stub FastMCP.run must not be called in tests")


def _pin_package_attrs(monkeypatch):
    """Re-bind ``reflect_mcp.<sub>`` attributes to the ``sys.modules`` entries.

    Another suite module (iter 73's mcp-blocked import sweep) re-imports
    submodules and restores ``sys.modules`` but not the package attributes, so
    ``from reflect_mcp import tools`` inside server.py could otherwise bind a
    different ``tools`` object than the one these tests patch.
    """
    import reflect_mcp

    for full, mod in list(sys.modules.items()):
        if full.startswith("reflect_mcp.") and full.count(".") == 1 and mod is not None:
            monkeypatch.setattr(reflect_mcp, full.split(".", 1)[1], mod, raising=False)


@pytest.fixture
def server(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    types_mod = types.ModuleType("mcp.types")
    fastmcp_mod.FastMCP = _StubFastMCP
    types_mod.CallToolResult = _StubCallToolResult
    types_mod.TextContent = _StubTextContent
    mcp_mod.server = server_mod
    mcp_mod.types = types_mod
    server_mod.fastmcp = fastmcp_mod
    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.types", types_mod)
    _pin_package_attrs(monkeypatch)
    sys.modules.pop("reflect_mcp.server", None)
    mod = importlib.import_module("reflect_mcp.server")
    try:
        yield mod
    finally:
        sys.modules.pop("reflect_mcp.server", None)


# ---------------------------------------------------------------------------
# EB1: signatures
# ---------------------------------------------------------------------------

EXTRACT_PARAMS = ["text", "field_threshold", "overall_threshold", "heuristics_only"]
VALIDATE_PARAMS = ["text", "data", "field_threshold", "overall_threshold", "heuristics_only"]


def _check_sig(fn, names):
    sig = inspect.signature(fn)
    assert list(sig.parameters) == names
    assert sig.parameters["field_threshold"].default is None
    assert sig.parameters["overall_threshold"].default is None
    assert sig.parameters["heuristics_only"].default is False
    assert sig.parameters["text"].default is inspect.Parameter.empty


class TestEB1Signatures:
    def test_tools_extract(self):
        _check_sig(tools.extract_tool, EXTRACT_PARAMS)

    def test_tools_validate(self):
        _check_sig(tools.validate_tool, VALIDATE_PARAMS)
        assert inspect.signature(tools.validate_tool).parameters["data"].default \
            is inspect.Parameter.empty

    def test_server_extract(self, server):
        _check_sig(server.extract, EXTRACT_PARAMS)

    def test_server_validate(self, server):
        _check_sig(server.validate, VALIDATE_PARAMS)


# ---------------------------------------------------------------------------
# EB3: effective gate unchanged
# ---------------------------------------------------------------------------

class TestEB3EffectiveGate:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("env,expected", [
        ((None, None), (0.70, 0.75)),
        (("0.9", "0.85"), (0.9, 0.85)),
        (("0", "0"), (0.70, 0.75)),
        (("0.95", "0.95"), (0.95, 0.95)),
        (("0.9", None), (0.9, 0.75)),
        ((None, "0.85"), (0.70, 0.85)),
        (("0.5", "0.5"), (0.70, 0.75)),
        (("  ", ""), (0.70, 0.75)),
    ])
    @pytest.mark.parametrize("omit", ["absent", "none_kw"])
    def test_omitted_is_max_default_floor(self, monkeypatch, tool, env, expected, omit):
        if env[0] is not None:
            monkeypatch.setenv(FIELD_ENV, env[0])
        if env[1] is not None:
            monkeypatch.setenv(OVERALL_ENV, env[1])
        rec = _install(monkeypatch)
        kw = {} if omit == "absent" else {"field_threshold": None, "overall_threshold": None}
        result = _call(tool, **kw)
        assert result["status"] in {"ok", "needs_review"}, result
        assert len(rec.calls) == 1
        assert _built(rec) == pytest.approx(expected)

    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("env", [(None, None), ("0", "0"), ("0.9", "0.85")])
    @pytest.mark.parametrize("args", [(0.0, 0.0), (0.5, 0.9), (0.7, 0.75), (0.95, 0.99), (1.0, 1.0)])
    def test_explicit_is_max_arg_floor(self, monkeypatch, tool, env, args):
        floor = [0.70, 0.75]
        if env[0] is not None:
            monkeypatch.setenv(FIELD_ENV, env[0])
            floor[0] = float(env[0])
        if env[1] is not None:
            monkeypatch.setenv(OVERALL_ENV, env[1])
            floor[1] = float(env[1])
        rec = _install(monkeypatch)
        _call(tool, *args)
        assert _built(rec) == pytest.approx((max(args[0], floor[0]), max(args[1], floor[1])))

    def test_omitting_one_threshold_only(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        tools.extract_tool(SOURCE, field_threshold=0.99)
        assert _built(rec) == pytest.approx((0.99, 0.85))
        tools.extract_tool(SOURCE, overall_threshold=0.99)
        assert _built(rec) == pytest.approx((0.9, 0.99))


# ---------------------------------------------------------------------------
# EB4: omitted thresholds never add a raise sentence
# ---------------------------------------------------------------------------

class TestEB4NoNoteWhenOmitted:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("ho", [False, True])
    @pytest.mark.parametrize("explicit_none", [False, True])
    def test_env_095_omitted(self, monkeypatch, tool, ho, explicit_none):
        monkeypatch.setenv(FIELD_ENV, "0.95")
        monkeypatch.setenv(OVERALL_ENV, "0.95")
        rec = _install(monkeypatch)
        kw = {"heuristics_only": True} if ho else {}
        if explicit_none:
            kw.update(field_threshold=None, overall_threshold=None)
        result = _call(tool, **kw)
        assert result["status"] in {"ok", "needs_review"}, result
        assert RAISE not in result["notes"]
        assert result["notes"] == _direct_notes(tool, 0.95, 0.95, heuristics_only=ho)
        assert _built(rec) == pytest.approx((0.95, 0.95))

    @pytest.mark.parametrize("tool", TOOLS)
    def test_no_env_omitted_has_no_note(self, monkeypatch, tool):
        _install(monkeypatch)
        result = _call(tool)
        assert RAISE not in result["notes"]
        assert result["notes"] == _direct_notes(tool, 0.70, 0.75)

    def test_validate_heuristics_only_no_provider_under_floor(self, monkeypatch):
        # Real build_default_pipeline (no stub): heuristics_only validate needs no provider.
        monkeypatch.setenv(FIELD_ENV, "0.95")
        monkeypatch.setenv(OVERALL_ENV, "0.95")
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), heuristics_only=True)
        assert result["status"] == "ok", result
        assert RAISE not in result["notes"]
        assert result["notes"] == _direct_notes("validate", 0.95, 0.95, heuristics_only=True)


# ---------------------------------------------------------------------------
# EB5: explicit below-floor args still noted (byte-identical to HEAD)
# ---------------------------------------------------------------------------

class TestEB5ExplicitStillNoted:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_env_floor_explicit_defaults(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        _install(monkeypatch)
        result = _call(tool, 0.7, 0.75)
        suffix = (" field_threshold raised from 0.7 to 0.9. "
                  "overall_threshold raised from 0.75 to 0.85.")
        assert result["notes"].endswith(suffix)
        assert result["notes"] == _direct_notes(tool, 0.9, 0.85) + suffix

    @pytest.mark.parametrize("tool", TOOLS)
    def test_no_env_zero_args(self, monkeypatch, tool):
        _install(monkeypatch)
        result = _call(tool, 0.0, 0.0)
        suffix = (" field_threshold raised from 0.0 to 0.7. "
                  "overall_threshold raised from 0.0 to 0.75.")
        assert result["notes"] == _direct_notes(tool, 0.70, 0.75) + suffix

    def test_explicit_at_floor_not_noted(self, monkeypatch):
        _install(monkeypatch)
        result = tools.extract_tool(SOURCE, 0.7, 0.75)
        assert RAISE not in result["notes"]


# ---------------------------------------------------------------------------
# EB6: mixed explicit/omitted
# ---------------------------------------------------------------------------

class TestEB6Mixed:
    def test_overall_only_explicit(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        result = tools.extract_tool(SOURCE, overall_threshold=0.5)
        assert result["notes"].endswith(" overall_threshold raised from 0.5 to 0.85.")
        assert "field_threshold raised" not in result["notes"]
        assert result["notes"] == (
            _direct_notes("extract", 0.9, 0.85) + " overall_threshold raised from 0.5 to 0.85."
        )
        assert _built(rec) == pytest.approx((0.9, 0.85))

    def test_field_only_explicit_validate(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        result = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), 0.1, None)
        assert result["notes"].endswith(" field_threshold raised from 0.1 to 0.9.")
        assert "overall_threshold raised" not in result["notes"]
        assert _built(rec) == pytest.approx((0.9, 0.85))


# ---------------------------------------------------------------------------
# EB7: invalid explicit values / malformed env still error
# ---------------------------------------------------------------------------

class TestEB7Invalid:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("kw,name", [
        ({"field_threshold": 1.5}, "field_threshold"),
        ({"field_threshold": -0.1}, "field_threshold"),
        ({"field_threshold": float("nan")}, "field_threshold"),
        ({"overall_threshold": 1.5}, "overall_threshold"),
        ({"overall_threshold": -0.1}, "overall_threshold"),
        ({"overall_threshold": float("nan")}, "overall_threshold"),
        ({"field_threshold": None, "overall_threshold": 1.5}, "overall_threshold"),
    ])
    def test_bad_explicit(self, monkeypatch, tool, kw, name):
        rec = _install(monkeypatch)
        result = _call(tool, **kw)
        _assert_envelope(result)
        assert name in result["error_message"]
        assert rec.calls == []

    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("var", [FIELD_ENV, OVERALL_ENV])
    @pytest.mark.parametrize("raw", ["abc", "nan", "1.5", "-0.1"])
    @pytest.mark.parametrize("explicit_none", [False, True])
    def test_malformed_env_with_omitted_args(self, monkeypatch, tool, var, raw, explicit_none):
        monkeypatch.setenv(var, raw)
        rec = _install(monkeypatch)
        kw = {"field_threshold": None, "overall_threshold": None} if explicit_none else {}
        result = _call(tool, **kw)
        _assert_envelope(result)
        assert var in result["error_message"]
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB8: server forwards None defaults positionally
# ---------------------------------------------------------------------------

class TestEB8ServerForwarding:
    def test_extract_forwards_none(self, server, monkeypatch):
        calls = []
        monkeypatch.setattr(tools, "extract_tool",
                            lambda *a, **k: calls.append((a, k)) or {"status": "ok"})
        asyncio.run(server.extract("doc"))
        assert calls == [(("doc", None, None, False), {})]

    def test_validate_forwards_none(self, server, monkeypatch):
        calls = []
        monkeypatch.setattr(tools, "validate_tool",
                            lambda *a, **k: calls.append((a, k)) or {"status": "ok"})
        asyncio.run(server.validate("doc", {}))
        assert calls == [(("doc", {}, None, None, False), {})]

    def test_server_extract_omitted_end_to_end_no_note(self, server, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.95")
        monkeypatch.setenv(OVERALL_ENV, "0.95")
        _install(monkeypatch)
        res = asyncio.run(server.extract(SOURCE))
        assert res.isError is False
        assert RAISE not in res.structuredContent["notes"]


# ---------------------------------------------------------------------------
# EB10: CLI untouched, eval unchanged
# ---------------------------------------------------------------------------

def _cli(args, env_extra=None, stdin=None):
    env = dict(os.environ)
    env.pop(FIELD_ENV, None)
    env.pop(OVERALL_ENV, None)
    env["COLUMNS"] = "80"
    env.update(env_extra or {})
    code = ("import sys; sys.argv = ['reflect-mcp'] + sys.argv[1:]; "
            "from reflect_mcp.__main__ import main; sys.exit(main())")
    return subprocess.run([sys.executable, "-c", code, *args], input=stdin, env=env,
                          capture_output=True, text=True, timeout=120, cwd=REPO)


class TestEB10CliAndEval:
    @pytest.mark.parametrize("verb,digest", [
        ("extract", HEAD_EXTRACT_HELP_SHA256),
        ("validate", HEAD_VALIDATE_HELP_SHA256),
    ])
    def test_help_byte_identical(self, verb, digest):
        proc = _cli([verb, "--help"])
        assert proc.returncode == 0, proc.stderr
        assert hashlib.sha256(proc.stdout.encode()).hexdigest() == digest, proc.stdout

    def test_cli_ignores_env_floor(self):
        args = ["validate", "--heuristics-only", "--field-threshold", "0",
                "--overall-threshold", "0", "--data", json.dumps(CORRECT_EXTRACTION)]
        plain = _cli(args, stdin=SOURCE)
        floored = _cli(args, {FIELD_ENV: "0.95", OVERALL_ENV: "0.95"}, stdin=SOURCE)
        assert plain.returncode == floored.returncode == 0, (plain.stderr, floored.stderr)
        a, b = json.loads(plain.stdout), json.loads(floored.stdout)
        a.pop("trace_id", None)
        b.pop("trace_id", None)
        assert a == b
        assert RAISE not in b["notes"]

    def test_frozen_eval_stdout_identical(self):
        proc = frozen_eval()
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_FROZEN_EVAL_SHA256


# ---------------------------------------------------------------------------
# AC: README states the rule
# ---------------------------------------------------------------------------

class TestReadme:
    def test_tool_rows_have_no_numeric_arg_defaults(self):
        rows = [ln for ln in README.read_text().splitlines()
                if ln.startswith("| `extract` |") or ln.startswith("| `validate` |")]
        assert len(rows) >= 2
        for ln in rows:
            assert "field_threshold=0.7" not in ln
            assert "overall_threshold=0.75" not in ln
            assert "field_threshold=null" in ln and "overall_threshold=null" in ln

    def test_floor_paragraph_states_omitted_rule(self):
        para = next(ln for ln in README.read_text().splitlines()
                    if ln.startswith("**Operator threshold floor.**"))
        assert "omitted" in para.lower()
        assert "null" in para


# ---------------------------------------------------------------------------
# EB2 + EB9: real SDK (skipped in the default offline venv)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_server(monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    _pin_package_attrs(monkeypatch)
    sys.modules.pop("reflect_mcp.server", None)
    try:
        yield importlib.import_module("reflect_mcp.server")
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _accepts(prop):
    branches = prop.get("anyOf") or prop.get("oneOf") or [prop]
    kinds = set()
    for b in branches:
        t = b.get("type")
        kinds.update(t if isinstance(t, list) else [t])
    return kinds


class TestRealSDK:
    def test_eb2_list_tools_input_schema(self, real_server):
        listed = asyncio.run(real_server.mcp.list_tools())
        by_name = {t.name: t for t in listed}
        assert set(by_name) == {"extract", "validate"}
        for t in by_name.values():
            schema = t.inputSchema
            for name in ("field_threshold", "overall_threshold"):
                prop = schema["properties"][name]
                assert "default" in prop and prop["default"] is None, prop
                assert prop["default"] not in (0.7, 0.75)
                kinds = _accepts(prop)
                assert "null" in kinds and "number" in kinds, prop
                assert name not in schema.get("required", [])

    @pytest.mark.parametrize("extra", [{}, {"field_threshold": None},
                                       {"field_threshold": None, "overall_threshold": None}])
    def test_eb9_call_tool_request_no_raise_note(self, real_server, monkeypatch, extra):
        mtypes = pytest.importorskip("mcp.types")
        monkeypatch.setenv(FIELD_ENV, "0.95")
        monkeypatch.setenv(OVERALL_ENV, "0.95")
        rec = _install(monkeypatch)
        handler = real_server.mcp._mcp_server.request_handlers[mtypes.CallToolRequest]
        args = {"text": SOURCE, "data": dict(CORRECT_EXTRACTION), **extra}
        req = mtypes.CallToolRequest(
            method="tools/call",
            params=mtypes.CallToolRequestParams(name="validate", arguments=args),
        )
        out = asyncio.run(handler(req))
        res = getattr(out, "root", out)
        assert res.isError is False, res
        sc = res.structuredContent
        assert sc["status"] in {"ok", "needs_review"}, sc
        assert RAISE not in sc["notes"]
        assert sc["notes"] == _direct_notes("validate", 0.95, 0.95)
        assert _built(rec) == pytest.approx((0.95, 0.95))
