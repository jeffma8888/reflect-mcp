"""Black-box behavioral tests for iter-73: typed MCP results (CallToolResult).

Covers the Expected Behaviors from pm.md:
  1. extract("") -> isError True, structuredContent is the error envelope,
     content is one text item whose JSON equals structuredContent.
  2. heuristics-only validate yielding ok -> isError False, text == structured.
  3. heuristics-only validate yielding needs_review -> isError False, total flagged.
  4. structuredContent equals the tools.* dict (no key added or dropped).
  5. ToolResult JSON schema: required == ["status"], status enum typed.
  6. ToolResult validates the 1-3 results; rejects {"status": "maybe"} and {}.
  7. Return annotation is Annotated[CallToolResult, ToolResult].
  8. Parallel validate calls still overlap (< 0.5 s for 4 x 0.2 s).
  9. pyproject server extra is mcp>=1.28,<2 and uv.lock agrees (offline check).
 10. Real-SDK list_tools() outputSchema check (importorskip("mcp")).
 11. README MCP section documents outputSchema / structuredContent / isError.
 Plus the acceptance criterion "server.py is the only mcp importer", checked
 black-box: every other reflect_mcp module imports with `mcp` blocked.

Isolation contract: server.py is imported with a stub MCP SDK (incl. mcp.types)
installed in sys.modules -- no real SDK, no network. Only public interfaces are
driven; no implementation source is read.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pkgutil
import re
import shutil
import subprocess
import sys
import time
import types
import typing
from pathlib import Path

import pytest

import reflect_mcp
from reflect_mcp import tools

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
PYPROJECT = REPO / "pyproject.toml"

TEXT = "ACME CORP, Invoice #INV-1001, Total: $108.50"
OK_DATA = {"vendor_name": "ACME CORP", "invoice_number": "INV-1001", "total": 108.5}
NR_DATA = {"vendor_name": "ACME CORP", "invoice_number": "INV-1001", "total": 999.0}
STATUSES = {"ok", "needs_review", "error"}


# ---------------------------------------------------------------------------
# Stub SDK fixture (FastMCP.tool() returns the function unchanged)
# ---------------------------------------------------------------------------

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


@pytest.fixture
def stub_types(monkeypatch):
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
    return types_mod


@pytest.fixture
def server(stub_types):
    sys.modules.pop("reflect_mcp.server", None)
    mod = importlib.import_module("reflect_mcp.server")
    try:
        yield mod
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _call(coro):
    return asyncio.run(coro)


def _assert_text_mirrors_structured(res):
    assert isinstance(res.content, list)
    assert len(res.content) == 1
    item = res.content[0]
    assert item.type == "text"
    assert json.loads(item.text) == res.structuredContent


def _no_trace(d):
    d = dict(d)
    d.pop("trace_id", None)
    return d


# ---------------------------------------------------------------------------
# EB1: error result
# ---------------------------------------------------------------------------

class TestEB1ErrorResult:
    def test_is_error_true(self, server):
        res = _call(server.extract(""))
        assert res.isError is True

    def test_structured_is_exact_error_envelope(self, server):
        sc = _call(server.extract("")).structuredContent
        assert isinstance(sc, dict)
        assert set(sc) == {"status", "error_message"}
        assert sc["status"] == "error"
        assert isinstance(sc["error_message"], str) and sc["error_message"].strip()

    def test_content_is_one_text_item_mirroring_structured(self, server):
        _assert_text_mirrors_structured(_call(server.extract("")))

    def test_validate_error_also_sets_is_error(self, server):
        res = _call(server.validate("", {}))
        assert res.isError is True
        assert res.structuredContent["status"] == "error"
        _assert_text_mirrors_structured(res)

    def test_returns_call_tool_result_type(self, server, stub_types):
        assert isinstance(_call(server.extract("")), stub_types.CallToolResult)


# ---------------------------------------------------------------------------
# EB2: ok result
# ---------------------------------------------------------------------------

class TestEB2OkResult:
    def test_ok(self, server):
        res = _call(server.validate(TEXT, dict(OK_DATA), heuristics_only=True))
        assert res.structuredContent["status"] == "ok", res.structuredContent
        assert res.isError is False

    def test_ok_text_mirrors_structured(self, server):
        _assert_text_mirrors_structured(
            _call(server.validate(TEXT, dict(OK_DATA), heuristics_only=True))
        )

    def test_ok_has_no_error_message_key(self, server):
        sc = _call(server.validate(TEXT, dict(OK_DATA), heuristics_only=True)).structuredContent
        assert "error_message" not in sc


# ---------------------------------------------------------------------------
# EB3: needs_review result
# ---------------------------------------------------------------------------

class TestEB3NeedsReview:
    def test_needs_review(self, server):
        res = _call(server.validate(TEXT, dict(NR_DATA), heuristics_only=True))
        sc = res.structuredContent
        assert sc["status"] == "needs_review", sc
        assert res.isError is False
        assert "total" in [f["field"] for f in sc["flagged_fields"]]

    def test_needs_review_text_mirrors_structured(self, server):
        _assert_text_mirrors_structured(
            _call(server.validate(TEXT, dict(NR_DATA), heuristics_only=True))
        )


# ---------------------------------------------------------------------------
# EB4: structuredContent == tools.* dict
# ---------------------------------------------------------------------------

class TestEB4SameAsTools:
    def test_error_equals_tools(self, server):
        assert _call(server.extract("")).structuredContent == tools.extract_tool("")

    @pytest.mark.parametrize("data", [OK_DATA, NR_DATA], ids=["ok", "needs_review"])
    def test_validate_equals_tools(self, server, data):
        got = _call(server.validate(TEXT, dict(data), heuristics_only=True)).structuredContent
        want = tools.validate_tool(TEXT, dict(data), 0.70, 0.75, True)
        # Same key set (no key added, none dropped); trace_id is per call.
        assert set(got) == set(want)
        assert _no_trace(got) == _no_trace(want)

    @pytest.mark.parametrize("data", [OK_DATA, NR_DATA], ids=["ok", "needs_review"])
    def test_no_none_valued_keys_added(self, server, data):
        got = _call(server.validate(TEXT, dict(data), heuristics_only=True)).structuredContent
        want = tools.validate_tool(TEXT, dict(data), 0.70, 0.75, True)
        added_none = {k for k, v in got.items() if v is None and k not in want}
        assert added_none == set()


# ---------------------------------------------------------------------------
# EB5 / EB6: ToolResult schema + validation
# ---------------------------------------------------------------------------

def _adapter():
    pydantic = pytest.importorskip("pydantic")
    return pydantic.TypeAdapter(tools.ToolResult)


def _resolve(schema, node):
    ref = node.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        return schema.get("$defs", {}).get(name, node)
    return node


class TestEB5Schema:
    def test_required_is_only_status(self):
        schema = _adapter().json_schema()
        assert schema.get("required") == ["status"]

    def test_status_enum(self):
        schema = _adapter().json_schema()
        status = _resolve(schema, schema["properties"]["status"])
        assert set(status["enum"]) == STATUSES
        assert len(status["enum"]) == 3

    def test_envelope_keys_are_properties(self):
        props = set(_adapter().json_schema()["properties"])
        expected = {
            "status", "data", "field_confidence", "overall_confidence",
            "flagged_fields", "field_provenance", "trace_id", "critic_summary",
            "notes", "error_message",
        }
        assert expected <= props

    def test_typed_dict_from_typing_extensions_shape(self):
        assert tools.ToolResult.__required_keys__ == frozenset({"status"})


class TestEB6Validation:
    def test_validates_tool_outputs(self, server):
        ta = _adapter()
        outs = [
            _call(server.extract("")).structuredContent,
            _call(server.validate(TEXT, dict(OK_DATA), heuristics_only=True)).structuredContent,
            _call(server.validate(TEXT, dict(NR_DATA), heuristics_only=True)).structuredContent,
        ]
        assert [o["status"] for o in outs] == ["error", "ok", "needs_review"]
        for o in outs:
            ta.validate_python(o)

    @pytest.mark.parametrize("bad", [{"status": "maybe"}, {}], ids=["maybe", "empty"])
    def test_rejects_bad(self, bad):
        pydantic = pytest.importorskip("pydantic")
        with pytest.raises(pydantic.ValidationError):
            _adapter().validate_python(bad)


# ---------------------------------------------------------------------------
# EB7: return annotation
# ---------------------------------------------------------------------------

class TestEB7Annotation:
    @pytest.mark.parametrize("name", ["extract", "validate"])
    def test_annotated_call_tool_result(self, server, stub_types, name):
        hint = typing.get_type_hints(getattr(server, name), include_extras=True)["return"]
        assert typing.get_origin(hint) is typing.Annotated
        assert hint.__metadata__ == (tools.ToolResult,)
        assert typing.get_args(hint)[0] is stub_types.CallToolResult


# ---------------------------------------------------------------------------
# EB8: parallel calls still overlap
# ---------------------------------------------------------------------------

def _sleepy(text, *args, **kwargs):
    time.sleep(0.2)
    return {"status": "ok", "text": text}


class TestEB8Overlap:
    def test_validate_gather_overlaps(self, server, monkeypatch):
        monkeypatch.setattr(tools, "validate_tool", _sleepy)

        async def main():
            t0 = time.perf_counter()
            rs = await asyncio.gather(*(server.validate(f"doc{i}", {}) for i in range(4)))
            return rs, time.perf_counter() - t0

        rs, elapsed = asyncio.run(main())
        assert elapsed < 0.5, elapsed
        assert [r.structuredContent["text"] for r in rs] == [f"doc{i}" for i in range(4)]
        assert all(r.isError is False for r in rs)


# ---------------------------------------------------------------------------
# EB9: dependency pin + lock agreement
# ---------------------------------------------------------------------------

class TestEB9Pin:
    def test_server_extra_pin(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        m = re.search(r"(?ms)^server\s*=\s*\[(.*?)\]", text)
        assert m, "no `server` extra in pyproject.toml"
        specs = re.findall(r'"([^"]+)"', m.group(1))
        mcp_specs = [s.replace(" ", "") for s in specs if s.startswith("mcp")]
        assert mcp_specs == ["mcp>=1.28,<2"]

    def test_lock_records_specifier(self):
        lock = (REPO / "uv.lock").read_text(encoding="utf-8")
        assert re.search(r'name = "mcp",[^}]*specifier = ">=1\.28,<2"', lock), \
            "uv.lock does not record mcp>=1.28,<2"

    def test_uv_lock_check_offline(self):
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv not on PATH")
        proc = subprocess.run(
            [uv, "lock", "--check", "--offline"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# EB10: real SDK (skipped in the default offline venv)
# ---------------------------------------------------------------------------

class TestEB10RealSDK:
    def test_list_tools_output_schema(self):
        pytest.importorskip("mcp")
        sys.modules.pop("reflect_mcp.server", None)
        try:
            srv = importlib.import_module("reflect_mcp.server")
            listed = asyncio.run(srv.mcp.list_tools())
        finally:
            sys.modules.pop("reflect_mcp.server", None)
        by_name = {t.name: t for t in listed}
        assert set(by_name) == {"extract", "validate"}
        for t in by_name.values():
            schema = t.outputSchema
            assert schema["required"] == ["status"]
            status = _resolve(schema, schema["properties"]["status"])
            assert set(status["enum"]) == STATUSES


# ---------------------------------------------------------------------------
# AC: server.py is the only mcp importer (black-box: import with mcp blocked)
# ---------------------------------------------------------------------------

class TestOnlyServerImportsMcp:
    def test_other_modules_import_without_mcp(self, monkeypatch):
        for name in ["mcp", "mcp.server", "mcp.server.fastmcp", "mcp.types"]:
            monkeypatch.setitem(sys.modules, name, None)
        names = [
            m.name for m in pkgutil.iter_modules(reflect_mcp.__path__)
            if m.name not in {"server", "__main__"}
        ]
        assert "tools" in names
        for n in names:
            full = f"reflect_mcp.{n}"
            saved = sys.modules.pop(full, None)
            try:
                importlib.import_module(full)
            finally:
                # Re-import also rebinds the package attribute; restore both so
                # later mock.patch("reflect_mcp.<n>.X") hits the live module.
                if saved is not None:
                    sys.modules[full] = saved
                    setattr(reflect_mcp, n, saved)


# ---------------------------------------------------------------------------
# EB11: README
# ---------------------------------------------------------------------------

class TestEB11Readme:
    def _para(self):
        text = README.read_text(encoding="utf-8")
        hits = [p for p in text.split("\n\n") if "outputSchema" in p]
        assert hits, "README never mentions outputSchema"
        return hits[0]

    def test_mentions_required_status_enum(self):
        p = self._para()
        assert "status" in p and "required" in p
        for s in ("`ok`", "`needs_review`", "`error`"):
            assert s in p

    def test_mentions_structured_content(self):
        assert "structuredContent" in self._para()

    def test_mentions_is_error_true(self):
        assert "isError: true" in self._para()

    def test_both_tools(self):
        assert re.search(r"\bBoth tools\b", self._para())
