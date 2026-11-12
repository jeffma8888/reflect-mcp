"""Black-box behavioral tests for iter-91 (#74): MCP host hints.

Covers the Expected Behaviors from pm.md:
  1. ``initialize`` returns a non-empty server ``instructions`` string of at
     most 600 chars naming ``needs_review``, ``flagged_fields``, ``ok``,
     ``error`` and ``validate``.
  2. Two-way tie: the ``status`` enum ``extract`` publishes in its
     ``outputSchema`` is exactly {ok, needs_review, error}, and every enum
     value appears in the instructions.
  3. The instructions state the escalation rule (``confirm`` + ``human``) and
     that ``validate`` does not re-extract.
  4. Both tools carry the spec'd human ``title``.
  5. Both tools carry exactly the four hints readOnly/destructive/idempotent/
     openWorld = True/False/True/True.
  6. ``serverInfo`` name/version unchanged (``reflect-mcp`` / ``__version__``).
  7. Nothing else moves: tool names, descriptions (literal digest measured on
     the pre-change tree), and a heuristics-only ``validate`` call. The iter-85
     inputSchema/outputSchema digest and iter-69 docstring pins cover the rest.
  8. Stub-SDK compatibility: server.py imports exactly {CallToolResult,
     TextContent} from ``mcp.types`` and passes ``annotations`` as a plain
     dict (``ast`` over the module, never executed text).
  9. README section 4 ``**Host hints.**`` paragraph follows ``**Typed
     results.**`` and is pinned two-way to the server's annotation keys.
 10. This module runs no git subprocess and reads no git revision (``ast``
     self-check).

Isolation contract: only the public surface is driven -- an in-process,
offline MCP session over the real installed SDK (no provider key), the README,
and an ``ast`` structural check of server.py's imports/keyword kinds that the
spec itself names. The description digest below is a literal measured by
listing tools on a pre-change copy of the source tree; no revision is read at
test time.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

import reflect_mcp

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
SERVER_PY = REPO / "src" / "reflect_mcp" / "server.py"
THIS_FILE = Path(__file__).resolve()

EXPECTED_TITLES = {
    "extract": "Extract and verify invoice data",
    "validate": "Verify pre-extracted invoice data",
}
EXPECTED_HINTS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
# sha256 of json.dumps(sorted([{name, description}], key=name), sort_keys=True)
# measured with the real SDK on the pre-iter-91 tree (descriptions must not move).
PRE_CHANGE_DESCRIPTIONS_SHA256 = (
    "f3882b1a5096899e3e79d5d10ce42750d6c28f214609c53eda343c9ba022be01"
)
MAX_INSTRUCTIONS_CHARS = 600


@pytest.fixture
def surface(monkeypatch):
    """Initialize + tools/list (+ one validate call) over an in-memory session."""
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    memory = pytest.importorskip("mcp.shared.memory")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REFLECT_MCP_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("reflect_mcp.server", None)
    try:
        server = importlib.import_module("reflect_mcp.server")
        if not hasattr(server.mcp, "_mcp_server"):
            pytest.skip("stub FastMCP in sys.modules; real SDK required")

        async def run():
            async with memory.create_connected_server_and_client_session(
                server.mcp._mcp_server
            ) as session:
                init = await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                call = await session.call_tool(
                    "validate",
                    {
                        "text": "Total: 10.00",
                        "data": {"total": 10.0},
                        "heuristics_only": True,
                    },
                )
                return init, tools, call

        init, tools, call = asyncio.run(run())
        yield {"init": init, "tools": tools, "call": call}
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _hints(tool) -> dict:
    ann = tool.annotations
    assert ann is not None, f"{tool.name}: annotations is None"
    return ann.model_dump(exclude_none=True, exclude={"title"})


# --- EB1 -------------------------------------------------------------------


def test_eb1_instructions_nonempty_bounded_and_names_contract(surface):
    instr = surface["init"].instructions
    assert isinstance(instr, str) and instr.strip(), f"instructions={instr!r}"
    assert len(instr) <= MAX_INSTRUCTIONS_CHARS, len(instr)
    for needle in ("needs_review", "flagged_fields", "ok", "error", "validate"):
        assert needle in instr, f"{needle!r} missing from instructions: {instr!r}"


# --- EB2 -------------------------------------------------------------------


def test_eb2_status_enum_tied_two_way_to_instructions(surface):
    schema = surface["tools"]["extract"].outputSchema
    assert schema is not None, "extract publishes no outputSchema"
    enum = schema["properties"]["status"]["enum"]
    assert sorted(enum) == sorted(["ok", "needs_review", "error"]), enum
    instr = surface["init"].instructions or ""
    missing = [v for v in enum if v not in instr]
    assert not missing, f"status values not in instructions: {missing}"


def test_eb2_validate_publishes_same_status_enum(surface):
    # Ambiguity note: the spec ties only extract's enum; validate shares the
    # envelope, so its enum must also be fully covered by the instructions.
    schema = surface["tools"]["validate"].outputSchema
    assert schema is not None
    enum = schema["properties"]["status"]["enum"]
    instr = surface["init"].instructions or ""
    assert [v for v in enum if v not in instr] == []


# --- EB3 -------------------------------------------------------------------


def test_eb3_instructions_state_escalation_and_validate_rule(surface):
    low = (surface["init"].instructions or "").lower()
    assert "confirm" in low and "human" in low, low
    assert "validate" in low, low
    assert ("never re-extract" in low) or ("does not re-extract" in low), low


# --- EB4 -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_TITLES))
def test_eb4_tool_titles(surface, name):
    assert surface["tools"][name].title == EXPECTED_TITLES[name]


# --- EB5 -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_TITLES))
def test_eb5_tool_annotations_exact_hints(surface, name):
    tool = surface["tools"][name]
    hints = _hints(tool)
    assert hints == EXPECTED_HINTS, hints
    ann = tool.annotations
    assert ann.readOnlyHint is True
    assert ann.destructiveHint is False
    assert ann.idempotentHint is True
    assert ann.openWorldHint is True


# --- EB6 -------------------------------------------------------------------


def test_eb6_server_info_unchanged(surface):
    info = surface["init"].serverInfo
    assert info.name == "reflect-mcp"
    assert info.version == reflect_mcp.__version__


# --- EB7 -------------------------------------------------------------------


def test_eb7_tool_name_set_exact(surface):
    assert set(surface["tools"]) == {"extract", "validate"}


def test_eb7_descriptions_unchanged(surface):
    rows = sorted(
        ({"name": t.name, "description": t.description} for t in surface["tools"].values()),
        key=lambda r: r["name"],
    )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == PRE_CHANGE_DESCRIPTIONS_SHA256, rows


def test_eb7_validate_call_behavior_unchanged(surface):
    call = surface["call"]
    assert call.isError is False
    assert call.structuredContent is not None
    assert call.structuredContent["status"] in {"ok", "needs_review"}, call.structuredContent


# --- EB8 -------------------------------------------------------------------


def _server_tree() -> ast.Module:
    return ast.parse(SERVER_PY.read_text(encoding="utf-8"))


def test_eb8_mcp_types_import_set_exact():
    names = set()
    for node in ast.walk(_server_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "mcp.types":
            names.update(a.name for a in node.names)
    assert names == {"CallToolResult", "TextContent"}, names


def test_eb8_annotations_passed_as_plain_dict():
    tree = _server_tree()
    module_dicts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    module_dicts[tgt.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            module_dicts[node.target.id] = node.value
    ann_values = [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "annotations"
    ]
    assert len(ann_values) >= 2, "expected annotations= on both tool decorators"
    for value in ann_values:
        is_dict = isinstance(value, ast.Dict) or (
            isinstance(value, ast.Name) and value.id in module_dicts
        )
        assert is_dict, f"annotations= is not a plain dict literal: {ast.dump(value)}"
    used = {
        n.id if isinstance(n, ast.Name) else n.attr
        for n in ast.walk(tree)
        if isinstance(n, (ast.Name, ast.Attribute))
    }
    assert "ToolAnnotations" not in used


# --- EB9 -------------------------------------------------------------------


def _section4() -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^## 4\. Use it in Claude Desktop\s*$(.*?)(?=^## )", text, re.M | re.S)
    assert m, "README section 4 not found"
    return m.group(1)


def _paragraphs(section: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]


def _host_hints_paragraph() -> str:
    paras = _paragraphs(_section4())
    hits = [p for p in paras if p.startswith("**Host hints.**")]
    assert len(hits) == 1, f"expected one **Host hints.** paragraph, got {len(hits)}"
    return hits[0]


def test_eb9_host_hints_paragraph_follows_typed_results():
    paras = _paragraphs(_section4())
    idx = [i for i, p in enumerate(paras) if p.startswith("**Host hints.**")]
    typed = [i for i, p in enumerate(paras) if p.startswith("**Typed results.**")]
    assert len(idx) == 1 and len(typed) == 1, (idx, typed)
    assert idx[0] == typed[0] + 1, (idx, typed)


def test_eb9_host_hints_paragraph_names_terms_and_open_world_reason():
    para = _host_hints_paragraph()
    for needle in ("instructions", *EXPECTED_HINTS):
        assert needle in para, f"{needle!r} missing from Host hints paragraph"
    assert "extract" in para and "external model" in para.lower(), para


def test_eb9_host_hints_paragraph_two_way_with_server_keys(surface):
    para = _host_hints_paragraph()
    server_keys = set()
    for tool in surface["tools"].values():
        server_keys.update(_hints(tool))
    assert server_keys, "server sets no annotation keys"
    missing = sorted(k for k in server_keys if k not in para)
    assert not missing, f"server keys absent from README: {missing}"
    readme_hints = set(re.findall(r"\b[a-z][A-Za-z]*Hint\b", para))
    extra = sorted(readme_hints - server_keys)
    assert not extra, f"README names hints the server does not set: {extra}"


# --- EB10 ------------------------------------------------------------------

_GIT_CMD = re.compile(r"^\s*git\s+\S")
_PROC_MODULES = {"subprocess", "pygit2", "git", "dulwich"}


def test_eb10_module_uses_no_git_or_subprocess():
    tree = ast.parse(THIS_FILE.read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _PROC_MODULES:
                    problems.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _PROC_MODULES:
                problems.append(f"from {node.module} import")
        elif isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "os"
                and f.attr in {"system", "popen", "execv", "execvp", "spawnv"}
            ):
                problems.append(f"os.{f.attr}()")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _GIT_CMD.match(node.value):
                problems.append(f"command literal {node.value!r}")
    assert not problems, problems
