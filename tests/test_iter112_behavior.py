"""Black-box behavioral tests for iter-112: host-visible MCP prose agrees with the payload.

Since iter 111 a Critic-failure error envelope carries the Generator's
*unverified* ``data``. The MCP ``instructions`` and the ``extract`` tool
description (the only prose a host model reads) must say so, and must say
that ``overall_confidence`` is a synthetic 1.0 when no Critic ran.

  1  instructions: no "nothing was extracted"; mention unverified + validate; <= 600 chars
  2  extract description: error shape {status, error_message} + Critic-failure unverified data
  3  extract description: overall_confidence line says synthetic 1.0 when no Critic ran
  4  drift: the live payload (Critic-failure envelope has data; heuristics_only -> 1.0)
     agrees with the prose, in the same test
  5  README + schema envelope docstring qualify "stored error row"
  6  behaviour unchanged: validate description, schemas, payloads pinned to
     digests measured on the pre-change tree (fixed literals, never git show HEAD)

Offline: ScriptedLLMClient only, no keys, no network. Server-surface tests skip
when the optional ``[server]`` extra (``mcp``) is absent.

Isolation contract: written from the spec, README, existing tests and the
product's runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_mod
from reflect_mcp import ReflectivePipeline
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"

G = ('{"vendor_name":"Acme","invoice_number":"INV-1","invoice_date":"2024-01-05",'
     '"currency":"USD","subtotal":100.0,"tax":10.0,"total":110.0}')
T = "Acme\nInvoice INV-1\nDate 2024-01-05\nSubtotal 100.00\nTax 10.00\nTotal 110.00 USD\n"
BAD = "not json"
STORED_ROW = "stored error row from an input, provider or Generator failure"

# sha256 digests measured on an export of the pre-change tree (d758b8f).
# They pin behaviour that iter 112 must NOT move (EB6).
PRE_VALIDATE_DESCRIPTION_SHA256 = "4378c5c06f3db2dc1ede3c347de07a5143149ad2b60082667635dad125810556"
# Re-hashed iter 115 #81: validate.data publishes the InvoiceData field properties.
PRE_TOOL_SCHEMAS_SHA256 = "fd7534ae2dcffe26480dbb22a9f055f613148fd887cafb14e7cd66ddafcb4aae"
PRE_MODEL_JSON_SCHEMA_SHA256 = "e25030a2013c33f5d110fc25f638957abfa3ff8de56abb9e3d73eaae9eb12d3e"
PRE_PAYLOADS_SHA256 = "404b3c2c32b6afbc45a4aba600d7cbfc04c935951d0dfd088f94428fe01e4b96"


def _flat(s: str) -> str:
    """Collapse whitespace so a phrase split by a docstring line break still matches."""
    return " ".join(s.split())


def _sha(obj) -> str:
    blob = re.sub(r"https://errors\.pydantic\.dev/\S+", "<url>",
                  json.dumps(obj, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- helpers

def _pipe(gen=None, critic=None) -> ReflectivePipeline:
    g = Generator(gen) if gen is not None else None
    c = Critic(critic) if critic is not None else None
    return ReflectivePipeline(g, c)


@contextlib.contextmanager
def _factory(fn):
    """Patch ``build_default_pipeline`` wherever it is bound (iter64/111 pattern)."""
    real = pipeline_mod.build_default_pipeline
    patched = [(pipeline_mod, "build_default_pipeline", real)]
    pipeline_mod.build_default_pipeline = fn
    for name in ("reflect_mcp", "reflect_mcp.tools", "reflect_mcp.cli", "reflect_mcp.__main__"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if getattr(mod, "build_default_pipeline", None) is real:
            patched.append((mod, "build_default_pipeline", real))
            mod.build_default_pipeline = fn
    try:
        yield
    finally:
        for mod, attr, val in patched:
            setattr(mod, attr, val)


def _tool(name: str, factory, *args, **kwargs) -> dict:
    from reflect_mcp import tools

    with _factory(factory):
        out = getattr(tools, name)(*args, **kwargs)
    out = dict(out)
    out.pop("trace_id", None)
    return out


def _critic_fail_factory():
    gen, crit = ScriptedLLMClient([G]), ScriptedLLMClient([BAD, BAD])
    p = _pipe(gen, crit)
    return (lambda *a, **k: p), gen, crit


def _golden(i: int = 0) -> dict:
    lines = [json.loads(x) for x in GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[i]


def _heur_factory(doc: dict):
    p = _pipe(ScriptedLLMClient([json.dumps(doc["expected"])]), None)
    return lambda *a, **k: p


@pytest.fixture
def surface(monkeypatch):
    """Real MCP server surface: instructions + tools/list (skip without the extra)."""
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REFLECT_MCP_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("reflect_mcp.server", None)
    try:
        server = importlib.import_module("reflect_mcp.server")
        if not hasattr(server.mcp, "_mcp_server"):
            pytest.skip("stub FastMCP in sys.modules; real SDK required")
        tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
        yield {
            "server": server,
            "instructions": server.mcp.instructions,
            "tools": tools,
            "extract": tools["extract"].description or "",
            "validate": tools["validate"].description or "",
        }
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _confidence_lines(desc: str) -> list[str]:
    return [ln for ln in desc.splitlines() if "overall_confidence" in ln]


# --------------------------------------------------------------------------- EB1

def test_eb1_instructions_drop_nothing_was_extracted(surface):
    flat = _flat(surface["instructions"])
    assert "nothing was extracted" not in flat
    assert "no extraction happened" not in flat


def test_eb1_instructions_mention_unverified_and_validate(surface):
    ins = surface["instructions"]
    assert "unverified" in ins
    assert "validate" in ins


def test_eb1_instructions_cap_and_existing_terms(surface):
    ins = surface["instructions"]
    assert len(ins) <= 600, len(ins)
    assert "needs_review" in ins
    assert "flagged_fields" in ins


def test_eb1_instructions_equal_module_constant(surface):
    assert surface["instructions"] == surface["server"]._INSTRUCTIONS


# --------------------------------------------------------------------------- EB2

def test_eb2_extract_description_drops_no_extraction_happened(surface):
    flat = _flat(surface["extract"])
    assert "no extraction happened" not in flat
    assert "nothing was extracted" not in flat


def test_eb2_extract_description_mentions_unverified_and_validate(surface):
    desc = surface["extract"]
    assert "unverified" in desc
    assert "validate" in desc


def test_eb2_extract_description_states_error_shape(surface):
    flat = _flat(surface["extract"])
    assert re.search(r"\{\s*status\s*,\s*error_message\s*\}", flat), flat
    assert 'status == "error"' in flat


def test_eb2_extract_description_states_critic_failure_salvage(surface):
    flat = _flat(surface["extract"])
    # one sentence ties Critic failure -> Generator's unverified data
    sentences = re.split(r"(?<=[.:])\s+", flat)
    hits = [s for s in sentences
            if "critic failure" in s.lower() and "generator" in s.lower()
            and "unverified" in s and "data" in s]
    assert hits, flat


def test_eb2_extract_docstring_equals_tool_description(surface):
    import inspect

    doc = inspect.getdoc(surface["server"].extract) or ""
    assert _flat(doc) == _flat(surface["extract"])


# --------------------------------------------------------------------------- EB3

def test_eb3_overall_confidence_line_says_synthetic_without_critic(surface):
    lines = _confidence_lines(surface["extract"])
    assert lines, surface["extract"]
    good = [ln for ln in lines if "1.0" in ln and "no critic" in ln.lower()]
    assert good, lines
    assert any("synthetic" in ln.lower() for ln in good), good


def test_eb3_overall_confidence_still_describes_critic_value(surface):
    lines = _confidence_lines(surface["extract"])
    assert any("critic" in ln.lower() and "overall confidence" in ln.lower() for ln in lines), lines


# --------------------------------------------------------------------------- EB4 (drift, payload-tied)

def test_eb4_critic_failure_payload_matches_prose(surface):
    fac, gen, crit = _critic_fail_factory()
    out = _tool("extract_tool", fac, T)
    assert out["status"] == "error"
    assert "data" in out
    assert out["data"] == InvoiceData.model_validate_json(G).model_dump()
    assert isinstance(out.get("error_message"), str) and out["error_message"]
    # the very strings a host reads must describe this payload
    for prose in (surface["instructions"], surface["extract"]):
        assert "unverified" in prose and "validate" in prose
        assert "nothing was extracted" not in _flat(prose)
        assert "no extraction happened" not in _flat(prose)
    # and the advice works: passing the whole envelope to validate re-verifies it
    from reflect_mcp import tools

    back = dict(tools.validate_tool(T, out, heuristics_only=True))
    assert back["status"] in ("ok", "needs_review")
    assert back["data"] == out["data"]


def test_eb4_heuristics_only_payload_is_synthetic_one(surface):
    doc = _golden(0)
    out = _tool("extract_tool", _heur_factory(doc), doc["text"], heuristics_only=True)
    assert out["status"] in ("ok", "needs_review"), out
    assert out["overall_confidence"] == 1.0
    lines = _confidence_lines(surface["extract"])
    assert any("1.0" in ln and "no critic" in ln.lower() and "synthetic" in ln.lower()
               for ln in lines), lines


def test_eb4_payload_facts_hold_without_server_extra():
    """The tools-level half of the drift check needs no ``mcp`` extra."""
    fac, _, _ = _critic_fail_factory()
    out = _tool("extract_tool", fac, T)
    assert out["status"] == "error" and "data" in out
    doc = _golden(0)
    heur = _tool("extract_tool", _heur_factory(doc), doc["text"], heuristics_only=True)
    assert heur["overall_confidence"] == 1.0


# --------------------------------------------------------------------------- EB5

def test_eb5_readme_qualifies_stored_error_row_twice():
    text = _flat(README.read_text(encoding="utf-8"))
    assert text.count(STORED_ROW) >= 2, text.count(STORED_ROW)


def test_eb5_readme_no_unqualified_stored_error_row():
    text = _flat(README.read_text(encoding="utf-8"))
    total = text.count("stored error row")
    assert total == text.count(STORED_ROW), (total, text.count(STORED_ROW))


def test_eb5_schema_unwrap_docstring_qualified():
    doc = _flat(InvoiceData._unwrap_result_envelope.__doc__ or "")
    assert "Generator failure" in doc, doc
    assert "input" in doc and "provider" in doc, doc


# --------------------------------------------------------------------------- EB6

def _payloads() -> dict:
    from reflect_mcp import tools

    fac, _, _ = _critic_fail_factory()
    doc = _golden(0)
    s = {
        "extract_critic_fail": _tool("extract_tool", fac, T),
        "extract_heur_golden": _tool("extract_tool", _heur_factory(doc), doc["text"],
                                     heuristics_only=True),
        "extract_empty": _tool("extract_tool", _heur_factory(doc), ""),
        "extract_bad_threshold": _tool("extract_tool", _heur_factory(doc), T,
                                       field_threshold=2.0),
    }
    for name, data in (("plain", json.loads(G)), ("envelope", s["extract_critic_fail"]),
                       ("error_row", {"status": "error", "error_message": "x"}),
                       ("typo", {"vendr_name": "Acme"})):
        out = dict(tools.validate_tool(T, data, heuristics_only=True))
        out.pop("trace_id", None)
        s[f"validate_{name}"] = out
    return s


def _tool_schemas(tools: dict) -> dict:
    return {n: {"input": t.inputSchema, "output": t.outputSchema} for n, t in sorted(tools.items())}


def test_eb6_validate_description_unchanged(surface):
    assert hashlib.sha256(surface["validate"].encode("utf-8")).hexdigest() == \
        PRE_VALIDATE_DESCRIPTION_SHA256


def test_eb6_tool_schemas_unchanged(surface):
    assert set(surface["tools"]) == {"extract", "validate"}
    assert _sha(_tool_schemas(surface["tools"])) == PRE_TOOL_SCHEMAS_SHA256


def test_eb6_model_json_schema_unchanged():
    assert _sha(InvoiceData.model_json_schema()) == PRE_MODEL_JSON_SCHEMA_SHA256


def test_eb6_tool_payloads_unchanged():
    assert _sha(_payloads()) == PRE_PAYLOADS_SHA256
