"""Black-box behavioral tests for iter-115 (#81): ``validate.data`` publishes its fields.

The MCP ``validate`` tool's ``data`` argument used to be advertised in
``tools/list`` as a bare ``{"type": "object", "additionalProperties": true}``.
It now carries ``properties`` equal to ``InvoiceData.model_json_schema()["properties"]``
so a host model learns the field names, types and descriptions, while
argument *validation* (``call_tool``) is unchanged.

  1  data.properties == InvoiceData.model_json_schema()["properties"]
  2  data keeps type=object, additionalProperties=true, no ``required``
  3  self-contained: no ``$ref`` anywhere, no ``$defs`` in validate's inputSchema
  4  rest of tools/list unchanged (extract entry, validate entry minus the new key,
     instructions), pinned to digests measured on the pre-change tree
  5  call_tool validate, correct bare fields -> same envelope as pre-change
  6  call_tool validate, stored result envelope as data -> same as pre-change
  7  call_tool validate, data={"totl": 1} -> same unknown-field error envelope
  8  non-object data (list / string) -> same outcome class as pre-change
  9  (checked outside the suite: CLI/schema/heuristics/gate files and make eval)
  10 drift guard: properties recomputed from InvoiceData at test time

Offline: no keys, no network; ``heuristics_only=True`` builds no provider client.
Server-surface tests skip when the optional ``[server]`` extra (``mcp``) is absent.

Isolation contract: written from the spec, README, existing tests and the
product's runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import re
import sys

import pytest

from reflect_mcp.schema import InvoiceData

T = "Acme\nInvoice INV-1\nDate 2024-01-05\nSubtotal 100.00\nTax 10.00\nTotal 110.00 USD\n"
FIELDS = {"vendor_name": "Acme", "invoice_number": "INV-1", "invoice_date": "2024-01-05",
          "currency": "USD", "subtotal": 100.0, "tax": 10.0, "total": 110.0}
WRONG = dict(FIELDS, total=999.0)
SEVEN = {"vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency"}

# sha256 digests measured on a `git archive` export of the pre-change tree
# (484eec4) by running this module's own helpers. Fixed literals, never git show HEAD.
PRE = {
    "extract_entry":
        "c070dc9fa5b8c4b6c796abfb40f39be501a877c697cfa792624735d813acbd2a",
    "validate_entry_minus_data_properties":
        "a06a39b1ac845dd7aa666e5bed3de68e47be5fc6f532eded7a68820ab8b36408",
    "instructions":
        "32dfd5748c68a9e7f0f300588751056106c7257178b710575a1e40c1611e4625",
    "tool_names":
        "3eff02403709fd340287dbf33113613fd76d1167c6048bd5575e630d801bc0d0",
    "eb5_bare_ok":
        "49e471acb1db1b856b43d8075315413065c0567d88384b07a7251cea62a2504f",
    "eb5_bare_wrong":
        "576c038f0c2b057f8553a62f7d92a56d306279940bf30efbb5eec7372b682c0f",
    "eb6_stored_envelope":
        "576c038f0c2b057f8553a62f7d92a56d306279940bf30efbb5eec7372b682c0f",
    "eb7_typo":
        "045dfd5ab67dfa6fdd63476e00664dee8b4c8911cdf20681cff3192aa3d6849e",
    "eb8_list":
        "13f24d32623dd6a148c682cb2b7405116477974d59350e0923e06e0d5617ab27",
    "eb8_string":
        "69094c18bf7dec05dc8d49e9821f0b6497da38fd23067a49de03d43a6bd7daad",
}


def _sha(obj) -> str:
    blob = re.sub(r"https://errors\.pydantic\.dev/\S+", "<url>",
                  json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_server():
    sys.modules.pop("reflect_mcp.server", None)
    server = importlib.import_module("reflect_mcp.server")
    if not hasattr(server.mcp, "_mcp_server"):
        raise RuntimeError("stub FastMCP")
    return server


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REFLECT_MCP_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    try:
        try:
            srv = _load_server()
        except RuntimeError:
            pytest.skip("stub FastMCP in sys.modules; real SDK required")
        yield srv
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _entries(server) -> dict:
    tools = asyncio.run(server.mcp.list_tools())
    return {t.name: json.loads(json.dumps(t.model_dump(mode="json"))) for t in tools}


def _data_schema(server) -> dict:
    return _entries(server)["validate"]["inputSchema"]["properties"]["data"]


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()
                if k not in ("trace_id", "ts", "elapsed_ms", "duration_ms", "latency_ms")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _norm_result(res):
    """Normalise FastMCP.call_tool's return to plain JSON (trace_id/timing stripped)."""
    if hasattr(res, "model_dump"):
        dumped = res.model_dump(mode="json")
        blocks = dumped.get("content") or []
        return {
            "kind": "result",
            "isError": dumped.get("isError"),
            "structured": _strip(dumped.get("structuredContent")),
            "content": [_strip(_maybe_json(b.get("text"))) for b in blocks],
        }
    blocks = res[0] if isinstance(res, tuple) else res
    return {"kind": "result", "isError": None, "structured": None,
            "content": [_strip(_maybe_json(getattr(b, "text", None))) for b in blocks]}


def _maybe_json(txt):
    try:
        return json.loads(txt)
    except (TypeError, ValueError):
        return txt


def _payload(outcome: dict) -> dict:
    """The tool's envelope: structuredContent, else the single JSON text block."""
    if outcome.get("structured"):
        return outcome["structured"]
    return outcome["content"][0]


def _call(server, args: dict) -> dict:
    try:
        res = asyncio.run(server.mcp.call_tool("validate", args))
    except Exception as exc:  # noqa: BLE001 - the outcome class is what we pin
        msg = re.sub(r"https://errors\.pydantic\.dev/\S+", "<url>", str(exc))
        return {"kind": "raised", "type": type(exc).__name__, "message": msg}
    return _norm_result(res)


def _scenarios() -> dict:
    stored = {"status": "ok", "data": dict(WRONG), "notes": "stored", "overall_confidence": 0.9}
    return {
        "eb5_bare_ok": {"text": T, "data": dict(FIELDS), "heuristics_only": True},
        "eb5_bare_wrong": {"text": T, "data": dict(WRONG), "heuristics_only": True},
        "eb6_stored_envelope": {"text": T, "data": stored, "heuristics_only": True},
        "eb7_typo": {"text": T, "data": {"totl": 1}, "heuristics_only": True},
        "eb8_list": {"text": T, "data": [1, 2], "heuristics_only": True},
        "eb8_string": {"text": T, "data": "total=110", "heuristics_only": True},
    }


def _outcomes(server) -> dict:
    return {name: _call(server, args) for name, args in _scenarios().items()}


def _surface_digests(server) -> dict:
    entries = _entries(server)
    val = copy.deepcopy(entries["validate"])
    val["inputSchema"]["properties"]["data"].pop("properties", None)
    out = {
        "extract_entry": _sha(entries["extract"]),
        "validate_entry_minus_data_properties": _sha(val),
        "instructions": _sha(server.mcp.instructions),
        "tool_names": _sha(sorted(entries)),
    }
    for name, outcome in _outcomes(server).items():
        out[name] = _sha(outcome)
    return out


# --------------------------------------------------------------------------- EB1

def test_eb1_data_properties_equal_invoice_schema(server):
    assert _data_schema(server).get("properties") == InvoiceData.model_json_schema()["properties"]


def test_eb1_seven_field_keys(server):
    assert set(_data_schema(server)["properties"]) == SEVEN


def test_eb1_field_types_descriptions_defaults(server):
    props = _data_schema(server)["properties"]
    for name in ("subtotal", "tax", "total"):
        assert props[name]["anyOf"] == [{"type": "number"}, {"type": "null"}], name
    for name in ("vendor_name", "invoice_number", "invoice_date", "currency"):
        assert {"type": "null"} in props[name]["anyOf"], name
        assert {"type": "string"} in props[name]["anyOf"], name
    for name, spec in props.items():
        assert spec.get("default", "MISSING") is None, name
        assert isinstance(spec.get("description"), str) and spec["description"].strip(), name
    assert props["vendor_name"]["description"] == "Legal/trading name of the seller."
    assert props["total"]["description"] == "Grand total charged."


# --------------------------------------------------------------------------- EB2

def test_eb2_object_open_and_no_required(server):
    data = _data_schema(server)
    assert data["type"] == "object"
    assert data["additionalProperties"] is True
    assert "required" not in data


def test_eb2_data_stays_optional_at_tool_level(server):
    # Pre-change the validate tool required both text and data; publishing the
    # fields must not move ``data``'s own optionality inside its schema.
    schema = _entries(server)["validate"]["inputSchema"]
    assert "data" in schema["properties"]
    assert "required" not in schema["properties"]["data"]


def test_eb2_client_side_schema_accepts_bare_and_stored_and_partial(server):
    jsonschema = pytest.importorskip("jsonschema")
    schema = _entries(server)["validate"]["inputSchema"]
    for data in (dict(FIELDS), {}, {"total": 110.0},
                 {"status": "ok", "data": dict(FIELDS), "notes": "x"},
                 {"vendor_name": None, "total": None}):
        jsonschema.validate({"text": T, "data": data}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"text": T, "data": [1, 2]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"text": T, "data": {"total": [1]}}, schema)


# --------------------------------------------------------------------------- EB3

def test_eb3_no_ref_anywhere_in_validate_schema(server):
    schema = _entries(server)["validate"]["inputSchema"]
    assert len(schema["properties"]["data"]["properties"]) == len(SEVEN)
    blob = json.dumps(schema)
    assert "$ref" not in blob
    assert "$defs" not in schema
    assert "definitions" not in schema
    assert "$ref" not in json.dumps(_data_schema(server))


# --------------------------------------------------------------------------- EB4

def test_eb4_extract_entry_unchanged(server):
    assert _sha(_entries(server)["extract"]) == PRE["extract_entry"]


def test_eb4_validate_entry_minus_new_key_unchanged(server):
    val = copy.deepcopy(_entries(server)["validate"])
    assert "properties" in val["inputSchema"]["properties"]["data"]
    val["inputSchema"]["properties"]["data"].pop("properties")
    assert _sha(val) == PRE["validate_entry_minus_data_properties"]


def test_eb4_instructions_and_tool_names_unchanged(server):
    assert _sha(server.mcp.instructions) == PRE["instructions"]
    assert _sha(sorted(_entries(server))) == PRE["tool_names"]


def test_eb4_only_data_gained_properties(server):
    entries = _entries(server)
    props = entries["validate"]["inputSchema"]["properties"]
    for name, spec in props.items():
        if name != "data":
            assert "properties" not in spec, name
    assert "vendor_name" not in json.dumps(entries["extract"]["inputSchema"])


# --------------------------------------------------------------------------- EB5

def test_eb5_bare_correct_fields_ok_and_unchanged(server):
    out = _call(server, _scenarios()["eb5_bare_ok"])
    env = _payload(out)
    assert out["kind"] == "result" and env["status"] == "ok", env
    assert env["data"] == FIELDS
    assert env["field_confidence"] == {k: 1.0 for k in SEVEN}
    assert env["flagged_fields"] == []
    assert _sha(out) == PRE["eb5_bare_ok"]


def test_eb5_bare_wrong_total_flagged_and_unchanged(server):
    out = _call(server, _scenarios()["eb5_bare_wrong"])
    env = _payload(out)
    assert env["status"] == "needs_review", env
    assert [f["field"] for f in env["flagged_fields"]] == ["total"]
    assert _sha(out) == PRE["eb5_bare_wrong"]


# --------------------------------------------------------------------------- EB6

def test_eb6_stored_envelope_verifies_inner_data(server):
    out = _call(server, _scenarios()["eb6_stored_envelope"])
    env = _payload(out)
    assert out["kind"] == "result" and out["isError"] is not True
    assert env["status"] == "needs_review", env
    assert env["data"] == WRONG
    assert [f["field"] for f in env["flagged_fields"]] == ["total"]
    # Same verdict as passing the inner fields bare.
    assert env == _payload(_call(server, _scenarios()["eb5_bare_wrong"]))
    assert _sha(out) == PRE["eb6_stored_envelope"]


# --------------------------------------------------------------------------- EB7

def test_eb7_typo_key_is_unknown_field_envelope_not_arg_error(server):
    out = _call(server, _scenarios()["eb7_typo"])
    assert out["kind"] == "result", out
    env = _payload(out)
    assert env["status"] == "error"
    assert "unknown field(s): totl" in env["error_message"]
    assert _sha(out) == PRE["eb7_typo"]


def test_eb7_wrong_typed_field_still_reaches_tool(server):
    # json_schema_extra publishes types only; a string amount is still coerced by
    # the tool's own InvoiceData validation, not rejected by argument validation.
    out = _call(server, {"text": T, "data": dict(FIELDS, total="110.00"),
                         "heuristics_only": True})
    assert out["kind"] == "result", out
    env = _payload(out)
    assert env["status"] == "ok", env
    assert env["data"]["total"] == 110.0


# --------------------------------------------------------------------------- EB8

@pytest.mark.parametrize("name", ["eb8_list", "eb8_string"])
def test_eb8_non_object_data_rejected_as_before(server, name):
    out = _call(server, _scenarios()[name])
    assert out["kind"] == "raised", out
    assert out["type"] == "ToolError"
    assert "valid dictionary" in out["message"]
    assert _sha(out) == PRE[name]


# --------------------------------------------------------------------------- EB10

def test_eb10_drift_guard_recomputed_at_test_time(server):
    live = InvoiceData.model_json_schema()["properties"]
    published = _data_schema(server)["properties"]
    assert set(published) == set(InvoiceData.model_fields)
    assert published == live
    for name, field in InvoiceData.model_fields.items():
        assert published[name].get("description") == field.description, name


def test_eb10_published_copy_is_not_the_live_model_dict(server):
    # Mutating a list_tools() result must not corrupt InvoiceData's own schema.
    before = copy.deepcopy(InvoiceData.model_json_schema())
    tools = asyncio.run(server.mcp.list_tools())
    for t in tools:
        if t.name == "validate":
            t.inputSchema["properties"]["data"].get("properties", {}).pop("total", None)
    assert InvoiceData.model_json_schema() == before
