"""Black-box behavioral tests for iter-111: Critic-failure salvage.

When the Critic stage of ``extract`` fails with ``CriticError``, the error
envelope (MCP ``extract_tool``, CLI ``extract``, ``extract --batch`` rows) also
carries the Generator's *unverified* extraction as ``data``; the library's
``CriticError`` exposes it as ``.data`` (``None`` from ``.validate()``).

  1  library extract: CriticError.data == Generator dump, 1 Gen + 2 Critic calls
  2  library validate: CriticError.data is None
  3  MCP extract_tool envelope {status, error_message, data}
  4  MCP round trip: envelope -> validate_tool == validate_tool(envelope["data"])
  5  CLI single doc: exit 3, envelope with data; pipe into validate --data -
  6  CLI batch: error row {id, data, status, error_message}, id first
  7  only a CriticError inside extract adds data (all other errors unchanged)
  8  success paths and schemas unchanged
  9  golden-derived: salvage == plain replay data; re-validate == plain replay
 10  make eval figures and golden files unchanged

Offline: ScriptedLLMClient only, no keys, no network. Pre-change bytes are pinned
as fixed literals / sha256 digests measured on an export of the pre-change tree
(never a ``git show HEAD`` baseline), so the tests survive the ship commit.

Isolation contract: written from the spec, README, existing tests and the
product's runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_mod
from reflect_mcp import ReflectivePipeline
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.schema import CritiqueResult, ExtractionResult, FieldVerdict, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency"]
G = ('{"vendor_name":"Acme","invoice_number":"INV-1","invoice_date":"2024-01-05",'
     '"currency":"USD","subtotal":100.0,"tax":10.0,"total":110.0}')
T = "Acme\nInvoice INV-1\nDate 2024-01-05\nSubtotal 100.00\nTax 10.00\nTotal 110.00 USD\n"
BAD = "not json"
CRITIC_MSG = "Critic produced invalid critique: No JSON object found in model output: 'not json'"


def _critique(conf: float) -> str:
    return json.dumps({
        "field_verdicts": [
            {"field": f, "confidence": conf, "grounded": True, "rationale": "ok"} for f in FIELDS
        ],
        "overall_confidence": conf,
        "summary": "ok",
    })


GOODC = _critique(0.95)
LOWC = _critique(0.30)
GDICT = InvoiceData.model_validate_json(G).model_dump()


class _DataErr(Exception):
    """A foreign exception that happens to carry a ``.data`` attribute."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.data = {"x": 1}


class _Raiser:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        raise self.exc


def _pipe(gen=None, critic=None) -> ReflectivePipeline:
    g = Generator(gen) if gen is not None else None
    c = Critic(critic) if critic is not None else None
    return ReflectivePipeline(g, c)


def _fail_pipe() -> tuple[ReflectivePipeline, ScriptedLLMClient, ScriptedLLMClient]:
    g, c = ScriptedLLMClient([G]), ScriptedLLMClient([BAD, BAD])
    return _pipe(g, c), g, c


@contextlib.contextmanager
def _factory(fn):
    """Patch ``build_default_pipeline`` wherever it is bound (iter64 pattern)."""
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


def _fixed(p: ReflectivePipeline):
    return lambda *a, **k: p


def _main(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    from reflect_mcp.__main__ import main

    out, err = io.StringIO(), io.StringIO()
    old = sys.argv, sys.stdin
    sys.argv, sys.stdin = ["reflect-mcp", *argv], io.StringIO(stdin)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main()
            except SystemExit as e:
                rc = e.code
    finally:
        sys.argv, sys.stdin = old
    return (0 if rc is None else rc), out.getvalue(), err.getvalue()


def _norm_out(stdout: str) -> list[dict]:
    """Parse stdout as one JSON doc or as JSONL; drop the random trace_id."""
    stdout = stdout.strip()
    if not stdout:
        return []
    try:
        docs = [json.loads(stdout)]
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    for d in docs:
        if isinstance(d, dict):
            d.pop("trace_id", None)
    return docs


def _err_lines(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if line.startswith("Error")]


def _digest(obj) -> str:
    # pydantic's docs URL embeds its version; keep the pin independent of it.
    blob = re.sub(r"https://errors\.pydantic\.dev/\S+", "<url>", json.dumps(obj, ensure_ascii=False))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tool(name: str, factory, *args, **kwargs) -> dict:
    from reflect_mcp import tools

    with _factory(factory):
        out = getattr(tools, name)(*args, **kwargs)
    out = dict(out)
    out.pop("trace_id", None)
    return out


def _cli(factory, argv: list[str], stdin: str = "") -> dict:
    with _factory(factory):
        rc, out, err = _main(argv, stdin)
    return {"rc": rc, "out": _norm_out(out), "err": _err_lines(err)}


def _batch_line(i: str | None, text: str = T, data: dict | None = None) -> str:
    rec: dict = {} if i is None else {"id": i}
    rec["text"] = text
    if data is not None:
        rec["data"] = data
    return json.dumps(rec)


@pytest.fixture()
def doc(tmp_path: Path) -> str:
    p = tmp_path / "invoice.txt"
    p.write_text(T, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- EB1


def test_eb1_library_extract_criticerror_carries_generator_data():
    p, g, c = _fail_pipe()
    with pytest.raises(CriticError) as ei:
        p.extract(T)
    assert str(ei.value) == CRITIC_MSG
    assert ei.value.data == GDICT
    assert set(ei.value.data) == set(FIELDS)
    assert len(g.calls) == 1
    assert len(c.calls) == 2


def test_eb1_data_includes_null_fields():
    partial = '{"vendor_name":"Acme","total":110.0}'
    p = _pipe(ScriptedLLMClient([partial]), ScriptedLLMClient([BAD, BAD]))
    with pytest.raises(CriticError) as ei:
        p.extract(T)
    assert ei.value.data == InvoiceData.model_validate_json(partial).model_dump()
    assert set(ei.value.data) == set(FIELDS)
    assert ei.value.data["invoice_number"] is None


def test_eb1_plain_criticerror_defaults_to_none():
    assert getattr(CriticError("x"), "data", "MISSING") is None


# --------------------------------------------------------------------------- EB2


def test_eb2_library_validate_criticerror_data_is_none():
    c = ScriptedLLMClient([BAD, BAD])
    with pytest.raises(CriticError) as ei:
        _pipe(None, c).validate(T, InvoiceData.model_validate_json(G))
    assert str(ei.value) == CRITIC_MSG
    assert ei.value.data is None
    assert len(c.calls) == 2


# --------------------------------------------------------------------------- EB3


def test_eb3_mcp_extract_tool_envelope_has_data():
    p, _, _ = _fail_pipe()
    out = _tool("extract_tool", _fixed(p), T)
    assert set(out) == {"status", "error_message", "data"}
    assert out["status"] == "error"
    assert out["error_message"] == CRITIC_MSG
    assert out["data"] == GDICT
    json.dumps(out)


# --------------------------------------------------------------------------- EB4


def test_eb4_mcp_round_trip_reverifies_without_generator():
    p, _, _ = _fail_pipe()
    env = _tool("extract_tool", _fixed(p), T)
    assert "data" in env

    g1 = ScriptedLLMClient([G])
    via_env = _tool("validate_tool", _fixed(_pipe(g1, ScriptedLLMClient([GOODC]))), T, env)
    g2 = ScriptedLLMClient([G])
    via_data = _tool("validate_tool", _fixed(_pipe(g2, ScriptedLLMClient([GOODC]))), T, env["data"])
    assert via_env == via_data
    assert via_env["status"] in ("ok", "needs_review")
    assert via_env["data"] == GDICT
    assert len(g1.calls) == 0 and len(g2.calls) == 0


# --------------------------------------------------------------------------- EB5


def test_eb5_cli_single_doc_envelope_and_pipe(doc):
    p, _, _ = _fail_pipe()
    with _factory(_fixed(p)):
        rc, out, _ = _main(["extract", "--file", doc])
    assert rc == 3
    env = json.loads(out)  # exactly one JSON object on stdout
    assert set(env) == {"status", "error_message", "data"}
    assert env["status"] == "error"
    assert env["error_message"] == CRITIC_MSG
    assert env["data"] == GDICT

    gp = ScriptedLLMClient([G])
    piped = _cli(_fixed(_pipe(gp, ScriptedLLMClient([GOODC]))),
                 ["validate", "--file", doc, "--data", "-"], out)
    inline = _cli(_fixed(_pipe(ScriptedLLMClient([G]), ScriptedLLMClient([GOODC]))),
                  ["validate", "--file", doc, "--data", json.dumps(GDICT)])
    assert piped["rc"] == inline["rc"] == 0
    assert piped["out"] == inline["out"]
    assert piped["out"][0]["status"] == "ok"
    assert len(gp.calls) == 0


# --------------------------------------------------------------------------- EB6


def test_eb6_batch_error_row_carries_data_id_first():
    def fac(*a, **k):
        return _pipe(ScriptedLLMClient([G, G, G]),
                     ScriptedLLMClient([GOODC, BAD, BAD, GOODC]))

    batch = "\n".join(_batch_line(i) for i in "abc") + "\n"
    with _factory(fac):
        rc, out, _ = _main(["extract", "--batch"], batch)
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    b = rows[1]
    assert list(b)[0] == "id"
    assert set(b) == {"id", "data", "status", "error_message"}
    assert b["status"] == "error" and b["error_message"] == CRITIC_MSG
    assert b["data"] == GDICT
    assert rc == 3  # pre-change exit code for the same script (measured)

    # rows a/c equal the rows of an all-ok batch of the same records
    def ok(*a, **k):
        return _pipe(ScriptedLLMClient([G, G, G]), ScriptedLLMClient([GOODC, GOODC, GOODC]))

    ok_rows = _cli(ok, ["extract", "--batch"], batch)["out"]
    norm = _norm_out(out)
    assert norm[0] == ok_rows[0] and norm[2] == ok_rows[2]
    assert ok_rows[0]["status"] == "ok"


def test_eb6_batch_record_without_id():
    fac = _fixed(_pipe(ScriptedLLMClient([G]), ScriptedLLMClient([BAD, BAD])))
    res = _cli(fac, ["extract", "--batch"], _batch_line(None) + "\n")
    assert res["rc"] == 3
    assert len(res["out"]) == 1
    row = res["out"][0]
    assert set(row) == {"data", "status", "error_message"}
    assert row["data"] == GDICT and row["status"] == "error"


# --------------------------------------------------------------------------- EB7
# Each scenario's (exit code, stdout, stderr "Error" lines) / tool envelope is
# pinned to the digest measured on the pre-change tree.


def _gen_err_pipe(*a, **k):
    return _pipe(ScriptedLLMClient([BAD, BAD]), ScriptedLLMClient([GOODC]))


def _critic_conn_pipe(*a, **k):
    return _pipe(ScriptedLLMClient([G]), _Raiser(ConnectionError("network down")))


def _gen_dataerr_pipe(*a, **k):
    return _pipe(_Raiser(_DataErr("foreign boom")), ScriptedLLMClient([GOODC]))


def _factory_err(*a, **k):
    raise RuntimeError("factory exploded")


def _critic_fail_validate_pipe(*a, **k):
    return _pipe(ScriptedLLMClient([G]), ScriptedLLMClient([BAD, BAD]))


def _ok_pipe(*a, **k):
    return _pipe(ScriptedLLMClient([G]), ScriptedLLMClient([GOODC]))


def _error_scenarios(doc: str) -> dict:
    gd = json.dumps(GDICT)
    vbatch = _batch_line("v", data=GDICT) + "\n"
    s = {}
    for name, fac in (("gen_err", _gen_err_pipe), ("critic_conn", _critic_conn_pipe),
                      ("gen_dataerr", _gen_dataerr_pipe), ("factory_err", _factory_err)):
        s[f"cli_extract_{name}"] = _cli(fac, ["extract", "--file", doc])
        s[f"cli_batch_{name}"] = _cli(fac, ["extract", "--batch"], _batch_line("x") + "\n")
        s[f"mcp_extract_{name}"] = _tool("extract_tool", fac, T)
    v = _critic_fail_validate_pipe
    s["cli_validate_critic_fail"] = _cli(v, ["validate", "--file", doc, "--data", gd])
    s["cli_validate_batch_critic_fail"] = _cli(v, ["validate", "--batch"], vbatch)
    s["mcp_validate_critic_fail"] = _tool("validate_tool", v, T, GDICT)
    s["cli_extract_empty"] = _cli(_ok_pipe, ["extract"], "")
    s["mcp_extract_empty"] = _tool("extract_tool", _ok_pipe, "")
    s["cli_extract_bad_threshold"] = _cli(_ok_pipe, ["extract", "--file", doc, "--field-threshold", "2"])
    s["mcp_extract_bad_threshold"] = _tool("extract_tool", _ok_pipe, T, field_threshold=2.0)
    return s


EB7_DIGESTS: dict[str, str] = {
    "cli_extract_gen_err": "87348a4c930bc894c8727e2f2e95360cfa8391e0510df94cfe30175ffe61d936",
    "cli_batch_gen_err": "87cd06274d18c0a4e24f0f76cf453387ab04b9b4af47f7109c899cb04c955737",
    "mcp_extract_gen_err": "0621760495c65e160bce7ac5f5662fa209585283722aff3239deaef6aad76e2b",
    "cli_extract_critic_conn": "5e636446965c49e0da5253234d451841a8548aa554740db41fb64c49f85d64e6",
    "cli_batch_critic_conn": "d12df3cab9bad131d86c3d8ed2fb0bcb8f77da32fe1292de23d00e210db81451",
    "mcp_extract_critic_conn": "14c45795184b3e2cefb3dc92e8e8815377b84c5ea27dc0725dbd022f2a1b9360",
    "cli_extract_gen_dataerr": "1aa569a4cf56be120665e6c4b30ad8f61c615d9aec1d3f3ef1b97bd108cd53a6",
    "cli_batch_gen_dataerr": "3a528b9d501cfb34d321f20322044d8503966731b4c719ba6d29406c54dfc2f9",
    "mcp_extract_gen_dataerr": "e18e1cbc68aee0a822b44616c6ea8ddfbdb3da7df14c873fadd35c1f39fbe90a",
    "cli_extract_factory_err": "ae40e02c757b91d225f2662937269b18eddf125dcd0d5e92fda57d45d9fa59fb",
    "cli_batch_factory_err": "ae40e02c757b91d225f2662937269b18eddf125dcd0d5e92fda57d45d9fa59fb",
    "mcp_extract_factory_err": "dbf6e2ca3065cfcc4cc3444ecb91409c7f3a5890bfebca85f7b4810f566f06b6",
    "cli_validate_critic_fail": "a7eeebb195a12aca619f50887e7a33112ec2f63a385b1aee88fe8498e451ef59",
    "cli_validate_batch_critic_fail": "83fa83a496f270baa8f421b373f41bf5104e22ea44ea8bbc61a31492eb9d3686",
    "mcp_validate_critic_fail": "15468e2cb49f5e690bd384ac0445ba1174d485c846a6d9facabc1424c2b31cb3",
    "cli_extract_empty": "9c4a7a87b13837d0e3cf6f5a1006bfaf8fbbbe5f3e210ebdec66a99d7108df48",
    "mcp_extract_empty": "16c98bc7665ee99abeb3acdaebc9c4f47b40bc70b3a9ec657e8790429bacb4fb",
    "cli_extract_bad_threshold": "b3e867eceb38e51762fe8abfdf210e9f45183d92fd4447b8035df2099aa820f9",
    "mcp_extract_bad_threshold": "94a028a4cc9f7234bc878bb5326c15621f879bb09755e099d598aa83d6d11696",
}


def test_eb7_non_critic_errors_never_carry_data(doc):
    s = _error_scenarios(doc)
    for name, res in s.items():
        envs = res["out"] if name.startswith("cli_") else [res]
        for env in envs:
            assert "data" not in env, (name, env)
            assert env.get("status") == "error", (name, env)
            assert set(env) - {"id"} == {"status", "error_message"}, (name, env)


def test_eb7_foreign_data_attribute_not_leaked():
    env = _tool("extract_tool", _gen_dataerr_pipe, T)
    assert env["status"] == "error"
    assert "data" not in env
    assert "foreign boom" in env["error_message"]


def test_eb7_library_critic_provider_error_unwrapped():
    with pytest.raises(ConnectionError):
        _critic_conn_pipe().extract(T)


def test_eb7_library_generator_error_unchanged():
    with pytest.raises(GeneratorError) as ei:
        _gen_err_pipe().extract(T)
    assert getattr(ei.value, "data", None) is None


def test_eb7_error_bytes_match_pre_change(doc):
    s = _error_scenarios(doc)
    got = {k: _digest(v) for k, v in s.items()}
    assert set(got) == set(EB7_DIGESTS)
    for k in got:
        assert got[k] == EB7_DIGESTS[k], (k, s[k])


# --------------------------------------------------------------------------- EB8


def _success_scenarios(doc: str) -> dict:
    gd = json.dumps(GDICT)

    def lo(*a, **k):
        return _pipe(ScriptedLLMClient([G]), ScriptedLLMClient([LOWC]))

    def batch_mix(*a, **k):
        return _pipe(ScriptedLLMClient([G, G]), ScriptedLLMClient([GOODC, LOWC]))

    eb = "\n".join(_batch_line(i) for i in "pq") + "\n"
    vb = "\n".join(_batch_line(i, data=GDICT) for i in "pq") + "\n"
    s = {}
    for tag, fac in (("ok", _ok_pipe), ("nr", lo)):
        s[f"cli_extract_{tag}"] = _cli(fac, ["extract", "--file", doc])
        s[f"cli_validate_{tag}"] = _cli(fac, ["validate", "--file", doc, "--data", gd])
        s[f"mcp_extract_{tag}"] = _tool("extract_tool", fac, T)
        s[f"mcp_validate_{tag}"] = _tool("validate_tool", fac, T, GDICT)
    s["cli_extract_heur"] = _cli(_ok_pipe, ["extract", "--file", doc, "--heuristics-only"])
    s["cli_validate_heur"] = _cli(_ok_pipe, ["validate", "--file", doc, "--data", gd, "--heuristics-only"])
    s["mcp_extract_heur"] = _tool("extract_tool", _ok_pipe, T, heuristics_only=True)
    s["mcp_validate_heur"] = _tool("validate_tool", _ok_pipe, T, GDICT, heuristics_only=True)
    s["cli_extract_batch"] = _cli(batch_mix, ["extract", "--batch"], eb)
    s["cli_validate_batch"] = _cli(batch_mix, ["validate", "--batch"], vb)
    return s


EB8_DIGESTS: dict[str, str] = {
    "cli_extract_ok": "e3e215a79852174c3a26ce4b0d53613e274e6800d28af527c48e7490157531e1",
    "cli_validate_ok": "e3e215a79852174c3a26ce4b0d53613e274e6800d28af527c48e7490157531e1",
    "mcp_extract_ok": "148837f51d19089fa31f7a4912376a66b12d90ddbac237a471e4cde0eeaeb9f6",
    "mcp_validate_ok": "148837f51d19089fa31f7a4912376a66b12d90ddbac237a471e4cde0eeaeb9f6",
    "cli_extract_nr": "005d1a2e7d3d3991ddb6a7d6266f7b0103f0242f2f456fb127e16a06ad8a3676",
    "cli_validate_nr": "005d1a2e7d3d3991ddb6a7d6266f7b0103f0242f2f456fb127e16a06ad8a3676",
    "mcp_extract_nr": "88d49d96c7404cecf95ee4adba6aa40fa08d37f80a0bb1c1f7d8e04519652f74",
    "mcp_validate_nr": "88d49d96c7404cecf95ee4adba6aa40fa08d37f80a0bb1c1f7d8e04519652f74",
    "cli_extract_heur": "e3e215a79852174c3a26ce4b0d53613e274e6800d28af527c48e7490157531e1",
    "cli_validate_heur": "e3e215a79852174c3a26ce4b0d53613e274e6800d28af527c48e7490157531e1",
    "mcp_extract_heur": "148837f51d19089fa31f7a4912376a66b12d90ddbac237a471e4cde0eeaeb9f6",
    "mcp_validate_heur": "148837f51d19089fa31f7a4912376a66b12d90ddbac237a471e4cde0eeaeb9f6",
    "cli_extract_batch": "447770c073bcc898bcefd4f9aba73427f08de82979bc1400e63b98fe5901c583",
    "cli_validate_batch": "447770c073bcc898bcefd4f9aba73427f08de82979bc1400e63b98fe5901c583",
}


def test_eb8_success_bytes_match_pre_change(doc):
    s = _success_scenarios(doc)
    assert s["cli_extract_ok"]["rc"] == 0 and s["cli_extract_nr"]["rc"] == 1
    assert s["mcp_extract_nr"]["status"] == "needs_review"
    got = {k: _digest(v) for k, v in s.items()}
    assert set(got) == set(EB8_DIGESTS)
    for k in got:
        assert got[k] == EB8_DIGESTS[k], (k, s[k])


@pytest.mark.parametrize("model,sha", [
    (InvoiceData, "e25030a2"),
    (CritiqueResult, "bc64cbf7"),
    (FieldVerdict, "7fb4decf"),
    (ExtractionResult, "174131d8"),
])
def test_eb8_schema_unchanged(model, sha):
    blob = json.dumps(model.model_json_schema(), sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:8] == sha


# --------------------------------------------------------------------------- EB9


def _golden():
    gold = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = json.loads(REPLAY.read_text(encoding="utf-8"))
    return gold, rep


def _dump(r) -> dict:
    d = r.model_dump()
    d.pop("trace_id", None)
    return d


def test_eb9_golden_salvage_equals_plain_replay_data():
    gold, rep = _golden()
    assert len(gold) == 13
    for row in gold:
        e = rep[row["id"]]
        plain = _pipe(ScriptedLLMClient([e["generator_raw"]]),
                      ScriptedLLMClient([e["critic_raw"]])).extract(row["text"])
        with pytest.raises(CriticError) as ei:
            _pipe(ScriptedLLMClient([e["generator_raw"]]),
                  ScriptedLLMClient([BAD, BAD])).extract(row["text"])
        assert ei.value.data == plain.data.model_dump(), row["id"]


def test_eb9_golden_revalidate_salvage_equals_plain_replay():
    gold, rep = _golden()
    for row in gold:
        e = rep[row["id"]]
        plain = _pipe(ScriptedLLMClient([e["generator_raw"]]),
                      ScriptedLLMClient([e["critic_raw"]])).extract(row["text"])
        with pytest.raises(CriticError) as ei:
            _pipe(ScriptedLLMClient([e["generator_raw"]]),
                  ScriptedLLMClient([BAD, BAD])).extract(row["text"])
        salvaged = InvoiceData.model_validate(ei.value.data)
        c = ScriptedLLMClient([e["critic_raw"]])
        again = _pipe(None, c).validate(row["text"], salvaged)
        assert _dump(again) == _dump(plain), row["id"]
        assert len(c.calls) == 1


# --------------------------------------------------------------------------- EB10


def test_eb10_make_eval_unchanged():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    lines = r.stdout.splitlines()
    for line in (
        "field accuracy    : 91.2%  (83/91)",
        "silent-wrong      : 0",
        "catch rate        : 100.0%  (8/8)",
    ):
        assert line in lines, line


@pytest.mark.parametrize("path,sha", [
    (GOLDEN, "43843904de624a1677542329761a3c47190f17da2d957484eb5f9602db10fe30"),
    (REPLAY, "193a25c6def6a4f1d88dba23caf93fd4ebe24798afe8d3e8ceca6b58a51eacd8"),
])
def test_eb10_golden_files_byte_identical(path, sha):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


# --------------------------------------------------------------------------- docs


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_docs_readme_error_contract_mentions_exc_data():
    text = _readme()
    sec = text[text.index("**Error contract.**"):]
    para = sec.split("\n\n")[0]
    assert "exc.data" in para and "unverified" in para


def test_docs_readme_cli_envelope_recipe():
    text = _readme()
    assert "| reflect-mcp validate --file DOC --data -" in text
    assert "unverified" in text.split("Exit codes (both verbs)")[1].split("\n\n")[0]


def test_docs_toolresult_docstring_mentions_data():
    from reflect_mcp.tools import ToolResult

    doc = ToolResult.__doc__ or ""
    assert "an error row carries just status + error_message" not in doc
    assert "data" in doc and "Critic" in doc
