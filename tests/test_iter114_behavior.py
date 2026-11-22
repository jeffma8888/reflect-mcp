"""Black-box behavioral tests for iter-114 (roadmap #119): amount fields reject
non-finite numbers, so every CLI / batch / MCP output is strict RFC-8259 JSON.

``InvoiceData.subtotal`` / ``tax`` / ``total`` reject ``NaN``, ``Infinity``,
``-Infinity``, overflowing literals such as ``1e400`` and the string forms
``"NaN"`` / ``"inf"`` with a named schema error (pydantic ``finite_number``),
following the iter-103 bool-amount precedent. A Generator reply carrying a
non-finite amount gets the iter-106 feedback retry.

Covers EB1-EB13 from pm.md. Isolation contract: only public interfaces are
driven (the CLI as a subprocess and in-process via ``__main__.main``, the MCP
server's CallTool request handler, the schema models, Generator / Critic /
ReflectivePipeline with list-backed scripted clients). No network, no secrets;
the MCP extra is ``importorskip``-guarded.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflect_mcp import pipeline as pipeline_mod
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
AMOUNTS = ("subtotal", "tax", "total")
SRC = "Total: 108.50\n"
PREFIX = "Error: --data failed InvoiceData validation"

# Pinned literals (fixed values, not git baselines -> ship-safe, H1).
GOLDEN_SHA256 = "43843904de624a1677542329761a3c47190f17da2d957484eb5f9602db10fe30"
# Same literal + hashing recipe as iter-112's HEAD pin: schema dict unchanged.
PRE_MODEL_JSON_SCHEMA_SHA256 = "e25030a2013c33f5d110fc25f638957abfa3ff8de56abb9e3d73eaae9eb12d3e"

NONFINITE_TOKEN = re.compile(r"(?<![\w'\"])-?(NaN|Infinity)(?![\w'\"])")


# ---- helpers ---------------------------------------------------------------
def _env() -> dict:
    drop = ("REFLECT_", "AWS_")
    return {
        k: v for k, v in os.environ.items()
        if not k.startswith(drop) and not k.endswith("_API_KEY")
    }


def _cli(args: list[str], cwd: Path, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        input=stdin, capture_output=True, text=True, env=_env(),
        cwd=str(cwd), timeout=120,
    )


def _main(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    """Drive the CLI in-process (fast; same entry point as the console script)."""
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


def _reject_constant(tok: str):
    raise ValueError(f"non-finite JSON token on stdout: {tok}")


def _strict(text: str):
    """RFC-8259 parse: a bare NaN / Infinity / -Infinity token raises."""
    return json.loads(text, parse_constant=_reject_constant)


def _strict_lines(stdout: str) -> list:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    return [_strict(ln) for ln in lines]


def _strict_stdout(stdout: str) -> list:
    """stdout is either one (possibly pretty-printed) JSON doc or JSONL."""
    s = stdout.strip()
    if not s:
        return []
    try:
        return [_strict(s)]
    except ValueError as exc:
        if "non-finite" in str(exc):
            raise
        return _strict_lines(s)


@pytest.fixture
def srcfile(tmp_path) -> Path:
    p = tmp_path / "src.txt"
    p.write_text(SRC, encoding="utf-8")
    return p


def _validate_raw(srcfile: Path, payload: str) -> subprocess.CompletedProcess:
    return _cli(["validate", "--heuristics-only", "--file", str(srcfile),
                 "--data", payload], cwd=srcfile.parent)


def _assert_schema_rejection(r, field: str):
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert r.stdout == "", r.stdout
    assert any(ln.startswith(PREFIX) for ln in r.stderr.splitlines()), r.stderr
    assert field in r.stderr, r.stderr
    assert "finite number" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


class _RecordingClient:
    """List-backed LLMClient that records every complete() call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._replies.pop(0)


NAN_REPLY = json.dumps(CORRECT_EXTRACTION).replace('"total": 108.5', '"total": NaN')
GOOD_REPLY = json.dumps(CORRECT_EXTRACTION)


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


def test_nan_reply_fixture_really_carries_nan():
    assert '"total": NaN' in NAN_REPLY


# ---- EB1: CLI --data NaN ---------------------------------------------------
def test_eb1_cli_nan_total_exits_2(srcfile):
    r = _validate_raw(srcfile, '{"total": NaN}')
    _assert_schema_rejection(r, "total")


# ---- EB2: other non-finite spellings ---------------------------------------
@pytest.mark.parametrize("payload,field", [
    ('{"total": Infinity}', "total"),
    ('{"tax": -Infinity}', "tax"),
    ('{"subtotal": 1e400}', "subtotal"),
    ('{"total": "NaN"}', "total"),
    ('{"total": "inf"}', "total"),
    ('{"total": "-Infinity"}', "total"),
])
def test_eb2_cli_nonfinite_variants_exit_2(srcfile, payload, field):
    r = _validate_raw(srcfile, payload)
    _assert_schema_rejection(r, field)


def test_eb2_in_process_cli_matches_subprocess(srcfile):
    rc, out, err = _main(["validate", "--heuristics-only", "--file", str(srcfile),
                          "--data", '{"tax": -Infinity}'])
    assert rc == 2 and out == "", (rc, out, err)
    assert "tax" in err and "finite number" in err


# ---- EB3: finite amounts unchanged -----------------------------------------
@pytest.mark.parametrize("payload", ['{"total": 108.5}', '{"total": "108.50"}'])
def test_eb3_finite_total_ok(srcfile, payload):
    r = _validate_raw(srcfile, payload)
    assert r.returncode == 0, (r.stdout, r.stderr)
    (doc,) = _strict_stdout(r.stdout)
    assert doc["status"] == "ok", doc
    assert doc["data"]["total"] == 108.5


@pytest.mark.parametrize("payload", ['{"total": 0}', '{"tax": -5.0}', '{"total": 1e308}'])
def test_eb3_finite_edge_values_accepted(srcfile, payload):
    r = _validate_raw(srcfile, payload)
    assert r.returncode != 2, (r.stdout, r.stderr)
    assert "failed InvoiceData validation" not in r.stderr
    docs = _strict_stdout(r.stdout)
    assert len(docs) == 1
    assert docs[0]["status"] != "error", docs[0]


@pytest.mark.parametrize("value", [0, -5.0, 1e308, "108.50"])
def test_eb3_model_accepts_finite(value):
    assert InvoiceData.model_validate({"total": value}).total == float(value)


# ---- EB4: batch -------------------------------------------------------------
BATCH = (
    '{"id":1,"text":"Total: 108.50","data":{"total":NaN}}\n'
    '{"id":2,"text":"Total: 108.50","data":{"total":108.5}}\n'
)


def test_eb4_batch_nan_row_error_finite_row_ok(tmp_path):
    r = _cli(["validate", "--heuristics-only", "--batch"], cwd=tmp_path, stdin=BATCH)
    assert r.returncode == 3, (r.stdout, r.stderr)
    rows = _strict_lines(r.stdout)
    assert len(rows) == 2, r.stdout
    first, second = rows
    assert list(first)[:2] == ["id", "status"], first
    assert first["id"] == 1 and first["status"] == "error", first
    assert "total" in first["error_message"] and "finite number" in first["error_message"]
    assert second["id"] == 2 and second["status"] == "ok", second


def test_eb4_batch_infinity_and_string_rows(tmp_path):
    batch = (
        '{"id":"a","text":"Tax: 1","data":{"tax":-Infinity}}\n'
        '{"id":"b","text":"Subtotal: 1","data":{"subtotal":1e400}}\n'
        '{"id":"c","text":"Total: 1","data":{"total":"inf"}}\n'
    )
    r = _cli(["validate", "--heuristics-only", "--batch"], cwd=tmp_path, stdin=batch)
    assert r.returncode == 3, (r.stdout, r.stderr)
    rows = _strict_lines(r.stdout)
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    for row, field in zip(rows, ("tax", "subtotal", "total")):
        assert row["status"] == "error", row
        assert field in row["error_message"] and "finite number" in row["error_message"]


# ---- EB5: strict-JSON invariant (and the checker itself discriminates) -----
def test_eb5_strict_checker_rejects_bare_tokens_only():
    for bad in ('{"total": NaN}', '{"total": Infinity}', '{"total": -Infinity}'):
        with pytest.raises(ValueError):
            _strict(bad)
    # The words inside a string are fine (e.g. pydantic's input_value='NaN').
    assert _strict('{"m": "input_value=\'NaN\' Infinity"}')["m"].startswith("input_value")


@pytest.mark.parametrize("payload", [
    '{"total": NaN}', '{"total": Infinity}', '{"tax": -Infinity}',
    '{"subtotal": 1e400}', '{"total": "NaN"}', '{"total": 108.5}',
])
def test_eb5_single_stdout_always_strict(srcfile, payload):
    r = _validate_raw(srcfile, payload)
    _strict_stdout(r.stdout)
    assert not NONFINITE_TOKEN.search(r.stdout.replace("'NaN'", "").replace("'inf'", "")), r.stdout


# ---- EB6: stored envelope ---------------------------------------------------
def _stored_envelope(tmp_path: Path) -> dict:
    p = tmp_path / "ok.txt"
    p.write_text(SRC, encoding="utf-8")
    r = _cli(["validate", "--heuristics-only", "--file", str(p),
              "--data", '{"total": 108.5}'], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    (env,) = _strict_stdout(r.stdout)
    assert env["status"] == "ok" and isinstance(env.get("data"), dict)
    return env


def test_eb6_envelope_with_nan_inner_total_exits_2(srcfile, tmp_path):
    env = _stored_envelope(tmp_path)
    text = json.dumps(env)
    assert '"total": 108.5' in text
    text = text.replace('"total": 108.5', '"total": NaN')
    r = _validate_raw(srcfile, text)
    _assert_schema_rejection(r, "total")


def test_eb6_envelope_with_finite_inner_total_still_ok(srcfile, tmp_path):
    env = _stored_envelope(tmp_path)
    r = _validate_raw(srcfile, json.dumps(env))
    assert r.returncode == 0, r.stderr
    (doc,) = _strict_stdout(r.stdout)
    assert doc["data"]["total"] == 108.5


# ---- EB7: MCP ---------------------------------------------------------------
@pytest.fixture
def real_server(monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server.fastmcp")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REFLECT_MCP_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("reflect_mcp.server", None)
    try:
        server = importlib.import_module("reflect_mcp.server")
        if not hasattr(server.mcp, "_mcp_server"):
            pytest.skip("stub FastMCP in sys.modules; real SDK required")
        yield server
    finally:
        sys.modules.pop("reflect_mcp.server", None)


def _call_validate(server, data: dict):
    mtypes = pytest.importorskip("mcp.types")
    handler = server.mcp._mcp_server.request_handlers[mtypes.CallToolRequest]
    args = {"text": SRC, "data": data, "heuristics_only": True}
    req = mtypes.CallToolRequest(
        method="tools/call",
        params=mtypes.CallToolRequestParams(name="validate", arguments=args),
    )
    out = asyncio.run(handler(req))
    return getattr(out, "root", out)


def _text_payload(res) -> str:
    texts = [c.text for c in (res.content or []) if getattr(c, "type", None) == "text"]
    assert texts, res
    return texts[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "NaN", "-Infinity"])
def test_eb7_mcp_nonfinite_total_is_error_and_channels_agree(real_server, value):
    res = _call_validate(real_server, {"total": value})
    text = _text_payload(res)
    doc = _strict(text)  # no bare NaN / Infinity token in the text channel
    assert doc["status"] == "error", doc
    assert "total" in doc["error_message"], doc
    sc = res.structuredContent
    if sc is not None:
        assert sc["status"] == doc["status"], (sc, doc)


def test_eb7_mcp_finite_total_still_ok(real_server):
    res = _call_validate(real_server, {"total": 108.5})
    doc = _strict(_text_payload(res))
    assert doc["status"] == "ok", doc
    assert doc["data"]["total"] == 108.5


# ---- EB8: library -----------------------------------------------------------
@pytest.mark.parametrize("field", AMOUNTS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "NaN", "inf"])
def test_eb8_model_rejects_nonfinite(field, value):
    with pytest.raises(ValidationError) as ei:
        InvoiceData.model_validate({field: value})
    errs = ei.value.errors()
    assert any(e["type"] == "finite_number" and tuple(e["loc"]) == (field,) for e in errs), errs


def test_eb8_null_amounts_still_accepted():
    d = InvoiceData.model_validate({"subtotal": None, "tax": None, "total": None})
    assert (d.subtotal, d.tax, d.total) == (None, None, None)


# ---- EB9: Generator feedback retry -----------------------------------------
def test_eb9_generator_nan_reply_is_retried_with_feedback():
    gen = _RecordingClient([NAN_REPLY, GOOD_REPLY])
    crit = _RecordingClient([critique_all(0.95, 0.9)])
    result = ReflectivePipeline(Generator(gen, max_retries=1), Critic(crit)).extract(SOURCE)
    assert len(gen.calls) == 2
    assert result.data.total == 108.5
    second = gen.calls[1]["user"]
    assert second != gen.calls[0]["user"]
    feedback = second[len(gen.calls[0]["user"]):]
    assert "total" in feedback and "finite" in feedback, feedback
    assert len(crit.calls) == 1
    crit_prompt = crit.calls[0]["system"] + crit.calls[0]["user"]
    assert "NaN" not in crit_prompt and "Infinity" not in crit_prompt


# ---- EB10: persistent NaN ---------------------------------------------------
def test_eb10_persistent_nan_raises_generator_error_after_2_calls():
    gen = _RecordingClient([NAN_REPLY, NAN_REPLY])
    crit = _RecordingClient([])
    with pytest.raises(GeneratorError):
        ReflectivePipeline(Generator(gen, max_retries=1), Critic(crit)).extract(SOURCE)
    assert len(gen.calls) == 2
    assert crit.calls == []


def test_eb10_cli_extract_nan_error_envelope_is_strict_json():
    gen = _RecordingClient([NAN_REPLY, NAN_REPLY])
    p = ReflectivePipeline(Generator(gen, max_retries=1), Critic(_RecordingClient([])))
    with _factory(lambda *a, **k: p):
        rc, out, err = _main(["extract"], stdin=SOURCE)
    assert len(gen.calls) == 2, (rc, out, err)
    assert rc == 3, (rc, out, err)
    docs = _strict_stdout(out)
    assert len(docs) == 1 and docs[0]["status"] == "error", out


# ---- EB11: golden-derived mutations ----------------------------------------
def _golden_rows() -> list[dict]:
    return [json.loads(ln) for ln in GOLDEN.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def test_eb11_golden_dataset_byte_identical():
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_eb11_golden_unmodified_expected_has_no_schema_error(tmp_path):
    rows = _golden_rows()
    assert rows
    for row in rows:
        src = tmp_path / f"{row['id']}.txt"
        src.write_text(row["text"], encoding="utf-8")
        rc, out, err = _main(["validate", "--heuristics-only", "--file", str(src),
                              "--data", json.dumps(row["expected"])])
        assert rc != 2, (row["id"], err)
        assert "failed InvoiceData validation" not in err, (row["id"], err)
        _strict_stdout(out)


def test_eb11_golden_nonfinite_mutations_exit_2(tmp_path):
    total = 0
    for row in _golden_rows():
        src = tmp_path / f"{row['id']}.txt"
        src.write_text(row["text"], encoding="utf-8")
        per_row = 0
        fields = [f for f in AMOUNTS if row["expected"].get(f) is not None]
        for field in fields:
            for token in ("NaN", "Infinity", "1e400"):
                data = dict(row["expected"], **{field: "__X__"})
                payload = json.dumps(data).replace('"__X__"', token)
                assert token in payload
                rc, out, err = _main(["validate", "--heuristics-only", "--file", str(src),
                                      "--data", payload])
                assert rc == 2, (row["id"], field, token, rc, out, err)
                assert out == "", (row["id"], field, token, out)
                assert field in err and "finite number" in err, (row["id"], field, token, err)
                per_row += 1
        if fields:
            assert per_row >= 1, row["id"]
        total += per_row
    assert total > 0


# ---- EB12: schema + eval unchanged -----------------------------------------
def _sha(obj) -> str:
    blob = re.sub(r"https://errors\.pydantic\.dev/\S+", "<url>",
                  json.dumps(obj, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_eb12_model_json_schema_unchanged():
    assert _sha(InvoiceData.model_json_schema()) == PRE_MODEL_JSON_SCHEMA_SHA256


def test_eb12_eval_unchanged():
    r = subprocess.run([sys.executable, str(REPO / "eval" / "run_eval.py")],
                       capture_output=True, text=True, env=_env(), cwd=str(REPO),
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = r.stdout.splitlines()
    for line in (
        "examples          : 13",
        "field accuracy    : 91.2%  (83/91)",
        "silent-wrong      : 0",
    ):
        assert line in lines, (line, r.stdout[-2000:])


# ---- EB13: README §5 ---------------------------------------------------------
def _readme_section(n: int) -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = re.search(rf"^## {n}\. .*$", text, re.M)
    assert start, n
    nxt = re.search(r"^## ", text[start.end():], re.M)
    return text[start.start(): start.end() + (nxt.start() if nxt else len(text))]


def test_eb13_readme_cli_section_documents_nonfinite_amounts():
    sec = " ".join(_readme_section(5).split())
    assert "NaN" in sec and "finite" in sec
    assert "1e400" in sec
    assert "strict JSON" in sec
