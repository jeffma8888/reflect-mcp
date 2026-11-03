"""Black-box behavioral tests for iter-68: stored result envelopes are verified
by their data, never as an all-null ok.

Covers Expected Behaviors 1-10 from the iter-68 spec:
  1  MCP: an ok-envelope is verified by its inner data (== bare data verdict)
  2  MCP: a needs_review-envelope wrapping good data is ok
  3  MCP: an error envelope (with or without an extra id) is a schema error
  4  an envelope whose data is null / missing is a schema error
  5  CLI single: exit 1 on the wrapped bad data; exit 2 + stderr on an error row
  6  CLI batch: the 3-line mixed case, exit 3, c=1 == c=3
  7  round trip: extract result -> validate == validating result.data directly
  8  non-envelopes (bare data, unknown status, status+field key, {}) are
     byte-identical to HEAD
  9  library: InvoiceData.model_validate unwraps / raises ValidationError
 10  nothing else moves: eval stdout, ExtractionResult fields, extract output,
     cli.py line count

HEAD constants were measured by running the product from a ``git archive
HEAD`` copy (4326d92) before the change was committed. That copy returned
``ok`` for the BAD ok-envelope, which confirms the copy loaded its own source.

Offline: the real heuristics-only pipeline, or ScriptedLLMClient-backed
pipelines. No network, no SDK, no secrets.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pydantic
import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import ExtractionResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests._golden import frozen_eval, frozen_replay, frozen_rows

REPO = Path(__file__).resolve().parents[1]

T = "ACME CORP\nInvoice INV-1\nTotal: $10.00"
BAD = {"vendor_name": "WRONG CO", "total": 999.0}
GOOD = {"vendor_name": "ACME CORP", "total": 10.0}

NO_DATA_ERROR = "a reflect-mcp 'error' result envelope has no data object to verify"
NO_DATA_OK = "a reflect-mcp 'ok' result envelope has no data object to verify"
SCHEMA_PREFIX = "data failed InvoiceData validation:"

# --- HEAD constants (measured on a `git archive HEAD` copy, pre-iter-68) ---
HEAD_EVAL_STDOUT_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
HEAD_CLI_LINES = 373  # iter 75: re-pinned 360 -> 373 for the bounded 2N batch window
HEAD_RESULT_FIELDS = [
    "critic_summary", "data", "field_confidence", "field_provenance",
    "flagged_fields", "notes", "overall_confidence", "status", "trace_id",
]
# sha256 of sorted-keys JSON of validate_tool(T, data, heuristics_only=True)
# with trace_id popped.
HEAD_NON_ENVELOPE_DIGESTS = {
    "bad": "eb22fab6b3b9b06d1393e2c2760112e3f4f853586526e1d698f330bae479215f",
    "paid": "185893af931fb8b202cca5549d8573b797a0d2a987a1586cffeab90945379298",
    "mixed": "f5d20f0b21a855d783931723ce04b5be562cd6055884692487ad8cce9af182d0",
    "empty": "185893af931fb8b202cca5549d8573b797a0d2a987a1586cffeab90945379298",
    "good": "0e4b12cbb757cd30906af85f91cc41a17d9c3f7ff05bd16575b5a356addb80b4",
}
# sha256 over {row_id:critic_on -> digest(extract result minus trace_id)} for
# the 8 golden rows x {critic off, critic on}, scripted from replay fixtures.
HEAD_EXTRACT_MATRIX_SHA256 = "b3a87b0e94df00bcdfcc4c4369b9c9d127014640b8ac3b0cf64f3d9a56929776"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("REFLECT_FIELD_THRESHOLD", raising=False)
    monkeypatch.delenv("REFLECT_OVERALL_THRESHOLD", raising=False)


def _v(data) -> dict:
    out = tools.validate_tool(T, data, heuristics_only=True)
    out.pop("trace_id", None)
    return out


def _flags(d: dict) -> list[str]:
    return [f["field"] for f in d.get("flagged_fields", [])]


def _digest(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _cli(args: list[str], stdin: str = "") -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c",
           "import sys, runpy; sys.modules['anthropic'] = None; "
           "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
           "runpy.run_module('reflect_mcp', run_name='__main__')", *args]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=120)


@pytest.fixture
def tfile(tmp_path) -> Path:
    p = tmp_path / "t.txt"
    p.write_text(T, encoding="utf-8")
    return p


def _scripted_pipe(reply: dict) -> ReflectivePipeline:
    return ReflectivePipeline(Generator(ScriptedLLMClient([json.dumps(reply)])), None)


# ---------------------------------------------------------------------------
# EB-1: MCP ok-envelope verified by its data
# ---------------------------------------------------------------------------

def test_eb1_ok_envelope_verified_by_data():
    out = _v({"status": "ok", "data": BAD, "trace_id": "t"})
    assert out["status"] == "needs_review"
    assert _flags(out) == ["vendor_name", "total"]
    assert out["data"]["vendor_name"] == "WRONG CO"
    assert out["data"]["total"] == 999.0


def test_eb1_ok_envelope_equals_bare_data():
    env = _v({"status": "ok", "data": BAD, "trace_id": "t"})
    bare = _v(dict(BAD))
    assert bare["status"] == "needs_review"  # non-vacuous comparison
    assert env == bare


# ---------------------------------------------------------------------------
# EB-2: needs_review envelope of good data -> ok
# ---------------------------------------------------------------------------

def test_eb2_needs_review_envelope_with_good_data_is_ok():
    out = _v({"status": "needs_review", "data": GOOD})
    assert out["status"] == "ok"
    assert out["flagged_fields"] == []
    assert out["data"]["vendor_name"] == "ACME CORP"
    assert out["data"]["total"] == 10.0
    assert out == _v(dict(GOOD))


# ---------------------------------------------------------------------------
# EB-3: error envelope rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", [
    {"status": "error", "error_message": "boom"},
    {"status": "error", "error_message": "boom", "id": 3},
], ids=["plain", "with_id"])
def test_eb3_error_envelope_rejected(env):
    out = _v(env)
    assert out["status"] == "error"
    assert out["error_message"].startswith(SCHEMA_PREFIX), out["error_message"]
    assert NO_DATA_ERROR in out["error_message"]
    assert "data" not in out


# ---------------------------------------------------------------------------
# EB-4: envelope with non-dict / missing data rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", [
    {"status": "ok", "data": None},
    {"status": "ok"},
    {"status": "ok", "data": "not a dict"},
    {"status": "ok", "data": [1, 2]},
], ids=["null", "missing", "string", "list"])
def test_eb4_ok_envelope_without_data_object_rejected(env):
    out = _v(env)
    assert out["status"] == "error"
    assert out["error_message"].startswith(SCHEMA_PREFIX), out["error_message"]
    assert NO_DATA_OK in out["error_message"]


def test_eb4_needs_review_envelope_without_data_names_its_status():
    out = _v({"status": "needs_review"})
    assert out["status"] == "error"
    assert "a reflect-mcp 'needs_review' result envelope has no data object to verify" in out["error_message"]


# ---------------------------------------------------------------------------
# EB-5: CLI single
# ---------------------------------------------------------------------------

def test_eb5_cli_single_ok_envelope_exit1(tfile):
    proc = _cli(["validate", "--heuristics-only", "--file", str(tfile),
                 "--data", '{"status":"ok","data":{"vendor_name":"WRONG CO","total":999}}'])
    assert proc.returncode == 1, proc.stderr[-2000:]
    out = json.loads(proc.stdout)  # single validate pretty-prints
    assert out["status"] == "needs_review"
    assert _flags(out) == ["vendor_name", "total"]


def test_eb5_cli_single_error_envelope_exit2(tfile):
    proc = _cli(["validate", "--heuristics-only", "--file", str(tfile),
                 "--data", '{"status":"error","error_message":"boom"}'])
    assert proc.returncode == 2, (proc.stdout, proc.stderr[-2000:])
    assert "--data failed InvoiceData validation" in proc.stderr
    assert "has no data object to verify" in proc.stderr
    assert proc.stdout.strip() == ""


def test_eb5_cli_single_bare_good_data_counterfactual_exit0(tfile):
    proc = _cli(["validate", "--heuristics-only", "--file", str(tfile),
                 "--data", json.dumps(GOOD)])
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-6: CLI batch
# ---------------------------------------------------------------------------

BATCH = "".join(json.dumps(r) + "\n" for r in [
    {"id": 1, "text": T, "data": {"status": "ok", "data": BAD}},
    {"id": 2, "text": T, "data": {"status": "error", "error_message": "boom"}},
    {"id": 3, "text": T, "data": {"status": "needs_review", "data": GOOD}},
])


def _batch_rows(c: int) -> tuple[int, list[dict]]:
    proc = _cli(["validate", "--batch", "--heuristics-only", "--batch-concurrency", str(c)], BATCH)
    rows = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    for r in rows:
        r.pop("trace_id", None)
    return proc.returncode, rows


@pytest.mark.parametrize("c", [1, 3])
def test_eb6_batch_mixed_rows(c):
    code, rows = _batch_rows(c)
    assert len(rows) == 3, rows
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert rows[0]["status"] == "needs_review"
    assert _flags(rows[0]) == ["vendor_name", "total"]
    assert list(rows[1]) == ["id", "status", "error_message"]
    assert rows[1]["status"] == "error"
    assert "has no data object to verify" in rows[1]["error_message"]
    assert rows[2]["status"] == "ok"
    assert rows[2]["flagged_fields"] == []
    assert code == 3


def test_eb6_batch_concurrency_parity():
    code1, rows1 = _batch_rows(1)
    code3, rows3 = _batch_rows(3)
    assert len(rows1) == 3
    assert rows1 == rows3
    assert code1 == code3 == 3


# ---------------------------------------------------------------------------
# EB-7: round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply,status", [(BAD, "needs_review"), (GOOD, "ok")], ids=["bad", "good"])
@pytest.mark.parametrize("dump_mode", ["python", "json"])
def test_eb7_round_trip_library_dump(reply, status, dump_mode):
    result = _scripted_pipe(reply).extract(T)
    envelope = result.model_dump() if dump_mode == "python" else result.model_dump(mode="json")
    via_env = _v(envelope)
    direct = _v(result.data.model_dump())
    assert direct["status"] == status
    assert via_env["status"] == direct["status"]
    assert _flags(via_env) == _flags(direct)
    if status == "needs_review":
        assert _flags(direct) == ["vendor_name", "total"]


def test_eb7_round_trip_cli_extract_stdout(monkeypatch, tfile):
    """CLI extract stdout (stub pipeline, in-process) fed back to CLI validate."""
    import reflect_mcp.cli as cli

    monkeypatch.setattr(pipeline_module, "build_default_pipeline",
                        lambda *a, **k: _scripted_pipe(BAD))
    out = io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", io.StringIO()):
        code = cli.run_extract(["--heuristics-only", "--file", str(tfile)])
    stored = json.loads(out.getvalue())
    assert stored["status"] == "needs_review" and code == 1
    proc = _cli(["validate", "--heuristics-only", "--file", str(tfile), "--data", json.dumps(stored)])
    direct = _cli(["validate", "--heuristics-only", "--file", str(tfile), "--data", json.dumps(stored["data"])])
    assert proc.returncode == direct.returncode == 1, proc.stderr[-2000:]
    a, b = json.loads(proc.stdout), json.loads(direct.stdout)
    assert a["status"] == b["status"] == "needs_review"
    assert _flags(a) == _flags(b) == ["vendor_name", "total"]


# ---------------------------------------------------------------------------
# EB-8: non-envelopes unchanged from HEAD
# ---------------------------------------------------------------------------

NON_ENVELOPES = {
    "bad": dict(BAD),
    "paid": {"status": "paid"},
    "mixed": {"status": "ok", "vendor_name": "WRONG CO", "data": GOOD},
    "empty": {},
    "good": dict(GOOD),
}


@pytest.mark.parametrize("name", list(NON_ENVELOPES))
def test_eb8_non_envelope_byte_identical_to_head(name):
    assert _digest(_v(NON_ENVELOPES[name])) == HEAD_NON_ENVELOPE_DIGESTS[name]


def test_eb8_non_envelope_verdicts():
    assert (_v(dict(BAD))["status"], _flags(_v(dict(BAD)))) == ("needs_review", ["vendor_name", "total"])
    paid = _v({"status": "paid"})
    assert paid["status"] == "ok" and all(v is None for v in paid["data"].values())
    mixed = _v({"status": "ok", "vendor_name": "WRONG CO", "data": GOOD})
    assert mixed["status"] == "needs_review" and _flags(mixed) == ["vendor_name"]
    assert mixed["data"]["vendor_name"] == "WRONG CO" and mixed["data"]["total"] is None
    empty = _v({})
    assert empty["status"] == "ok" and all(v is None for v in empty["data"].values())


# ---------------------------------------------------------------------------
# EB-9: library
# ---------------------------------------------------------------------------

def test_eb9_library_unwraps_envelope():
    got = InvoiceData.model_validate({"status": "ok", "data": GOOD})
    assert got == InvoiceData(**GOOD)
    assert got.vendor_name == "ACME CORP" and got.total == 10.0


def test_eb9_library_error_envelope_raises_validation_error():
    with pytest.raises(pydantic.ValidationError) as ei:
        InvoiceData.model_validate({"status": "error"})
    assert NO_DATA_ERROR in str(ei.value)


# ---------------------------------------------------------------------------
# EB-10: nothing else moves
# ---------------------------------------------------------------------------

def test_eb10_eval_stdout_byte_identical_to_head():
    proc = frozen_eval(timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_EVAL_STDOUT_SHA256


def test_eb10_extraction_result_fields_unchanged():
    assert sorted(ExtractionResult.model_fields) == HEAD_RESULT_FIELDS


def test_eb10_extract_output_unchanged_over_golden_matrix():
    replay = frozen_replay()
    rows = frozen_rows()
    assert len(rows) == 8
    out = {}
    for crit in (False, True):
        for r in rows:
            fx = replay[r["id"]]
            critic = Critic(ScriptedLLMClient([fx["critic_raw"]])) if crit else None
            p = ReflectivePipeline(Generator(ScriptedLLMClient([fx["generator_raw"]])), critic)
            d = p.extract(r["text"]).model_dump(mode="json")
            d.pop("trace_id")
            out[f"{r['id']}:{int(crit)}"] = _digest(d)
    assert len(out) == 16
    assert _digest(out) == HEAD_EXTRACT_MATRIX_SHA256


def test_eb10_cli_line_count_unchanged():
    lines = (REPO / "src" / "reflect_mcp" / "cli.py").read_text().splitlines()
    assert len(lines) == HEAD_CLI_LINES
