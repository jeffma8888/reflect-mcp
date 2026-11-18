"""Black-box behavioral tests for iter-103: a JSON boolean is not an amount.

``InvoiceData.subtotal`` / ``.tax`` / ``.total`` must reject ``true``/``false``
with a validation error instead of silently coercing them to ``1.0``/``0.0``.

Covers the Expected Behaviors from pm.md:
  1-3  CLI single ``validate --heuristics-only``: a boolean total/tax/subtotal
       exits 2, prints nothing on stdout, names the field and ``boolean``.
  4    Non-bool numerics (int, float, numeric string) and null still pass.
  5    MCP/library ``validate_tool``: ``status: error`` with the schema prefix,
       no ``data`` key.
  6    ``--batch``: the boolean row is an error row, the numeric row is ok,
       order kept, exit 3 (worst row).
  7    Python API: ``model_validate`` / constructor raise ``ValidationError``;
       int and numeric string still coerce.
  8    Generator: a bool-amount reply is a parse failure that is retried;
       a persistent one raises ``GeneratorError`` after max_retries+1 calls.
  9    Golden-derived: every non-null amount in every golden ``expected`` swapped
       to True/False is an error naming the field; the unmodified answers stay ok.
  10   Stored-envelope unwrap still works; an envelope with a boolean
       ``data.total`` exits 2 with ``boolean``.
  11   ``eval/run_eval.py`` keeps rc 0 and its 91.2% score.

Isolation contract: drives only the public CLI, ``validate_tool``,
``InvoiceData``, ``Generator`` and the golden dataset; never reads
implementation source. No network, no secrets, no optional extra imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.schema import InvoiceData
from reflect_mcp.tools import validate_tool

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
SRC = "ACME\nTotal: 1.00 USD\n"
PREFIX = "data failed InvoiceData validation"
AMOUNTS = ("subtotal", "tax", "total")


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


@pytest.fixture
def srcfile(tmp_path) -> Path:
    p = tmp_path / "src.txt"
    p.write_text(SRC, encoding="utf-8")
    return p


def _validate_file(srcfile: Path, data) -> subprocess.CompletedProcess:
    payload = data if isinstance(data, str) else json.dumps(data)
    return _cli(["validate", "--heuristics-only", "--file", str(srcfile),
                 "--data", payload], cwd=srcfile.parent)


class _RecordingClient:
    """List-backed LLMClient that records every complete() call."""

    def __init__(self, replies, repeat_last: bool = False):
        self._replies = list(replies)
        self._repeat = repeat_last
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if self._repeat and len(self._replies) == 1:
            return self._replies[0]
        return self._replies.pop(0)


# ---- EB1-3: CLI single -----------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("total", True), ("total", False),
    ("tax", False), ("tax", True),
    ("subtotal", True), ("subtotal", False),
])
def test_eb1_3_cli_bool_amount_exits_2(srcfile, field, value):
    r = _validate_file(srcfile, {field: value, "currency": "USD"})
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert r.stdout == ""
    assert "--data failed InvoiceData validation" in r.stderr
    assert field in r.stderr
    assert "boolean" in r.stderr


def test_eb1_raw_literal_true_spec_payload(srcfile):
    # exact payload text from the spec (JSON literal, not json.dumps)
    r = _validate_file(srcfile, '{"total":true,"currency":"USD"}')
    assert r.returncode == 2, r.stdout
    assert r.stdout == ""
    assert "status" not in r.stdout


# ---- EB4: non-bool numerics unchanged --------------------------------------
@pytest.mark.parametrize("payload", [
    '{"total":1,"currency":"USD"}',
    '{"total":1.0,"currency":"USD"}',
    '{"total":"1.00","currency":"USD"}',
])
def test_eb4_numeric_totals_still_ok(srcfile, payload):
    r = _validate_file(srcfile, payload)
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = json.loads(r.stdout)
    assert out["status"] == "ok"
    assert out["data"]["total"] == 1.0
    assert not isinstance(out["data"]["total"], bool)


def test_eb4_null_total_still_accepted(srcfile):
    r = _validate_file(srcfile, '{"total":null,"currency":"USD"}')
    assert r.returncode != 2, r.stderr
    out = json.loads(r.stdout)
    assert out["status"] != "error"
    assert out["data"]["total"] is None


# ---- EB5: MCP / library ----------------------------------------------------
@pytest.mark.parametrize("field", AMOUNTS)
def test_eb5_validate_tool_bool_is_error(field):
    out = validate_tool(text="ACME\nTotal: 1.00 USD", data={field: True},
                        heuristics_only=True)
    assert out["status"] == "error"
    msg = out["error_message"]
    assert msg.startswith(PREFIX)
    assert field in msg and "boolean" in msg
    assert "data" not in out


def test_eb5_counterpart_numeric_is_ok():
    out = validate_tool(text="ACME\nTotal: 1.00 USD",
                        data={"total": 1.0, "currency": "USD"}, heuristics_only=True)
    assert out["status"] == "ok"
    assert out["data"]["total"] == 1.0


# ---- EB6: batch ------------------------------------------------------------
def test_eb6_batch_bool_row_error_numeric_row_ok(tmp_path):
    rows = [
        {"id": "r1", "text": "ACME\nTotal: 1.00 USD", "data": {"total": True}},
        {"id": "r2", "text": "ACME\nTotal: 1.00 USD", "data": {"total": 1.0}},
    ]
    stdin = "".join(json.dumps(r) + "\n" for r in rows)
    r = _cli(["validate", "--batch", "--heuristics-only"], cwd=tmp_path, stdin=stdin)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2, r.stdout
    r1, r2 = (json.loads(ln) for ln in lines)
    assert r1["id"] == "r1" and r1["status"] == "error"
    assert "boolean" in r1["error_message"]
    assert r2["id"] == "r2" and r2["status"] == "ok"
    assert r.returncode == 3, (r.returncode, r.stderr)


# ---- EB7: Python API -------------------------------------------------------
@pytest.mark.parametrize("field", AMOUNTS)
@pytest.mark.parametrize("value", [True, False])
def test_eb7_model_rejects_bool(field, value):
    with pytest.raises(ValidationError) as ei:
        InvoiceData.model_validate({field: value})
    assert "boolean" in str(ei.value)
    with pytest.raises(ValidationError):
        InvoiceData(**{field: value})


def test_eb7_int_and_numeric_string_still_coerce():
    d = InvoiceData.model_validate({"total": 5, "tax": "0.5"})
    assert d.total == 5.0 and isinstance(d.total, float)
    assert d.tax == 0.5
    assert InvoiceData(total=None).total is None


# ---- EB8: Generator --------------------------------------------------------
BOOL_REPLY = '{"vendor_name":"ACME","total":true,"currency":"USD"}'
GOOD_REPLY = '{"vendor_name":"ACME","total":1.0,"currency":"USD"}'


def test_eb8_generator_persistent_bool_raises_after_retries():
    client = _RecordingClient([BOOL_REPLY], repeat_last=True)
    with pytest.raises(GeneratorError):
        Generator(client, max_retries=1).generate(SRC)
    assert client.calls == 2


def test_eb8_generator_recovers_on_retry():
    client = _RecordingClient(['{"total":true}', GOOD_REPLY])
    out = Generator(client, max_retries=1).generate(SRC)
    data = out if isinstance(out, InvoiceData) else getattr(out, "data", out)
    assert data.total == 1.0
    assert client.calls == 2


# ---- EB9: golden-derived ---------------------------------------------------
def _golden_rows() -> list[dict]:
    return [json.loads(ln) for ln in GOLDEN.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def test_eb9_golden_unmodified_expected_ok():
    rows = _golden_rows()
    assert rows
    for row in rows:
        out = validate_tool(text=row["text"], data=row["expected"], heuristics_only=True)
        assert out["status"] == "ok", (row["id"], out)


def test_eb9_golden_bool_swaps_are_errors():
    swaps = 0
    for row in _golden_rows():
        for field in AMOUNTS:
            if row["expected"].get(field) is None:
                continue
            for value in (True, False):
                data = dict(row["expected"], **{field: value})
                out = validate_tool(text=row["text"], data=data, heuristics_only=True)
                assert out["status"] == "error", (row["id"], field, value, out)
                assert field in out["error_message"], (row["id"], field)
                assert "boolean" in out["error_message"]
                swaps += 1
    assert swaps > 0


# ---- EB10: stored envelope -------------------------------------------------
def _envelope(data: dict) -> dict:
    out = validate_tool(text="ACME\nTotal: 1.00 USD", data=data, heuristics_only=True)
    assert out["status"] == "ok"
    return out


def test_eb10_numeric_envelope_still_unwraps(srcfile):
    env = _envelope({"total": 1.0, "currency": "USD"})
    r = _validate_file(srcfile, env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["status"] == "ok"
    assert out["data"]["total"] == 1.0


def test_eb10_bool_inside_envelope_exits_2(srcfile):
    env = _envelope({"total": 1.0, "currency": "USD"})
    env["data"]["total"] = True
    r = _validate_file(srcfile, env)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert r.stdout == ""
    assert "boolean" in r.stderr


# ---- EB11: eval keeps rc 0 and its score -----------------------------------
def test_eb11_eval_rc0_score_unchanged(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "eval" / "run_eval.py")],
                       capture_output=True, text=True, env=_env(), cwd=str(REPO),
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "91.2%" in r.stdout
