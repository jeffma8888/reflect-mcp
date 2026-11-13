"""Black-box behavioral tests for iter-67: batch rows echo the input record's id.

Covers Expected Behaviors 1-7 from the iter-67 spec:
  1  result rows echo the id as their first key; rest == no-id output; exit unchanged
  2  the id value is echoed verbatim, JSON type preserved
  3  error rows of a parsed object echo the id (missing text / data, bad data,
     pipeline exception)
  4  rows with no id are byte-identical to HEAD (no id key, non-JSON, non-object,
     build-failure envelopes)
  5  order + blank-line skip unchanged; c=1 and c=3 byte-identical
  6  single-document paths, MCP tools, ExtractionResult fields, eval stdout unchanged
  7  the spec's 7-line mixed reference case (validate --batch --heuristics-only)

HEAD constants (eval stdout digest, single-validate digest, ExtractionResult
field set, non-object envelopes) were measured by running the product from a
``git archive HEAD`` copy before the change was committed.

Offline: deterministic fake pipelines are patched in at
``reflect_mcp.pipeline.build_default_pipeline``, or the real heuristics-only
pipeline is used (no Critic, no Generator, no network, no secrets).

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
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.schema import ExtractionResult
from tests._golden import frozen_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
VERBS = ["run_extract", "run_validate"]
CONCS = [1, 3]

GOOD_TEXT = "ACME CORP\nInvoice INV-1\nTotal: $10.00"
GOOD_DATA = {"vendor_name": "ACME CORP", "total": 10.0}

# Measured on a `git archive HEAD` copy (pre-iter-67).
HEAD_EVAL_STDOUT_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
HEAD_SINGLE_VALIDATE_SHA256 = "064924ea0cf864802b98dcecaf62a042af0f25ec52d740731bd0ab25a2982db3"
HEAD_RESULT_FIELDS = [
    "critic_summary", "data", "field_confidence", "field_provenance",
    "flagged_fields", "notes", "overall_confidence", "status", "trace_id",
]
MISSING_TEXT_LINE = '{"status": "error", "error_message": "missing \\"text\\" key"}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(status: str, text: str) -> ExtractionResult:
    return ExtractionResult(
        status=status,
        data={},
        field_confidence={},
        overall_confidence=0.9 if status == "ok" else 0.3,
        flagged_fields=[],
        critic_summary=f"summary:{text}",
        trace_id=f"trace:{text}",
        notes="",
    )


class FakePipeline:
    """Deterministic: result depends only on text; 'boom' in text raises."""

    def __init__(self):
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _go(self, text):
        with self._lock:
            self.calls.append(text)
        if "boom" in text:
            raise RuntimeError(f"pipeline boom:{text}")
        return _result("needs_review" if "review" in text else "ok", text)

    def extract(self, text, *a, **k):
        return self._go(text)

    def validate(self, text, *a, **k):
        return self._go(text)


def _run(func_name: str, args: list[str], stdin: str = "", factory=None):
    """Run a cli entry point in-process; return (code, stdout, stderr)."""
    import reflect_mcp.cli as cli

    func = getattr(cli, func_name)
    out, err = io.StringIO(), io.StringIO()
    patches = [patch("sys.stdin", io.StringIO(stdin)), patch("sys.stdout", out), patch("sys.stderr", err)]
    if factory is not None:
        patches.append(patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory))
    for p in patches:
        p.start()
    try:
        code = func(args)
    finally:
        for p in reversed(patches):
            p.stop()
    return code, out.getvalue(), err.getvalue()


def _fake_run(verb: str, stdin: str, c: int = 1):
    fake = FakePipeline()
    return _run(verb, ["--batch", "--batch-concurrency", str(c)], stdin, factory=lambda *a, **k: fake)


def _lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.split("\n") if ln != ""]


def _pairs(line: str) -> list[tuple]:
    """Ordered top-level [key, value] pairs of a serialized JSON object
    (json.loads preserves key order)."""
    return [list(kv) for kv in json.loads(line).items()]


def _rec(text=None, data=GOOD_DATA, **extra) -> dict:
    rec = dict(extra)
    if text is not None:
        rec["text"] = text
    if data is not None:
        rec["data"] = data
    return rec


def _jsonl(*records) -> str:
    return "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records)


def _no_trace(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k != "trace_id"}


# ---------------------------------------------------------------------------
# EB-1: result rows echo the id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("text,status,code", [("plain doc", "ok", 0), ("please review", "needs_review", 1)])
def test_eb1_result_row_is_id_plus_single_output(verb, c, text, status, code):
    code_id, out_id, _ = _fake_run(verb, _jsonl(_rec(text, id="r-1")), c)
    code_no, out_no, _ = _fake_run(verb, _jsonl(_rec(text)), c)
    rows_id, rows_no = _lines(out_id), _lines(out_no)
    assert len(rows_id) == 1 and len(rows_no) == 1, (out_id, out_no)
    pairs = _pairs(rows_id[0])
    assert pairs[0] == ["id", "r-1"], rows_id[0]
    assert dict(pairs[1:]) == json.loads(rows_no[0])
    assert json.loads(rows_id[0]) == {"id": "r-1", **json.loads(rows_no[0])}
    assert json.loads(rows_id[0])["status"] == status
    assert code_id == code_no == code


@pytest.mark.parametrize("c", CONCS)
def test_eb1_real_heuristics_only_validate_row(c, tmp_path):
    """Real offline pipeline: id row minus id == no-id row (trace_id aside)."""
    f = tmp_path / "in.jsonl"
    f.write_text(_jsonl(_rec(GOOD_TEXT, id="a"), _rec(GOOD_TEXT)), encoding="utf-8")
    code, out, _ = _run("run_validate", ["--batch", "--heuristics-only", "--batch-concurrency", str(c), "--file", str(f)])
    rows = _lines(out)
    assert len(rows) == 2, out
    assert _pairs(rows[0])[0] == ["id", "a"]
    with_id, without = json.loads(rows[0]), json.loads(rows[1])
    assert "id" not in without
    assert with_id.pop("id") == "a"
    assert _no_trace(with_id) == _no_trace(without)
    assert with_id["status"] == "ok"
    assert code == 0


# ---------------------------------------------------------------------------
# EB-2: value echoed verbatim, JSON type preserved
# ---------------------------------------------------------------------------

ID_VALUES = ["a", 7, "7", 7.5, None, True, {"k": [1]}, [1, "x"], ""]


@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("value", ID_VALUES, ids=[repr(v) for v in ID_VALUES])
def test_eb2_id_value_and_type_preserved(verb, value):
    code, out, _ = _fake_run(verb, _jsonl(_rec("plain doc", id=value)))
    rows = _lines(out)
    assert len(rows) == 1, out
    pairs = _pairs(rows[0])
    assert pairs[0][0] == "id", rows[0]
    got = json.loads(rows[0])["id"]
    assert got == value
    assert type(got) is type(value)
    # serialized exactly as the JSON encoding of the input value
    assert rows[0].startswith('{"id": ' + json.dumps(value) + ", ")
    assert code == 0


def test_eb2_string_seven_not_retyped():
    _, out, _ = _fake_run("run_extract", _jsonl(_rec("doc", id="7"), _rec("doc", id=7)))
    rows = _lines(out)
    assert len(rows) == 2, out
    assert json.loads(rows[0])["id"] == "7" and isinstance(json.loads(rows[0])["id"], str)
    assert json.loads(rows[1])["id"] == 7 and isinstance(json.loads(rows[1])["id"], int)


# ---------------------------------------------------------------------------
# EB-3: error rows of a parsed object echo the id
# ---------------------------------------------------------------------------

def _error_cases(verb: str):
    cases = [
        ("missing_text", _rec(None, data={}), 'missing "text" key'),
        ("pipeline_exception", _rec("boom doc"), "pipeline boom:boom doc"),
    ]
    if verb == "run_validate":
        cases.append(("missing_data", _rec("x", data=None), 'missing "data" key'))
        cases.append(("bad_data", _rec("x", data={"total": "abc"}), None))
    return cases


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb3_error_rows_echo_id(verb, c):
    for name, rec, msg in _error_cases(verb):
        code_id, out_id, _ = _fake_run(verb, _jsonl({"id": 42, **rec}), c)
        code_no, out_no, _ = _fake_run(verb, _jsonl(rec), c)
        rows_id, rows_no = _lines(out_id), _lines(out_no)
        assert len(rows_id) == 1 and len(rows_no) == 1, (name, out_id, out_no)
        no = json.loads(rows_no[0])
        assert no["status"] == "error", (name, no)
        if msg is not None:
            assert no["error_message"] == msg, name
        assert _pairs(rows_id[0]) == [["id", 42], ["status", "error"], ["error_message", no["error_message"]]], name
        assert code_id == code_no == 3, name


def test_eb3_bad_data_message_is_invoice_validation():
    _, out, _ = _fake_run("run_validate", _jsonl({"id": "z", "text": "x", "data": {"total": "abc"}}))
    rows = _lines(out)
    assert len(rows) == 1, out
    row = json.loads(rows[0])
    assert list(row) == ["id", "status", "error_message"]
    assert row["id"] == "z" and row["status"] == "error"
    assert row["error_message"].startswith("1 validation error for InvoiceData")


def test_eb3_spec_example_validate_missing_text():
    p = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only"],
        input='{"id": 7, "data": {}}\n', capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=120,
    )
    assert p.stdout == '{"id": 7, "status": "error", "error_message": "missing \\"text\\" key"}\n', p.stdout
    assert p.returncode == 3


# ---------------------------------------------------------------------------
# EB-4: rows with no id are byte-identical to HEAD
# ---------------------------------------------------------------------------

NO_ID_LINES = [
    ("not_object_list", "[1]", MISSING_TEXT_LINE),
    ("not_object_list_of_id", '[{"id": 1, "text": "x"}]', MISSING_TEXT_LINE),
    ("not_object_str", '"x"', MISSING_TEXT_LINE),
    ("not_object_num", "5", MISSING_TEXT_LINE),
    ("not_object_null", "null", MISSING_TEXT_LINE),
    ("invalid_json", "not json",
     '{"status": "error", "error_message": "invalid JSON: Expecting value: line 1 column 1 (char 0)"}'),
    # whitespace_only dropped in iter 96: such lines now emit no row (tests/test_iter96_behavior.py EB5/EB6)
]


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("name,line,expected", NO_ID_LINES, ids=[n[0] for n in NO_ID_LINES])
def test_eb4_unparsed_or_non_object_rows_unchanged(verb, c, name, line, expected):
    code, out, _ = _fake_run(verb, line + "\n", c)
    assert out == expected + "\n"
    assert code == 3


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb4_records_without_top_level_id_get_no_id(verb, c):
    recs = [_rec("plain doc"), _rec("plain doc", meta={"id": 1}), _rec("plain doc", ID=5), _rec("boom doc")]
    _, out, _ = _fake_run(verb, _jsonl(*recs), c)
    rows = [json.loads(r) for r in _lines(out)]
    assert len(rows) == 4, out
    for row in rows:
        assert "id" not in row, row
    assert rows[0] == _no_trace(rows[0]) | {"trace_id": "trace:plain doc"}
    assert rows[3] == {"status": "error", "error_message": "pipeline boom:boom doc"}


@pytest.mark.parametrize("c", [None, 1, 3])
@pytest.mark.parametrize("verb", VERBS)
def test_eb4_build_failure_envelope_has_no_id(verb, c):
    def factory(*a, **k):
        raise ValueError("boom-config")

    flags = [] if c is None else ["--batch-concurrency", str(c)]
    code, out, _ = _run(verb, ["--batch", *flags], _jsonl(_rec("a", id=1), _rec("b", id=2)), factory=factory)
    assert out == '{"status": "error", "error_message": "boom-config"}\n'
    assert code == 3


@pytest.mark.parametrize("verb", VERBS)
def test_eb4_per_thread_build_failure_rows_have_no_id(verb):
    fake = FakePipeline()
    n = []
    lock = threading.Lock()

    def factory(*a, **k):
        with lock:
            n.append(1)
            first = len(n) == 1
        if first:
            return fake
        raise RuntimeError("late")

    recs = [_rec(f"doc {i}", id=i) for i in range(6)]
    code, out, _ = _run(verb, ["--batch", "--batch-concurrency", "3"], _jsonl(*recs), factory=factory)
    rows = [json.loads(r) for r in _lines(out)]
    assert len(rows) == 6, out
    for i, row in enumerate(rows):
        if row["status"] == "error" and row.get("error_message") == "late":
            assert row == {"status": "error", "error_message": "late"}
        else:
            assert row["id"] == i and row["status"] == "ok", row


# ---------------------------------------------------------------------------
# EB-5: order, blank-line skip, c=1 vs c=3 byte-identity
# ---------------------------------------------------------------------------

def _ordered_input() -> str:
    lines = []
    for i in range(12):
        if i % 4 == 1:
            lines.append("")
        text = "please review" if i % 3 == 0 else ("boom doc" if i % 5 == 2 else f"doc {i}")
        lines.append(json.dumps(_rec(text, id=f"k{i}")))
    lines.insert(5, "not json")
    lines.insert(8, json.dumps(_rec("no id doc")))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("verb", VERBS)
def test_eb5_order_and_blank_skip(verb):
    inp = _ordered_input()
    expected_ids = []
    for ln in inp.split("\n"):
        if ln == "":
            continue
        try:
            obj = json.loads(ln)
        except ValueError:
            expected_ids.append("<none>")
            continue
        expected_ids.append(obj.get("id", "<none>"))
    for c in CONCS:
        code, out, _ = _fake_run(verb, inp, c)
        rows = [json.loads(r) for r in _lines(out)]
        assert len(rows) == len(expected_ids) == 14, out
        assert [r.get("id", "<none>") for r in rows] == expected_ids
        assert code == 3


@pytest.mark.parametrize("verb", VERBS)
def test_eb5_c1_and_c3_byte_identical(verb):
    inp = _ordered_input()
    code1, out1, _ = _fake_run(verb, inp, 1)
    code3, out3, _ = _fake_run(verb, inp, 3)
    assert len(_lines(out1)) == 14
    assert out1 == out3
    assert code1 == code3 == 3


# ---------------------------------------------------------------------------
# EB-6: single-document paths, tools, ExtractionResult, eval unchanged
# ---------------------------------------------------------------------------

ID_TEXT = 'id: 7\n{"id": 7}\nACME CORP\nInvoice INV-1\nTotal: $10.00'


def _digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_eb6_single_validate_unchanged_with_id_in_text():
    code, out, _ = _run("run_validate", ["--heuristics-only", "--data", json.dumps(GOOD_DATA)], ID_TEXT)
    d = json.loads(out)
    assert "id" not in d
    assert _digest(_no_trace(d)) == HEAD_SINGLE_VALIDATE_SHA256
    assert code == 0


def test_eb6_single_extract_unchanged_with_id_in_text():
    fake = FakePipeline()
    code, out, _ = _run("run_extract", [], ID_TEXT, factory=lambda *a, **k: fake)
    d = json.loads(out)
    assert "id" not in d
    assert d == json.loads(_result("ok", ID_TEXT).model_dump_json())
    assert code == 0


def test_eb6_mcp_validate_tool_unchanged():
    from reflect_mcp import tools

    r = tools.validate_tool(ID_TEXT, dict(GOOD_DATA), heuristics_only=True)
    assert isinstance(r, dict)
    assert "id" not in r
    assert _digest(_no_trace(r)) == HEAD_SINGLE_VALIDATE_SHA256


def test_eb6_extraction_result_gains_no_field():
    assert sorted(ExtractionResult.model_fields) == HEAD_RESULT_FIELDS


def test_eb6_eval_stdout_byte_identical_to_head():
    p = frozen_eval(timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert hashlib.sha256(p.stdout.encode()).hexdigest() == HEAD_EVAL_STDOUT_SHA256


# ---------------------------------------------------------------------------
# EB-7: mixed-input reference case
# ---------------------------------------------------------------------------

REFERENCE_LINES = [
    '{"id":"a","text":"ACME CORP\\nInvoice INV-1\\nTotal: $10.00","data":{"vendor_name":"ACME CORP","total":10.0}}',
    "",
    "not json",
    '{"id":7,"data":{}}',
    '{"id":"c","text":"x"}',
    '{"id":null,"text":"x","data":{"total":"abc"}}',
    '{"text":"x","data":{}}',
]


@pytest.mark.parametrize("c", [None, 1, 3])
def test_eb7_reference_case(tmp_path, c):
    f = tmp_path / "ref.jsonl"
    f.write_text("\n".join(REFERENCE_LINES) + "\n", encoding="utf-8")
    flags = [] if c is None else ["--batch-concurrency", str(c)]
    p = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--batch", "--heuristics-only", "--file", str(f), *flags],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=120,
    )
    rows = _lines(p.stdout)
    assert len(rows) == 6, p.stdout
    objs = [json.loads(r) for r in rows]

    assert _pairs(rows[0])[0] == ["id", "a"] and objs[0]["status"] == "ok"

    assert "id" not in objs[1]
    assert objs[1]["status"] == "error" and objs[1]["error_message"].startswith("invalid JSON: ")

    assert objs[2] == {"id": 7, "status": "error", "error_message": 'missing "text" key'}
    assert list(objs[2]) == ["id", "status", "error_message"]

    assert objs[3] == {"id": "c", "status": "error", "error_message": 'missing "data" key'}
    assert list(objs[3]) == ["id", "status", "error_message"]

    assert list(objs[4]) == ["id", "status", "error_message"]
    assert objs[4]["id"] is None and objs[4]["status"] == "error"
    assert objs[4]["error_message"].startswith("1 validation error for InvoiceData")

    assert "id" not in objs[5] and objs[5]["status"] == "ok"
    assert p.returncode == 3
