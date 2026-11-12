"""Black-box behavioral tests for iter-76: honest ``ok`` notes.

An ``ok`` result's ``notes`` name every committed field a heuristic penalized
that still cleared a (lowered) field threshold, and say nothing was verified
when no field was committed. ``status`` and every confidence value are
unchanged. Covers Expected Behaviors 1-9 of pm.md.

Isolation contract: these tests drive the public interface only (``gate.decide``,
``ReflectivePipeline`` with ``ScriptedLLMClient``, the MCP tool functions, the
CLI entry points and ``eval/run_eval.py``) plus the golden data files, README and
PRODUCT.md. No implementation source was read. The HEAD constants below (the
minus-notes digest, the needs_review-notes digest and the default-threshold
golden notes) were captured by running this module's own ``_matrix()`` against a
``git archive HEAD`` tree (``PYTHONPATH=<head>/src:.``) before commit.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig, decide
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import EXTRACTION_FIELDS, CritiqueResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests._golden import FROZEN_EVAL_SHA256, frozen_eval

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
GOLDEN = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"

OK_NOTE = "All fields cleared the confidence thresholds."
HO_OK_NOTE = (
    "No Critic ran: every field passed the deterministic heuristic checks only; "
    "field confidences are heuristic factors, not a Critic's judgement."
)
HO_PREFIX = "No Critic ran: field confidences are heuristic factors, not a Critic's judgement."
NV_CRITIC = "No field was committed (every value is null), so nothing was verified."
NV_HO = "No Critic ran. No field was committed (every value is null), so nothing was verified."
NR_70 = "1 field(s) below the 70% confidence threshold require human review before use."
NR_25 = "1 field(s) below the 25% confidence threshold require human review before use."

T25 = GateConfig(field_threshold=0.25, overall_threshold=0.0)

SRC = (
    "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: $100.00\n"
    "Tax: $8.50\nTotal: $108.50\nPaid in USD"
)
CLEAN = {
    "vendor_name": "ACME CORP", "invoice_number": "INV-1001", "invoice_date": "2026-03-14",
    "subtotal": 100.0, "tax": 8.5, "total": 108.5, "currency": "USD",
}
CASES = {
    "t999": dict(CLEAN, total=999.0),
    "two": dict(CLEAN, subtotal=555.0, total=999.0),
    "date_total": dict(CLEAN, invoice_date="2026-09-30", total=999.0),
    "clean": CLEAN,
    "empty": {},
    "totl": {"totl": 999},
    "allnull": {k: None for k in CLEAN},
}

# Captured from the HEAD tree (see module docstring).
# iter 79: re-pinned over the matrix minus ho/totl/* (was a1b7f221... over all rows);
# the subset digest is identical on the iter-78 HEAD tree and the iter-79 tree.
# iter 89: re-pinned minus */allnull/* (was b82058ac...); identical on the iter-88 HEAD tree.
HEAD_MINUS_NOTES_SHA256 = "2a889287e14327e5b17206e57a1bbd51553edc5e10c49d918d4bacef064b51b3"
HEAD_NR_NOTES_SHA256 = "bd9e9b5b0b4df28e07239d8e018a09a330f6a3cef3afff9c04854def11330184"
HEAD_DEFAULT_NOTES = {
    "inv-001": OK_NOTE, "inv-002": OK_NOTE, "inv-003": OK_NOTE, "inv-005": OK_NOTE,
    "inv-004": NR_70, "inv-006": NR_70, "inv-007": NR_70, "inv-008": NR_70, "inv-009": NR_70,
}
GOLDEN_IDS = tuple(sorted(HEAD_DEFAULT_NOTES))


# ---------------------------------------------------------------- helpers
def _dump(r) -> dict:
    d = r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r)
    d = dict(d)
    d.pop("trace_id", None)
    return d


def _status(d: dict) -> str:
    s = d["status"]
    return getattr(s, "value", s)


def _rows_by_id() -> dict:
    return {json.loads(ln)["id"]: json.loads(ln) for ln in GOLDEN.read_text().splitlines() if ln.strip()}


def _replay_pipe(rid: str, cfg: GateConfig | None = None) -> ReflectivePipeline:
    entry = json.loads(REPLAY.read_text())[rid]
    kw = {"gate_config": cfg} if cfg is not None else {}
    return ReflectivePipeline(
        Generator(ScriptedLLMClient([entry["generator_raw"]])),
        Critic(ScriptedLLMClient([entry["critic_raw"]])),
        **kw,
    )


def _golden(rid: str, cfg: GateConfig | None = None) -> dict:
    return _dump(_replay_pipe(rid, cfg).extract(_rows_by_id()[rid]["text"]))


def _critic_raw(conf: float = 0.95) -> str:
    verdicts = [{"field": f, "confidence": conf, "grounded": True, "rationale": "x"} for f in CLEAN]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.95, "summary": "ok"})


def _critique(conf: float = 0.95) -> CritiqueResult:
    return CritiqueResult.model_validate(json.loads(_critic_raw(conf)))


def _model(data: dict) -> InvoiceData:
    return InvoiceData(**{k: v for k, v in data.items() if k in EXTRACTION_FIELDS})


def _critic_validate(data: dict, thr: float) -> dict:
    pipe = ReflectivePipeline(None, Critic(ScriptedLLMClient([_critic_raw()])),
                              gate_config=GateConfig(field_threshold=thr, overall_threshold=0.0))
    return _dump(pipe.validate(SRC, _model(data)))


def _tools():
    return importlib.import_module("reflect_mcp.tools")


def _ho(data: dict, thr: float) -> dict:
    with patch.dict(os.environ, {"REFLECT_FIELD_THRESHOLD": "0", "REFLECT_OVERALL_THRESHOLD": "0"}):
        return _dump(_tools().validate_tool(SRC, dict(data), thr, 0.0, True))


def _expected_items(text: str, data: dict) -> list[tuple[str, float]]:
    h = run_heuristics(text, _model(data))
    m = _model(data)
    return [(f, h.factor_for(f)) for f in EXTRACTION_FIELDS
            if getattr(m, f) is not None and h.factor_for(f) < 1.0]


def _clause(items: list[tuple[str, float]], thr: float) -> str:
    joined = ", ".join(f"{f} (factor {x:.0%})" for f, x in items)
    return f" Heuristics penalized {joined}; each penalized field still cleared the {thr:.0%} field threshold."


def _matrix() -> dict:
    """Every probed input -> result minus trace_id (runs unchanged on HEAD)."""
    out = {}
    for rid in GOLDEN_IDS:
        out[f"golden/{rid}/default"] = _golden(rid)
        out[f"golden/{rid}/t25"] = _golden(rid, T25)
    for name, data in CASES.items():
        for thr in (0.0, 0.30, 0.70):
            out[f"ho/{name}/{thr}"] = _ho(data, thr)
        for thr in (0.25, 0.70):
            out[f"critic/{name}/{thr}"] = _critic_validate(data, thr)
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _minus_notes(m: dict) -> dict:
    return {k: {a: b for a, b in v.items() if a != "notes"} for k, v in m.items()}


def _nr_notes(m: dict) -> dict:
    return {k: v["notes"] for k, v in m.items() if _status(v) == "needs_review"}


@pytest.fixture(scope="module")
def matrix() -> dict:
    return _matrix()


# ---------------------------------------------------------------- EB1
@pytest.mark.parametrize("rid", ["inv-001", "inv-002", "inv-003", "inv-005"])
def test_eb1_golden_ok_rows_keep_head_note(rid):
    r = _golden(rid)
    assert _status(r) == "ok"
    assert r["notes"] == OK_NOTE


@pytest.mark.parametrize("thr", [0.25, 0.70])
def test_eb1_critic_clean_validate_keeps_head_note(thr):
    r = _critic_validate(CLEAN, thr)
    assert _status(r) == "ok"
    assert r["notes"] == OK_NOTE


def test_eb1_decide_clean_keeps_head_note():
    r = decide(_model(CLEAN), _critique(), run_heuristics(SRC, _model(CLEAN)), "t1")
    assert r.status == "ok"
    assert r.notes == OK_NOTE


# ---------------------------------------------------------------- EB2
def test_eb2_constant_value_unchanged():
    from reflect_mcp import gate

    assert gate.HEURISTICS_ONLY_OK_NOTE == HO_OK_NOTE


@pytest.mark.parametrize("thr", [0.0, 0.30, 0.70])
def test_eb2_heuristics_only_clean_keeps_constant(thr):
    r = _ho(CLEAN, thr)
    assert _status(r) == "ok"
    assert r["notes"] == HO_OK_NOTE


def test_eb2_decide_no_critic_clean_keeps_constant():
    h = run_heuristics(SRC, _model(CLEAN))
    r = decide(_model(CLEAN), _critique(), h, "t2", critic_ran=False)
    assert r.status == "ok"
    assert r.notes == HO_OK_NOTE


# ---------------------------------------------------------------- EB3
@pytest.mark.parametrize("rid,field", [
    ("inv-006", "invoice_date"), ("inv-008", "total"), ("inv-009", "invoice_date"),
])
def test_eb3_golden_replay_at_025_names_penalized_field(rid, field):
    r = _golden(rid, T25)
    assert _status(r) == "ok"
    assert r["notes"] == (
        f"{OK_NOTE} Heuristics penalized {field} (factor 30%); "
        "each penalized field still cleared the 25% field threshold."
    )


def test_eb3_inv008_exact_spec_example():
    r = _golden("inv-008", T25)
    assert r["notes"] == (
        "All fields cleared the confidence thresholds. Heuristics penalized total "
        "(factor 30%); each penalized field still cleared the 25% field threshold."
    )


def test_eb3_critic_validate_total_999():
    r = _critic_validate(CASES["t999"], 0.25)
    assert _status(r) == "ok"
    assert r["notes"] == OK_NOTE + _clause(_expected_items(SRC, CASES["t999"]), 0.25)
    assert "total (factor 30%)" in r["notes"]


# ---------------------------------------------------------------- EB4
def test_eb4_heuristics_only_total_999_at_030():
    r = _ho(CASES["t999"], 0.30)
    assert _status(r) == "ok"
    assert r["field_confidence"]["total"] == 0.3
    assert "total (factor 30%)" in r["notes"]
    assert "30% field threshold" in r["notes"]
    assert "every field passed" not in r["notes"]
    assert r["notes"] == HO_PREFIX + _clause([("total", 0.3)], 0.30)


def test_eb4_heuristics_only_threshold_zero_prints_zero_percent():
    r = _ho(CASES["t999"], 0.0)
    assert _status(r) == "ok"
    assert r["notes"] == HO_PREFIX + _clause([("total", 0.3)], 0.0)
    assert "0% field threshold" in r["notes"]


def test_eb4_decide_no_critic_penalized():
    m = _model(CASES["t999"])
    r = decide(m, _critique(), run_heuristics(SRC, m), "t4",
               GateConfig(field_threshold=0.25, overall_threshold=0.0), critic_ran=False)
    assert r.status == "ok"
    assert r.notes.startswith(HO_PREFIX + " Heuristics penalized ")
    assert "every field passed" not in r.notes


# ---------------------------------------------------------------- EB5
@pytest.mark.parametrize("name", ["empty", "totl", "allnull"])
@pytest.mark.parametrize("thr", [0.0, 0.30, 0.70])
def test_eb5_heuristics_only_nothing_committed(name, thr):
    r = _ho(CASES[name], thr)
    if name == "totl":  # iter 79: an unknown caller key is a schema error, not an all-null ok
        assert _status(r) == "error" and "unknown field(s): totl" in r["error_message"]
        return
    if name == "allnull" and thr == 0.70:  # iter 89: SRC labels a total, so the null total is flagged
        assert _status(r) == "needs_review" and [f["field"] for f in r["flagged_fields"]] == ["total"]
        return
    assert _status(r) == "ok"
    assert r["notes"] == NV_HO
    assert "cleared" not in r["notes"] and "passed" not in r["notes"]


@pytest.mark.parametrize("name", ["empty", "totl", "allnull"])
@pytest.mark.parametrize("thr", [0.25, 0.70])
def test_eb5_critic_nothing_committed(name, thr):
    r = _critic_validate(CASES[name], thr)
    if name == "allnull" and thr == 0.70:  # iter 89: SRC labels a total, so the null total is flagged
        assert _status(r) == "needs_review" and [f["field"] for f in r["flagged_fields"]] == ["total"]
        return
    assert _status(r) == "ok"
    assert r["notes"] == NV_CRITIC
    assert "cleared" not in r["notes"] and "passed" not in r["notes"]


def test_eb5_decide_both_paths_all_null():
    m = InvoiceData()
    h = run_heuristics(SRC, m)
    assert decide(m, _critique(), h, "t5").notes == NV_CRITIC
    assert decide(m, _critique(), h, "t5", critic_ran=False).notes == NV_HO


# ---------------------------------------------------------------- EB6
@pytest.mark.parametrize("name", ["two", "date_total"])
def test_eb6_several_penalized_fields_listed_in_order(name):
    items = _expected_items(SRC, CASES[name])
    assert len(items) >= 2, items  # the probe really penalizes 2+ committed fields
    r = _ho(CASES[name], 0.30)
    assert _status(r) == "ok"
    assert r["notes"] == HO_PREFIX + _clause(items, 0.30)
    c = _critic_validate(CASES[name], 0.25)
    assert _status(c) == "ok"
    assert c["notes"] == OK_NOTE + _clause(items, 0.25)
    pos = [c["notes"].index(f"{f} (factor ") for f, _ in items]
    assert pos == sorted(pos)


def test_eb6_two_fields_literal():
    r = _ho(CASES["two"], 0.30)
    assert r["notes"] == (
        HO_PREFIX + " Heuristics penalized subtotal (factor 35%), total (factor 30%); "
        "each penalized field still cleared the 30% field threshold."
    )


# ---------------------------------------------------------------- EB7
def test_eb7_everything_but_notes_identical_to_head(matrix):
    # iter 79: the ho/totl rows are now schema errors (asserted in EB5); pin the rest.
    rest = {k: v for k, v in matrix.items() if not k.startswith("ho/totl/")}
    rest = {k: v for k, v in rest.items() if "/allnull/" not in k}  # iter 89: null total flagged (EB5)
    assert _sha(_minus_notes(rest)) == HEAD_MINUS_NOTES_SHA256


def test_eb7_needs_review_notes_identical_to_head(matrix):
    nr = _nr_notes({k: v for k, v in matrix.items() if "/allnull/" not in k})  # iter 89: see EB5
    assert nr, "matrix must contain needs_review results"
    assert _sha(nr) == HEAD_NR_NOTES_SHA256


def test_eb7_penalized_cleared_status_still_ok(matrix):
    for key in ("golden/inv-006/t25", "golden/inv-008/t25", "golden/inv-009/t25",
                "ho/t999/0.3", "critic/t999/0.25"):
        assert _status(matrix[key]) == "ok", key
        assert matrix[key]["flagged_fields"] == [], key


# ---------------------------------------------------------------- EB8
@pytest.mark.parametrize("rid", GOLDEN_IDS)
def test_eb8_default_threshold_golden_notes_equal_head(rid):
    assert _golden(rid)["notes"] == HEAD_DEFAULT_NOTES[rid]


def test_eb8_frozen_eval_stdout_identical():
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == FROZEN_EVAL_SHA256


def test_eb8_live_eval_exit_zero():
    proc = subprocess.run([sys.executable, str(EVAL_DIR / "run_eval.py")], capture_output=True,
                          text=True, cwd=str(REPO), timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert "Heuristics penalized" not in proc.stdout
    assert "nothing was verified" not in proc.stdout


# ---------------------------------------------------------------- EB9
INV008_NOTE = (
    "All fields cleared the confidence thresholds. Heuristics penalized total "
    "(factor 30%); each penalized field still cleared the 25% field threshold."
)


def _inv008_factory(*a, **k):
    entry = json.loads(REPLAY.read_text())["inv-008"]
    crit = Critic(ScriptedLLMClient([entry["critic_raw"]])) if k.get("critic_enabled", True) else None
    gen = Generator(ScriptedLLMClient([entry["generator_raw"]]))
    return ReflectivePipeline(gen, crit, gate_config=k.get("gate_config"))


def _cli_inproc(verb: str, args: list[str], stdin: str = "") -> tuple[int, str]:
    cli = importlib.import_module("reflect_mcp.cli")
    func = getattr(cli, "run_extract" if verb == "extract" else "run_validate")
    out = io.StringIO()
    with patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=_inv008_factory), \
            patch("sys.stdin", io.StringIO(stdin)), patch("sys.stdout", out), \
            patch("sys.stderr", io.StringIO()):
        code = func(args)
    return code, out.getvalue()


def _cli_sub(args: list[str], stdin: str = "") -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c",
           "import sys, runpy; sys.modules['anthropic'] = None; "
           "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
           "runpy.run_module('reflect_mcp', run_name='__main__')", *args]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=120)


def _ho_t999_note() -> str:
    return HO_PREFIX + _clause([("total", 0.3)], 0.30)


def test_eb9_cli_validate_heuristics_only_single(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(SRC)
    proc = _cli_sub(["validate", "--heuristics-only", "--file", str(p), "--data",
                     json.dumps(CASES["t999"]), "--field-threshold", "0.3", "--overall-threshold", "0"])
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    assert out["status"] == "ok"
    assert out["notes"] == _ho_t999_note()


def test_eb9_cli_validate_batch_rows():
    lines = "".join(json.dumps({"id": i, "text": SRC, "data": d}) + "\n"
                    for i, d in enumerate([CASES["t999"], {}, CLEAN]))
    proc = _cli_sub(["validate", "--batch", "--heuristics-only", "--field-threshold", "0.3",
                     "--overall-threshold", "0"], stdin=lines)
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    rows = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert [r["notes"] for r in rows] == [_ho_t999_note(), NV_HO, HO_OK_NOTE]


def test_eb9_cli_extract_single_and_batch(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(_rows_by_id()["inv-008"]["text"])
    flags = ["--field-threshold", "0.25", "--overall-threshold", "0"]
    code, out = _cli_inproc("extract", ["--file", str(p), *flags])
    assert code == 0, out
    assert json.loads(out)["notes"] == INV008_NOTE
    line = json.dumps({"text": _rows_by_id()["inv-008"]["text"]}) + "\n"
    code, out = _cli_inproc("extract", ["--batch", *flags], stdin=line)
    assert code == 0, out
    assert json.loads(out.splitlines()[0])["notes"] == INV008_NOTE


def test_eb9_mcp_validate_and_extract_tools():
    assert _ho(CASES["t999"], 0.30)["notes"] == _ho_t999_note()
    tools = _tools()
    pipeline_mod = importlib.import_module("reflect_mcp.pipeline")
    with patch.dict(os.environ, {"REFLECT_FIELD_THRESHOLD": "0", "REFLECT_OVERALL_THRESHOLD": "0"}), \
            patch.object(pipeline_mod, "build_default_pipeline", side_effect=_inv008_factory):
        r = _dump(tools.extract_tool(_rows_by_id()["inv-008"]["text"], 0.25, 0.0))
    assert _status(r) == "ok"
    assert r["notes"] == INV008_NOTE


# ---------------------------------------------------------------- docs / ACs
def test_readme_documents_honest_ok_notes():
    text = (REPO / "README.md").read_text()
    assert "Heuristics penalized" in text
    assert "nothing was verified" in text


def test_product_md_row_and_ledger():
    text = (REPO / "PRODUCT.md").read_text()
    assert any(ln.startswith("- iter 76") for ln in text.splitlines())
    rows = [ln for ln in text.splitlines() if ln.startswith("|") and "65" in ln.split("|")[1]]
    assert rows and "DONE" in rows[0] and "76" in rows[0], rows


def test_gate_module_line_budget():
    n = len((REPO / "src" / "reflect_mcp" / "gate.py").read_text().splitlines())
    assert n <= 185, n
