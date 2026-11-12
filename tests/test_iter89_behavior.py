"""Black-box behavioral tests for iter-89: an explicit null ``total`` on a labelled source.

Covers Expected Behaviors of pm.md (#39, scoped to ``total``):

  1. CLI ``validate --heuristics-only`` with ``"total": null`` -> needs_review, exit 1,
     one flagged entry (total, null, 0.3) whose reason names the labelled value.
  2. Omitted ``total`` key -> stdout byte-identical to base, exit code identical.
  3. No total label -> same status / exit / field_confidence.total as base.
  4. Two labelled totals -> reason lists both in first-occurrence order.
  5. Null subtotal/tax are out of scope -> equal to base.
  6. extract path (stub Generator; stub Critic without a total verdict, and heuristics-only)
     and the MCP tool functions give the same verdict.
  7. field_provenance gains no total key; other fields' confidence/provenance equal base.
  8. Golden row inv-013 + replay; ``make eval`` PASS, silent-wrong 0, 8/8 caught; base src
     against the new dataset reports inv-013 total SILENT and exits non-zero; README block.
  9. A null inv-001 ``tax`` mutation stays SILENT and exits 1.
 10. All-null on a labelled-total source -> needs_review, total the only flagged field;
     all-null without a total label keeps base's ok.
 11. README states the rule.

Base (7237e5a) counterfactual: ``git archive 7237e5a src`` into a temp dir and *execute* that
tree's CLI with ``PYTHONPATH=<copy>/src`` (never read). Isolation: no implementation source,
engineer/reviewer notes or diff were read.

EB6 ambiguity: "``round(overall_confidence * 0.3, 4)``" is read as the stub Critic's
``overall_confidence`` (the only overall the Critic supplies for a field it omits).
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
DATASET = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"
README = REPO / "README.md"

SRC = (
    "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: $540.00\n"
    "Tax: $43.20\nTotal: 583.20\nPaid in USD\n"
)
SRC_TWO = SRC + "Total: 600.00\n"
SRC_NOLABEL = (
    "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: $540.00\n"
    "Tax: $43.20\nPaid in USD\n"
)
OTHERS = {
    "vendor_name": "ACME CORP", "invoice_number": "INV-1001", "invoice_date": "2026-03-14",
    "subtotal": 540.0, "tax": 43.2, "currency": "USD",
}
FULL = dict(OTHERS, total=583.2)
NULL_TOTAL = dict(OTHERS, total=None)
REASON_ONE = "source labels total as 583.20, extraction left it null"
REASON_TWO = "source labels total as 583.20, 600.00, extraction left it null"


# ---------------------------------------------------------------- helpers
# The last commit before this ship. A floating ``HEAD`` becomes the ship itself once it
# is committed (the iter-89 PRESHIP revert), so every counterfactual names this sha.
BASE_SHA = "7237e5a"


def _git_ok() -> bool:
    """Skip, never error, when the base commit is absent (shallow CI checkout, archive copy)."""
    r = subprocess.run(
        ["git", "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], cwd=REPO, capture_output=True, text=True
    )
    return r.returncode == 0


@pytest.fixture(scope="module")
def head_tree(tmp_path_factory):
    if not _git_ok():
        pytest.skip(f"base commit {BASE_SHA} not in the object store")
    dest = tmp_path_factory.mktemp("head")
    blob = subprocess.run(["git", "archive", BASE_SHA, "src"], cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)
    return dest


def _cli(args: list[str], text: str, tmp: Path, pythonpath: str | None = None) -> tuple[int, str]:
    src = tmp / "src.txt"
    src.write_text(text, encoding="utf-8")
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("REFLECT_"):
            env.pop(k)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args, "--file", str(src)],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(tmp),
    )
    return r.returncode, r.stdout


def _validate(data: dict, text: str, tmp: Path, pythonpath: str | None = None) -> tuple[int, str]:
    return _cli(["validate", "--heuristics-only", "--data", json.dumps(data)], text, tmp, pythonpath)


def _j(out: str) -> dict:
    d = json.loads(out)
    d.pop("trace_id", None)
    return d


def _flag(d: dict, field: str) -> list[dict]:
    return [f for f in d.get("flagged_fields", []) if f["field"] == field]


def _dump(r) -> dict:
    d = r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r)
    d = dict(d)
    d.pop("trace_id", None)
    return d


def _critic_raw_no_total(overall: float = 0.95) -> str:
    verdicts = [{"field": f, "confidence": 0.95, "grounded": True, "rationale": "x"} for f in OTHERS]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": overall, "summary": "ok"})


# ---------------------------------------------------------------- EB1
def test_eb1_explicit_null_total_flagged(tmp_path):
    code, out = _validate(NULL_TOTAL, SRC, tmp_path)
    d = _j(out)
    assert d["status"] == "needs_review", d
    assert code == 1
    assert len(d["flagged_fields"]) == 1, d["flagged_fields"]
    f = d["flagged_fields"][0]
    assert f["field"] == "total" and f["value"] is None and f["confidence"] == 0.3, f
    assert REASON_ONE in f["reason"], f
    assert d["field_confidence"]["total"] == 0.3
    assert d["data"]["total"] is None


def test_eb1_red_on_head(head_tree, tmp_path):
    code, out = _validate(NULL_TOTAL, SRC, tmp_path, str(head_tree / "src"))
    d = _j(out)
    assert d["status"] == "ok" and code == 0, d  # the silent wrong answer this iteration removes


# ---------------------------------------------------------------- EB2
def test_eb2_omitted_key_byte_identical_to_head(head_tree, tmp_path):
    code, out = _validate(OTHERS, SRC, tmp_path)
    hcode, hout = _validate(OTHERS, SRC, tmp_path, str(head_tree / "src"))
    assert (code, _j(out)) == (hcode, _j(hout))
    assert _j(out)["status"] == "ok" and code == 0
    assert not _flag(_j(out), "total")


# ---------------------------------------------------------------- EB3
def test_eb3_no_label_no_flag(head_tree, tmp_path):
    code, out = _validate(NULL_TOTAL, SRC_NOLABEL, tmp_path)
    hcode, hout = _validate(NULL_TOTAL, SRC_NOLABEL, tmp_path, str(head_tree / "src"))
    d, h = _j(out), _j(hout)
    assert not _flag(d, "total"), d
    assert (d["status"], code, d["field_confidence"].get("total")) == (
        h["status"], hcode, h["field_confidence"].get("total"))


# ---------------------------------------------------------------- EB4
def test_eb4_two_labels_listed_in_order(tmp_path):
    code, out = _validate(NULL_TOTAL, SRC_TWO, tmp_path)
    d = _j(out)
    fl = _flag(d, "total")
    assert code == 1 and len(fl) == 1, d
    assert REASON_TWO in fl[0]["reason"], fl


# ---------------------------------------------------------------- EB5
@pytest.mark.parametrize("field", ["subtotal", "tax"])
def test_eb5_subtotal_tax_null_out_of_scope(field, head_tree, tmp_path):
    data = dict(FULL, **{field: None})
    code, out = _validate(data, SRC, tmp_path)
    hcode, hout = _validate(data, SRC, tmp_path, str(head_tree / "src"))
    d, h = _j(out), _j(hout)
    assert not _flag(d, field), d
    assert (d["status"], code, d["flagged_fields"]) == (h["status"], hcode, h["flagged_fields"])


# ---------------------------------------------------------------- EB6
def _gen_raw() -> str:
    return json.dumps(NULL_TOTAL)


def _pipe(critic: bool) -> ReflectivePipeline:
    crit = Critic(ScriptedLLMClient([_critic_raw_no_total()])) if critic else None
    kw = {} if critic else {"critic_enabled": False}
    try:
        return ReflectivePipeline(Generator(ScriptedLLMClient([_gen_raw()])), crit, **kw)
    except TypeError:
        return ReflectivePipeline(Generator(ScriptedLLMClient([_gen_raw()])), crit)


@pytest.mark.parametrize("critic", [True, False], ids=["stub-critic", "heuristics-only"])
def test_eb6_extract_path_flags_null_total(critic):
    d = _dump(_pipe(critic).extract(SRC))
    status = getattr(d["status"], "value", d["status"])
    assert status == "needs_review", d
    fl = _flag(d, "total")
    assert len(fl) == 1 and fl[0]["value"] is None, d
    assert REASON_ONE in fl[0]["reason"]
    assert d["data"]["total"] is None
    if critic:
        assert d["field_confidence"]["total"] == round(0.95 * 0.3, 4), d["field_confidence"]
    else:
        assert d["field_confidence"]["total"] == 0.3


def _tools():
    return importlib.import_module("reflect_mcp.tools")


def test_eb6_mcp_validate_tool_same_verdict(tmp_path):
    _, out = _validate(NULL_TOTAL, SRC, tmp_path)
    cli = _j(out)
    tools = _tools()
    params = inspect.signature(tools.validate_tool).parameters
    kw = {"heuristics_only": True} if "heuristics_only" in params else {}
    r = _dump(tools.validate_tool(SRC, dict(NULL_TOTAL), **kw))
    assert getattr(r["status"], "value", r["status"]) == "needs_review", r
    assert r["flagged_fields"] == cli["flagged_fields"]


def test_eb6_mcp_extract_tool_same_verdict():
    tools = _tools()
    pipeline_mod = importlib.import_module("reflect_mcp.pipeline")

    def factory(*a, **k):
        return ReflectivePipeline(Generator(ScriptedLLMClient([_gen_raw()])),
                                  Critic(ScriptedLLMClient([_critic_raw_no_total()])),
                                  **({"gate_config": k["gate_config"]} if "gate_config" in k else {}))

    with patch.object(pipeline_mod, "build_default_pipeline", side_effect=factory):
        r = _dump(tools.extract_tool(SRC))
    direct = _dump(_pipe(True).extract(SRC))
    assert getattr(r["status"], "value", r["status"]) == "needs_review", r
    assert r["flagged_fields"] == direct["flagged_fields"]


# ---------------------------------------------------------------- EB7
def test_eb7_no_other_envelope_changes(head_tree, tmp_path):
    _, out = _validate(NULL_TOTAL, SRC, tmp_path)
    _, hout = _validate(NULL_TOTAL, SRC, tmp_path, str(head_tree / "src"))
    d, h = _j(out), _j(hout)
    assert "total" not in (d.get("field_provenance") or {}), d.get("field_provenance")
    assert d.get("field_provenance") == h.get("field_provenance")
    for f in OTHERS:
        assert d["field_confidence"][f] == h["field_confidence"][f], f


# ---------------------------------------------------------------- EB8
def _rows() -> list[dict]:
    return [json.loads(ln) for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_eb8_golden_row_and_replay():
    rows = {r["id"]: r for r in _rows()}
    assert "inv-013" in rows
    assert "Total" in rows["inv-013"]["text"] or "total" in rows["inv-013"]["text"].lower()
    entry = json.loads(REPLAY.read_text(encoding="utf-8"))["inv-013"]
    gen = json.loads(entry["generator_raw"])
    assert "total" in gen and gen["total"] is None
    exp = rows["inv-013"]["expected"]
    assert exp.get("total") is not None
    for k, v in exp.items():
        if k != "total":
            assert gen.get(k) == v, (k, gen.get(k), v)


@pytest.fixture(scope="module")
def make_eval():
    r = subprocess.run(["make", "-s", "eval"], cwd=REPO, capture_output=True, text=True, timeout=300)
    return r


def test_eb8_make_eval_pass(make_eval):
    out = make_eval.stdout
    assert make_eval.returncode == 0, out + make_eval.stderr
    assert "PASS" in out
    assert "SILENT" not in out
    assert "8/8" in out, out


def test_eb8_head_src_on_new_dataset_is_silent(head_tree, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(head_tree / "src", work / "src")
    shutil.copytree(EVAL_DIR, work / "eval", ignore=shutil.ignore_patterns("__pycache__"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env["PYTHONPATH"] = str(work / "src")
    r = subprocess.run([sys.executable, str(work / "eval" / "run_eval.py")], cwd=str(work),
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0, r.stdout
    assert any("inv-013" in ln and "total" in ln and "SILENT" in ln for ln in r.stdout.splitlines()), r.stdout


def test_eb8_readme_eval_block_matches(make_eval):
    text = README.read_text(encoding="utf-8")
    body = [ln for ln in make_eval.stdout.splitlines() if ln.strip()]
    assert body
    joined = "\n".join(body)
    assert joined in text, "README eval block differs from make eval output"


# ---------------------------------------------------------------- EB9
def test_eb9_null_tax_mutation_stays_silent(tmp_path):
    work = tmp_path / "w"
    shutil.copytree(REPO / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(EVAL_DIR, work / "eval", ignore=shutil.ignore_patterns("__pycache__"))
    rp = work / "eval" / "fixtures" / "replay.json"
    data = json.loads(rp.read_text(encoding="utf-8"))
    gen = json.loads(data["inv-001"]["generator_raw"])
    gen["tax"] = None
    data["inv-001"]["generator_raw"] = json.dumps(gen)
    rp.write_text(json.dumps(data), encoding="utf-8")
    exp_tax = {r["id"]: r for r in _rows()}["inv-001"]["expected"]["tax"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env["PYTHONPATH"] = str(work / "src")
    r = subprocess.run([sys.executable, str(work / "eval" / "run_eval.py")], cwd=str(work),
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1, r.stdout
    assert f"[inv-001] tax: expected {exp_tax} but got None" in r.stdout, r.stdout
    line = [ln for ln in r.stdout.splitlines() if "[inv-001] tax:" in ln]
    assert line and "(SILENT)" in line[0]


# ---------------------------------------------------------------- EB10
ALLNULL = {k: None for k in FULL}


def test_eb10_all_null_on_labelled_total(tmp_path):
    code, out = _validate(ALLNULL, SRC, tmp_path)
    d = _j(out)
    assert d["status"] == "needs_review" and code == 1, d
    assert [f["field"] for f in d["flagged_fields"]] == ["total"], d["flagged_fields"]


def test_eb10_all_null_no_label_keeps_head(head_tree, tmp_path):
    code, out = _validate(ALLNULL, SRC_NOLABEL, tmp_path)
    hcode, hout = _validate(ALLNULL, SRC_NOLABEL, tmp_path, str(head_tree / "src"))
    d, h = _j(out), _j(hout)
    assert d["status"] == "ok" and code == 0, d
    assert (code, d) == (hcode, h)
    assert "nothing was verified" in d["notes"]


# ---------------------------------------------------------------- EB11
def test_eb11_readme_states_rule():
    text = README.read_text(encoding="utf-8")
    low = text.lower()
    i = low.find("null")
    windows = []
    start = 0
    while True:
        i = low.find("omitted", start)
        if i < 0:
            break
        windows.append(low[max(0, i - 600): i + 600])
        start = i + 1
    hits = [w for w in windows if "total" in w and "null" in w and "label" in w]
    assert hits, "README does not state the explicit-null vs omitted total rule"
    assert any("only" in w for w in hits), "README does not name total as the only field covered"
