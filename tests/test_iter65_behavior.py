"""Black-box behavioral tests for iter-65: a heuristics-only ``ok`` says no Critic ran.

An ``ok`` result that no Critic produced carries ``HEURISTICS_ONLY_OK_NOTE``
instead of "All fields cleared the confidence thresholds.", and the Critic-off
path logs ``critic.skipped`` (``reason="heuristics_only"``) instead of a
fabricated ``critique.done``. Covers Expected Behaviors 1-12 of pm.md.

Isolation contract: tests drive the public package API (ReflectivePipeline,
gate.decide, the MCP tool functions, the CLI and the eval script) plus the
golden data files and the README, asserting on observable output only. No
implementation source was read. The HEAD constants below (minus-notes payload
digests, HEAD needs_review notes, HEAD CLI exit code) were captured by running
the HEAD tree (``git archive HEAD``) before commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import gate, tools
from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig, decide
from reflect_mcp.generator import Generator
from reflect_mcp.heuristics import run_heuristics
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import CritiqueResult, InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"
README = REPO / "README.md"

sys.path.insert(0, str(REPO / "eval"))
import run_eval  # noqa: E402

NOTE_OK = "All fields cleared the confidence thresholds."
NOTE_HO = (
    "No Critic ran: every field passed the deterministic heuristic checks only; "
    "field confidences are heuristic factors, not a Critic's judgement."
)
NR_NOTE = "1 field(s) below the 70% confidence threshold require human review before use."

OK_IDS = ("inv-001", "inv-002", "inv-003", "inv-005", "inv-007")
NR_IDS = ("inv-004", "inv-006", "inv-008")

# sha256 of json.dumps({row_id: model_dump(mode="json") minus trace_id AND notes},
# sort_keys=True) over the 8 golden rows, captured from the HEAD tree. Identical
# for validate and extract. Everything except `notes` must stay byte-identical.
HEAD_MINUS_NOTES_SHA256 = {
    "off": "3708cc11d664ecaf235364964dbad64c7927e6d8b0d4bbd4f5c56958e583f431",
    "on": "a2a84ffd154a9aec29086c24f41a5a5724157957b92711e8a9ac023cc22ae1df",
}
# HEAD notes per golden row with the Critic ON (must be unchanged).
HEAD_ON_NOTES = {
    "inv-001": NOTE_OK, "inv-002": NOTE_OK, "inv-003": NOTE_OK, "inv-005": NOTE_OK,
    "inv-004": NR_NOTE, "inv-006": NR_NOTE, "inv-007": NR_NOTE, "inv-008": NR_NOTE,
}

FIELDS = ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency")

ALL_UNGROUNDED = {
    "vendor_name": "GLOBEX INC",
    "invoice_number": "ZZ-0000",
    "invoice_date": None,
    "subtotal": 999.0,
    "tax": 99.0,
    "total": 1098.0,
    "currency": "EUR",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def events(self) -> list[str]:
        return [r.getMessage() for r in self.records]

    def named(self, name: str) -> list[dict]:
        return [dict(getattr(r, "fields", {}) or {}) for r in self.records if r.getMessage() == name]


@pytest.fixture
def logcap():
    lg = logging.getLogger("reflect_mcp")
    h = _Capture()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def _gen(raws=()) -> Generator:
    return Generator(ScriptedLLMClient(list(raws)))


def _crit(raws=()) -> Critic:
    return Critic(ScriptedLLMClient(list(raws)))


def _status(result) -> str:
    s = result.status if not isinstance(result, dict) else result["status"]
    return getattr(s, "value", s)


def _clean_data() -> InvoiceData:
    return InvoiceData(**CORRECT_EXTRACTION)


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _golden_rows() -> list[dict]:
    rep = _replay()
    rows = []
    for row in run_eval.load_golden():
        fx = rep[row["id"]]
        rows.append({"id": row["id"], "text": row["text"], "fx": fx,
                     "data": json.loads(fx["generator_raw"])})
    assert len(rows) == 8
    return rows


def _golden_results(critic_on: bool, mode: str) -> dict:
    out = {}
    for row in _golden_rows():
        fx = row["fx"]
        crit = _crit([fx["critic_raw"]]) if critic_on else None
        pipe = ReflectivePipeline(_gen([fx["generator_raw"]]), crit)
        if mode == "validate":
            out[row["id"]] = pipe.validate(row["text"], InvoiceData(**row["data"]))
        else:
            out[row["id"]] = pipe.extract(row["text"])
    return out


def _minus_notes_digest(results: dict) -> str:
    dumps = {}
    for rid, r in results.items():
        d = r.model_dump(mode="json")
        d.pop("trace_id", None)
        d.pop("notes", None)
        dumps[rid] = d
    return hashlib.sha256(json.dumps(dumps, sort_keys=True).encode()).hexdigest()


def _critique(per_field=0.95, overall=0.95) -> CritiqueResult:
    verdicts = [{"field": f, "confidence": per_field, "grounded": per_field >= 0.7,
                 "rationale": f"{f} check"} for f in FIELDS]
    return CritiqueResult.model_validate({"field_verdicts": verdicts, "overall_confidence": overall})


def _dump(result) -> dict:
    d = result.model_dump(mode="json")
    return d


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------

def test_constant_exact_text():
    assert gate.HEURISTICS_ONLY_OK_NOTE == NOTE_HO
    assert NOTE_HO.startswith("No Critic ran")


# ---------------------------------------------------------------------------
# EB-1: Critic off -> ok with NOTE_HO (validate and extract)
# ---------------------------------------------------------------------------

def test_eb1_validate_critic_off_ok_note():
    r = ReflectivePipeline(_gen(), None).validate(SOURCE, _clean_data())
    assert _status(r) == "ok"
    assert r.notes == NOTE_HO
    assert "All fields cleared" not in r.notes


def test_eb1_validate_without_generator_ok_note():
    r = ReflectivePipeline(None, None).validate(SOURCE, _clean_data())
    assert _status(r) == "ok"
    assert r.notes == NOTE_HO


def test_eb1_extract_critic_off_ok_note():
    r = ReflectivePipeline(_gen([json.dumps(CORRECT_EXTRACTION)]), None).extract(SOURCE)
    assert _status(r) == "ok"
    assert r.notes == NOTE_HO
    assert "All fields cleared" not in r.notes


# ---------------------------------------------------------------------------
# EB-2: Critic present, all-high -> NOTE_OK unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb2_critic_ok_note_unchanged(mode):
    pipe = ReflectivePipeline(_gen([json.dumps(CORRECT_EXTRACTION)]),
                              _crit([critique_all(0.95, 0.9)]))
    r = pipe.validate(SOURCE, _clean_data()) if mode == "validate" else pipe.extract(SOURCE)
    assert _status(r) == "ok"
    assert r.notes == NOTE_OK
    assert "No Critic ran" not in r.notes


# ---------------------------------------------------------------------------
# EB-3: gate.decide keyword-only critic_ran
# ---------------------------------------------------------------------------

def test_eb3_decide_default_returns_note_ok():
    r = decide(_clean_data(), _critique(), run_heuristics(SOURCE, _clean_data()), "t1")
    assert _status(r) == "ok"
    assert r.notes == NOTE_OK


def test_eb3_decide_critic_ran_true_equals_default():
    h = run_heuristics(SOURCE, _clean_data())
    a = decide(_clean_data(), _critique(), h, "t1")
    b = decide(_clean_data(), _critique(), h, "t1", critic_ran=True)
    assert _dump(a) == _dump(b)


@pytest.mark.parametrize("cfg", [None, GateConfig(), GateConfig(field_threshold=0.5, overall_threshold=0.5)])
def test_eb3_decide_critic_ran_false_only_notes_differ(cfg):
    h = run_heuristics(SOURCE, _clean_data())
    a = _dump(decide(_clean_data(), _critique(), h, "t1", cfg))
    b = _dump(decide(_clean_data(), _critique(), h, "t1", cfg, critic_ran=False))
    assert a["notes"] == NOTE_OK
    assert b["notes"] == NOTE_HO
    a.pop("notes"), b.pop("notes")
    assert a == b


def test_eb3_decide_critic_ran_false_needs_review_keeps_notes():
    h = run_heuristics(SOURCE, _clean_data())
    low = _critique(per_field=0.4, overall=0.4)
    a = _dump(decide(_clean_data(), low, h, "t1"))
    b = _dump(decide(_clean_data(), low, h, "t1", critic_ran=False))
    assert a["status"] == "needs_review"
    assert a == b


def test_eb3_critic_ran_is_keyword_only():
    h = run_heuristics(SOURCE, _clean_data())
    with pytest.raises(TypeError):
        decide(_clean_data(), _critique(), h, "t1", GateConfig(), False)


# ---------------------------------------------------------------------------
# EB-4: Critic-off needs_review notes and all other fields unchanged vs HEAD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb4_critic_off_needs_review_notes_unchanged(mode):
    res = _golden_results(critic_on=False, mode=mode)
    for rid in NR_IDS:
        assert _status(res[rid]) == "needs_review"
        assert res[rid].notes == NR_NOTE, rid


@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb4_critic_off_everything_but_notes_matches_head(mode):
    res = _golden_results(critic_on=False, mode=mode)
    assert _minus_notes_digest(res) == HEAD_MINUS_NOTES_SHA256["off"]
    for r in res.values():
        assert r.critic_summary == ""


def test_eb4_fabricated_needs_review_notes_untouched():
    text = "ACME Corp invoice INV-1 total 10.00 USD"
    data = InvoiceData(invoice_number="ZZZ-999", vendor_name="Initech", total=99999,
                       currency="EUR", invoice_date="2031-01-01")
    r = ReflectivePipeline(_gen(), None).validate(text, data)
    assert _status(r) == "needs_review"
    assert "No Critic ran" not in r.notes
    assert re.fullmatch(
        r"\d+ field\(s\) below the 70% confidence threshold require human review before use\.",
        r.notes,
    ), r.notes
    assert r.critic_summary == ""


# ---------------------------------------------------------------------------
# EB-5: log stream, Critic off
# ---------------------------------------------------------------------------

def test_eb5_validate_log_stream(logcap):
    r = ReflectivePipeline(_gen(), None).validate(SOURCE, _clean_data())
    assert logcap.events() == ["validate.start", "heuristics.done", "critic.skipped", "decide.done"]
    skipped = logcap.named("critic.skipped")
    assert len(skipped) == 1
    assert skipped[0].get("reason") == "heuristics_only"
    assert skipped[0].get("trace_id") == r.trace_id
    assert logcap.named("critique.done") == []


def test_eb5_extract_log_stream(logcap):
    r = ReflectivePipeline(_gen([json.dumps(CORRECT_EXTRACTION)]), None).extract(SOURCE)
    ev = logcap.events()
    # Spec reading: extract emits generate.start/done in place of validate.start.
    assert ev[:2] == ["generate.start", "generate.done"]
    assert ev[-3:] == ["heuristics.done", "critic.skipped", "decide.done"]
    assert "critique.done" not in ev
    skipped = logcap.named("critic.skipped")
    assert len(skipped) == 1
    assert skipped[0].get("reason") == "heuristics_only"
    assert skipped[0].get("trace_id") == r.trace_id


def test_eb5_needs_review_also_logs_critic_skipped(logcap):
    rows = {r["id"]: r for r in _golden_rows()}
    row = rows["inv-004"]
    ReflectivePipeline(_gen(), None).validate(row["text"], InvoiceData(**row["data"]))
    assert logcap.events().count("critique.done") == 0
    assert [f.get("reason") for f in logcap.named("critic.skipped")] == ["heuristics_only"]


def test_eb5_golden_critic_off_zero_critique_done(logcap):
    _golden_results(critic_on=False, mode="validate")
    ev = logcap.events()
    assert ev.count("critique.done") == 0
    assert ev.count("critic.skipped") == 8
    assert all(f.get("reason") == "heuristics_only" for f in logcap.named("critic.skipped"))


# ---------------------------------------------------------------------------
# EB-6: doomed short-circuit and normal Critic path
# ---------------------------------------------------------------------------

def test_eb6_doomed_short_circuit_reason(logcap):
    critic_client = ScriptedLLMClient([critique_all(0.95, 0.9)])
    pipe = ReflectivePipeline(_gen([json.dumps(ALL_UNGROUNDED)]), Critic(critic_client))
    r = pipe.extract(SOURCE)
    assert len(critic_client.calls) == 0
    assert _status(r) == "needs_review"
    skipped = logcap.named("critic.skipped")
    assert len(skipped) == 1
    assert skipped[0].get("reason") == "all_fields_doomed"
    assert skipped[0].get("trace_id") == r.trace_id
    assert logcap.events().count("critique.done") == 0


def test_eb6_normal_critic_path(logcap):
    critic_client = ScriptedLLMClient([critique_all(0.95, 0.9)])
    pipe = ReflectivePipeline(_gen(), Critic(critic_client))
    pipe.validate(SOURCE, _clean_data())
    assert len(critic_client.calls) == 1
    ev = logcap.events()
    assert ev == ["validate.start", "heuristics.done", "critique.done", "decide.done"]
    assert ev.count("critic.skipped") == 0


# ---------------------------------------------------------------------------
# EB-7 / EB-8: golden coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb7_golden_critic_off_notes(mode):
    res = _golden_results(critic_on=False, mode=mode)
    for rid in OK_IDS:
        assert _status(res[rid]) == "ok", rid
        assert res[rid].notes == NOTE_HO, rid
    for rid in NR_IDS:
        assert _status(res[rid]) == "needs_review", rid
        assert res[rid].notes == NR_NOTE, rid


def test_eb7_inv007_wrong_vendor_is_no_longer_called_cleared():
    res = _golden_results(critic_on=False, mode="validate")
    r = res["inv-007"]
    assert r.data.vendor_name == "Contoso Ltd"  # the known-wrong value
    assert _status(r) == "ok"
    assert "All fields cleared" not in r.notes
    assert r.notes.startswith("No Critic ran")


@pytest.mark.parametrize("mode", ["validate", "extract"])
def test_eb7_golden_critic_on_unchanged(mode):
    res = _golden_results(critic_on=True, mode=mode)
    assert {rid: r.notes for rid, r in res.items()} == HEAD_ON_NOTES
    assert _minus_notes_digest(res) == HEAD_MINUS_NOTES_SHA256["on"]


def test_eb8_iter62_digest_pins():
    import tests.test_iter62_behavior as t62
    assert t62.HEAD_PAYLOAD_SHA256["on"] == (
        "251385698ef79f3e6ef8dc68b41c02cfeac1e55c603a00935c2e1451aa7cdccc")
    assert t62.HEAD_PAYLOAD_SHA256["off"] == (
        "b47dfcfc313d6c6b4319f1ba3e2d98c126d0da2c2cf12b416640f9062f6f27c5")
    for on in (True, False):
        digest, _ = t62._payload_digest(on)
        assert digest == t62.HEAD_PAYLOAD_SHA256["on" if on else "off"]


# ---------------------------------------------------------------------------
# EB-9: CLI batch, offline, no SDK
# ---------------------------------------------------------------------------

def _cli_env() -> dict:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}


@pytest.fixture(scope="module")
def cli_batch(tmp_path_factory):
    rows = _golden_rows()
    path = tmp_path_factory.mktemp("b65") / "golden.jsonl"
    path.write_text("\n".join(json.dumps({"text": r["text"], "data": r["data"]}) for r in rows) + "\n")
    cmd = [sys.executable, "-c",
           "import sys, runpy; sys.modules['anthropic'] = None; "
           "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
           "runpy.run_module('reflect_mcp', run_name='__main__')",
           "validate", "--batch", "--heuristics-only", "--file", str(path)]
    return subprocess.run(cmd, input="", capture_output=True, text=True, env=_cli_env(),
                          cwd=str(REPO), timeout=120)


def _stderr_records(stderr: str) -> list[dict]:
    out = []
    for ln in stderr.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def test_eb9_cli_exit_code_same_as_head(cli_batch):
    assert cli_batch.returncode == 1, cli_batch.stderr[-2000:]


def test_eb9_cli_stdout_notes(cli_batch):
    objs = [json.loads(ln) for ln in cli_batch.stdout.splitlines() if ln.strip()]
    assert len(objs) == 8
    ok = [o for o in objs if o["status"] == "ok"]
    assert len(ok) == 5
    assert all(o["notes"] == NOTE_HO for o in ok)
    nr = [o for o in objs if o["status"] == "needs_review"]
    assert len(nr) == 3
    assert all(o["notes"] == NR_NOTE for o in nr)


def test_eb9_cli_stderr_events(cli_batch):
    recs = _stderr_records(cli_batch.stderr)
    skipped = [r for r in recs if r.get("event") == "critic.skipped"]
    assert len(skipped) == 8
    assert all(r.get("reason") == "heuristics_only" for r in skipped)
    assert [r for r in recs if r.get("event") == "critique.done"] == []
    assert cli_batch.stderr.count('"reason": "heuristics_only"') == 8


# ---------------------------------------------------------------------------
# EB-10: MCP tools
# ---------------------------------------------------------------------------

FIELD_ENV = "REFLECT_FIELD_THRESHOLD"
OVERALL_ENV = "REFLECT_OVERALL_THRESHOLD"


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv(FIELD_ENV, raising=False)
    monkeypatch.delenv(OVERALL_ENV, raising=False)


def _install_scripted(monkeypatch, gen_raw: str, critic_raw: str | None = None):
    def _factory(gate_config=None, **kwargs):
        gen = Generator(ScriptedLLMClient([gen_raw]))
        critic = None
        if kwargs.get("critic_enabled", True):
            critic = Critic(ScriptedLLMClient([critic_raw or critique_all(0.95, 0.95)]))
        return ReflectivePipeline(gen, critic, gate_config=gate_config)

    monkeypatch.setattr(pipeline_module, "build_default_pipeline", _factory)


def test_eb10_validate_tool_golden_clean_row(_clean_env):
    row = {r["id"]: r for r in _golden_rows()}["inv-001"]
    res = tools.validate_tool(row["text"], dict(row["data"]), heuristics_only=True)
    assert res["status"] == "ok", res
    assert res["notes"] == NOTE_HO


def test_eb10_validate_tool_with_raise_suffix(_clean_env):
    row = {r["id"]: r for r in _golden_rows()}["inv-001"]
    res = tools.validate_tool(row["text"], dict(row["data"]), 0.0, 0.0, heuristics_only=True)
    assert res["status"] == "ok", res
    assert res["notes"] == (
        NOTE_HO + " field_threshold raised from 0.0 to 0.7. "
        "overall_threshold raised from 0.0 to 0.75."
    )


def test_eb10_validate_tool_env_floor_suffix(monkeypatch, _clean_env):
    monkeypatch.setenv(FIELD_ENV, "0.9")
    monkeypatch.setenv(OVERALL_ENV, "0.85")
    res = tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), heuristics_only=True)
    assert res["status"] == "ok", res
    assert res["notes"] == (
        NOTE_HO + " field_threshold raised from 0.7 to 0.9. "
        "overall_threshold raised from 0.75 to 0.85."
    )


def test_eb10_extract_tool_scripted(monkeypatch, _clean_env):
    _install_scripted(monkeypatch, json.dumps(CORRECT_EXTRACTION))
    res = tools.extract_tool(SOURCE, heuristics_only=True)
    assert res["status"] == "ok", res
    assert res["notes"] == NOTE_HO
    res2 = tools.extract_tool(SOURCE, 0.5, 0.9, heuristics_only=True)
    assert res2["notes"] == NOTE_HO + " field_threshold raised from 0.5 to 0.7."


def test_eb10_tool_critic_on_keeps_note_ok(monkeypatch, _clean_env):
    _install_scripted(monkeypatch, json.dumps(CORRECT_EXTRACTION))
    res = tools.extract_tool(SOURCE)
    assert res["status"] == "ok", res
    assert res["notes"] == NOTE_OK


# ---------------------------------------------------------------------------
# EB-11: make eval unchanged
# ---------------------------------------------------------------------------

def test_eb11_eval_stdout_unchanged():
    import tests.test_iter62_behavior as t62
    proc = subprocess.run([sys.executable, str(REPO / "eval" / "run_eval.py")],
                          capture_output=True, text=True, cwd=str(REPO), timeout=120,
                          env={**_cli_env()})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == t62.HEAD_EVAL_STDOUT


# ---------------------------------------------------------------------------
# EB-12: README §5 paragraph
# ---------------------------------------------------------------------------

def _heuristics_only_paragraph() -> str:
    text = README.read_text()
    paras = [p for p in text.split("\n\n") if p.lstrip().startswith("**Heuristics-only mode**")]
    assert len(paras) == 1, "expected exactly one 'Heuristics-only mode' paragraph"
    return paras[0]


def test_eb12_readme_paragraph_mentions_no_critic_and_skipped():
    p = _heuristics_only_paragraph()
    assert NOTE_HO.split(":")[0] == "No Critic ran"
    assert "No Critic ran" in p
    assert "critic.skipped" in p
    assert "heuristics_only" in p
    assert "critique.done" in p
    assert "critic_summary" in p
