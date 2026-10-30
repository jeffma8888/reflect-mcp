"""Black-box behavioral tests for iter-61: operator-pinned gate floor on MCP tools.

Covers the Expected Behaviors from pm.md:
  1. Default floor (env unset) 0.70 / 0.75; args are clamped up, never down.
  2. Env floor from REFLECT_FIELD_THRESHOLD / REFLECT_OVERALL_THRESHOLD,
     each var independent; env read at call time.
  3. Operator opt-out (env 0) restores the caller's low thresholds.
  4. ``notes`` records each raise (field first, then overall); otherwise
     notes are exactly what the pipeline returns for the effective thresholds.
  5. Invalid requested args are still error envelopes (not clamped).
  6. Malformed env fails closed with an ``invalid thresholds:`` envelope.
  7. Empty-text check wins over a malformed env.
  8. Golden adversarial-caller pass at (0.0, 0.0), env unset: caught 4 / silent 0.
  9. Golden opt-out counterfactual (env 0): caught 0 / silent 4.
 10. Docs: server.py docstrings (AST only) + README §4; table rows unchanged;
     ``make eval`` stdout unchanged.

Isolation contract: tests drive ``reflect_mcp.tools`` through its public
functions with a monkeypatched ``reflect_mcp.pipeline.build_default_pipeline``
(pipelines over ``ScriptedLLMClient``), env set via monkeypatch. server.py is
only parsed with ``ast`` to read its tool docstrings; it is never imported.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_module
from reflect_mcp import tools
from reflect_mcp.critic import Critic
from reflect_mcp.gate import GateConfig
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient
from tests.conftest import CORRECT_EXTRACTION, SOURCE, critique_all

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "src" / "reflect_mcp" / "server.py"
README = REPO / "README.md"
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"

sys.path.insert(0, str(REPO / "eval"))
import run_eval  # noqa: E402

FIELD_ENV = "REFLECT_FIELD_THRESHOLD"
OVERALL_ENV = "REFLECT_OVERALL_THRESHOLD"

# README §4 tool table rows as they were at HEAD before iter 61 (must stay byte-identical).
HEAD_TABLE_ROWS = (
    "| Tool | Args | Returns |",
    "| --- | --- | --- |",
    "| `extract` | `text: str`, `field_threshold=0.70`, `overall_threshold=0.75`, "
    "`heuristics_only=False` | `{ status, data, field_confidence, overall_confidence, "
    "flagged_fields, field_provenance, critic_summary, trace_id, notes }` |",
    "| `validate` | `text: str`, `data: dict`, `field_threshold=0.70`, "
    "`overall_threshold=0.75`, `heuristics_only=False` | Same `ExtractionResult` "
    "envelope as `extract` |",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Never let the developer's shell env leak into these tests."""
    monkeypatch.delenv(FIELD_ENV, raising=False)
    monkeypatch.delenv(OVERALL_ENV, raising=False)


class Recorder:
    """Recording factory: stores every gate_config and builds a scripted pipeline."""

    def __init__(self, gen_raw: str | None = None, critic_raw: str | None = None):
        self.gen_raw = gen_raw if gen_raw is not None else json.dumps(CORRECT_EXTRACTION)
        self.critic_raw = critic_raw if critic_raw is not None else critique_all(0.95, 0.95)
        self.calls: list[tuple] = []

    def __call__(self, gate_config=None, **kwargs):
        self.calls.append((gate_config, kwargs))
        gen = Generator(ScriptedLLMClient([self.gen_raw]))
        critic = None
        if kwargs.get("critic_enabled", True):
            critic = Critic(ScriptedLLMClient([self.critic_raw]))
        return ReflectivePipeline(gen, critic, gate_config=gate_config)


def _install(monkeypatch, rec=None):
    rec = rec or Recorder()
    monkeypatch.setattr(pipeline_module, "build_default_pipeline", rec)
    return rec


def _built(rec):
    gc = rec.calls[-1][0]
    return gc.field_threshold, gc.overall_threshold


def _call(tool, *args, **kw):
    if tool == "extract":
        return tools.extract_tool(SOURCE, *args, **kw)
    return tools.validate_tool(SOURCE, dict(CORRECT_EXTRACTION), *args, **kw)


def _assert_envelope(result):
    assert isinstance(result, dict)
    assert set(result) == {"status", "error_message"}, result
    assert result["status"] == "error"
    assert isinstance(result["error_message"], str) and result["error_message"]


def _direct_notes(tool, field, overall, gen_raw=None, critic_raw=None):
    """Notes the pipeline itself returns for the given effective thresholds."""
    rec = Recorder(gen_raw, critic_raw)
    gen = Generator(ScriptedLLMClient([rec.gen_raw]))
    crit = Critic(ScriptedLLMClient([rec.critic_raw]))
    p = ReflectivePipeline(gen, crit, gate_config=GateConfig(
        field_threshold=field, overall_threshold=overall))
    if tool == "extract":
        return p.extract(SOURCE).notes
    return p.validate(SOURCE, InvoiceData(**CORRECT_EXTRACTION)).notes


TOOLS = ["extract", "validate"]


# ---------------------------------------------------------------------------
# EB-1: default floor (env unset)
# ---------------------------------------------------------------------------

class TestEB1DefaultFloor:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("args,expected", [
        ((0.0, 0.0), (0.70, 0.75)),
        ((0.9, 0.8), (0.9, 0.8)),
        ((), (0.70, 0.75)),
        ((0.5, 0.9), (0.70, 0.9)),
        ((0.95, 0.1), (0.95, 0.75)),
        ((0.70, 0.75), (0.70, 0.75)),
    ])
    def test_clamped_to_default_floor(self, monkeypatch, tool, args, expected):
        rec = _install(monkeypatch)
        result = _call(tool, *args)
        assert result["status"] != "error", result
        assert _built(rec) == pytest.approx(expected)

    @pytest.mark.parametrize("tool", TOOLS)
    def test_whitespace_env_counts_as_unset(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "   ")
        monkeypatch.setenv(OVERALL_ENV, "")
        rec = _install(monkeypatch)
        result = _call(tool, 0.0, 0.0)
        assert result["status"] != "error", result
        assert _built(rec) == pytest.approx((0.70, 0.75))


# ---------------------------------------------------------------------------
# EB-2: env floor
# ---------------------------------------------------------------------------

class TestEB2EnvFloor:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_env_floor_applies_to_default_args(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        _call(tool)
        assert _built(rec) == pytest.approx((0.9, 0.85))

    @pytest.mark.parametrize("tool", TOOLS)
    def test_stricter_args_win_over_env(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        _call(tool, 0.95, 0.99)
        assert _built(rec) == pytest.approx((0.95, 0.99))

    @pytest.mark.parametrize("tool", TOOLS)
    def test_field_env_only(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        rec = _install(monkeypatch)
        _call(tool)
        assert _built(rec) == pytest.approx((0.9, 0.75))

    @pytest.mark.parametrize("tool", TOOLS)
    def test_overall_env_only(self, monkeypatch, tool):
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        rec = _install(monkeypatch)
        _call(tool, 0.0, 0.0)
        assert _built(rec) == pytest.approx((0.70, 0.85))

    def test_env_read_at_call_time(self, monkeypatch):
        rec = _install(monkeypatch)
        tools.extract_tool(SOURCE, 0.0, 0.0)
        assert _built(rec) == pytest.approx((0.70, 0.75))
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        tools.extract_tool(SOURCE, 0.0, 0.0)
        assert _built(rec) == pytest.approx((0.9, 0.85))
        monkeypatch.setenv(FIELD_ENV, "0")
        monkeypatch.setenv(OVERALL_ENV, "0")
        tools.extract_tool(SOURCE, 0.0, 0.0)
        assert _built(rec) == pytest.approx((0.0, 0.0))


# ---------------------------------------------------------------------------
# EB-3: operator opt-out
# ---------------------------------------------------------------------------

class TestEB3OptOut:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_env_zero_restores_low_args(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0")
        monkeypatch.setenv(OVERALL_ENV, "0")
        rec = _install(monkeypatch)
        result = _call(tool, 0.0, 0.0)
        assert result["status"] != "error", result
        assert _built(rec) == pytest.approx((0.0, 0.0))

    @pytest.mark.parametrize("tool", TOOLS)
    def test_env_half_arg_point_six(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "0.5")
        monkeypatch.setenv(OVERALL_ENV, "0.5")
        rec = _install(monkeypatch)
        _call(tool, 0.6, 0.6)
        assert _built(rec) == pytest.approx((0.6, 0.6))

    def test_opt_out_notes_have_no_raise_text(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0")
        monkeypatch.setenv(OVERALL_ENV, "0")
        _install(monkeypatch)
        result = tools.extract_tool(SOURCE, 0.0, 0.0)
        assert "raised from" not in result["notes"]
        assert result["notes"] == _direct_notes("extract", 0.0, 0.0)


# ---------------------------------------------------------------------------
# EB-4: notes record a raise
# ---------------------------------------------------------------------------

class TestEB4Notes:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_both_raised_suffix_exact(self, monkeypatch, tool):
        _install(monkeypatch)
        result = _call(tool, 0.0, 0.0)
        base = _direct_notes(tool, 0.70, 0.75)
        assert result["notes"] == (
            base + " field_threshold raised from 0.0 to 0.7. "
            "overall_threshold raised from 0.0 to 0.75."
        )
        assert result["notes"].endswith(
            "field_threshold raised from 0.0 to 0.7. overall_threshold raised from 0.0 to 0.75."
        )

    @pytest.mark.parametrize("tool", TOOLS)
    def test_only_field_raised(self, monkeypatch, tool):
        _install(monkeypatch)
        result = _call(tool, 0.5, 0.9)
        base = _direct_notes(tool, 0.70, 0.9)
        assert result["notes"] == base + " field_threshold raised from 0.5 to 0.7."

    @pytest.mark.parametrize("tool", TOOLS)
    def test_only_overall_raised(self, monkeypatch, tool):
        _install(monkeypatch)
        result = _call(tool, 0.95, 0.1)
        base = _direct_notes(tool, 0.95, 0.75)
        assert result["notes"] == base + " overall_threshold raised from 0.1 to 0.75."

    def test_env_floor_numbers_in_notes(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.85")
        _install(monkeypatch)
        result = tools.extract_tool(SOURCE)
        base = _direct_notes("extract", 0.9, 0.85)
        assert result["notes"] == (
            base + " field_threshold raised from 0.7 to 0.9. "
            "overall_threshold raised from 0.75 to 0.85."
        )

    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("args", [(), (0.9, 0.8), (0.70, 0.75)])
    def test_no_raise_notes_unchanged(self, monkeypatch, tool, args):
        _install(monkeypatch)
        result = _call(tool, *args)
        eff = args or (0.70, 0.75)
        assert result["notes"] == _direct_notes(tool, *eff)
        assert "raised from" not in result["notes"]

    def test_raise_on_needs_review_result_keeps_status(self, monkeypatch):
        # Low critic confidence: the floor (not the 0.0 arg) decides the status.
        _install(monkeypatch, Recorder(critic_raw=critique_all(0.6, 0.6)))
        result = tools.extract_tool(SOURCE, 0.0, 0.0)
        assert result["status"] == "needs_review", result
        assert result["notes"].endswith("overall_threshold raised from 0.0 to 0.75.")


# ---------------------------------------------------------------------------
# EB-5: invalid requested args still error (checked before clamping)
# ---------------------------------------------------------------------------

class TestEB5InvalidArgs:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("kw,name", [
        ({"field_threshold": -0.1}, "field_threshold"),
        ({"field_threshold": 1.5}, "field_threshold"),
        ({"field_threshold": float("nan")}, "field_threshold"),
        ({"overall_threshold": 2.0}, "overall_threshold"),
        ({"overall_threshold": -0.5}, "overall_threshold"),
        ({"overall_threshold": float("nan")}, "overall_threshold"),
    ])
    def test_bad_arg_is_error_not_clamped(self, monkeypatch, tool, kw, name):
        rec = _install(monkeypatch)
        result = _call(tool, **kw)
        _assert_envelope(result)
        assert name in result["error_message"]
        assert rec.calls == []

    def test_negative_arg_error_even_with_env_floor(self, monkeypatch):
        monkeypatch.setenv(FIELD_ENV, "0.9")
        monkeypatch.setenv(OVERALL_ENV, "0.9")
        rec = _install(monkeypatch)
        result = tools.extract_tool(SOURCE, field_threshold=-0.1)
        _assert_envelope(result)
        assert "field_threshold" in result["error_message"]
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-6: malformed env fails closed
# ---------------------------------------------------------------------------

class TestEB6MalformedEnv:
    @pytest.mark.parametrize("tool", TOOLS)
    @pytest.mark.parametrize("var", [FIELD_ENV, OVERALL_ENV])
    @pytest.mark.parametrize("raw", ["abc", "nan", "1.5", "-0.1", "inf"])
    def test_malformed_env_envelope(self, monkeypatch, tool, var, raw):
        monkeypatch.setenv(var, raw)
        rec = _install(monkeypatch)
        result = _call(tool)
        _assert_envelope(result)
        assert result["error_message"].startswith("invalid thresholds:"), result
        assert var in result["error_message"]
        assert rec.calls == []

    @pytest.mark.parametrize("tool", TOOLS)
    def test_malformed_env_with_valid_strict_args(self, monkeypatch, tool):
        monkeypatch.setenv(FIELD_ENV, "abc")
        rec = _install(monkeypatch)
        result = _call(tool, 0.99, 0.99)
        _assert_envelope(result)
        assert result["error_message"].startswith("invalid thresholds:")
        assert FIELD_ENV in result["error_message"]
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-7: empty-text check unchanged, wins over malformed env
# ---------------------------------------------------------------------------

class TestEB7EmptyText:
    @pytest.mark.parametrize("text", ["", "   \n\t"])
    def test_extract_empty_with_bad_env(self, monkeypatch, text):
        monkeypatch.setenv(FIELD_ENV, "abc")
        rec = _install(monkeypatch)
        result = tools.extract_tool(text)
        _assert_envelope(result)
        assert result["error_message"] == "input is empty"
        assert rec.calls == []

    def test_validate_empty_with_bad_env(self, monkeypatch):
        monkeypatch.setenv(OVERALL_ENV, "1.5")
        rec = _install(monkeypatch)
        result = tools.validate_tool("  ", dict(CORRECT_EXTRACTION))
        _assert_envelope(result)
        assert result["error_message"] == "input is empty"
        assert rec.calls == []

    def test_extract_empty_env_unset(self, monkeypatch):
        rec = _install(monkeypatch)
        result = tools.extract_tool("", 0.0, 0.0)
        _assert_envelope(result)
        assert result["error_message"] == "input is empty"
        assert rec.calls == []


# ---------------------------------------------------------------------------
# EB-8 / EB-9: golden adversarial-caller pass
# ---------------------------------------------------------------------------

WRONG = {
    "inv-004": "total",
    "inv-006": "invoice_date",
    "inv-007": "vendor_name",
    "inv-008": "total",
}


def _golden_rows():
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _adversarial_pass(monkeypatch, field_arg=0.0, overall_arg=0.0):
    replay = json.loads(REPLAY.read_text())
    caught = silent = false_alarms = 0
    statuses: dict[str, str] = {}
    wrong_seen: dict[str, set] = {}
    for row in _golden_rows():
        entry = replay[row["id"]]

        def factory(gate_config=None, _e=entry, **kw):
            return ReflectivePipeline(
                Generator(ScriptedLLMClient([_e["generator_raw"]])),
                Critic(ScriptedLLMClient([_e["critic_raw"]])),
                gate_config=gate_config,
            )

        monkeypatch.setattr(pipeline_module, "build_default_pipeline", factory)
        res = tools.extract_tool(row["text"], field_arg, overall_arg)
        assert res["status"] in ("ok", "needs_review"), res
        statuses[row["id"]] = res["status"]
        flagged = {x["field"] for x in res["flagged_fields"]}
        needs_review = res["status"] == "needs_review"
        wrong = {f for f, exp in row["expected"].items()
                 if not run_eval._equal(res["data"].get(f), exp)}
        wrong_seen[row["id"]] = wrong
        for f in row["expected"]:
            is_wrong = f in wrong
            is_flagged = f in flagged
            if is_wrong and is_flagged and needs_review:
                caught += 1
            elif is_wrong:
                silent += 1
            elif is_flagged:
                false_alarms += 1
    return caught, silent, false_alarms, statuses, wrong_seen


def test_eb8_golden_adversarial_caller_is_caught(monkeypatch):
    caught, silent, fa, statuses, wrong = _adversarial_pass(monkeypatch, 0.0, 0.0)
    assert {k: v for k, v in wrong.items() if v} == {k: {f} for k, f in WRONG.items()}
    assert (caught, silent, fa) == (4, 0, 0)
    for rid in WRONG:
        assert statuses[rid] == "needs_review", (rid, statuses[rid])


def test_eb8_mid_knob_is_also_caught(monkeypatch):
    # (0.5, 0.5) gave 3 caught / 1 silent at HEAD per the PM probe.
    caught, silent, fa, _, _ = _adversarial_pass(monkeypatch, 0.5, 0.5)
    assert (caught, silent, fa) == (4, 0, 0)


def test_eb9_golden_opt_out_counterfactual(monkeypatch):
    monkeypatch.setenv(FIELD_ENV, "0")
    monkeypatch.setenv(OVERALL_ENV, "0")
    caught, silent, _, statuses, _ = _adversarial_pass(monkeypatch, 0.0, 0.0)
    assert (caught, silent) == (0, 4)
    for rid in WRONG:
        assert statuses[rid] == "ok", (rid, statuses[rid])


# ---------------------------------------------------------------------------
# EB-10: docs
# ---------------------------------------------------------------------------

def _server_docstring(name):
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not found in server.py")


def _readme_section4():
    text = README.read_text(encoding="utf-8")
    start = text.index("## 4.")
    end = text.index("## 5.", start)
    return text[start:end]


class TestEB10Docs:
    @pytest.mark.parametrize("name", ["extract", "validate"])
    def test_server_docstring_mentions_floor(self, name):
        doc = _server_docstring(name)
        assert FIELD_ENV in doc and OVERALL_ENV in doc, doc
        assert "0.70" in doc and "0.75" in doc, doc
        assert "floor" in doc.lower() or "raised" in doc.lower(), doc

    def test_readme_section4_names_env_vars(self):
        sec = _readme_section4()
        assert FIELD_ENV in sec and OVERALL_ENV in sec
        assert "0.70" in sec and "0.75" in sec
        assert "floor" in sec.lower()
        assert "raised" in sec.lower()

    def test_readme_table_rows_byte_identical(self):
        lines = _readme_section4().splitlines()
        for row in HEAD_TABLE_ROWS:
            assert row in lines, row

    def test_make_eval_stdout_unchanged(self):
        from tests.test_iter60_behavior import EXPECTED_STDOUT
        proc = subprocess.run(
            [sys.executable, str(REPO / "eval" / "run_eval.py")],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout == EXPECTED_STDOUT
