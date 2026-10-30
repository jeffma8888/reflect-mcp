"""Black-box behavioral tests for iter-62: delete the `heuristics_enabled` knob.

No constructible ReflectivePipeline can skip the deterministic checks any more.
Covers Expected Behaviors 1-9 of pm.md:

  1. `heuristics_enabled=` keyword is rejected with TypeError (any value).
  2. A positional 4th constructor argument is rejected with TypeError.
  3. `__init__` signature is exactly (self, generator, critic, gate_config=None);
     a pipeline has no `heuristics_enabled` attribute.
  4. `heuristics._number_in_text` is gone; `_find_number_span` and
     `run_heuristics` remain.
  5. Fabricated data on the Critic-less path is flagged on all 5 fields.
  6. Golden pass, Critic on: caught 4 / silent 0 / FA 0.
  7. Golden pass, Critic off: caught 3 / silent 1 / FA 0 (inv-007 vendor_name).
  8. `eval/run_eval.py` exits 0 with stdout byte-identical to HEAD's.
  9. Golden payloads (minus trace_id) are identical to HEAD's, for both the
     Critic-on and Critic-off pipelines; `__all__` unchanged.

Isolation contract: tests drive the public package API, the golden data files
and the eval CLI, asserting on observable output. No implementation source was
read. The HEAD constants below (eval stdout, payload digests, `__all__`) were
captured by running the HEAD 4829966 tree (`git archive HEAD`) before commit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

import reflect_mcp
import reflect_mcp.heuristics as heuristics
from reflect_mcp.critic import Critic
from reflect_mcp import GateConfig
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.schema import InvoiceData
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "eval" / "fixtures" / "replay.json"

sys.path.insert(0, str(REPO / "eval"))
import run_eval  # noqa: E402

FIELDS = ("vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "currency")

# Captured from HEAD 4829966 (pre-deletion) eval stdout.
HEAD_EVAL_STDOUT = """\
==============================================================
reflect-mcp eval  |  mode: OFFLINE (replay fixtures)
==============================================================
examples          : 8
field accuracy    : 92.9%  (52/56)
exact-match acc   : 50.0%  (4/8)
needs-review rate : 50.0%  (4/8)
catch rate        : 100.0%  (4/4)
silent-wrong      : 0
false alarms      : 0
--------------------------------------------------------------
field-level failures (where it breaks):
  [inv-004] total: expected 583.2 but got 683.2  (flagged)
  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'  (flagged)
  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'  (flagged)
  [inv-008] total: expected 75.0 but got 45.0  (flagged)
==============================================================
PASS: field accuracy 92.9% >= required 90.0%
PASS: silent-wrong fields 0 <= allowed 0
"""

# sha256 of json.dumps({row_id: model_dump(mode="json") minus trace_id}, sort_keys=True)
# over all golden rows, captured from HEAD 4829966.
HEAD_PAYLOAD_SHA256 = {
    "on": "251385698ef79f3e6ef8dc68b41c02cfeac1e55c603a00935c2e1451aa7cdccc",
    "off": "8afccd2fd53069231967bf1f6957159844d4954d6c3f5b2bdb34795842cfcacb",
}
HEAD_PAYLOAD_KEYS = {
    "critic_summary", "data", "field_confidence", "field_provenance",
    "flagged_fields", "notes", "overall_confidence", "status",
}
HEAD_ALL = [
    "Critic", "CriticError", "CritiqueResult", "EXTRACTION_FIELDS", "ExtractionResult",
    "FieldVerdict", "FlaggedField", "GateConfig", "Generator", "GeneratorError",
    "InvoiceData", "ProvenanceSpan", "ReflectivePipeline", "__version__",
    "build_default_pipeline", "decide",
]

FAB_TEXT = "ACME Corp invoice INV-1 total 10.00 USD"


def _fab_data() -> InvoiceData:
    return InvoiceData(
        invoice_number="ZZZ-999", vendor_name="Initech", total=99999,
        currency="EUR", invoice_date="2031-01-01",
    )


def _gen(raws=()) -> Generator:
    return Generator(ScriptedLLMClient(list(raws)))


def _crit(raws=()) -> Critic:
    return Critic(ScriptedLLMClient(list(raws)))


def _status(result) -> str:
    s = result.status
    return getattr(s, "value", s)


def _replay() -> dict:
    return json.loads(REPLAY.read_text())


def _golden_results(critic_on: bool) -> dict[str, object]:
    rep = _replay()
    out = {}
    for row in run_eval.load_golden():
        fx = rep[row["id"]]
        crit = _crit([fx["critic_raw"]]) if critic_on else None
        pipe = ReflectivePipeline(_gen([fx["generator_raw"]]), crit)
        out[row["id"]] = (row, pipe.extract(row["text"]))
    return out


def _score(critic_on: bool):
    caught, silent, fa = [], [], []
    for rid, (row, result) in _golden_results(critic_on).items():
        flagged = {f.field for f in result.flagged_fields}
        data = result.data.model_dump() if hasattr(result.data, "model_dump") else dict(result.data)
        for field in FIELDS:
            wrong = not run_eval._equal(data.get(field), row["expected"].get(field))
            if wrong and field in flagged:
                caught.append((rid, field))
            elif wrong:
                silent.append((rid, field))
            elif field in flagged:
                fa.append((rid, field))
    return caught, silent, fa


def _payload_digest(critic_on: bool) -> tuple[str, dict]:
    dumps = {}
    for rid, (_row, result) in _golden_results(critic_on).items():
        d = result.model_dump(mode="json")
        d.pop("trace_id", None)
        dumps[rid] = d
    s = json.dumps(dumps, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest(), dumps


# ---------------------------------------------------------------------------
# EB-1: keyword rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "with_critic,value",
    [(True, False), (False, False), (False, True), (True, True)],
)
def test_eb1_heuristics_enabled_keyword_rejected(with_critic, value):
    crit = _crit() if with_critic else None
    with pytest.raises(TypeError):
        ReflectivePipeline(_gen(), crit, heuristics_enabled=value)


def test_eb1_keyword_rejected_alongside_gate_config():
    with pytest.raises(TypeError):
        ReflectivePipeline(_gen(), None, gate_config=GateConfig(), heuristics_enabled=False)


# ---------------------------------------------------------------------------
# EB-2: positional 4th argument rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fourth", [False, True])
def test_eb2_positional_fourth_argument_rejected(fourth):
    with pytest.raises(TypeError):
        ReflectivePipeline(_gen(), None, GateConfig(), fourth)


def test_eb2_three_positional_arguments_still_accepted():
    pipe = ReflectivePipeline(_gen(), None, GateConfig())
    assert isinstance(pipe, ReflectivePipeline)


# ---------------------------------------------------------------------------
# EB-3: signature
# ---------------------------------------------------------------------------

def test_eb3_signature_exact():
    sig = inspect.signature(ReflectivePipeline.__init__)
    assert list(sig.parameters) == ["self", "generator", "critic", "gate_config"]
    assert sig.parameters["gate_config"].default is None


@pytest.mark.parametrize("with_critic", [True, False])
def test_eb3_no_heuristics_enabled_attribute(with_critic):
    pipe = ReflectivePipeline(_gen(), _crit() if with_critic else None)
    assert hasattr(pipe, "heuristics_enabled") is False
    assert hasattr(ReflectivePipeline, "heuristics_enabled") is False


# ---------------------------------------------------------------------------
# EB-4: dead helper gone, live helpers kept
# ---------------------------------------------------------------------------

def test_eb4_number_in_text_removed():
    assert hasattr(heuristics, "_number_in_text") is False


def test_eb4_live_helpers_still_present():
    assert callable(getattr(heuristics, "_find_number_span", None))
    from reflect_mcp.heuristics import run_heuristics  # noqa: F401
    assert callable(run_heuristics)


# ---------------------------------------------------------------------------
# EB-5: fabricated data is caught on the Critic-less path
# ---------------------------------------------------------------------------

def test_eb5_fabricated_data_flagged_without_critic():
    result = ReflectivePipeline(_gen(), None).validate(FAB_TEXT, _fab_data())
    assert _status(result) == "needs_review"
    assert {f.field for f in result.flagged_fields} == {
        "invoice_number", "vendor_name", "total", "currency", "invoice_date",
    }


# ---------------------------------------------------------------------------
# EB-6 / EB-7: golden pass for every constructible pipeline
# ---------------------------------------------------------------------------

def test_eb6_golden_critic_on_catches_all():
    caught, silent, fa = _score(critic_on=True)
    assert (len(caught), len(silent), len(fa)) == (4, 0, 0), (caught, silent, fa)
    assert sorted(caught) == [
        ("inv-004", "total"), ("inv-006", "invoice_date"),
        ("inv-007", "vendor_name"), ("inv-008", "total"),
    ]


def test_eb7_golden_critic_off_one_silent():
    caught, silent, fa = _score(critic_on=False)
    assert (len(caught), len(silent), len(fa)) == (3, 1, 0), (caught, silent, fa)
    assert silent == [("inv-007", "vendor_name")]


def test_eb7_critic_off_not_all_rows_ok():
    # The deleted knob made all 8 rows `ok`; the heuristics must now keep some
    # rows out of `ok` even with no Critic.
    statuses = [_status(r) for _row, r in _golden_results(critic_on=False).values()]
    assert statuses.count("ok") < len(statuses), statuses


# ---------------------------------------------------------------------------
# EB-8: eval stdout byte-identical to HEAD
# ---------------------------------------------------------------------------

def test_eb8_eval_stdout_identical_to_head():
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "run_eval.py")],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == HEAD_EVAL_STDOUT


# ---------------------------------------------------------------------------
# EB-9 (+EB-10 surface): payloads and __all__ unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", ["on", "off"])
def test_eb9_golden_payloads_identical_to_head(cfg):
    digest, dumps = _payload_digest(critic_on=(cfg == "on"))
    for rid, d in dumps.items():
        assert set(d) == HEAD_PAYLOAD_KEYS, (rid, sorted(d))
    assert digest == HEAD_PAYLOAD_SHA256[cfg]


def test_eb9_public_all_unchanged():
    assert sorted(reflect_mcp.__all__) == HEAD_ALL
