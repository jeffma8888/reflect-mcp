"""Black-box behavioral tests for iter-58: README §3 Demo is the real inv-004 replay.

Covers the Expected Behaviors from pm.md:
  1. §3 holds exactly one ```text block, byte-equal to golden inv-004 text (has 583.20).
  2. §3 holds exactly one ```json block, strict JSON; no ```jsonc block left in §3.
  3. The parsed json block has exactly the ExtractionResult key set.
  4. Re-running the inv-004 replay (trace_id dropped) == the parsed README block,
     with the spec's pinned spot values.
  5. The README block's trace_id is a string starting with ``trace-``.
  6. §3 prose (outside fences) names inv-004, 583.20, 683.2 and the source/model mismatch.
  7. The old demo phrasing is absent from the whole README.
  8. The pin is real: the old one-liner reason / old Globex input make the checks fail.
  9. The checks need no network, no credentials and no ``mcp`` import.
 (10. ``make eval`` unchanged -- verified by the tester outside pytest.)

Isolation contract: tests read only README.md, eval/golden/dataset.jsonl and
eval/fixtures/replay.json, and drive the public pipeline over ScriptedLLMClient.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reflect_mcp.critic import Critic
from reflect_mcp.generator import Generator
from reflect_mcp.pipeline import ReflectivePipeline
from reflect_mcp.scripted import ScriptedLLMClient

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
GOLDEN = REPO / "eval" / "golden" / "dataset.jsonl"
REPLAY = REPO / "eval" / "fixtures" / "replay.json"

EXPECTED_KEYS = {
    "status",
    "data",
    "field_confidence",
    "overall_confidence",
    "flagged_fields",
    "field_provenance",
    "trace_id",
    "critic_summary",
    "notes",
}

FOUR_PART_REASON = (
    "683.20 does not reconcile with 540.00 + 43.20 and is not in the source; "
    "critic marked the value not grounded in the source; "
    "amount 683.2 not found in source text; "
    "subtotal(540.0) + tax(43.2) != total(683.2)"
)
OLD_REASON = "683.20 does not reconcile with 540.00 + 43.20; not found in source"
OLD_INPUT = (
    "Hi team, bill from Globex Ltd. Ref GLX/2026/778, dated 2026-02-28.\n"
    " 540.00 before tax, tax 43.20, amount due 683.20. Pay in GBP."
)

FENCE_RE = re.compile(r"^```([A-Za-z0-9_-]*)[ \t]*\n(.*?)\n```[ \t]*$", re.S | re.M)


# ---------------------------------------------------------------- helpers


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _golden_row(row_id: str) -> dict:
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] == row_id:
                return row
    raise AssertionError(f"golden row {row_id} missing")


def _section3(readme: str) -> str:
    start = readme.find("\n## 3. Demo")
    assert start != -1, "README has no '## 3. Demo' heading"
    end = readme.find("\n## 4.", start + 1)
    assert end != -1, "README has no '## 4.' heading after §3"
    return readme[start:end]


def _blocks(section: str, lang: str) -> list[str]:
    return [body for info, body in FENCE_RE.findall(section) if info == lang]


def _prose(section: str) -> str:
    return FENCE_RE.sub("", section)


def _replay_run() -> dict:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))["inv-004"]
    pipeline = ReflectivePipeline(
        Generator(ScriptedLLMClient([replay["generator_raw"]])),
        Critic(ScriptedLLMClient([replay["critic_raw"]])),
    )
    return pipeline.extract(_golden_row("inv-004")["text"]).model_dump(mode="json")


def check_text_block(readme: str) -> None:
    blocks = _blocks(_section3(readme), "text")
    assert len(blocks) == 1, f"expected exactly one ```text block in §3, got {len(blocks)}"
    golden = _golden_row("inv-004")["text"]
    assert blocks[0] == golden
    assert "583.20" in blocks[0]


def parsed_json_block(readme: str) -> dict:
    section = _section3(readme)
    blocks = _blocks(section, "json")
    assert len(blocks) == 1, f"expected exactly one ```json block in §3, got {len(blocks)}"
    assert "```jsonc" not in section
    return json.loads(blocks[0])


def check_matches_replay(readme: str) -> None:
    block = parsed_json_block(readme)
    run = _replay_run()
    block = {k: v for k, v in block.items() if k != "trace_id"}
    run = {k: v for k, v in run.items() if k != "trace_id"}
    assert block == run


# ---------------------------------------------------------------- behavior 1


def test_b1_single_text_block_is_golden_inv004_verbatim():
    check_text_block(_readme())


# ---------------------------------------------------------------- behavior 2


def test_b2_single_strict_json_block_and_no_jsonc():
    section = _section3(_readme())
    assert "```jsonc" not in section
    blocks = _blocks(section, "json")
    assert len(blocks) == 1
    body = blocks[0]
    assert "//" not in body
    assert '"..."' not in body
    parsed = json.loads(body)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------- behavior 3


def test_b3_json_block_has_exact_result_key_set():
    block = parsed_json_block(_readme())
    assert set(block) == EXPECTED_KEYS
    # Same key set as the real model_dump(mode="json").
    assert set(_replay_run()) == EXPECTED_KEYS


# ---------------------------------------------------------------- behavior 4


def test_b4_replay_rerun_equals_readme_block():
    check_matches_replay(_readme())


def test_b4_pinned_spot_values():
    block = parsed_json_block(_readme())
    run = _replay_run()
    for out in (block, run):
        assert out["status"] == "needs_review"
        assert out["data"]["total"] == 683.2
        assert out["field_confidence"]["total"] == 0.12
        assert out["overall_confidence"] == 0.7
        assert len(out["flagged_fields"]) == 1
        flag = out["flagged_fields"][0]
        assert flag["field"] == "total"
        assert flag["reason"] == FOUR_PART_REASON
        assert out["field_provenance"]["total"] == []
        assert out["critic_summary"] == "Total appears hallucinated; escalate."


def test_b4_displayed_provenance_spans_slice_to_values():
    """Every non-empty span in the README block slices the golden text to the field value."""
    block = parsed_json_block(_readme())
    text = _golden_row("inv-004")["text"]
    checked = 0
    for field, spans in block["field_provenance"].items():
        for span in spans:
            shown = text[span["start"] : span["end"]]
            value = block["data"][field]
            if isinstance(value, float):
                assert float(shown.replace(",", "")) == value, (field, shown)
            else:
                assert shown == value, (field, shown)
            checked += 1
    assert checked >= 6


# ---------------------------------------------------------------- behavior 5


def test_b5_trace_id_is_trace_prefixed_string():
    block = parsed_json_block(_readme())
    assert isinstance(block["trace_id"], str)
    assert block["trace_id"].startswith("trace-")
    # The real run also emits a trace- id, so the placeholder mirrors reality.
    assert _replay_run()["trace_id"].startswith("trace-")


# ---------------------------------------------------------------- behavior 6


def test_b6_prose_names_row_and_mismatch():
    prose = _prose(_section3(_readme()))
    for needle in ("inv-004", "583.20", "683.2"):
        assert needle in prose, needle
    assert "source" in prose.lower()
    assert re.search(r"generator|model", prose, re.I)
    # It must not suggest the source itself says 683.20.
    assert "amount due 683.20" not in prose
    assert not re.search(r"source[^.\n]{0,40}683\.20?", prose, re.I)


# ---------------------------------------------------------------- behavior 7


def test_b7_old_demo_phrasing_absent_everywhere():
    readme = _readme()
    assert "amount due 683.20" not in readme
    assert '; not found in source"' not in readme


# ---------------------------------------------------------------- behavior 8


def test_b8_old_reason_in_json_block_breaks_the_pin():
    readme = _readme()
    assert FOUR_PART_REASON in readme
    mutated = readme.replace(FOUR_PART_REASON, OLD_REASON)
    assert mutated != readme
    with pytest.raises(AssertionError):
        check_matches_replay(mutated)


def test_b8_old_globex_input_in_text_block_breaks_the_pin():
    readme = _readme()
    golden = _golden_row("inv-004")["text"]
    assert golden in readme
    mutated = readme.replace(golden, OLD_INPUT)
    assert mutated != readme
    with pytest.raises(AssertionError):
        check_text_block(mutated)


def test_b8_jsonc_style_block_fails_strict_parse():
    readme = _readme()
    section = _section3(readme)
    body = _blocks(section, "json")[0]
    mutated = readme.replace(body, "// output\n" + body)
    with pytest.raises(json.JSONDecodeError):
        parsed_json_block(mutated)


# ---------------------------------------------------------------- behavior 9


def test_b9_checks_run_offline_without_mcp():
    code = (
        "import socket, sys\n"
        "def _blocked(*a, **k):\n"
        "    raise RuntimeError('network blocked')\n"
        "socket.socket.connect = _blocked\n"
        "socket.create_connection = _blocked\n"
        "sys.path.insert(0, %r)\n"
        "import tests.test_iter58_behavior as t\n"
        "r = t._readme()\n"
        "t.check_text_block(r)\n"
        "t.check_matches_replay(r)\n"
        "assert not any(m == 'mcp' or m.startswith('mcp.') for m in sys.modules), "
        "sorted(m for m in sys.modules if m.startswith('mcp'))\n"
        "print('OK')\n"
    ) % str(REPO)
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
