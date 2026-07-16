#!/usr/bin/env python3
"""Tiny eval harness for the reflective extraction pipeline.

Runs the FULL pipeline (generate -> heuristics -> critique -> gate) over a
golden-file dataset and reports:

  - field-level accuracy   (correct fields / total fields)
  - exact-match accuracy   (examples with every field correct)
  - needs-review rate      (examples the gate escalated)
  - a table of exactly WHERE it failed

By default it runs OFFLINE using recorded model outputs in
`eval/fixtures/replay.json`, so anyone can `make eval` with no API keys and get
deterministic numbers. Pass `--live` to run against a real provider configured
via REFLECT_PROVIDER (+ credentials).

Exit code is non-zero if field accuracy drops below --min-accuracy (default 0.90),
so this doubles as an offline eval-CI gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflect_mcp.critic import Critic  # noqa: E402
from reflect_mcp.generator import Generator  # noqa: E402
from reflect_mcp.pipeline import ReflectivePipeline  # noqa: E402
from reflect_mcp.schema import EXTRACTION_FIELDS  # noqa: E402
from reflect_mcp.scripted import ScriptedLLMClient  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"


@dataclass
class EvalMetrics:
    mode: str
    examples: int = 0
    total_fields: int = 0
    correct_fields: int = 0
    exact_matches: int = 0
    reviewed: int = 0
    failures: list[tuple[str, str, object, object]] = dc_field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        return self.correct_fields / self.total_fields if self.total_fields else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.examples if self.examples else 0.0

    @property
    def review_rate(self) -> float:
        return self.reviewed / self.examples if self.examples else 0.0


def _equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), abs_tol=0.005)
    return a == b


def load_golden() -> list[dict]:
    rows = []
    with open(GOLDEN) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _offline_pipeline(example_id: str, replay: dict) -> ReflectivePipeline:
    entry = replay[example_id]
    gen = Generator(ScriptedLLMClient([entry["generator_raw"]]))
    crit = Critic(ScriptedLLMClient([entry["critic_raw"]]))
    return ReflectivePipeline(gen, crit)


def evaluate(live: bool = False) -> EvalMetrics:
    """Run the pipeline over the golden set and return metrics (no printing)."""
    golden = load_golden()
    replay = {} if live else json.loads(REPLAY.read_text())
    mode = ("LIVE (" + os.getenv("REFLECT_PROVIDER", "anthropic") + ")") if live else "OFFLINE (replay fixtures)"
    m = EvalMetrics(mode=mode, examples=len(golden))

    for row in golden:
        eid, text, expected = row["id"], row["text"], row["expected"]
        if live:
            from reflect_mcp.pipeline import build_default_pipeline

            pipeline = build_default_pipeline()
        else:
            pipeline = _offline_pipeline(eid, replay)
        result = pipeline.extract(text)
        got = result.data.model_dump()

        all_ok = True
        for field in EXTRACTION_FIELDS:
            m.total_fields += 1
            if _equal(got.get(field), expected.get(field)):
                m.correct_fields += 1
            else:
                all_ok = False
                m.failures.append((eid, field, expected.get(field), got.get(field)))
        if all_ok:
            m.exact_matches += 1
        if result.status == "needs_review":
            m.reviewed += 1
    return m


def run(live: bool, min_accuracy: float) -> int:
    m = evaluate(live=live)
    print("=" * 62)
    print(f"reflect-mcp eval  |  mode: {m.mode}")
    print("=" * 62)
    print(f"examples          : {m.examples}")
    print(f"field accuracy    : {m.field_accuracy:.1%}  ({m.correct_fields}/{m.total_fields})")
    print(f"exact-match acc   : {m.exact_accuracy:.1%}  ({m.exact_matches}/{m.examples})")
    print(f"needs-review rate : {m.review_rate:.1%}  ({m.reviewed}/{m.examples})")
    print("-" * 62)
    if m.failures:
        print("field-level failures (where it breaks):")
        for eid, field, exp, got in m.failures:
            print(f"  [{eid}] {field}: expected {exp!r} but got {got!r}")
    else:
        print("no field-level failures.")
    print("=" * 62)

    if m.field_accuracy < min_accuracy:
        print(f"FAIL: field accuracy {m.field_accuracy:.1%} < required {min_accuracy:.1%}")
        return 1
    print(f"PASS: field accuracy {m.field_accuracy:.1%} >= required {min_accuracy:.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the reflect-mcp extraction eval.")
    ap.add_argument("--live", action="store_true", help="Use a real provider instead of replay fixtures.")
    ap.add_argument("--min-accuracy", type=float, default=0.90, help="Fail below this field accuracy.")
    args = ap.parse_args()
    return run(args.live, args.min_accuracy)


if __name__ == "__main__":
    raise SystemExit(main())
