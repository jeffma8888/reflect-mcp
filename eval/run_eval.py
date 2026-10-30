#!/usr/bin/env python3
"""Tiny eval harness for the reflective extraction pipeline.

Runs the FULL pipeline (generate -> heuristics -> critique -> gate) over a
golden-file dataset and reports:

  - field-level accuracy   (correct fields / total fields)
  - exact-match accuracy   (examples with every field correct)
  - needs-review rate      (examples the gate escalated)
  - catch rate             (wrong fields the gate named in `flagged_fields`)
  - silent-wrong count     (wrong fields returned WITHOUT being flagged -- the
                            "silent wrong answer" the product promises never to give)
  - false-alarm count      (correct fields flagged anyway; reported, never gated)
  - a table of exactly WHERE it failed, each row tagged (flagged) or (SILENT)

By default it runs OFFLINE using recorded model outputs in
`eval/fixtures/replay.json`, so anyone can `make eval` with no API keys and get
deterministic numbers. Pass `--live` to run against a real provider configured
via REFLECT_PROVIDER (+ credentials).

Exit code is non-zero if field accuracy drops below --min-accuracy (default 0.90)
OR if silent-wrong fields exceed --max-silent-wrong (default 0), so this doubles
as an offline eval-CI gate on both accuracy and the trust contract: a wrong field
the gate flagged is the product working; a wrong field returned unflagged is the
product failing, however high accuracy stays.
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
    # Trust-contract counters: every failure is either caught (flagged) or silent.
    caught: int = 0
    silent_wrong: int = 0
    false_alarms: int = 0
    silent_failures: list[tuple[str, str, object, object]] = dc_field(default_factory=list)
    # Parallel to `failures`: True where the gate flagged that wrong field. Recorded
    # where the classification is decided, so the readout tag can never drift from it.
    failure_flagged: list[bool] = dc_field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        return self.correct_fields / self.total_fields if self.total_fields else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.examples if self.examples else 0.0

    @property
    def review_rate(self) -> float:
        return self.reviewed / self.examples if self.examples else 0.0

    @property
    def catch_rate(self) -> float | None:
        """Share of wrong fields the gate flagged; None when nothing was wrong.

        None rather than 1.0 because "caught every error" is vacuous with zero
        errors, and the readout should say so (`n/a`) instead of implying a result.
        """
        wrong = self.caught + self.silent_wrong
        return self.caught / wrong if wrong else None


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
        flagged = {f.field for f in result.flagged_fields}

        all_ok = True
        for field in EXTRACTION_FIELDS:
            m.total_fields += 1
            if _equal(got.get(field), expected.get(field)):
                m.correct_fields += 1
                if field in flagged:
                    m.false_alarms += 1
                continue
            all_ok = False
            failure = (eid, field, expected.get(field), got.get(field))
            m.failures.append(failure)
            m.failure_flagged.append(field in flagged)
            if field in flagged:
                m.caught += 1
            else:
                m.silent_wrong += 1
                m.silent_failures.append(failure)
        if all_ok:
            m.exact_matches += 1
        if result.status == "needs_review":
            m.reviewed += 1
    return m


def _catch_rate_line(m: EvalMetrics) -> str:
    """Format the catch-rate readout, printing `n/a` when no field was wrong."""
    wrong = m.caught + m.silent_wrong
    rate = "n/a" if m.catch_rate is None else f"{m.catch_rate:.1%}"
    return f"catch rate        : {rate}  ({m.caught}/{wrong})"


def run(live: bool, min_accuracy: float, max_silent_wrong: int = 0) -> int:
    """Print the readout; return 0 only if BOTH the accuracy and silent-wrong checks pass.

    Both verdict lines are always printed so a failing run shows every reason at once.
    """
    m = evaluate(live=live)
    print("=" * 62)
    print(f"reflect-mcp eval  |  mode: {m.mode}")
    print("=" * 62)
    print(f"examples          : {m.examples}")
    print(f"field accuracy    : {m.field_accuracy:.1%}  ({m.correct_fields}/{m.total_fields})")
    print(f"exact-match acc   : {m.exact_accuracy:.1%}  ({m.exact_matches}/{m.examples})")
    print(f"needs-review rate : {m.review_rate:.1%}  ({m.reviewed}/{m.examples})")
    print(_catch_rate_line(m))
    print(f"silent-wrong      : {m.silent_wrong}")
    print(f"false alarms      : {m.false_alarms}")
    print("-" * 62)
    if m.failures:
        print("field-level failures (where it breaks):")
        for (eid, field, exp, got), was_flagged in zip(m.failures, m.failure_flagged):
            tag = "(flagged)" if was_flagged else "(SILENT)"
            print(f"  [{eid}] {field}: expected {exp!r} but got {got!r}  {tag}")
    else:
        print("no field-level failures.")
    print("=" * 62)

    accuracy_ok = m.field_accuracy >= min_accuracy
    if accuracy_ok:
        print(f"PASS: field accuracy {m.field_accuracy:.1%} >= required {min_accuracy:.1%}")
    else:
        print(f"FAIL: field accuracy {m.field_accuracy:.1%} < required {min_accuracy:.1%}")
    silent_ok = m.silent_wrong <= max_silent_wrong
    if silent_ok:
        print(f"PASS: silent-wrong fields {m.silent_wrong} <= allowed {max_silent_wrong}")
    else:
        print(f"FAIL: silent-wrong fields {m.silent_wrong} > allowed {max_silent_wrong}")
    return 0 if accuracy_ok and silent_ok else 1


def _non_negative_int(raw: str) -> int:
    """argparse type: a negative allowance would make the silent-wrong gate unpassable."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {raw!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the reflect-mcp extraction eval.")
    ap.add_argument("--live", action="store_true", help="Use a real provider instead of replay fixtures.")
    ap.add_argument("--min-accuracy", type=float, default=0.90, help="Fail below this field accuracy.")
    ap.add_argument(
        "--max-silent-wrong",
        type=_non_negative_int,
        default=0,
        help="Fail when more than N wrong fields are returned without being flagged.",
    )
    args = ap.parse_args()
    return run(args.live, args.min_accuracy, args.max_silent_wrong)


if __name__ == "__main__":
    raise SystemExit(main())
