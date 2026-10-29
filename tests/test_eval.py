"""The eval harness runs deterministically offline over the golden set."""

from __future__ import annotations

import sys
from pathlib import Path

# Make eval/run_eval.py importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import run_eval  # noqa: E402


def test_offline_eval_produces_expected_metrics():
    m = run_eval.evaluate(live=False)
    assert m.examples == 6
    assert m.total_fields == 42  # 6 examples x 7 fields
    # inv-004 injects a wrong total, inv-006 an ungrounded date -> two misses.
    assert m.correct_fields == 40
    assert 0.90 <= m.field_accuracy < 1.0
    assert m.exact_matches == 4  # 4/6 exact
    # Both injected-error examples must be routed to human review.
    assert m.reviewed == 2
    assert ("inv-004", "total", 583.2, 683.2) in m.failures
    assert ("inv-006", "invoice_date", "2026-03-14", "2026-04-14") in m.failures


def test_eval_run_gate_passes_offline():
    # The CLI gate returns 0 (pass) at the default 0.90 accuracy bar.
    assert run_eval.run(live=False, min_accuracy=0.90) == 0
