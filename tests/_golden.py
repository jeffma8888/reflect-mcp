"""Frozen golden subset: the 8 rows (inv-001..inv-008) HEAD-identity pins read.

WHY this exists: many behavior tests pin "eval output is byte-identical to the
iteration that shipped them" or "the golden set has exactly 8 rows". Pinned
against the *live* dataset, every new golden row broke ~80 of them, which made
the quality bar ("every gate change ships golden-set coverage") unaffordable.
These pins now read a frozen subset, so they keep their exact numbers forever,
and only a few deliberately live pins move when a row is added.

The subset is guarded, not trusted: a frozen row that is deleted raises (it is
never silently skipped) and an edited one changes `FROZEN_DATASET_SHA256`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
GOLDEN = EVAL_DIR / "golden" / "dataset.jsonl"
REPLAY = EVAL_DIR / "fixtures" / "replay.json"

FROZEN_IDS: tuple[str, ...] = tuple(f"inv-00{i}" for i in range(1, 9))
# sha256 of `python eval/run_eval.py` stdout over exactly the frozen rows.
FROZEN_EVAL_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
# sha256 of the frozen rows' original lines, joined "\n" with a trailing "\n".
FROZEN_DATASET_SHA256 = "ffd8faeb8935e65d15e2edf8f62e6cd310a9d3261497f5ed17b1e8a01c2e6dfc"


def frozen_lines(path: Path | str | None = None) -> list[str]:
    """Return the frozen rows' raw dataset lines in `FROZEN_IDS` order.

    Raw lines (not re-serialized dicts) so the content hash detects any edit.
    """
    lines = [ln for ln in Path(path or GOLDEN).read_text().splitlines() if ln.strip()]
    by_id = {json.loads(ln)["id"]: ln for ln in lines}
    missing = [rid for rid in FROZEN_IDS if rid not in by_id]
    assert not missing, f"frozen golden ids missing from dataset: {missing}"
    return [by_id[rid] for rid in FROZEN_IDS]


def frozen_dataset_text(path: Path | str | None = None) -> str:
    """The frozen subset as JSONL text; its sha256 is `FROZEN_DATASET_SHA256`."""
    return "\n".join(frozen_lines(path)) + "\n"


def frozen_rows(path: Path | str | None = None) -> list[dict]:
    """Parsed frozen rows, in `FROZEN_IDS` order (later golden rows excluded)."""
    return [json.loads(ln) for ln in frozen_lines(path)]


def frozen_replay() -> dict:
    """The replay fixture restricted to `FROZEN_IDS`."""
    replay = json.loads(REPLAY.read_text())
    return {rid: replay[rid] for rid in FROZEN_IDS}


def write_frozen_dataset(directory: Path | str) -> Path:
    """Write the frozen subset to `<directory>/dataset.jsonl` and return its path."""
    target = Path(directory) / "dataset.jsonl"
    target.write_text(frozen_dataset_text())
    return target


def frozen_eval(*args: str, timeout: float = 180) -> subprocess.CompletedProcess[str]:
    """Run `eval/run_eval.py`'s `main()` over the frozen subset in a subprocess.

    WHY a subprocess that patches `run_eval.GOLDEN`: `load_golden()` reads the
    module global at call time, so the frozen eval needs no `--golden` flag and
    no edit to the shipped harness or the committed dataset.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset = write_frozen_dataset(tmp)
        code = (
            "import sys; sys.path.insert(0, {eval_dir!r}); import run_eval; "
            "run_eval.GOLDEN = run_eval.Path({dataset!r}); "
            "sys.argv = ['run_eval.py'] + sys.argv[1:]; sys.exit(run_eval.main())"
        ).format(eval_dir=str(EVAL_DIR), dataset=str(dataset))
        return subprocess.run(
            [sys.executable, "-c", code, *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def frozen_eval_sha256(*args: str) -> str:
    """sha256 of `frozen_eval(*args)` stdout (the HEAD-identity fingerprint)."""
    return hashlib.sha256(frozen_eval(*args).stdout.encode()).hexdigest()


def use_frozen_golden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path | None = None) -> Path:
    """Point in-process `run_eval.GOLDEN` at a frozen-only dataset; return its path.

    `monkeypatch` undoes the patch at test teardown, so live-eval tests in the
    same session still see the committed dataset.
    """
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import run_eval  # noqa: PLC0415 -- importable only once eval/ is on sys.path

    directory = tmp_path if tmp_path is not None else Path(tempfile.mkdtemp())
    dataset = write_frozen_dataset(directory)
    monkeypatch.setattr(run_eval, "GOLDEN", dataset)
    return dataset


@contextmanager
def frozen_golden() -> Iterator[Path]:
    """Context-manager form of `use_frozen_golden` for non-fixture call sites."""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as tmp:
        yield use_frozen_golden(mp, Path(tmp))
