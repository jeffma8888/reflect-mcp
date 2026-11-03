"""Black-box behavioral tests for iter-64: `validate --batch`.

Covers Expected Behaviors 1-10 from the iter-64 spec:
  1  corpus run: one compact JSON object per non-blank line, file or stdin
  2  per-line result == single-document `validate` output (golden corpus)
  3  exit code is the worst line
  4  per-line error envelopes do not abort the batch
  5  --batch-concurrency parity on a 200-line corpus; requires --batch
  6  flag rules (--data vs --batch)
  7  no Generator: build_default_pipeline kwargs; no-SDK subprocess
  8  build failure is a single error envelope
  9  throughput bound
  10 discoverability (--help, README)

The golden JSONL is built at test time from eval/golden/dataset.jsonl and
eval/fixtures/replay.json (`generator_raw`). Offline: no network, no secrets;
the provider SDK is either a fake module or forced absent.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import reflect_mcp.pipeline as pipeline_mod
from tests._golden import frozen_replay, frozen_rows

REPO_ROOT = Path(__file__).resolve().parent.parent

# iter 66: party-role grounding moves inv-007 (bill-to vendor) to needs_review.
EXPECTED_STATUS = {
    "inv-001": "ok", "inv-002": "ok", "inv-003": "ok", "inv-005": "ok",
    "inv-004": "needs_review", "inv-006": "needs_review", "inv-007": "needs_review",
    "inv-008": "needs_review",
}

TEXT = "Invoice INV-1\nVendor: Acme\nDate: 2024-01-05\nTotal: 10.00 USD\n"
DATA = {
    "invoice_number": "INV-1",
    "vendor_name": "Acme",
    "invoice_date": "2024-01-05",
    "total": 10.0,
    "currency": "USD",
}
GEN_MODEL = "gen-model-x"
CRIT_MODEL = "crit-model-y"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _golden_rows() -> list[dict]:
    replay = frozen_replay()
    rows = []
    for row in frozen_rows():
        data = json.loads(replay[row["id"]]["generator_raw"])
        rows.append({"id": row["id"], "text": row["text"], "data": data})
    assert len(rows) == 8
    return rows


def _golden_jsonl(repeat: int = 1) -> str:
    rows = _golden_rows()
    out = []
    for _ in range(repeat):
        out.extend(json.dumps({"text": r["text"], "data": r["data"]}) for r in rows)
    return "\n".join(out) + "\n"


def _env(**extra: str) -> dict:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT / "src")}
    env.update(extra)
    return env


def _run(args: list[str], stdin: str | None = None, block_sdk: bool = False, **env_extra: str):
    if block_sdk:
        cmd = [sys.executable, "-c",
               "import sys, runpy; sys.modules['anthropic'] = None; "
               "sys.argv = ['reflect_mcp'] + sys.argv[1:]; "
               "runpy.run_module('reflect_mcp', run_name='__main__')", *args]
    else:
        cmd = [sys.executable, "-m", "reflect_mcp", *args]
    return subprocess.run(
        cmd, input=stdin if stdin is not None else "", capture_output=True, text=True,
        env=_env(**env_extra), cwd=str(REPO_ROOT), timeout=120,
    )


def _lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.strip()]


def _objs(stdout: str) -> list[dict]:
    return [json.loads(ln) for ln in _lines(stdout)]


def _strip(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k != "trace_id"}


def _write(tmp_path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


@pytest.fixture(scope="module")
def golden_batch(tmp_path_factory):
    d = tmp_path_factory.mktemp("golden")
    path = d / "golden.jsonl"
    path.write_text(_golden_jsonl())
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", str(path)])
    return proc


@pytest.fixture(scope="module")
def corpus200(tmp_path_factory):
    d = tmp_path_factory.mktemp("c200")
    path = d / "c200.jsonl"
    path.write_text(_golden_jsonl(repeat=25))
    return str(path)


# ---------------------------------------------------------------------------
# EB-1: corpus run
# ---------------------------------------------------------------------------

def test_eb1_one_compact_line_per_input_line(golden_batch):
    proc = golden_batch
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    raw = proc.stdout.splitlines()
    nonblank = [ln for ln in raw if ln.strip()]
    assert len(nonblank) == 8, proc.stdout
    for ln in nonblank:
        obj = json.loads(ln)  # each line is a full JSON object on its own
        assert isinstance(obj, dict)
        assert "status" in obj


def test_eb1_order_preserved(golden_batch):
    got = [o["status"] for o in _objs(golden_batch.stdout)]
    want = [EXPECTED_STATUS[r["id"]] for r in _golden_rows()]
    assert got == want


def test_eb1_stdin_equals_file(golden_batch):
    proc = _run(["validate", "--batch", "--heuristics-only"], stdin=_golden_jsonl())
    assert proc.returncode == golden_batch.returncode == 1
    assert len(_objs(proc.stdout)) == 8
    assert [_strip(o) for o in _objs(proc.stdout)] == [_strip(o) for o in _objs(golden_batch.stdout)]


def test_eb1_blank_lines_skipped(tmp_path, golden_batch):
    # "Blank" read as the existing extract line contract (iter-38 EB-4): empty
    # lines. Whitespace-only lines are out of this test's scope (see tester.md).
    lines = _golden_jsonl().splitlines()
    padded = "\n\n" + "\n\n".join(lines) + "\n\n\n"
    path = _write(tmp_path, "padded.jsonl", padded)
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", path])
    assert len(_lines(proc.stdout)) == 8
    assert [_strip(o) for o in _objs(proc.stdout)] == [_strip(o) for o in _objs(golden_batch.stdout)]


# ---------------------------------------------------------------------------
# EB-2: same result as single mode
# ---------------------------------------------------------------------------

def test_eb2_golden_statuses(golden_batch):
    objs = _objs(golden_batch.stdout)
    assert len(objs) == 8, golden_batch.stdout + golden_batch.stderr
    for row, obj in zip(_golden_rows(), objs):
        assert obj["status"] == EXPECTED_STATUS[row["id"]], (row["id"], obj)


def test_eb2_each_line_equals_single_document_validate(tmp_path, golden_batch):
    objs = _objs(golden_batch.stdout)
    assert len(objs) == 8
    for i, (row, obj) in enumerate(zip(_golden_rows(), objs)):
        text_path = _write(tmp_path, f"doc{i}.txt", row["text"])
        single = _run(["validate", "--heuristics-only", "--file", text_path,
                       "--data", json.dumps(row["data"])])
        assert single.returncode in (0, 1), single.stderr
        single_obj = json.loads(single.stdout)
        assert _strip(obj) == _strip(single_obj), row["id"]


# ---------------------------------------------------------------------------
# EB-3: exit code is the worst line
# ---------------------------------------------------------------------------

def test_eb3_golden_exits_1(golden_batch):
    assert golden_batch.returncode == 1


def test_eb3_all_ok_exits_0(tmp_path):
    ok = [r for r in _golden_rows() if EXPECTED_STATUS[r["id"]] == "ok"]
    body = "\n".join(json.dumps({"text": r["text"], "data": r["data"]}) for r in ok) + "\n"
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", _write(tmp_path, "ok.jsonl", body)])
    assert proc.returncode == 0, proc.stdout
    assert [o["status"] for o in _objs(proc.stdout)] == ["ok"] * len(ok)


def test_eb3_any_error_exits_3(tmp_path):
    body = _golden_jsonl() + "not json\n"
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", _write(tmp_path, "e.jsonl", body)])
    assert proc.returncode == 3
    assert len(_lines(proc.stdout)) == 9


# ---------------------------------------------------------------------------
# EB-4: per-line errors don't abort
# ---------------------------------------------------------------------------

GOOD_LINE = json.dumps({"text": TEXT, "data": DATA})


@pytest.mark.parametrize(
    "bad, check",
    [
        ("{not json", lambda m: m.startswith("invalid JSON:")),
        (json.dumps([1, 2, 3]), lambda m: m == 'missing "text" key'),
        (json.dumps("just a string"), lambda m: m == 'missing "text" key'),
        (json.dumps({"data": DATA}), lambda m: m == 'missing "text" key'),
        (json.dumps({"text": TEXT}), lambda m: m == 'missing "data" key'),
        (json.dumps({"text": TEXT, "data": {"total": "abc"}}), lambda m: "total" in m),
        (json.dumps({"text": TEXT, "data": [1, 2]}), lambda m: bool(m.strip())),
    ],
    ids=["invalid-json", "list", "string", "no-text", "no-data", "bad-total", "data-list"],
)
def test_eb4_bad_line_envelope_in_position(tmp_path, bad, check):
    body = GOOD_LINE + "\n" + bad + "\n" + GOOD_LINE + "\n"
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", _write(tmp_path, "b.jsonl", body)])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    objs = _objs(proc.stdout)
    assert len(objs) == 3, proc.stdout
    assert objs[0]["status"] != "error"
    assert objs[1]["status"] == "error"
    assert isinstance(objs[1]["error_message"], str)
    assert check(objs[1]["error_message"]), objs[1]["error_message"]
    assert objs[2]["status"] != "error"
    assert _strip(objs[0]) == _strip(objs[2])


def test_eb4_bad_data_differs_from_good_data(tmp_path):
    """Counterfactual: the `data` really is validated per line (not ignored)."""
    body = json.dumps({"text": TEXT, "data": {"total": "abc"}}) + "\n" + GOOD_LINE + "\n"
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", _write(tmp_path, "c.jsonl", body)])
    statuses = [o["status"] for o in _objs(proc.stdout)]
    assert statuses[0] == "error" and statuses[1] != "error"


# ---------------------------------------------------------------------------
# EB-5: concurrency
# ---------------------------------------------------------------------------

def test_eb5_concurrency_matches_serial_200(corpus200):
    serial = _run(["validate", "--batch", "--heuristics-only", "--file", corpus200])
    conc = _run(["validate", "--batch", "--heuristics-only", "--batch-concurrency", "4", "--file", corpus200])
    s_objs, c_objs = _objs(serial.stdout), _objs(conc.stdout)
    assert len(s_objs) == 200
    assert [_strip(o) for o in c_objs] == [_strip(o) for o in s_objs]
    assert conc.returncode == serial.returncode == 1


def test_eb5_concurrency_without_batch_exits_2(tmp_path):
    text_path = _write(tmp_path, "t.txt", TEXT)
    proc = _run(["validate", "--heuristics-only", "--file", text_path,
                 "--data", json.dumps(DATA), "--batch-concurrency", "4"])
    assert proc.returncode == 2
    assert "--batch-concurrency requires --batch" in proc.stderr


# ---------------------------------------------------------------------------
# EB-6: flag rules
# ---------------------------------------------------------------------------

def test_eb6_data_with_batch_exits_2(tmp_path):
    path = _write(tmp_path, "g.jsonl", GOOD_LINE + "\n")
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", path, "--data", json.dumps(DATA)])
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert "--data" in proc.stderr
    assert proc.stdout == ""


def test_eb6_neither_data_nor_batch_exits_2(tmp_path):
    proc = _run(["validate", "--heuristics-only", "--file", _write(tmp_path, "t.txt", TEXT)])
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert "--data" in proc.stderr


def test_eb6_single_mode_unchanged(tmp_path):
    text_path = _write(tmp_path, "t.txt", TEXT)
    proc = _run(["validate", "--heuristics-only", "--file", text_path, "--data", json.dumps(DATA)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB-7: no Generator, no SDK
# ---------------------------------------------------------------------------

def _critique() -> str:
    verdicts = [
        {"field": f, "confidence": 0.99, "grounded": True, "rationale": "matches source"}
        for f in DATA
    ]
    return json.dumps({"field_verdicts": verdicts, "overall_confidence": 0.99, "summary": "ok"})


class _Counter:
    def __init__(self) -> None:
        self.built = 0
        self.calls: list = []


@pytest.fixture
def fake_sdk(monkeypatch):
    counter = _Counter()
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            model = kwargs.get("model")
            counter.calls.append(model)
            text = json.dumps(DATA) if model == GEN_MODEL else _critique()
            block = types.SimpleNamespace(text=text, type="text")
            return types.SimpleNamespace(content=[block], stop_reason="end_turn")

    class Anthropic:
        def __init__(self, *args, **kwargs) -> None:
            counter.built += 1
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.delenv("REFLECT_PROVIDER", raising=False)
    monkeypatch.delenv("REFLECT_FIELD_THRESHOLD", raising=False)
    monkeypatch.delenv("REFLECT_OVERALL_THRESHOLD", raising=False)
    monkeypatch.setenv("REFLECT_GENERATOR_MODEL", GEN_MODEL)
    monkeypatch.setenv("REFLECT_CRITIC_MODEL", CRIT_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    return counter


@pytest.fixture
def recorder(fake_sdk, monkeypatch):
    real = pipeline_mod.build_default_pipeline
    seen: list[dict] = []

    def _rec(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "build_default_pipeline", _rec)
    for name in ("reflect_mcp", "reflect_mcp.tools", "reflect_mcp.cli", "reflect_mcp.__main__"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if getattr(mod, "build_default_pipeline", None) is real:
            monkeypatch.setattr(mod, "build_default_pipeline", _rec)
    return seen


def _run_main(monkeypatch, argv: list[str], stdin: str | None = None) -> int:
    from reflect_mcp.__main__ import main

    monkeypatch.setattr(sys, "argv", ["reflect-mcp", *argv])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin or ""))
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code
    return 0 if rc is None else rc


N_LINES = 5


def _batch_input(n: int = N_LINES) -> str:
    return "\n".join([GOOD_LINE] * n) + "\n"


@pytest.mark.parametrize("ho", [False, True])
@pytest.mark.parametrize("conc", [1, 4])
def test_eb7_every_build_has_generator_disabled(recorder, monkeypatch, capsys, ho, conc):
    argv = ["validate", "--batch", "--batch-concurrency", str(conc)]
    if ho:
        argv.append("--heuristics-only")
    rc = _run_main(monkeypatch, argv, stdin=_batch_input())
    out = capsys.readouterr().out
    assert rc in (0, 1), out
    assert len(_lines(out)) == N_LINES
    assert recorder, "factory never called"
    for kw in recorder:
        assert kw.get("generator_enabled", "MISSING") is False, kw
        assert kw.get("critic_enabled", True) is (not ho), kw


def test_eb7_critic_on_concurrency_1_builds_once(recorder, fake_sdk, monkeypatch, capsys):
    rc = _run_main(monkeypatch, ["validate", "--batch"], stdin=_batch_input())
    out = capsys.readouterr().out
    assert rc == 0, out
    objs = _objs(out)
    assert [o["status"] for o in objs] == ["ok"] * N_LINES
    assert len(recorder) == 1
    # Critic only: one client, N critic calls, zero generator calls.
    assert fake_sdk.built == 1
    assert GEN_MODEL not in fake_sdk.calls
    assert fake_sdk.calls.count(CRIT_MODEL) == N_LINES


def test_eb7_extract_batch_kwargs_unchanged(recorder, monkeypatch, capsys):
    """Extract batch keeps HEAD's kwargs (no generator_enabled)."""
    stdin = "\n".join([json.dumps({"text": TEXT})] * 2) + "\n"
    _run_main(monkeypatch, ["extract", "--batch"], stdin=stdin)
    capsys.readouterr()
    assert recorder
    for kw in recorder:
        assert "generator_enabled" not in kw


@pytest.mark.parametrize("provider", [None, "bogus"])
def test_eb7_no_sdk_subprocess(tmp_path, provider):
    path = _write(tmp_path, "g.jsonl", _golden_jsonl())
    extra = {"REFLECT_PROVIDER": provider} if provider else {}
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", path], block_sdk=True, **extra)
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    objs = _objs(proc.stdout)
    assert len(objs) == 8
    assert all(o["status"] != "error" for o in objs)


def test_eb7_counterfactual_critic_on_no_sdk_errors(tmp_path):
    """The no-SDK arm is not vacuous: with the Critic on, the build fails."""
    path = _write(tmp_path, "g.jsonl", _golden_jsonl())
    proc = _run(["validate", "--batch", "--file", path], block_sdk=True)
    assert proc.returncode == 3


def test_eb7_cli_run_validate_accepts_batch(tmp_path, monkeypatch, capsys):
    """Acceptance: cli._build_validate_parser path (cli.run_validate(argv)) also takes --batch."""
    from reflect_mcp import cli

    path = _write(tmp_path, "g.jsonl", _golden_jsonl())
    try:
        rc = cli.run_validate(["--batch", "--heuristics-only", "--batch-concurrency", "2", "--file", path])
    except SystemExit as e:
        rc = e.code
    out = capsys.readouterr().out
    assert rc == 1
    assert [o["status"] for o in _objs(out)] == [EXPECTED_STATUS[r["id"]] for r in _golden_rows()]


# ---------------------------------------------------------------------------
# EB-8: build failure is one envelope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conc", ["1", "4"])
@pytest.mark.parametrize("empty", [False, True])
def test_eb8_bogus_provider_single_envelope(tmp_path, conc, empty):
    body = "" if empty else _golden_jsonl()
    path = _write(tmp_path, "g.jsonl", body)
    proc = _run(["validate", "--batch", "--batch-concurrency", conc, "--file", path],
                REFLECT_PROVIDER="bogus")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    objs = _objs(proc.stdout)
    assert len(objs) == 1, proc.stdout
    assert objs[0]["status"] == "error"
    msg = objs[0]["error_message"]
    assert "bogus" in msg or "REFLECT_PROVIDER" in msg, msg


def test_eb8_empty_input_heuristics_only(tmp_path):
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", _write(tmp_path, "e.jsonl", "")])
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
    proc2 = _run(["validate", "--batch", "--heuristics-only"], stdin="")
    assert proc2.returncode == 0
    assert proc2.stdout.strip() == ""


# ---------------------------------------------------------------------------
# EB-9: throughput
# ---------------------------------------------------------------------------

def test_eb9_200_lines_under_5s(corpus200):
    t0 = time.monotonic()
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", corpus200])
    elapsed = time.monotonic() - t0
    assert len(_lines(proc.stdout)) == 200
    assert proc.returncode == 1
    assert elapsed < 5.0, elapsed


# ---------------------------------------------------------------------------
# EB-10: discoverability
# ---------------------------------------------------------------------------

def test_eb10_validate_help_lists_batch_flags():
    proc = _run(["validate", "--help"])
    assert proc.returncode == 0
    options = proc.stdout.split("options:", 1)[-1]
    assert "--batch-concurrency" in options
    # The --batch option entry (not --batch-concurrency) mentions `data`.
    m = re.search(r"^\s+--batch\s+(.*?)(?=^\s+--|\Z)", options, re.M | re.S)
    assert m, options
    assert "data" in m.group(1)


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text()


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:bash|sh|shell)\n(.*?)```", text, re.S)


def test_eb10_readme_validate_batch_examples_parse():
    from reflect_mcp.__main__ import _build_top_parser

    cmds = []
    for block in _bash_blocks(_readme()):
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            seg = line.split("|")[-1].strip()
            if seg.startswith("reflect-mcp validate") and "--batch" in shlex.split(seg):
                cmds.append(seg)
    assert cmds, "no `reflect-mcp validate --batch` example in README bash blocks"
    for cmd in cmds:
        tokens = shlex.split(cmd)
        try:
            ns = _build_top_parser().parse_args(tokens[1:])
        except SystemExit as e:  # pragma: no cover - failure path
            pytest.fail(f"README example does not parse: {cmd!r} ({e.code})")
        assert getattr(ns, "batch", False) is True, cmd


def test_eb10_readme_batch_paragraph_states_line_shape():
    paras = [p for p in _readme().split("\n\n") if p.startswith("**Batch mode**")]
    assert paras, "no Batch mode paragraph"
    para = paras[0]
    assert "validate --batch" in para
    assert '{"text": "...", "data": {...}}' in para


def test_eb10_readme_heuristics_only_including_batch_mode_kept():
    paras = [p for p in _readme().split("\n\n") if p.startswith("**Heuristics-only mode**")]
    assert paras
    assert "including batch mode" in paras[0]
