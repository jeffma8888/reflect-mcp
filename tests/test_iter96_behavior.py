"""Black-box behavioral tests for iter-96: ``--batch`` ignores a line-1 UTF-8 BOM
and skips whitespace-only lines (roadmap #46).

Covers Expected Behaviors 1-10 from the iter-96 spec:
  1  a BOM on line 1 is ignored: stdout + exit byte-identical to the BOM-free run
     (both verbs, stdin and --file, --batch-concurrency 1 and 3)
  2  the id echo survives the BOM (first key "id" == "a")
  3  a BOM-only first line emits no row
  4  only line 1 is stripped: a BOM on line 2+ keeps its HEAD error row, exit 3
  5  whitespace-only lines (spaces, tabs, \\r) emit no row; other rows keep order
  6  exit code ignores skipped lines; an all-blank input prints nothing, exit 0
  7  malformed lines still produce their HEAD error rows, exit 3
  8  non-batch extract/validate unchanged
  9  footprint: cli.py stays 306 lines; frozen eval stdout byte-identical to HEAD
 10  README Batch mode paragraph mentions BOM and whitespace-only

HEAD constants (frozen-eval stdout digest, single-document validate digest)
were measured by running the product from an exported copy of the
pre-change commit; no git command runs at test time.

Offline: a deterministic fake pipeline is patched in at
``reflect_mcp.pipeline.build_default_pipeline``, or the real heuristics-only
pipeline is used (no Critic, no Generator, no network, no secrets).

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from reflect_mcp.schema import ExtractionResult
from tests._golden import frozen_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
VERBS = ["run_extract", "run_validate"]
CONCS = [1, 3]
BOM = "\ufeff"

# Measured on an exported copy of the pre-iter-96 commit.
HEAD_FROZEN_EVAL_SHA256 = "9df9499d6aa3a9b96961beee122f3408e0c6477f485258e54d609030ac18a7f7"
HEAD_SINGLE_VALIDATE_SHA256 = "9fcf3c9910017cb2d639518169cb2b3b92ddb7f7b861c856a3de42d0d69aa6c5"
HEAD_CLI_LINES = 306

MID_BOM_ROW = (
    '{"status": "error", "error_message": "invalid JSON: Unexpected UTF-8 BOM '
    '(decode using utf-8-sig): line 1 column 1 (char 0)"}'
)
MISSING_TEXT_ROW = '{"status": "error", "error_message": "missing \\"text\\" key"}'
INVALID_JSON_ROW = '{"status": "error", "error_message": "invalid JSON: Expecting value: line 1 column 1 (char 0)"}'

GOOD_TEXT = "ACME CORP\nInvoice INV-1\nTotal: $10.00"
GOOD_DATA = {"vendor_name": "ACME CORP", "total": 10.0}
SINGLE_TEXT = "ACME CORP\n   \n\t\nInvoice INV-1\nDate: 2026-03-14\nTotal: $10.00\n"
SINGLE_DATA = {"vendor_name": "ACME CORP", "invoice_number": "INV-1", "total": 10.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(status: str, text: str) -> ExtractionResult:
    return ExtractionResult(
        status=status,
        data={},
        field_confidence={},
        overall_confidence=0.9 if status == "ok" else 0.3,
        flagged_fields=[],
        critic_summary=f"summary:{text}",
        trace_id=f"trace:{text}",
        notes="",
    )


class FakePipeline:
    """Deterministic: result depends only on text; 'review' -> needs_review."""

    def __init__(self):
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _go(self, text):
        with self._lock:
            self.calls.append(text)
        return _result("needs_review" if "review" in text else "ok", text)

    def extract(self, text, *a, **k):
        return self._go(text)

    def validate(self, text, *a, **k):
        return self._go(text)


def _run(func_name: str, args: list[str], stdin: str = "", factory=None):
    """Run a cli entry point in-process; return (code, stdout, stderr)."""
    import reflect_mcp.cli as cli

    func = getattr(cli, func_name)
    out, err = io.StringIO(), io.StringIO()
    patches = [patch("sys.stdin", io.StringIO(stdin)), patch("sys.stdout", out), patch("sys.stderr", err)]
    if factory is not None:
        patches.append(patch("reflect_mcp.pipeline.build_default_pipeline", side_effect=factory))
    for p in patches:
        p.start()
    try:
        code = func(args)
    finally:
        for p in reversed(patches):
            p.stop()
    return code, out.getvalue(), err.getvalue()


def _fake(verb: str, stdin: str, c: int = 1):
    fake = FakePipeline()
    return _run(verb, ["--batch", "--batch-concurrency", str(c)], stdin, factory=lambda *a, **k: fake)


_TRACE = re.compile(r'"trace_id": "[^"]*"')


def _mask(out: str) -> str:
    return _TRACE.sub('"trace_id": "<T>"', out)


def _real(verb: str, c: int, route: str, payload: str, tmp_path: Path, name: str):
    """Real offline run: extract via the fake (no provider), validate heuristics-only."""
    args = ["--batch", "--batch-concurrency", str(c)]
    factory = None
    if verb == "run_validate":
        args.append("--heuristics-only")
    else:
        fake = FakePipeline()
        factory = lambda *a, **k: fake  # noqa: E731
    stdin = ""
    if route == "file":
        f = tmp_path / f"{name}.jsonl"
        f.write_bytes(payload.encode("utf-8"))
        args += ["--file", str(f)]
    else:
        stdin = payload
    code, out, _ = _run(verb, args, stdin, factory=factory)
    return code, _mask(out)


def _rec(text: str, **extra) -> str:
    rec = dict(extra)
    rec["text"] = text
    rec.setdefault("data", GOOD_DATA)
    return json.dumps(rec)


def _rows(out: str) -> list[str]:
    return [r for r in out.split("\n") if r]


CORPUS = "\n".join([
    _rec(GOOD_TEXT, id="a"),
    _rec(GOOD_TEXT + "\nreview me", id="b"),
    _rec(GOOD_TEXT),
]) + "\n"


# ---------------------------------------------------------------------------
# EB-1: BOM on line 1 is ignored, byte-identical to the BOM-free run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("route", ["stdin", "file"])
@pytest.mark.parametrize("verb", VERBS)
def test_eb1_leading_bom_byte_identical(verb, route, c, tmp_path):
    code_plain, out_plain = _real(verb, c, route, CORPUS, tmp_path, "plain")
    code_bom, out_bom = _real(verb, c, route, BOM + CORPUS, tmp_path, "bom")
    assert "BOM" not in out_bom
    assert len(_rows(out_bom)) == 3, out_bom
    assert all(json.loads(r).get("status") != "error" for r in _rows(out_bom)), out_bom
    assert out_bom == out_plain
    assert code_bom == code_plain


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb1_fake_bom_identical_and_text_unmangled(verb, c):
    plain = _fake(verb, CORPUS, c)
    bom = _fake(verb, BOM + CORPUS, c)
    assert bom == plain
    assert bom[0] == 1  # one needs_review row; worst severity, not 3
    assert BOM not in bom[1]


def test_eb1_real_validate_bom_corpus_exits_on_data(tmp_path):
    code, out = _real("run_validate", 1, "file", BOM + _rec(GOOD_TEXT, id="a") + "\n", tmp_path, "one")
    rows = _rows(out)
    assert len(rows) == 1
    assert json.loads(rows[0])["status"] == "ok", out
    assert code == 0


# ---------------------------------------------------------------------------
# EB-2: id echo survives the BOM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb2_id_first_key_after_bom(verb, c):
    _, out, _ = _fake(verb, BOM + _rec("doc", id="a") + "\n", c)
    rows = _rows(out)
    assert len(rows) == 1, out
    pairs = json.loads(rows[0], object_pairs_hook=lambda p: p)
    assert pairs[0] == ("id", "a")
    _, out_plain, _ = _fake(verb, _rec("doc", id="a") + "\n", c)
    assert out == out_plain


# ---------------------------------------------------------------------------
# EB-3: BOM-only first line is skipped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb3_bom_only_first_line_skipped(verb, c):
    plain = _fake(verb, CORPUS, c)
    got = _fake(verb, BOM + "\n" + CORPUS, c)
    assert got == plain
    assert len(_rows(got[1])) == 3


# ---------------------------------------------------------------------------
# EB-4: only line 1 is stripped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb4_mid_file_bom_still_errors(verb, c):
    first = _rec("doc one", id="a")
    third = _rec("doc three", id="c")
    code, out, _ = _fake(verb, first + "\n" + BOM + _rec("doc two", id="b") + "\n" + third + "\n", c)
    rows = _rows(out)
    assert len(rows) == 3, out
    assert rows[1] == MID_BOM_ROW
    _, ref, _ = _fake(verb, first + "\n" + third + "\n", c)
    assert [rows[0], rows[2]] == _rows(ref)
    assert code == 3


@pytest.mark.parametrize("verb", VERBS)
def test_eb4_bom_on_line1_and_line2_only_first_stripped(verb):
    code, out, _ = _fake(verb, BOM + _rec("doc", id="a") + "\n" + BOM + _rec("doc", id="b") + "\n")
    rows = _rows(out)
    assert len(rows) == 2
    assert json.loads(rows[0])["id"] == "a"
    assert rows[1] == MID_BOM_ROW
    assert code == 3


@pytest.mark.parametrize("verb", VERBS)
def test_eb4_double_bom_on_line1_is_not_widened(verb):
    """A one-shot prefix strip: a second U+FEFF on line 1 is still invalid JSON."""
    code, out, _ = _fake(verb, BOM + BOM + _rec("doc", id="a") + "\n")
    assert _rows(out) == [MID_BOM_ROW]
    assert code == 3


# ---------------------------------------------------------------------------
# EB-5: whitespace-only lines produce no row
# ---------------------------------------------------------------------------

WS_LINES = ["   ", "\t", " \r", "\r", " \t \r", ""]


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("ws", WS_LINES, ids=["spaces", "tab", "space_cr", "cr", "mixed", "empty"])
def test_eb5_whitespace_only_lines_emit_no_row(verb, c, ws):
    recs = [_rec(f"doc {i}", id=i) for i in range(4)]
    padded = ws + "\n" + recs[0] + "\n" + ws + "\n" + recs[1] + "\n" + recs[2] + "\n" + ws + "\n" + ws + "\n" + recs[3] + "\n" + ws + "\n"
    got = _fake(verb, padded, c)
    ref = _fake(verb, "\n".join(recs) + "\n", c)
    assert got == ref
    assert [json.loads(r)["id"] for r in _rows(got[1])] == [0, 1, 2, 3]
    assert got[0] == 0


@pytest.mark.parametrize("verb", VERBS)
def test_eb5_c1_c3_identical_with_padding(verb):
    recs = [_rec(f"doc {i}" + (" review" if i % 3 == 0 else ""), id=i) for i in range(9)]
    padded = "\n   \n".join(recs) + "\n\t\n"
    assert _fake(verb, padded, 1) == _fake(verb, padded, 3)


@pytest.mark.parametrize("verb", VERBS)
def test_eb5_crlf_corpus_with_blank_crlf_lines(verb):
    recs = [_rec("doc a", id="a"), _rec("doc b", id="b")]
    got = _fake(verb, recs[0] + "\r\n \r\n" + recs[1] + "\r\n")
    assert [json.loads(r)["id"] for r in _rows(got[1])] == ["a", "b"]
    assert got[0] == 0


# ---------------------------------------------------------------------------
# EB-6: exit code ignores skipped lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
def test_eb6_two_ok_with_spaces_between_exit_0(verb, c):
    code, out, _ = _fake(verb, _rec("doc a") + "\n   \n" + _rec("doc b") + "\n", c)
    assert len(_rows(out)) == 2
    assert all(json.loads(r)["status"] == "ok" for r in _rows(out))
    assert code == 0


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("payload", ["   \n\t\n \r\n", BOM + "\n   \n", BOM, BOM + "\n", ""],
                         ids=["ws_only", "bom_ws", "bom_bare", "bom_nl", "empty"])
def test_eb6_all_blank_input_prints_nothing_exit_0(verb, c, payload):
    code, out, _ = _fake(verb, payload, c)
    assert out == ""
    assert code == 0


def test_eb6_real_validate_ws_padded_file_exit_0(tmp_path):
    payload = _rec(GOOD_TEXT, id="a") + "\n   \n" + _rec(GOOD_TEXT, id="b") + "\n"
    code, out = _real("run_validate", 3, "file", BOM + payload, tmp_path, "pad")
    assert [json.loads(r)["id"] for r in _rows(out)] == ["a", "b"]
    assert code == 0


# ---------------------------------------------------------------------------
# EB-7: malformed lines still error
# ---------------------------------------------------------------------------

MALFORMED = [("not json", INVALID_JSON_ROW), ('"x"', MISSING_TEXT_ROW), ("5", MISSING_TEXT_ROW), ("null", MISSING_TEXT_ROW)]


@pytest.mark.parametrize("c", CONCS)
@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("line,expected", MALFORMED, ids=["invalid_json", "str", "num", "null"])
def test_eb7_malformed_lines_still_error(verb, c, line, expected):
    code, out, _ = _fake(verb, line + "\n", c)
    assert out == expected + "\n"
    assert code == 3
    code2, out2, _ = _fake(verb, "   \n" + _rec("doc", id="a") + "\n" + line + "\n\t\n", c)
    rows = _rows(out2)
    assert len(rows) == 2
    assert rows[1] == expected
    assert code2 == 3


@pytest.mark.parametrize("verb", VERBS)
def test_eb7_bom_then_malformed_line1(verb):
    """BOM-stripped line 1 that is still malformed gets the plain-JSON error, not a BOM one."""
    code, out, _ = _fake(verb, BOM + "not json\n")
    assert out == INVALID_JSON_ROW + "\n"
    assert code == 3


# ---------------------------------------------------------------------------
# EB-8: non-batch paths unchanged
# ---------------------------------------------------------------------------

def test_eb8_single_validate_unchanged_vs_head():
    code, out, _ = _run(
        "run_validate", ["--heuristics-only", "--data", json.dumps(SINGLE_DATA)], SINGLE_TEXT,
    )
    digest = hashlib.sha256(f"{code}\n{_mask(out)}".encode()).hexdigest()
    assert digest == HEAD_SINGLE_VALIDATE_SHA256, (code, out)


def test_eb8_single_extract_gets_whole_text_verbatim():
    fake = FakePipeline()
    text = BOM + "ACME\n   \n\tInvoice\n"
    code, out, _ = _run("run_extract", [], text, factory=lambda *a, **k: fake)
    assert fake.calls == [text]
    assert json.loads(out)["status"] == "ok"
    assert code == 0


# ---------------------------------------------------------------------------
# EB-9: footprint
# ---------------------------------------------------------------------------

def test_eb9_cli_line_count_unchanged():
    cli = REPO_ROOT / "src" / "reflect_mcp" / "cli.py"
    assert len(cli.read_text(encoding="utf-8").splitlines()) == HEAD_CLI_LINES


def test_eb9_frozen_eval_stdout_byte_identical_to_head():
    proc = frozen_eval()
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert hashlib.sha256(proc.stdout.encode()).hexdigest() == HEAD_FROZEN_EVAL_SHA256


# ---------------------------------------------------------------------------
# EB-10: README Batch mode paragraph
# ---------------------------------------------------------------------------

def _batch_paragraph() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    paras = [p for p in readme.split("\n\n") if p.lstrip().startswith("**Batch mode**")]
    assert len(paras) == 1, "expected exactly one **Batch mode** paragraph"
    return paras[0]


def test_eb10_readme_batch_paragraph_mentions_bom_and_whitespace_only():
    para = _batch_paragraph()
    assert "BOM" in para
    assert "whitespace-only" in para


# ---------------------------------------------------------------------------
# Self-check: this module stays committed-clone safe (no VCS calls, no machine paths)
# ---------------------------------------------------------------------------

def test_module_runs_no_vcs_and_has_no_machine_paths():
    src = Path(__file__).read_text(encoding="utf-8")
    for token in ("sub" + "process", "g" + "it " + "show", "g" + "it " + "archive", "/" + "Users/", "/" + "home/"):
        assert token not in src, f"forbidden token present: {token!r}"
