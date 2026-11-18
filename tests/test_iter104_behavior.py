"""Black-box behavioral tests for iter-104: ``--file -`` reads stdin.

``reflect-mcp extract|validate [--batch] --file -`` reads its input from stdin,
exactly as omitting ``--file`` does; a file literally named ``-`` stays
reachable as ``--file ./-``.

  1  single-row validate via ``--file -`` -> rc 0, status ok, total 250.0
  2  wrong total via ``--file -`` -> rc 1, needs_review, ``total`` flagged
  3  parity: ``--file -`` == no ``--file`` (stdout JSON minus trace_id, rc)
  4  batch: BOM + whitespace-only line, 2 rows a/ok b/needs_review, rc 1, parity
  5  empty stdin -> rc 2, ``Error: input is empty.``, same as no ``--file``
  6  extract ``--file -`` matches extract without ``--file`` (no SDK / key)
  7  a file literally named ``-`` is read via ``--file ./-``
  8  nonexistent path still rc 2 ``file not found``; a real file still read
  9  ``extract --help`` / ``validate --help`` byte-identical to the base tree
 10  README section 5 documents ``--file -`` and ``./-``

Offline, no keys, subprocesses only, no git at test time, no wall-clock bound.

Isolation contract: written from the spec, README, existing tests and the
product's own runtime behaviour only. No implementation source was read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT = "ACME Corp\nInvoice INV-1\nTotal: 250.00 USD\n"
DATA = {"vendor_name": "ACME Corp", "total": 250.0, "currency": "USD"}
BAD = {**DATA, "total": 999.0}
BATCH_TEXT = "ACME Corp Total: 250.00 USD"

# Measured from a `git archive` export of the pre-change tree with COLUMNS=80,
# after normalising argparse's 3.10 "optional arguments:" title to "options:".
HELP_SHA256 = {
    "extract": "446711f8a2c1ee58a6eb4301fae50c35ddf1cc73928ef68a12d3d9267eff3bc9",
    "validate": "6c318aec864ab2a63454240613d679b5c4222c5d8011163a3508eabc64bcc7a6",
}


def _env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.endswith("_API_KEY") or k.startswith("AWS_"))
    }
    env["COLUMNS"] = "80"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], stdin: bytes | None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        input=stdin if stdin is not None else b"",
        capture_output=True,
        env=_env(),
        cwd=str(cwd or REPO_ROOT),
        timeout=120,
    )


def _doc(proc: subprocess.CompletedProcess) -> dict:
    out = proc.stdout.decode("utf-8")
    assert out.strip(), f"empty stdout; stderr={proc.stderr!r}"
    return json.loads(out)


def _flagged(doc: dict) -> set[str]:
    return {f["field"] if isinstance(f, dict) else f for f in doc["flagged_fields"]}


def _strip(obj: dict) -> dict:
    obj = dict(obj)
    obj.pop("trace_id", None)
    return obj


def _validate_args(data: dict, *, dash: bool) -> list[str]:
    args = ["validate"]
    if dash:
        args += ["--file", "-"]
    return args + ["--data", json.dumps(data), "--heuristics-only"]


# --------------------------------------------------------------------------- EB1


def test_eb1_dash_reads_stdin_ok():
    proc = _run(_validate_args(DATA, dash=True), TEXT.encode())
    assert proc.returncode == 0, proc.stderr
    doc = _doc(proc)
    assert doc["status"] == "ok"
    assert doc["data"]["total"] == 250.0
    assert b"file not found" not in proc.stderr


def test_eb1_dash_equals_form_also_reads_stdin():
    # ``--file=-`` is the same argv token to argparse; reasonable reading of EB1.
    args = ["validate", "--file=-", "--data", json.dumps(DATA), "--heuristics-only"]
    proc = _run(args, TEXT.encode())
    assert proc.returncode == 0, proc.stderr
    assert _doc(proc)["status"] == "ok"


# --------------------------------------------------------------------------- EB2


def test_eb2_dash_wrong_total_needs_review():
    proc = _run(_validate_args(BAD, dash=True), TEXT.encode())
    assert proc.returncode == 1, proc.stderr
    doc = _doc(proc)
    assert doc["status"] == "needs_review"
    assert "total" in _flagged(doc)
    assert b"file not found" not in proc.stderr


# --------------------------------------------------------------------------- EB3


@pytest.mark.parametrize("data", [DATA, BAD], ids=["ok", "needs_review"])
def test_eb3_parity_with_plain_stdin(data):
    dash = _run(_validate_args(data, dash=True), TEXT.encode())
    plain = _run(_validate_args(data, dash=False), TEXT.encode())
    assert dash.returncode == plain.returncode
    a, b = _strip(_doc(dash)), _strip(_doc(plain))
    assert len(a) > 2, a  # non-vacuous: a real result document
    assert a == b


# --------------------------------------------------------------------------- EB4


def _batch_stdin() -> bytes:
    row_a = json.dumps({"id": "a", "text": BATCH_TEXT, "data": DATA})
    row_b = json.dumps({"id": "b", "text": BATCH_TEXT, "data": BAD})
    return b"\xef\xbb\xbf" + row_a.encode() + b"\n   \t\n" + row_b.encode() + b"\n"


def _batch_lines(proc: subprocess.CompletedProcess) -> list[dict]:
    lines = proc.stdout.decode("utf-8").splitlines()
    assert len(lines) == 2, proc.stdout
    return [json.loads(line) for line in lines]


def test_eb4_batch_dash_bom_and_blank_line():
    proc = _run(["validate", "--batch", "--heuristics-only", "--file", "-"], _batch_stdin())
    assert proc.returncode == 1, proc.stderr
    rows = _batch_lines(proc)
    assert [(r["id"], r["status"]) for r in rows] == [("a", "ok"), ("b", "needs_review")]


def test_eb4_batch_parity_with_plain_stdin():
    dash = _run(["validate", "--batch", "--heuristics-only", "--file", "-"], _batch_stdin())
    plain = _run(["validate", "--batch", "--heuristics-only"], _batch_stdin())
    assert dash.returncode == plain.returncode == 1
    a = [_strip(r) for r in _batch_lines(dash)]
    b = [_strip(r) for r in _batch_lines(plain)]
    assert a == b


# --------------------------------------------------------------------------- EB5


def test_eb5_empty_stdin_same_error():
    dash = _run(_validate_args(DATA, dash=True), b"")
    plain = _run(_validate_args(DATA, dash=False), b"")
    assert dash.returncode == 2
    assert "Error: input is empty." in dash.stderr.decode()
    assert "file not found" not in dash.stderr.decode()
    assert plain.returncode == 2
    assert "Error: input is empty." in plain.stderr.decode()


# --------------------------------------------------------------------------- EB6


def test_eb6_extract_dash_matches_plain():
    dash = _run(["extract", "--file", "-"], b"x")
    plain = _run(["extract"], b"x")
    assert "file not found" not in dash.stderr.decode()
    assert dash.returncode == plain.returncode
    a, b = _strip(_doc(dash)), _strip(_doc(plain))
    assert "status" in a
    assert (a.get("status"), a.get("error_message")) == (b.get("status"), b.get("error_message"))
    assert a == b


# --------------------------------------------------------------------------- EB7


def test_eb7_literal_dash_file_via_dot_slash(tmp_path):
    (tmp_path / "-").write_text(TEXT, encoding="utf-8")
    args = ["validate", "--file", "./-", "--data", json.dumps(DATA), "--heuristics-only"]
    proc = _run(args, b"", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _doc(proc)["status"] == "ok"


def test_eb7_literal_dash_file_content_is_what_is_read(tmp_path):
    # One-edit counterpart: the literal file disagrees with the data, while
    # stdin carries agreeing text -> the file (not stdin) decides the verdict.
    (tmp_path / "-").write_text("ACME Corp\nTotal: 999.00 USD\n", encoding="utf-8")
    args = ["validate", "--file", "./-", "--data", json.dumps(DATA), "--heuristics-only"]
    proc = _run(args, TEXT.encode(), cwd=tmp_path)
    assert proc.returncode == 1, proc.stderr
    assert _doc(proc)["status"] == "needs_review"


# --------------------------------------------------------------------------- EB8


def test_eb8_nonexistent_path_still_usage_error(tmp_path):
    missing = tmp_path / "nope.txt"
    args = ["validate", "--file", str(missing), "--data", json.dumps(DATA), "--heuristics-only"]
    proc = _run(args, TEXT.encode())
    assert proc.returncode == 2
    assert f"Error: file not found: {missing}" in proc.stderr.decode()


def test_eb8_real_file_still_read_over_stdin(tmp_path):
    f = tmp_path / "inv.txt"
    f.write_text(TEXT, encoding="utf-8")
    ok = _run(["validate", "--file", str(f), "--data", json.dumps(DATA), "--heuristics-only"], b"")
    assert ok.returncode == 0, ok.stderr
    assert _doc(ok)["status"] == "ok"
    bad = _run(["validate", "--file", str(f), "--data", json.dumps(BAD), "--heuristics-only"], b"")
    assert bad.returncode == 1
    assert "total" in _flagged(_doc(bad))


# --------------------------------------------------------------------------- EB9


@pytest.mark.parametrize("verb", ["extract", "validate"])
def test_eb9_help_byte_identical_to_base(verb):
    proc = _run([verb, "--help"], b"")
    assert proc.returncode == 0
    text = proc.stdout.decode("utf-8").replace("\noptional arguments:\n", "\noptions:\n")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == HELP_SHA256[verb], text


# --------------------------------------------------------------------------- EB10


def _readme_section5() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n## 5.")
    end = readme.index("\n## ", start + 1)
    return readme[start:end]


def test_eb10_readme_cli_section_documents_dash():
    sec = _readme_section5()
    hits = [p for p in sec.split("\n\n") if "--file -" in p and "./-" in p]
    assert hits, "README section 5 must document `--file -` (stdin) and `./-`"
    assert "stdin" in hits[0]
