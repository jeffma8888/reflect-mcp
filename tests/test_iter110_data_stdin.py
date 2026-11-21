"""Black-box behavioral tests for iter-110: ``validate --data -`` reads stdin.

``reflect-mcp validate --file DOC --data -`` reads the ``--data`` JSON (a bare
fields object or a stored result envelope) from stdin; the source text comes
from ``--file PATH``.

  1  bare fields on stdin (with / without trailing newline) == inline --data
  2  pretty-printed stored ok envelope on stdin == same text inline
  3  stored error envelope on stdin -> rc 2, schema error, no "status": "ok"
  4  empty / non-JSON stdin -> rc 2, ``Error: --data is not valid JSON``
  5  ``{"total": true}`` on stdin -> rc 2 schema error, same as inline
  6  ``--data -`` without --file, and ``--file - --data -`` -> rc 2, message
  7  ``--batch --data -`` never reads stdin: exits 2 on an OPEN pipe
  8  inline --data unchanged; ``--file -`` + inline data; extract has no --data
  9  extract/validate/top-level --help byte-identical to the base tree
 10  README section 5 documents ``--data -`` with a pipe recipe

Offline, no keys, subprocesses only, no git at test time, no wall-clock bound
(EB7 uses an 8 s *hang guard*, not a timing assertion).

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

TEXT = "ACME CORP\nInvoice INV-1\nDate 2024-01-05\nTotal 108.50\n"
FIELDS = '{"vendor_name":"ACME CORP","total":108.5}'

# Measured on a `git archive` export of the pre-change tree with COLUMNS=80,
# after normalising argparse's 3.10 "optional arguments:" title to "options:".
HELP_SHA256 = {
    "extract": "446711f8a2c1ee58a6eb4301fae50c35ddf1cc73928ef68a12d3d9267eff3bc9",
    "validate": "6c318aec864ab2a63454240613d679b5c4222c5d8011163a3508eabc64bcc7a6",
    "": "6379969155d9616fd0ef04d5508fadbba9d0d6a92fd81e1cbca49a17be23927f",
}

NEEDS_FILE = "--data - reads the fields from stdin, so pass the source text with --file PATH."
EXCLUSIVE = "Error: validate needs exactly one of --data or --batch."


def _env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("REFLECT_") or k.endswith("_API_KEY") or k.startswith("AWS_"))
    }
    env["COLUMNS"] = "80"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *args],
        input=(stdin if stdin is not None else "").encode("utf-8"),
        capture_output=True,
        env=_env(),
        cwd=str(REPO_ROOT),
        timeout=120,
    )


def _out(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout.decode("utf-8")


def _err(proc: subprocess.CompletedProcess) -> str:
    return proc.stderr.decode("utf-8")


def _doc(proc: subprocess.CompletedProcess) -> dict:
    out = _out(proc)
    assert out.strip(), f"empty stdout; stderr={_err(proc)!r}"
    doc = json.loads(out)
    doc.pop("trace_id", None)
    assert len(doc) > 2, doc  # guard: parity on a real envelope, not a stub
    return doc


@pytest.fixture()
def doc_path(tmp_path: Path) -> str:
    p = tmp_path / "invoice.txt"
    p.write_text(TEXT, encoding="utf-8")
    return str(p)


def _validate(doc_path: str, data: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return _run(["validate", "--file", doc_path, "--data", data, "--heuristics-only"], stdin)


# --------------------------------------------------------------------------- EB1


@pytest.mark.parametrize("suffix", ["", "\n"])
def test_eb1_bare_fields_on_stdin_match_inline(doc_path, suffix):
    piped = _validate(doc_path, "-", FIELDS + suffix)
    inline = _validate(doc_path, FIELDS)
    assert piped.returncode == 0, _err(piped)
    doc = _doc(piped)
    assert doc["status"] == "ok"
    assert doc["data"]["vendor_name"] == "ACME CORP"
    assert doc["data"]["total"] == 108.5
    assert inline.returncode == piped.returncode
    assert _doc(inline) == doc


def test_eb1_counterpart_stdin_is_what_gets_verified(doc_path):
    # A different stdin object changes the verified data: stdin is really read.
    other = _validate(doc_path, "-", '{"vendor_name":"OTHER LTD","total":108.5}')
    assert other.returncode in (0, 1), _err(other)
    assert _doc(other)["data"]["vendor_name"] == "OTHER LTD"


# --------------------------------------------------------------------------- EB2


def test_eb2_pretty_stored_ok_envelope_on_stdin(doc_path):
    first = _validate(doc_path, FIELDS)
    assert first.returncode == 0, _err(first)
    envelope = json.loads(_out(first))
    assert envelope["status"] == "ok" and isinstance(envelope.get("data"), dict)
    text = json.dumps(envelope, indent=2)
    assert text.count("\n") > 3  # several lines
    piped = _validate(doc_path, "-", text)
    inline = _validate(doc_path, text)
    assert piped.returncode == inline.returncode == 0, _err(piped)
    assert _doc(piped) == _doc(inline)
    assert _doc(piped)["data"]["total"] == 108.5


# --------------------------------------------------------------------------- EB3


def test_eb3_error_envelope_on_stdin_is_schema_error(doc_path):
    proc = _validate(doc_path, "-", '{"status":"error","error_message":"x"}')
    assert proc.returncode == 2, (_out(proc), _err(proc))
    err = _err(proc)
    assert "Error: --data failed InvoiceData validation" in err
    assert "has no data object to verify" in err
    assert '"status": "ok"' not in _out(proc)


# --------------------------------------------------------------------------- EB4


@pytest.mark.parametrize("stdin", ["", "not json"])
def test_eb4_empty_or_non_json_stdin(doc_path, stdin):
    proc = _validate(doc_path, "-", stdin)
    assert proc.returncode == 2, (_out(proc), _err(proc))
    assert "Error: --data is not valid JSON" in _err(proc)
    assert '"status": "ok"' not in _out(proc)


# --------------------------------------------------------------------------- EB5


def test_eb5_schema_invalid_on_stdin_matches_inline(doc_path):
    piped = _validate(doc_path, "-", '{"total": true}')
    inline = _validate(doc_path, '{"total": true}')
    assert piped.returncode == inline.returncode == 2
    assert "Error: --data failed InvoiceData validation" in _err(piped)
    assert "Error: --data failed InvoiceData validation" in _err(inline)
    assert _out(piped) == _out(inline)


# --------------------------------------------------------------------------- EB6


@pytest.mark.parametrize(
    "args",
    [
        ["validate", "--data", "-", "--heuristics-only"],
        ["validate", "--file", "-", "--data", "-", "--heuristics-only"],
    ],
    ids=["no-file", "file-dash"],
)
def test_eb6_data_dash_requires_file(args):
    proc = _run(args, FIELDS)
    assert proc.returncode == 2, (_out(proc), _err(proc))
    assert NEEDS_FILE in _err(proc)
    assert _out(proc) == ""


# --------------------------------------------------------------------------- EB7


def _run_open_stdin(args: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "reflect_mcp", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
        cwd=str(REPO_ROOT),
    )
    try:
        try:
            rc = proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(f"blocked on an open stdin pipe: {args}")
        return rc, proc.stderr.read().decode("utf-8")
    finally:
        for f in (proc.stdin, proc.stdout, proc.stderr):
            if f is not None:
                f.close()


@pytest.mark.parametrize("with_file", [True, False], ids=["file", "no-file"])
def test_eb7_batch_data_dash_never_reads_stdin(doc_path, with_file):
    args = ["validate", "--batch", "--data", "-"]
    if with_file:
        args += ["--file", doc_path]
    args.append("--heuristics-only")
    rc, err = _run_open_stdin(args)
    assert rc == 2, err
    assert EXCLUSIVE in err


def test_eb7_counterpart_non_batch_data_dash_reads_stdin(doc_path):
    # Without --batch the same flags DO read stdin: EOF-terminated input works.
    proc = _validate(doc_path, "-", FIELDS)
    assert proc.returncode == 0, _err(proc)
    assert EXCLUSIVE not in _err(proc)


# --------------------------------------------------------------------------- EB8


def test_eb8_inline_data_unchanged(doc_path):
    proc = _validate(doc_path, FIELDS)
    assert proc.returncode == 0, _err(proc)
    doc = _doc(proc)
    assert doc["status"] == "ok"
    assert doc["data"]["vendor_name"] == "ACME CORP" and doc["data"]["total"] == 108.5
    bad = _validate(doc_path, "not json")
    assert bad.returncode == 2 and "Error: --data is not valid JSON" in _err(bad)


def test_eb8_file_dash_with_inline_data_reads_text_from_stdin(doc_path):
    via_stdin = _run(["validate", "--file", "-", "--data", FIELDS, "--heuristics-only"], TEXT)
    via_file = _validate(doc_path, FIELDS)
    assert via_stdin.returncode == via_file.returncode == 0, _err(via_stdin)
    assert _doc(via_stdin) == _doc(via_file)


def test_eb8_extract_has_no_data_flag(doc_path):
    proc = _run(["extract", "--file", doc_path, "--data", "-"], FIELDS)
    assert proc.returncode == 2, (_out(proc), _err(proc))
    assert "unrecognized arguments" in _err(proc)
    assert _out(proc) == ""


# --------------------------------------------------------------------------- EB9


@pytest.mark.parametrize("verb", ["extract", "validate", ""], ids=["extract", "validate", "top"])
def test_eb9_help_byte_identical_to_base(verb):
    proc = _run([verb, "--help"] if verb else ["--help"])
    assert proc.returncode == 0
    text = _out(proc).replace("\noptional arguments:\n", "\noptions:\n")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == HELP_SHA256[verb], text


# --------------------------------------------------------------------------- EB10


def _readme_section5() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n## 5.")
    end = readme.index("\n## ", start + 1)
    return readme[start:end]


def test_eb10_readme_documents_data_dash_pipe():
    sec = _readme_section5()
    hits = [p for p in sec.split("\n\n") if "--data -" in p]
    assert hits, "README section 5 must document `--data -`"
    para = hits[0]
    assert "| reflect-mcp validate --file invoice.txt --data -" in para
    assert "stdin" in para
    assert "--file" in para and "required" in para
