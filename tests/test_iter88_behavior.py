"""Black-box behavioral tests for iter-88: ``--batch`` flushes each output row
as it is written, so a piped consumer sees finished rows immediately
(roadmap #63).

Covers the Expected Behaviors from pm.md:
  1. Serial streaming: one row written, stdin held open -> row readable < 5 s.
  2. Concurrent streaming (N=4): row 1, ~1 s pause, row 2, stdin held open ->
     row 1 readable < 5 s; after close, row 2 follows in input order.
  3. Same bytes: 3-line input (ok, needs_review, non-JSON) -> 3 compact JSON
     lines in input order, statuses ok/needs_review/error, exit 3.
  4. Exit contract: 5000-row file, reader closes after one line -> exit 0,
     never 120, no Traceback.
  5. README Batch mode paragraph says rows are ``flushed``; the ``| head -1``
     sentence still says the code is never more lenient and does not claim
     that at most one row may go unseen.
  6. Defuse: test_iter86 pins diff the fixed range ``BASE_SHA SHIP_SHA``
     (``SHIP_SHA = "0f27eeb"``); in a clone with this iteration COMMITTED its
     pins pass, while HEAD's old open-ended pin committed there fails.

Isolation contract: tests run the product as a subprocess
(``python -m reflect_mcp``) and read README.md only. No implementation source
was read. Every subprocess is killed in ``finally`` and every read is bounded
by ``select`` so no test can hang.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import ast
import shutil
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="select() on pipes is POSIX-only"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

TIMEOUT_S = 5.0


def _row(i: int) -> dict:
    return {
        "id": i,
        "text": f"Invoice INV-{i} from Acme. Total 10.00 USD",
        "data": {"vendor_name": "Acme", "total": 10.0},
    }


NR_ROW = {
    "text": "Invoice No: INV-1\nTotal: 10.00 USD",
    "data": {"invoice_number": "INV-1", "total": 99.0},
}


def _env() -> dict:
    """Subprocess env with the threshold-floor vars removed (they change status)."""
    env = dict(os.environ)
    for k in ("REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD"):
        env.pop(k, None)
    return env


def _spawn(extra: list[str] | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only", "--batch"]
    cmd += extra or []
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        env=_env(),
    )


def _send(proc: subprocess.Popen, obj: dict) -> None:
    proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _readable(proc: subprocess.Popen, timeout: float = TIMEOUT_S) -> bool:
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    return bool(ready)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:  # pragma: no cover - best effort cleanup
        pass
    for fh in (proc.stdin, proc.stdout):
        try:
            if fh is not None:
                fh.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# EB1: serial streaming
# ---------------------------------------------------------------------------

def test_eb1_serial_row_reaches_pipe_while_stdin_open():
    proc = _spawn()
    try:
        _send(proc, _row(1))
        t0 = time.monotonic()
        assert _readable(proc), "no row on stdout within 5 s while stdin is open"
        line = proc.stdout.readline().decode("utf-8")
        elapsed = time.monotonic() - t0
        assert elapsed < TIMEOUT_S + 1.0
        assert line.endswith("\n"), repr(line)
        obj = json.loads(line)
        assert isinstance(obj, dict)
        assert obj["id"] == 1
        assert obj["status"] == "ok", obj
        # Process is still alive (stdin open) -- streaming, not EOF-driven.
        assert proc.poll() is None
        # Closing stdin ends the run cleanly with no further rows.
        proc.stdin.close()
        rc = proc.wait(timeout=30)
        rest = proc.stdout.read().decode("utf-8")
        assert rc == 0
        assert rest.strip() == "", rest
    finally:
        _kill(proc)


# ---------------------------------------------------------------------------
# EB2: concurrent streaming
# ---------------------------------------------------------------------------

def test_eb2_concurrent_row1_reaches_pipe_before_eof_and_order_kept():
    proc = _spawn(["--batch-concurrency", "4"])
    try:
        _send(proc, _row(1))
        time.sleep(1.0)
        _send(proc, _row(2))
        assert _readable(proc), "no row on stdout within 5 s at --batch-concurrency 4"
        first = proc.stdout.readline().decode("utf-8")
        assert first.endswith("\n"), repr(first)
        obj1 = json.loads(first)
        assert isinstance(obj1, dict)
        assert obj1["id"] == 1
        assert obj1["status"] == "ok", obj1
        assert proc.poll() is None
        proc.stdin.close()
        rc = proc.wait(timeout=30)
        rest = [ln for ln in proc.stdout.read().decode("utf-8").splitlines() if ln.strip()]
        assert rc == 0
        assert len(rest) == 1, rest
        obj2 = json.loads(rest[0])
        assert obj2["id"] == 2
        assert obj2["status"] == "ok", obj2
    finally:
        _kill(proc)


# ---------------------------------------------------------------------------
# EB3: same bytes / shape (regression guard)
# ---------------------------------------------------------------------------

def _strip_trace(obj):
    if isinstance(obj, dict):
        return {k: _strip_trace(v) for k, v in obj.items() if k != "trace_id"}
    if isinstance(obj, list):
        return [_strip_trace(v) for v in obj]
    return obj


@pytest.mark.parametrize("concurrency", [None, 4], ids=["serial", "N4"])
def test_eb3_output_shape_and_order_unchanged(tmp_path, concurrency):
    src = tmp_path / "mixed.jsonl"
    src.write_text(
        json.dumps(_row(1)) + "\n" + json.dumps(NR_ROW) + "\n" + "this is not json\n",
        encoding="utf-8",
    )
    cmd = [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
           "--batch", "--file", str(src)]
    if concurrency is not None:
        cmd += ["--batch-concurrency", str(concurrency)]

    def run():
        return subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT),
                              env=_env(), timeout=60)

    r1, r2 = run(), run()
    assert r1.returncode == 3, r1.stderr[-2000:]
    out = r1.stdout.decode("utf-8")
    assert out.endswith("\n")
    lines = out.splitlines()
    assert len(lines) == 3, lines
    objs = [json.loads(ln) for ln in lines]
    for ln, obj in zip(lines, objs):
        assert isinstance(obj, dict)
        # A single compact json.dumps object: no indentation, re-dumps identically.
        assert ln == json.dumps(obj), ln
    assert [o["status"] for o in objs] == ["ok", "needs_review", "error"]
    assert objs[0]["id"] == 1
    assert "id" not in objs[2]
    # Only trace_id may differ between runs.
    objs2 = [json.loads(ln) for ln in r2.stdout.decode("utf-8").splitlines()]
    assert r2.returncode == 3
    assert [_strip_trace(o) for o in objs] == [_strip_trace(o) for o in objs2]


# ---------------------------------------------------------------------------
# EB4: exit contract under an early-closing reader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("concurrency", [None, 4], ids=["serial", "N4"])
def test_eb4_head_one_on_5000_rows_exits_zero(tmp_path, concurrency):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(5000)),
                    encoding="utf-8")
    cmd = [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
           "--batch", "--file", str(rows)]
    if concurrency is not None:
        cmd += ["--batch-concurrency", str(concurrency)]
    err_path = tmp_path / "stderr.txt"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=err, cwd=str(REPO_ROOT), env=_env())
        try:
            assert _readable(proc, 30.0)
            first = proc.stdout.readline().decode("utf-8")
            proc.stdout.close()
            rc = proc.wait(timeout=60)
        finally:
            _kill(proc)
    err_text = err_path.read_text(encoding="utf-8", errors="replace")
    assert rc == 0, (rc, err_text[-2000:])
    assert rc != 120
    assert "Traceback" not in err_text, err_text[-2000:]
    assert json.loads(first)["status"] == "ok"


# ---------------------------------------------------------------------------
# EB5: README
# ---------------------------------------------------------------------------

def _batch_paragraph() -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^\*\*Batch mode\*\*.*$", text, re.M)
    assert m, "README has no **Batch mode** paragraph"
    return m.group(0)


def test_eb5_readme_batch_paragraph_says_flushed():
    para = _batch_paragraph()
    assert "flushed" in para, para
    # The claim: each output *row* is flushed, so a consumer sees it immediately.
    # (A naive sentence split breaks on the "`... --batch`" ellipsis, so check a
    # bounded window around the word instead.)
    i = para.index("flushed")
    window = para[max(0, i - 60): i + 300]
    assert "row" in window and "immediately" in window, window


def test_eb5_readme_head_one_sentence_still_never_more_lenient():
    text = README.read_text(encoding="utf-8")
    paras = [p for p in text.split("\n") if "| head -1" in p]
    assert paras, "README lost its `| head -1` exit-code sentence"
    assert any("never more lenient" in p and "never 120" in p for p in paras), paras


def test_eb5_readme_head_one_sentence_does_not_claim_one_row_bound():
    text = README.read_text(encoding="utf-8")
    paras = [p for p in text.split("\n") if "| head -1" in p]
    assert paras
    for p in paras:
        assert not re.search(r"at most (one|1|a single) (output )?row", p, re.I), p


# ---------------------------------------------------------------------------
# EB6: the iter-86 scope pins diff a FIXED range (committed-state green)
# ---------------------------------------------------------------------------

ITER86 = REPO_ROOT / "tests" / "test_iter86_behavior.py"
PRE_DEFUSE_SHA = "0f27eeb"  # HEAD before this iteration: holds the open-ended pin
BASE86 = "dcda81c"


def _git_repo(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _module_consts(tree: ast.Module) -> dict:
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n.value.value
    return out


def _git_diff_calls(tree: ast.Module) -> list[ast.Call]:
    calls = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_git" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "diff"):
            calls.append(n)
    return calls


def _rev_args(call: ast.Call) -> list[str]:
    """Positional args between ``diff`` and ``--`` that are not option strings."""
    revs = []
    for a in call.args[1:]:
        if isinstance(a, ast.Constant) and a.value == "--":
            break
        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-"):
            continue
        revs.append(ast.unparse(a))
    return revs


def test_eb6_iter86_defines_fixed_ship_sha():
    consts = _module_consts(ast.parse(ITER86.read_text(encoding="utf-8")))
    assert consts.get("SHIP_SHA") == "0f27eeb", consts.get("SHIP_SHA")
    assert consts.get("BASE_SHA") == BASE86


def test_eb6_iter86_scope_pins_diff_fixed_range():
    tree = ast.parse(ITER86.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("test_eb6_no_new_dependency", "test_eb6_existing_test_edits_scoped"):
        assert name in funcs, name
        calls = _git_diff_calls(funcs[name])
        assert calls, f"{name} no longer runs git diff"
        for c in calls:
            assert _rev_args(c) == ["BASE_SHA", "SHIP_SHA"], (name, ast.unparse(c))


def test_eb6_no_open_range_git_diff_anywhere_in_iter86():
    tree = ast.parse(ITER86.read_text(encoding="utf-8"))
    calls = _git_diff_calls(tree)
    assert len(calls) >= 3, [ast.unparse(c) for c in calls]
    for c in calls:
        revs = _rev_args(c)
        assert len(revs) == 2, ("open-ended diff (worktree/HEAD)", ast.unparse(c))
        assert "HEAD" not in revs and "'HEAD'" not in revs, ast.unparse(c)


def _committed_clone(tmp_path: Path) -> Path:
    """Clone the repo and COMMIT the current working-tree state in the clone."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    if _git_repo("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout (archive copy)")
    for sha in (BASE86, PRE_DEFUSE_SHA):
        if _git_repo("cat-file", "-e", sha + "^{commit}").returncode != 0:
            pytest.skip("pinned commits unavailable (shallow clone)")
    clone = tmp_path / "clone"
    r = _git_repo("clone", "-q", str(REPO_ROOT), str(clone))
    assert r.returncode == 0, r.stderr
    changed = _git_repo("ls-files", "-m", "-o", "--exclude-standard", "-z").stdout
    for rel in sorted({x for x in changed.split("\0") if x}):
        src = REPO_ROOT / rel
        dst = clone / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    deleted = _git_repo("ls-files", "-d", "-z").stdout
    for rel in [x for x in deleted.split("\0") if x]:
        (clone / rel).unlink(missing_ok=True)
    _commit_all(clone, "iter-88 state committed")
    venv = REPO_ROOT / ".venv"
    if venv.exists():
        (clone / ".venv").symlink_to(venv)
    return clone


def _commit_all(clone: Path, msg: str) -> None:
    assert _git_repo("add", "-A", cwd=clone).returncode == 0
    r = _git_repo("-c", "user.name=tester", "-c", "user.email=tester@example.invalid",
                  "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty",
                  "--no-verify", "-m", msg, cwd=clone)
    assert r.returncode == 0, r.stderr
    st = _git_repo("status", "--porcelain", "--untracked-files=all", cwd=clone).stdout
    assert st.strip() == "", st


def _clone_env(clone: Path) -> dict:
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(clone / "src"), str(clone)])
    env.pop("PYTEST_ADDOPTS", None)
    return env


def _pytest_in(clone: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-rfs", "-p", "no:cacheprovider", *args],
        cwd=str(clone), env=_clone_env(clone), capture_output=True, text=True,
        timeout=300,
    )


def test_eb6_committed_clone_iter86_pins_green(tmp_path):
    clone = _committed_clone(tmp_path)
    # The clone's own source is what gets imported.
    probe = subprocess.run(
        [sys.executable, "-c", "import reflect_mcp; print(reflect_mcp.__file__)"],
        cwd=str(clone), env=_clone_env(clone), capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()).resolve().is_relative_to(clone.resolve()), probe.stdout
    # The iteration's new test file is part of the committed state.
    tracked = _git_repo("ls-files", "tests/", cwd=clone).stdout.split()
    assert "tests/test_iter88_behavior.py" in tracked
    r = _pytest_in(clone, "tests/test_iter86_behavior.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b\d+ passed\b", r.stdout), r.stdout[-2000:]
    assert not re.search(r"\b(skipped|failed|error)", r.stdout), r.stdout[-2000:]


def test_eb6_committed_clone_old_open_ended_pin_goes_red(tmp_path):
    clone = _committed_clone(tmp_path)
    old = _git_repo("show", f"{PRE_DEFUSE_SHA}:tests/test_iter86_behavior.py", cwd=clone)
    assert old.returncode == 0, old.stderr
    (clone / "tests" / "test_iter86_behavior.py").write_text(old.stdout, encoding="utf-8")
    _commit_all(clone, "restore pre-defuse iter-86 pins")
    r = _pytest_in(clone, "tests/test_iter86_behavior.py::test_eb6_existing_test_edits_scoped")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b1 failed\b", r.stdout), r.stdout[-2000:]
    assert "FAILED tests/test_iter86_behavior.py::test_eb6_existing_test_edits_scoped" in r.stdout
