"""Black-box behavioral tests for iter-90: re-land of #39 (explicit null ``total``) with the
iter-89 PRESHIP time bomb defused.

EB1-EB11 are covered by the preserved ``tests/test_iter89_behavior.py`` (the spec keeps that
file name). This module adds independent black-box coverage of the headline behavior and the
part of the spec that only shows up after a COMMIT -- EB12:

  12a. No test module this ship adds or edits passes ``HEAD``/``@`` as a revision to
       ``git archive``/``diff``/``show`` (AST scan, with positive/negative controls).
  12b. The base-tree probe is ``git cat-file -e 7237e5a^{commit}``, never ``rev-parse HEAD``.
  12c. In a ``git clone`` of the COMMITTED ship (``.venv`` symlinked, not committed) the
       iter-89 module is green -- the exact PRESHIP shape that reverted iter 89.
  12d. In a shallow (``--depth 1``) clone the base commit is absent and the base-tree tests
       SKIP rather than ERROR.
  12e. Counterfactual: re-arming the bomb (``git archive HEAD src``) in a committed clone
       turns the module red, i.e. 12c really discriminates.

Base means the fixed commit ``7237e5a``; nothing here names a floating ``HEAD`` revision.
Isolation: no implementation source, engineer/reviewer notes or diff were read.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
ITER89 = TESTS / "test_iter89_behavior.py"
BASE_SHA = "7237e5a"

# Test modules the #39 ship adds or edits (the patch's migration set + the new modules).
SHIP_TEST_MODULES = [
    "test_eval.py", "test_iter59_behavior.py", "test_iter60_behavior.py",
    "test_iter72_behavior.py", "test_iter76_behavior.py", "test_iter77_behavior.py",
    "test_iter80_behavior.py", "test_iter82_behavior.py", "test_iter83_behavior.py",
    "test_iter84_behavior.py", "test_iter89_behavior.py", "test_iter90_behavior.py",
]

SRC = (
    "ACME CORP\nInvoice #INV-1001\nDate: 2026-03-14\nSubtotal: $540.00\n"
    "Tax: $43.20\nTotal: 583.20\nPaid in USD\n"
)
OTHERS = {
    "vendor_name": "ACME CORP", "invoice_number": "INV-1001", "invoice_date": "2026-03-14",
    "subtotal": 540.0, "tax": 43.2, "currency": "USD",
}
REASON = "source labels total as 583.20, extraction left it null"


# ---------------------------------------------------------------- helpers
def _git(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("REFLECT_")}
    env.pop("PYTEST_ADDOPTS", None)
    return env


def _have_base() -> bool:
    if shutil.which("git") is None:
        return False
    if _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        return False
    return _git("cat-file", "-e", f"{BASE_SHA}^{{commit}}").returncode == 0


def _validate(data: dict, tmp: Path, pythonpath: str | None = None) -> tuple[int, dict]:
    f = tmp / "src.txt"
    f.write_text(SRC, encoding="utf-8")
    env = _env()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", "validate", "--heuristics-only",
         "--data", json.dumps(data), "--file", str(f)],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(tmp),
    )
    d = json.loads(r.stdout)
    d.pop("trace_id", None)
    return r.returncode, d


# ---------------------------------------------------------------- EB1 (independent)
def test_eb1_cli_explicit_null_total_needs_review(tmp_path):
    code, d = _validate(dict(OTHERS, total=None), tmp_path)
    assert (d["status"], code) == ("needs_review", 1), d
    assert [(f["field"], f["value"], f["confidence"]) for f in d["flagged_fields"]] == [
        ("total", None, 0.3)], d["flagged_fields"]
    assert REASON in d["flagged_fields"][0]["reason"]
    assert d["field_confidence"]["total"] == 0.3
    assert d["data"]["total"] is None


def test_eb1_base_sha_is_silent_ok(tmp_path):
    if not _have_base():
        pytest.skip(f"base commit {BASE_SHA} not in the object store")
    blob = _git_bytes("archive", BASE_SHA, "src")
    dest = tmp_path / "base"
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)
    code, d = _validate(dict(OTHERS, total=None), tmp_path, str(dest / "src"))
    assert (d["status"], code) == ("ok", 0), d


def _git_bytes(*args: str) -> bytes:
    r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, check=True)
    return r.stdout


# ---------------------------------------------------------------- EB12a: AST guard
_REV_VERBS = {"archive", "diff", "show"}
_FLOATING = re.compile(r"^(HEAD|@)([~^:.@{]|$)")


def _str_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value for v in node.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str))]
    return []


def _floating_rev_violations(source: str) -> list[str]:
    """Every git argv (list/tuple literal or ``_git(...)``-style call) whose verb is
    archive/diff/show and which carries a floating ``HEAD``/``@`` revision, plus any
    single-revision ``git diff <rev>`` (it compares against the moving worktree)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.List, ast.Tuple)):
            elts = node.elts
        elif isinstance(node, ast.Call):
            elts = node.args
        else:
            continue
        words = [(_str_parts(e) or [None])[0] for e in elts]  # None = non-literal (e.g. BASE_SHA)
        if words and words[0] == "git":
            elts, words = elts[1:], words[1:]
        verb_idx = next((i for i, w in enumerate(words[:5]) if w in _REV_VERBS), None)
        if verb_idx is None:
            continue
        revs = []
        for w in words[verb_idx + 1:]:
            if w == "--":
                break
            if w is not None and w.startswith("-"):
                continue
            revs.append(w)
        bad = any(w is not None and (_FLOATING.match(w) or "..HEAD" in w or "HEAD:" in w)
                  for w in revs)
        single = words[verb_idx] == "diff" and len(revs) == 1 and not (
            revs[0] is not None and ".." in revs[0])
        if bad or single:
            found.append(ast.unparse(node))
    return found


@pytest.mark.parametrize("snippet", [
    'subprocess.run(["git", "archive", "HEAD", "src"], cwd=REPO)',
    '_git("diff", "HEAD", "--", "src")',
    '_git("show", "HEAD:tests/x.py")',
    'subprocess.run(["git", "diff", "7237e5a..HEAD"])',
    '_git_repo("show", "@:README.md")',
    'subprocess.run(["git", "archive", "HEAD~1", "src"])',
    '_git("diff", "--name-only", BASE_SHA, "--", "tests/")',
])
def test_eb12a_guard_flags_floating_revisions(snippet):
    assert _floating_rev_violations(snippet), snippet


@pytest.mark.parametrize("snippet", [
    'subprocess.run(["git", "archive", BASE_SHA, "src"], cwd=REPO)',
    '_git("diff", "7237e5a", "0f27eeb", "--", "src")',
    '_git("cat-file", "-e", f"{BASE_SHA}^{{commit}}")',
    'x = ["HEAD", "archive"]',
])
def test_eb12a_guard_passes_fixed_revisions(snippet):
    assert _floating_rev_violations(snippet) == [], snippet


@pytest.mark.parametrize("name", SHIP_TEST_MODULES)
def test_eb12a_no_floating_rev_in_ship_test_modules(name):
    path = TESTS / name
    assert path.is_file(), name
    assert _floating_rev_violations(path.read_text(encoding="utf-8")) == [], name


# ---------------------------------------------------------------- EB12b: probe shape
def test_eb12b_iter89_probes_fixed_sha_with_cat_file():
    text = ITER89.read_text(encoding="utf-8")
    tree = ast.parse(text)
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
    assert consts.get("BASE_SHA") == BASE_SHA, consts.get("BASE_SHA")
    assert '"cat-file", "-e"' in text and "^{{commit}}" in text
    assert "rev-parse" not in text
    archives = [ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.List) and len(n.elts) >= 3
                and _str_parts(n.elts[0]) == ["git"] and _str_parts(n.elts[1]) == ["archive"]]
    assert archives and all("BASE_SHA" in a for a in archives), archives


# ---------------------------------------------------------------- EB12c-e: after a commit
def _committed_clone(tmp_path: Path, name: str = "clone") -> Path:
    if not _have_base():
        pytest.skip(f"base commit {BASE_SHA} unavailable (shallow/archive checkout)")
    clone = tmp_path / name
    r = _git("clone", "-q", str(REPO), str(clone))
    assert r.returncode == 0, r.stderr
    changed = _git("ls-files", "-m", "-o", "--exclude-standard", "-z").stdout
    for rel in sorted({x for x in changed.split("\0") if x}):
        src = REPO / rel
        if src.is_file() and not src.is_symlink():
            dst = clone / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in [x for x in _git("ls-files", "-d", "-z").stdout.split("\0") if x]:
        (clone / rel).unlink(missing_ok=True)
    _commit(clone, "ship committed")
    _link_venv(clone)
    return clone


def _link_venv(clone: Path) -> None:
    venv = REPO / ".venv"
    if venv.exists() and not (clone / ".venv").exists():
        (clone / ".venv").symlink_to(venv)


def _commit(clone: Path, msg: str) -> None:
    assert _git("add", "-A", cwd=clone).returncode == 0
    r = _git("-c", "user.name=tester", "-c", "user.email=tester@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "--no-verify",
             "-m", msg, cwd=clone)
    assert r.returncode == 0, r.stderr
    st = _git("status", "--porcelain", "--untracked-files=all", cwd=clone).stdout
    assert st.strip() == "", st


def _pytest_in(clone: Path, *args: str) -> subprocess.CompletedProcess:
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(clone / "src"), str(clone)])
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-rfEs", "-p", "no:cacheprovider", *args],
        cwd=str(clone), env=env, capture_output=True, text=True, timeout=300,
    )


def _imports_from(clone: Path) -> Path:
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(clone / "src"), str(clone)])
    r = subprocess.run([sys.executable, "-c", "import reflect_mcp; print(reflect_mcp.__file__)"],
                       cwd=str(clone), env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return Path(r.stdout.strip()).resolve()


def test_eb12c_committed_clone_iter89_module_green(tmp_path):
    clone = _committed_clone(tmp_path)
    assert _imports_from(clone).is_relative_to(clone.resolve())
    tracked = _git("ls-files", "tests/", cwd=clone).stdout.split()
    assert "tests/test_iter89_behavior.py" in tracked
    assert ".venv" not in _git("ls-files", cwd=clone).stdout.split()
    r = _pytest_in(clone, "tests/test_iter89_behavior.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b\d+ passed\b", r.stdout), r.stdout[-2000:]
    assert not re.search(r"\b\d+ (skipped|failed|errors?)\b", r.stdout), r.stdout[-2000:]


def test_eb12d_shallow_clone_skips_base_tests_without_error(tmp_path):
    full = _committed_clone(tmp_path, "full")
    shallow = tmp_path / "shallow"
    r = _git("clone", "-q", "--depth", "1", f"file://{full}", str(shallow))
    assert r.returncode == 0, r.stderr
    assert _git("cat-file", "-e", f"{BASE_SHA}^{{commit}}", cwd=shallow).returncode != 0
    _link_venv(shallow)
    r = _pytest_in(shallow, "tests/test_iter89_behavior.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b\d+ skipped\b", r.stdout), r.stdout[-2000:]
    assert re.search(r"\b\d+ passed\b", r.stdout), r.stdout[-2000:]
    assert not re.search(r"\b\d+ (failed|errors?)\b", r.stdout), r.stdout[-2000:]


def test_eb12e_rearmed_head_bomb_goes_red_after_commit(tmp_path):
    clone = _committed_clone(tmp_path)
    path = clone / "tests" / "test_iter89_behavior.py"
    text = path.read_text(encoding="utf-8")
    armed = text.replace('["git", "archive", BASE_SHA, "src"]', '["git", "archive", "HEAD", "src"]')
    assert armed != text, "fixed-sha archive argv not found in the iter-89 module"
    path.write_text(armed, encoding="utf-8")
    _commit(clone, "re-arm the iter-89 time bomb")
    r = _pytest_in(clone, "tests/test_iter89_behavior.py::test_eb1_red_on_head")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b1 failed\b", r.stdout), r.stdout[-2000:]
