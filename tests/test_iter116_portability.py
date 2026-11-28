"""Iter-116 tester checks: the suite carries no machine paths and runs anywhere.

Black-box companion to the engineer's guard (``tests/test_iter116_behavior.py``).
These checks are written from the spec, not from the guard: the repo-wide scan
below is an independent re-implementation, so a bug in the guard helper cannot
make both pass at once.

Every machine-path fragment is assembled by concatenation, because this file
lives under ``tests/`` and is itself in scope of the guard.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
GUARD = TESTS / "test_iter116_behavior.py"
CLI_FILES = ("test_iter37_behavior.py", "test_iter41_behavior.py")
EXEMPT_REL = "tests/test_iter80_behavior.py"

USERS = "/Us" + "ers/"
HOME = "/ho" + "me/"
WIN = "C:" + "\\"
FRAGMENTS = (USERS, HOME, WIN)


def _independent_scan(root: Path) -> list[str]:
    """Spec-derived scan (EB1/EB5): ``rel:line`` for each fragment hit."""
    hits: list[str] = []
    for scope in ("src", "tests"):
        base = root / scope
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel_parts = path.relative_to(root).parts
            if "__pycache__" in rel_parts:
                continue
            rel = "/".join(rel_parts)
            for i, line in enumerate(path.read_text("utf-8", "replace").splitlines(), 1):
                if any(f in line for f in FRAGMENTS):
                    hits.append(f"{rel}:{i}")
    return hits


def _load_guard():
    spec = importlib.util.spec_from_file_location("_iter116_guard_under_test", GUARD)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plant(root: Path, rel: str, body: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# EB1: no machine-path literal anywhere in src/ or tests/ except test_iter80
# ---------------------------------------------------------------------------


def test_scanner_self_check_detects_planted_fragments(tmp_path: Path) -> None:
    """Non-vacuity: the independent scanner really finds each fragment."""
    _plant(tmp_path, "tests/a.py", "x = 1\np = '" + USERS + "me/x'\n")
    _plant(tmp_path, "src/p/b.py", "q = '" + HOME + "u/'\n")
    _plant(tmp_path, "tests/c.py", "r = r'" + WIN + "dir'\n")
    _plant(tmp_path, "tests/__pycache__/d.py", "s = '" + USERS + "'\n")
    assert _independent_scan(tmp_path) == ["src/p/b.py:1", "tests/a.py:2", "tests/c.py:1"]


def test_eb1_only_test_iter80_carries_fragments() -> None:
    hits = _independent_scan(ROOT)
    offenders = [h for h in hits if not h.startswith(EXEMPT_REL + ":")]
    assert offenders == [], "machine-path literal(s):\n" + "\n".join(offenders)


def test_eb1_exemption_is_real() -> None:
    """test_iter80 still names the fragments verbatim (so the exemption matters)."""
    hits = _independent_scan(ROOT)
    assert any(h.startswith(EXEMPT_REL + ":") for h in hits)


@pytest.mark.parametrize("name", CLI_FILES)
def test_eb1_cli_files_have_no_users_or_home(name: str) -> None:
    text = (TESTS / name).read_text("utf-8")
    for frag in FRAGMENTS:
        assert frag not in text, f"{name} still contains {frag!r}"
    # The old pins also named a venv and a home dir via env keys.
    assert "VIRTUAL_ENV" not in text
    assert ".venv" not in text


# ---------------------------------------------------------------------------
# EB2: no cwd=/env= keyword on any subprocess run call in test_iter37/41
# ---------------------------------------------------------------------------


def _run_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name.endswith("run"):
                calls.append(node)
    return calls


def _is_module_cmd(node: ast.AST, assigns: dict[str, ast.AST]) -> bool:
    """True if ``node`` evaluates to ``[sys.executable, "-m", "reflect_mcp", ...]``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_module_cmd(node.left, assigns)
    if isinstance(node, ast.Name) and node.id in assigns:
        return _is_module_cmd(assigns[node.id], assigns)
    if isinstance(node, ast.List) and len(node.elts) >= 3:
        head = node.elts[0]
        head_src = ast.unparse(head)
        if isinstance(head, ast.Name) and head.id in assigns:
            head_src = ast.unparse(assigns[head.id])
        consts = [getattr(e, "value", None) for e in node.elts[1:3]]
        return head_src == "sys.executable" and consts == ["-m", "reflect_mcp"]
    return False


@pytest.mark.parametrize("name", CLI_FILES)
def test_eb2_no_cwd_or_env_and_module_invocation(name: str) -> None:
    tree = ast.parse((TESTS / name).read_text("utf-8"))
    assigns: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                assigns[t.id] = node.value
    calls = _run_calls(tree)
    assert calls, f"{name}: expected subprocess run calls"
    for call in calls:
        kws = {k.arg for k in call.keywords}
        assert "cwd" not in kws, f"{name}:{call.lineno} passes cwd="
        assert "env" not in kws, f"{name}:{call.lineno} passes env="
        assert call.args, f"{name}:{call.lineno} has no positional command"
        assert _is_module_cmd(call.args[0], assigns), (
            f"{name}:{call.lineno} command is not [sys.executable, '-m', 'reflect_mcp', ...]: "
            + ast.unparse(call.args[0])
        )


def test_eb2_checker_bites_on_cwd_env() -> None:
    """Non-vacuity: the AST checks flag the old pinned shape."""
    src = (
        "import subprocess, sys\n"
        "M = [sys.executable, '-m', 'reflect_mcp']\n"
        "subprocess.run(M + ['--help'], cwd='x', env={'PATH': 'y'})\n"
        "subprocess.run(['python', '-m', 'reflect_mcp'])\n"
    )
    tree = ast.parse(src)
    assigns = {"M": tree.body[1].value}
    calls = _run_calls(tree)
    assert len(calls) == 2
    assert {k.arg for k in calls[0].keywords} == {"cwd", "env"}
    assert _is_module_cmd(calls[0].args[0], assigns)
    assert not _is_module_cmd(calls[1].args[0], assigns)


# ---------------------------------------------------------------------------
# EB3 / EB4: 69 collected; all pass from a relocated copy
# ---------------------------------------------------------------------------


def _child_env(extra_pythonpath: Path | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    if extra_pythonpath is not None:
        env["PYTHONPATH"] = str(extra_pythonpath)
    return env


def test_eb3_collect_count_is_69() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
         "-p", "no:cacheprovider", *[f"tests/{n}" for n in CLI_FILES]],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=_child_env(),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"(\d+) tests? collected", r.stdout)
    assert m, r.stdout
    assert int(m.group(1)) == 69


def test_eb4_relocated_copy_passes(tmp_path: Path) -> None:
    dest = tmp_path / "relocated-checkout"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(ROOT / "src", dest / "src", ignore=ignore)
    shutil.copytree(ROOT / "tests", dest / "tests", ignore=ignore)
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")
    # The copy's path must share no component with the original checkout.
    assert str(ROOT) not in str(dest)
    assert ROOT.name not in dest.parts
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
         *[f"tests/{n}" for n in CLI_FILES]],
        cwd=dest, capture_output=True, text=True, timeout=300,
        env=_child_env(dest / "src"),
    )
    tail = r.stdout[-2000:] + r.stderr[-2000:]
    assert r.returncode == 0, tail
    assert re.search(r"\b69 passed\b", r.stdout), tail
    assert "failed" not in r.stdout.splitlines()[-1], tail


# ---------------------------------------------------------------------------
# EB5: the guard file exists, roots on __file__, scans itself, passes
# ---------------------------------------------------------------------------


def test_eb5_guard_file_exists_and_roots_on_file() -> None:
    assert GUARD.is_file()
    text = GUARD.read_text("utf-8")
    assert "Path(__file__)" in text
    for frag in FRAGMENTS:
        assert frag not in text, f"guard spells {frag!r} literally"
    assert str(ROOT) not in text


def test_eb5_guard_passes_on_real_tree() -> None:
    guard = _load_guard()
    assert guard.find_machine_paths(ROOT) == []


def test_eb5_guard_scans_its_own_file(tmp_path: Path) -> None:
    """Plant a literal fragment into a relocated copy of the guard file itself."""
    guard = _load_guard()
    rel = "tests/test_iter116_behavior.py"
    body = GUARD.read_text("utf-8") + "\nLEAK = '" + USERS + "someone/x'\n"
    _plant(tmp_path, rel, body)
    n = body.splitlines().index("LEAK = '" + USERS + "someone/x'") + 1
    assert guard.find_machine_paths(tmp_path) == [f"{rel}:{n}"]


def test_eb5_guard_test_fails_with_locator_message(tmp_path: Path) -> None:
    """The guard's repo test fails and names rel:line when a hit exists."""
    guard = _load_guard()
    _plant(tmp_path, "tests/leak.py", "a = 0\nb = '" + HOME + "u/p'\n")
    guard.ROOT = tmp_path
    with pytest.raises(AssertionError) as exc:
        guard.test_repo_has_no_machine_paths()
    assert "tests/leak.py:2" in str(exc.value)


# ---------------------------------------------------------------------------
# EB6: the guard bites on planted trees
# ---------------------------------------------------------------------------


def test_eb6_planted_tests_file_one_hit(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/x.py", "import os\n\ncwd=\"" + USERS + "someone/proj\"\n")
    assert guard.find_machine_paths(tmp_path) == ["tests/x.py:3"]


def test_eb6_planted_src_file_one_hit(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "src/pkg/y.py", "P = '" + HOME + "u/'\n")
    assert guard.find_machine_paths(tmp_path) == ["src/pkg/y.py:1"]


def test_eb6_windows_fragment_hits(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/w.py", "x = 1\ny = 2\nP = r'" + WIN + "proj'\n")
    assert guard.find_machine_paths(tmp_path) == ["tests/w.py:3"]


def test_eb6_non_python_ignored(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/notes.txt", USERS + "someone/proj\n")
    _plant(tmp_path, "src/pkg/readme.md", HOME + "u/\n")
    assert guard.find_machine_paths(tmp_path) == []


def test_eb6_exempt_file_ignored(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/test_iter80_behavior.py", "X = '" + USERS + "a'\n")
    assert guard.find_machine_paths(tmp_path) == []


def test_eb6_exemption_is_only_that_path(tmp_path: Path) -> None:
    """Only tests/test_iter80_behavior.py is exempt, not a same-named file elsewhere."""
    guard = _load_guard()
    _plant(tmp_path, "src/test_iter80_behavior.py", "X = '" + USERS + "a'\n")
    _plant(tmp_path, "tests/sub/test_iter80_behavior.py", "X = '" + HOME + "a'\n")
    assert guard.find_machine_paths(tmp_path) == [
        "src/test_iter80_behavior.py:1",
        "tests/sub/test_iter80_behavior.py:1",
    ]


def test_eb6_clean_tree_no_hits(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/clean.py", "x = '/usr/bin/env'\n")
    _plant(tmp_path, "src/pkg/__init__.py", "")
    assert guard.find_machine_paths(tmp_path) == []


def test_eb6_out_of_scope_dirs_ignored(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "eval/e.py", "X = '" + USERS + "a'\n")
    _plant(tmp_path, "scripts/s.py", "X = '" + HOME + "a'\n")
    assert guard.find_machine_paths(tmp_path) == []


def test_eb6_pycache_skipped(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/__pycache__/z.py", "X = '" + USERS + "a'\n")
    _plant(tmp_path, "src/pkg/__pycache__/z.py", "X = '" + HOME + "a'\n")
    assert guard.find_machine_paths(tmp_path) == []


def test_eb6_multiple_hits_all_named(tmp_path: Path) -> None:
    guard = _load_guard()
    _plant(tmp_path, "tests/m.py", "a = '" + USERS + "x'\nb = 1\nc = '" + HOME + "y'\n")
    _plant(tmp_path, "src/n.py", "d = '" + WIN + "z'\n")
    assert sorted(guard.find_machine_paths(tmp_path)) == [
        "src/n.py:1", "tests/m.py:1", "tests/m.py:3",
    ]


# ---------------------------------------------------------------------------
# PRODUCT.md acceptance rows (anchored on substrings, never line numbers)
# ---------------------------------------------------------------------------


def test_product_md_rows_for_iter116() -> None:
    text = (ROOT / "PRODUCT.md").read_text("utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("- iter 116 ")]
    assert len(rows) == 1, rows
    assert len(rows[0]) <= 120, len(rows[0])
    roadmap = [ln for ln in text.splitlines() if re.match(r"\|\s*136\s*\|", ln)]
    assert len(roadmap) == 1, roadmap
    assert "DONE (iter 116)" in roadmap[0]
    for row in rows + roadmap:
        for frag in FRAGMENTS:
            assert frag not in row
