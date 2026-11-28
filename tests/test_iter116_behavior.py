"""Repo-wide guard: no machine-path literals in any ``src/`` or ``tests/`` ``.py`` file.

The iter-80 guard only scans ``.github/workflows/ci.yml``, so hard-coded
checkout/venv/home paths crept into two CLI test files and turned 34 tests red
on every machine except the author's (CI included). This guard widens the
"publication-safe, no machine paths" rule to all Python sources and tests.

Why the fragments are assembled by concatenation: this file is itself scanned,
so spelling a fragment out literally would make the guard fail on itself.
``tests/test_iter80_behavior.py`` is the single exemption because its ci.yml
guard must name the fragments verbatim in a parametrize.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRAGMENTS: tuple[str, ...] = ("/Us" + "ers/", "/ho" + "me/", "C:" + "\\")
EXEMPT: frozenset[str] = frozenset({"tests/test_iter80_behavior.py"})
SCOPES: tuple[str, ...] = ("src", "tests")


def find_machine_paths(root: Path) -> list[str]:
    """Return ``relative/path:line`` for every fragment hit under ``root``.

    Takes the root as a parameter so the same logic that guards the real repo
    can be proven to bite on a planted ``tmp_path`` tree.
    """
    hits: list[str] = []
    for scope in SCOPES:
        base = root / scope
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts or rel in EXEMPT:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(frag in line for frag in FRAGMENTS):
                    hits.append(f"{rel}:{lineno}")
    return hits


def test_repo_has_no_machine_paths() -> None:
    hits = find_machine_paths(ROOT)
    assert not hits, "machine-path literal(s) found:\n" + "\n".join(hits)


def test_guard_scans_itself() -> None:
    """The guard file is in scope (not exempt) and contains no literal fragment."""
    me = Path(__file__).resolve().relative_to(ROOT).as_posix()
    assert me not in EXEMPT
    assert me not in find_machine_paths(ROOT)


def _plant(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_bites_on_planted_tests_file(tmp_path: Path) -> None:
    _plant(tmp_path, "tests/x.py", 'ok = 1\ncwd="' + "/Us" + 'ers/someone/proj"\n')
    assert find_machine_paths(tmp_path) == ["tests/x.py:2"]


def test_bites_on_planted_src_file(tmp_path: Path) -> None:
    _plant(tmp_path, "src/pkg/y.py", 'P = "' + "/ho" + 'me/u/"\n')
    assert find_machine_paths(tmp_path) == ["src/pkg/y.py:1"]


def test_ignores_non_python_and_exempt_and_clean(tmp_path: Path) -> None:
    frag = "/Us" + "ers/someone"
    _plant(tmp_path, "tests/notes.txt", frag + "\n")
    _plant(tmp_path, "tests/test_iter80_behavior.py", f'X = "{frag}"\n')
    _plant(tmp_path, "tests/clean.py", "x = 1\n")
    _plant(tmp_path, "src/__pycache__/z.py", f'X = "{frag}"\n')
    assert find_machine_paths(tmp_path) == []
