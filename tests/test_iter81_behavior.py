"""Black-box behavioral tests for iter-81: the in-process CLI entry points
(``cli.run_extract`` / ``cli.run_validate``) parse through the console
script's own argparse tree (``reflect_mcp.__main__._build_top_parser``), and
cli.py's duplicate parser builders are gone.

Covers the Expected Behaviors from pm.md:
  1. ``run_extract(["--help"])`` stdout == ``python -m reflect_mcp extract --help``.
  2. Same for validate.
  3. ``--bogus`` -> SystemExit(2), stderr byte-identical to the console's,
     last line ``reflect-mcp: error: unrecognized arguments: --bogus``.
  4. ``--field-threshold abc`` -> SystemExit(2), stderr equals the console's.
  5. The Namespace handed to ``run_*_from_args`` equals the console parse.
  6. ``_build_top_parser`` is resolved at call time, exactly once per call.
  7. ``argv=None`` still means ``sys.argv[1:]``.
  8. cli.py holds no parser builders, no ``ArgumentParser(`` / ``add_argument(``.
  9. ``import reflect_mcp.cli`` does not import ``reflect_mcp.__main__``.
 10. Shipped surface unchanged vs HEAD 2bb26a1 (``__main__.py``, console help,
     ``eval/``, ``make eval`` stdout).
 11. Test migrations are mechanical.

Isolation contract: tests drive the public CLI (subprocess ``python -m
reflect_mcp``) and the public in-process entry points with spies, and assert
observable output only. EB8's text checks grep the source file as the spec
demands; no implementation logic was read to write these tests.

Subprocesses inherit ``PYTHONPATH`` so the module can be pointed at a
``git archive`` tree for discrimination checks.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PY = REPO_ROOT / "src" / "reflect_mcp" / "cli.py"
SRC_DIR = REPO_ROOT / "src"
BASE_SHA = "2bb26a1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(**extra: str) -> dict:
    env = dict(os.environ)
    for k in ("REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD"):
        env.pop(k, None)
    env["COLUMNS"] = "80"
    env.update(extra)
    return env


def _console(*argv: str, cwd: Path | None = None, env: dict | None = None):
    r = subprocess.run(
        [sys.executable, "-m", "reflect_mcp", *argv],
        capture_output=True, text=True, timeout=60,
        cwd=str(cwd or REPO_ROOT), env=env or _env(),
    )
    return r.returncode, r.stdout, r.stderr


def _cli():
    return importlib.import_module("reflect_mcp.cli")


def _main_mod():
    return importlib.import_module("reflect_mcp.__main__")


@pytest.fixture
def cols80(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.delenv("REFLECT_FIELD_THRESHOLD", raising=False)
    monkeypatch.delenv("REFLECT_OVERALL_THRESHOLD", raising=False)


def _inproc(fn_name: str, argv, capsys):
    """Run cli.<fn_name>(argv) and return (exit_code, stdout, stderr)."""
    fn = getattr(_cli(), fn_name)
    capsys.readouterr()
    with pytest.raises(SystemExit) as ei:
        fn(argv)
    out, err = capsys.readouterr()
    code = ei.value.code
    return (0 if code is None else code), out, err


def _base_reachable() -> bool:
    if not (REPO_ROOT / ".git").exists():
        return False
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{BASE_SHA}^{{commit}}"],
                       capture_output=True, timeout=30)
    return r.returncode == 0


def _git_show(path: str) -> bytes:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{BASE_SHA}:{path}"],
                       capture_output=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def head_tree(tmp_path_factory):
    if not _base_reachable():
        pytest.skip(f"{BASE_SHA} not reachable (no git history)")
    dest = tmp_path_factory.mktemp("head81")
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", BASE_SHA],
                       capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(dest)
    return dest


# ---------------------------------------------------------------------------
# EB1 / EB2 -- help byte-identical to the console
# ---------------------------------------------------------------------------

class TestEB1EB2Help:
    @pytest.mark.parametrize("verb,fn", [("extract", "run_extract"), ("validate", "run_validate")])
    def test_help_matches_console(self, verb, fn, cols80, capsys):
        code, out, err = _inproc(fn, ["--help"], capsys)
        c_code, c_out, c_err = _console(verb, "--help")
        assert c_code == 0
        assert code == 0
        assert out == c_out
        assert out.startswith(f"usage: reflect-mcp {verb} ")

    def test_extract_help_has_no_test_path_only_description(self, cols80, capsys):
        _, out, _ = _inproc("run_extract", ["--help"], capsys)
        _, c_out, _ = _console("extract", "--help")
        line = "Extract structured data from text via the reflective pipeline."
        assert (line in out) == (line in c_out)

    def test_validate_help_has_no_test_path_only_description(self, cols80, capsys):
        _, out, _ = _inproc("run_validate", ["--help"], capsys)
        _, c_out, _ = _console("validate", "--help")
        line = "Verify pre-extracted data against source text"
        assert (line in out) == (line in c_out)


# ---------------------------------------------------------------------------
# EB3 / EB4 -- parse errors byte-identical to the console
# ---------------------------------------------------------------------------

class TestEB3UnknownFlag:
    @pytest.mark.parametrize("verb,fn", [("extract", "run_extract"), ("validate", "run_validate")])
    def test_bogus_matches_console(self, verb, fn, cols80, capsys):
        code, out, err = _inproc(fn, ["--bogus"], capsys)
        c_code, c_out, c_err = _console(verb, "--bogus")
        assert code == 2
        assert c_code == 2
        assert err == c_err
        assert err.rstrip("\n").splitlines()[-1] == "reflect-mcp: error: unrecognized arguments: --bogus"
        assert out == ""


class TestEB4InvalidValue:
    def test_extract_invalid_float_matches_console(self, cols80, capsys):
        argv = ["--field-threshold", "abc"]
        code, _, err = _inproc("run_extract", argv, capsys)
        c_code, _, c_err = _console("extract", *argv)
        assert code == 2 and c_code == 2
        assert err == c_err
        assert err.rstrip("\n").endswith(
            "reflect-mcp extract: error: argument --field-threshold: invalid float value: 'abc'"
        )

    def test_validate_invalid_float_matches_console(self, cols80, capsys):
        # Not discriminating vs HEAD (subparser-level error); kept as an invariant.
        argv = ["--field-threshold", "abc"]
        code, _, err = _inproc("run_validate", argv, capsys)
        c_code, _, c_err = _console("validate", *argv)
        assert code == 2 and c_code == 2
        assert err == c_err


# ---------------------------------------------------------------------------
# EB5 / EB6 / EB7 -- same Namespace, one parser resolved at call time
# ---------------------------------------------------------------------------

class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, args, *rest, **kw):
        self.calls.append(args)
        return 0


EXTRACT_ARGV = ["--heuristics-only", "--batch", "--batch-concurrency", "3"]
VALIDATE_ARGV = ["--data", "{}", "--field-threshold", "0.5"]


class TestEB5Namespace:
    def test_extract_namespace_equals_console_parse(self, monkeypatch, cols80):
        cli = _cli()
        spy = _Spy()
        monkeypatch.setattr(cli, "run_extract_from_args", spy)
        assert cli.run_extract(list(EXTRACT_ARGV)) == 0
        assert len(spy.calls) == 1
        expected = vars(_main_mod()._build_top_parser().parse_args(["extract", *EXTRACT_ARGV]))
        got = vars(spy.calls[0])
        assert got == expected
        assert got["command"] == "extract"
        assert got["heuristics_only"] is True
        assert got["batch_concurrency"] == 3

    def test_validate_namespace_equals_console_parse(self, monkeypatch, cols80):
        cli = _cli()
        spy = _Spy()
        monkeypatch.setattr(cli, "run_validate_from_args", spy)
        assert cli.run_validate(list(VALIDATE_ARGV)) == 0
        assert len(spy.calls) == 1
        expected = vars(_main_mod()._build_top_parser().parse_args(["validate", *VALIDATE_ARGV]))
        got = vars(spy.calls[0])
        assert got == expected
        assert got["command"] == "validate"
        assert got["field_threshold"] == 0.5


class TestEB6OneSourceOfTruth:
    @pytest.mark.parametrize("fn,spied,argv", [
        ("run_extract", "run_extract_from_args", ["--heuristics-only"]),
        ("run_validate", "run_validate_from_args", ["--data", "{}"]),
    ])
    def test_top_parser_called_once_per_call(self, monkeypatch, cols80, fn, spied, argv):
        cli = _cli()
        main_mod = _main_mod()
        real = main_mod._build_top_parser
        count = {"n": 0}

        def wrapper(*a, **k):
            count["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(main_mod, "_build_top_parser", wrapper)
        monkeypatch.setattr(cli, spied, _Spy())
        assert getattr(cli, fn)(list(argv)) == 0
        assert count["n"] == 1
        assert getattr(cli, fn)(list(argv)) == 0
        assert count["n"] == 2


class TestEB7ArgvNone:
    @pytest.mark.parametrize("fn,spied,cmd", [
        ("run_extract", "run_extract_from_args", "extract"),
        ("run_validate", "run_validate_from_args", "validate"),
    ])
    def test_argv_none_reads_sys_argv(self, monkeypatch, cols80, fn, spied, cmd):
        cli = _cli()
        spy = _Spy()
        monkeypatch.setattr(cli, spied, spy)
        monkeypatch.setattr(sys, "argv", ["reflect-mcp", "--heuristics-only"])
        assert getattr(cli, fn)(None) == 0
        assert len(spy.calls) == 1
        ns = vars(spy.calls[0])
        assert ns["heuristics_only"] is True
        assert ns["command"] == cmd


# ---------------------------------------------------------------------------
# EB8 / EB9 -- deletion is real; no eager import
# ---------------------------------------------------------------------------

class TestEB8Deletion:
    def test_helpers_not_defined(self):
        tree = ast.parse(CLI_PY.read_text(encoding="utf-8"))
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        for gone in ("_add_common_args", "_add_batch_args",
                     "_build_extract_parser", "_build_validate_parser"):
            assert gone not in names, gone
            assert not hasattr(_cli(), gone), gone

    def test_no_argparse_construction_in_cli(self):
        text = CLI_PY.read_text(encoding="utf-8")
        assert "ArgumentParser(" not in text
        assert "add_argument(" not in text

    def test_add_argument_only_in_main(self):
        hits = sorted(p.relative_to(SRC_DIR).as_posix()
                      for p in SRC_DIR.rglob("*.py")
                      if "add_argument(" in p.read_text(encoding="utf-8"))
        assert hits == ["reflect_mcp/__main__.py"]

    def test_cli_line_count_cap(self):
        n = len(CLI_PY.read_text(encoding="utf-8").splitlines())
        assert n <= 310


class TestEB9NoEagerImport:
    def test_import_cli_does_not_import_main(self):
        code = ("import sys, reflect_mcp.cli; "
                "print('reflect_mcp.__main__' in sys.modules)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=60, cwd=str(REPO_ROOT), env=_env())
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# EB10 -- shipped surface unchanged vs HEAD
# ---------------------------------------------------------------------------

class TestEB10ShippedSurface:
    def test_main_py_byte_identical_to_base(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        wt = (REPO_ROOT / "src" / "reflect_mcp" / "__main__.py").read_bytes()
        assert wt == _git_show("src/reflect_mcp/__main__.py")

    @pytest.mark.parametrize("argv", [["--help"], ["extract", "--help"], ["validate", "--help"]])
    def test_console_help_byte_identical_to_base(self, head_tree, argv):
        wt = _console(*argv)
        env = _env(PYTHONPATH=str(head_tree / "src"))
        base = _console(*argv, cwd=head_tree, env=env)
        assert wt[0] == base[0] == 0
        assert wt[1] == base[1]
        assert wt[1].startswith("usage: reflect-mcp")

    def test_eval_dir_unchanged(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet", BASE_SHA, "--", "eval"],
                           capture_output=True, timeout=30)
        assert r.returncode == 0
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "eval"],
                           capture_output=True, text=True, timeout=30)
        assert r.stdout.strip() == ""

    def test_eval_stdout_byte_identical_to_base(self, head_tree):
        wt = subprocess.run([sys.executable, "eval/run_eval.py"], capture_output=True, text=True,
                            timeout=120, cwd=str(REPO_ROOT), env=_env())
        base = subprocess.run([sys.executable, "eval/run_eval.py"], capture_output=True, text=True,
                              timeout=120, cwd=str(head_tree),
                              env=_env(PYTHONPATH=str(head_tree / "src")))
        assert wt.returncode == base.returncode == 0
        assert wt.stdout == base.stdout

    def test_iter74_help_pins_unedited(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        wt = (REPO_ROOT / "tests" / "test_iter74_behavior.py").read_bytes()
        assert wt == _git_show("tests/test_iter74_behavior.py")


# ---------------------------------------------------------------------------
# EB11 -- migrations are mechanical
# ---------------------------------------------------------------------------

def _test_names(src: str) -> set[str]:
    tree = ast.parse(src)
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")}


class TestEB11Migrations:
    def test_iter68_pin_matches_cli_and_names_iter81(self):
        text = (REPO_ROOT / "tests" / "test_iter68_behavior.py").read_text(encoding="utf-8")
        m = re.search(r"^HEAD_CLI_LINES\s*=\s*(\d+)(.*)$", text, re.M)
        assert m
        n = len(CLI_PY.read_text(encoding="utf-8").splitlines())
        assert int(m.group(1)) == n
        assert "iter 81" in m.group(2)

    def test_iter79_literal_repinned(self):
        text = (REPO_ROOT / "tests" / "test_iter79_behavior.py").read_text(encoding="utf-8")
        assert "== 375" not in text
        n = len(CLI_PY.read_text(encoding="utf-8").splitlines())
        assert f"== {n}" in text

    def test_iter78_cap_unedited(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        wt = (REPO_ROOT / "tests" / "test_iter78_behavior.py").read_bytes()
        assert wt == _git_show("tests/test_iter78_behavior.py")
        assert b"assert n < 380" in wt

    def test_cli_helpers_keeps_all_tests(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        wt = (REPO_ROOT / "tests" / "test_cli_helpers.py").read_text(encoding="utf-8")
        base = _git_show("tests/test_cli_helpers.py").decode("utf-8")
        assert _test_names(base) <= _test_names(wt)
        assert "_build_top_parser" in wt

    def test_no_existing_test_deleted_and_edits_scoped(self):
        if not _base_reachable():
            pytest.skip(f"{BASE_SHA} not reachable")
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--name-status", BASE_SHA, "--", "tests"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        rows = [ln.split("\t") for ln in r.stdout.splitlines() if ln.strip()]
        assert not [p for s, *p in rows if s.startswith("D")]
        modified = {p[-1] for s, *p in rows if s.startswith("M")}
        # Spec names 3 files; test_iter80's history check (base vs working tree)
        # is necessarily re-pointed by the first src change after iter 80.
        allowed = {"tests/test_cli_helpers.py", "tests/test_iter68_behavior.py",
                   "tests/test_iter79_behavior.py", "tests/test_iter80_behavior.py"}
        assert modified <= allowed, modified - allowed
        for path in modified:
            base = _git_show(path).decode("utf-8")
            wt = (REPO_ROOT / path).read_text(encoding="utf-8")
            assert _test_names(base) <= _test_names(wt), path
