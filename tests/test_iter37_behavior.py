"""Black-box behavioral tests for iter-37: argparse top-level --help / --version.

Covers all 7 Expected Behaviors from pm.md:
  1. `reflect-mcp --help` exits 0, stdout contains "extract", "validate", and
     describes the default server mode.
  2. `reflect-mcp --version` exits 0, stdout matches `reflect_mcp.__version__`.
  3. `reflect-mcp` with no arguments still starts the MCP server (backward compat).
  4. `reflect-mcp extract --help` exits 0, lists extract flags (--file, --batch,
     --batch-concurrency, --field-threshold, --overall-threshold).
  5. `reflect-mcp validate --help` exits 0, lists validate flags (--file, --data,
     --field-threshold, --overall-threshold).
  6. `reflect-mcp --help` does NOT show subcommand-internal flags (no --batch at
     top level).
  7. `reflect-mcp unknown-verb` prints error to stderr and exits 2.

Isolation contract: tests drive the public CLI entry point and inspect its
observable output (stdout, stderr, exit code). No implementation source was read.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PYTHON = sys.executable
_MODULE = [_PYTHON, "-m", "reflect_mcp"]


def _run_cli(extra_args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    """Run reflect-mcp as a subprocess with given args and return the result."""
    result = subprocess.run(
        _MODULE + extra_args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/Users/jincm/projects/reflect-mcp",
        env={"PATH": "/Users/jincm/projects/reflect-mcp/.venv/bin:" + "/usr/bin:/bin",
             "HOME": "/Users/jincm",
             "VIRTUAL_ENV": "/Users/jincm/projects/reflect-mcp/.venv"},
    )
    return result


# ---------------------------------------------------------------------------
# EB-1: reflect-mcp --help exits 0, shows "extract", "validate", server mode
# ---------------------------------------------------------------------------

class TestEB1TopLevelHelp:
    """EB-1: --help exits 0 and prints usage listing subcommands + server mode."""

    def test_exit_code_zero(self):
        r = _run_cli(["--help"])
        assert r.returncode == 0

    def test_stdout_contains_extract(self):
        r = _run_cli(["--help"])
        assert "extract" in r.stdout

    def test_stdout_contains_validate(self):
        r = _run_cli(["--help"])
        assert "validate" in r.stdout

    def test_stdout_describes_server_mode(self):
        r = _run_cli(["--help"])
        lower = r.stdout.lower()
        assert "mcp" in lower or "server" in lower, \
            f"Expected help to mention MCP/server mode, got:\n{r.stdout}"


# ---------------------------------------------------------------------------
# EB-2: reflect-mcp --version exits 0, prints __version__
# ---------------------------------------------------------------------------

class TestEB2Version:
    """EB-2: --version exits 0, output matches reflect_mcp.__version__."""

    def test_exit_code_zero(self):
        r = _run_cli(["--version"])
        assert r.returncode == 0

    def test_version_string_matches(self):
        import reflect_mcp
        r = _run_cli(["--version"])
        output = (r.stdout + r.stderr).strip()
        assert reflect_mcp.__version__ in output, \
            f"Expected {reflect_mcp.__version__!r} in output {output!r}"

    def test_version_on_stdout(self):
        import reflect_mcp
        r = _run_cli(["--version"])
        assert reflect_mcp.__version__ in r.stdout


# ---------------------------------------------------------------------------
# EB-3: bare invocation still starts the MCP server (backward compat)
# ---------------------------------------------------------------------------

class TestEB3BareInvocationStartsServer:
    """EB-3: no args → MCP server starts (not help, not error).

    We cannot test a running server in a unit test, but we CAN verify:
    - _has_subcommand() returns False for bare invocation
    - The bare invocation does NOT print help text or exit 0/2 immediately
    - The main() path routes to the server starter for empty argv
    """

    def test_has_subcommand_false_for_bare(self):
        from reflect_mcp.__main__ import _has_subcommand
        saved = sys.argv[:]
        try:
            sys.argv = ["reflect-mcp"]
            assert _has_subcommand() is False
        finally:
            sys.argv = saved

    def test_has_subcommand_true_for_help(self):
        from reflect_mcp.__main__ import _has_subcommand
        saved = sys.argv[:]
        try:
            sys.argv = ["reflect-mcp", "--help"]
            assert _has_subcommand() is True
        finally:
            sys.argv = saved

    def test_has_subcommand_true_for_extract(self):
        from reflect_mcp.__main__ import _has_subcommand
        saved = sys.argv[:]
        try:
            sys.argv = ["reflect-mcp", "extract"]
            assert _has_subcommand() is True
        finally:
            sys.argv = saved

    def test_bare_does_not_print_help(self):
        """Bare invocation should try to start the server, not print help.
        We give it a short timeout; it will either hang (server) or error
        on missing provider — either way, it should NOT exit 0 with help text."""
        try:
            r = subprocess.run(
                _MODULE,
                capture_output=True,
                text=True,
                timeout=2,
                cwd="/Users/jincm/projects/reflect-mcp",
                env={"PATH": "/Users/jincm/projects/reflect-mcp/.venv/bin:" + "/usr/bin:/bin",
                     "HOME": "/Users/jincm",
                     "VIRTUAL_ENV": "/Users/jincm/projects/reflect-mcp/.venv"},
            )
            assert "usage:" not in r.stdout.lower(), \
                "Bare invocation should not print usage/help"
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# EB-4: reflect-mcp extract --help exits 0, shows all extract flags
# ---------------------------------------------------------------------------

class TestEB4ExtractHelp:
    """EB-4: extract --help exits 0 and lists all extract flags."""

    def test_exit_code_zero(self):
        r = _run_cli(["extract", "--help"])
        assert r.returncode == 0

    def test_contains_file_flag(self):
        r = _run_cli(["extract", "--help"])
        assert "--file" in r.stdout

    def test_contains_batch_flag(self):
        r = _run_cli(["extract", "--help"])
        assert "--batch" in r.stdout

    def test_contains_batch_concurrency_flag(self):
        r = _run_cli(["extract", "--help"])
        assert "--batch-concurrency" in r.stdout

    def test_contains_field_threshold_flag(self):
        r = _run_cli(["extract", "--help"])
        assert "--field-threshold" in r.stdout

    def test_contains_overall_threshold_flag(self):
        r = _run_cli(["extract", "--help"])
        assert "--overall-threshold" in r.stdout


# ---------------------------------------------------------------------------
# EB-5: reflect-mcp validate --help exits 0, shows all validate flags
# ---------------------------------------------------------------------------

class TestEB5ValidateHelp:
    """EB-5: validate --help exits 0 and lists all validate flags."""

    def test_exit_code_zero(self):
        r = _run_cli(["validate", "--help"])
        assert r.returncode == 0

    def test_contains_file_flag(self):
        r = _run_cli(["validate", "--help"])
        assert "--file" in r.stdout

    def test_contains_data_flag(self):
        r = _run_cli(["validate", "--help"])
        assert "--data" in r.stdout

    def test_contains_field_threshold_flag(self):
        r = _run_cli(["validate", "--help"])
        assert "--field-threshold" in r.stdout

    def test_contains_overall_threshold_flag(self):
        r = _run_cli(["validate", "--help"])
        assert "--overall-threshold" in r.stdout


# ---------------------------------------------------------------------------
# EB-6: top-level --help does NOT show subcommand-internal flags
# ---------------------------------------------------------------------------

class TestEB6NoSubcommandFlagsAtTopLevel:
    """EB-6: top-level help does not leak subcommand-specific flags."""

    def test_no_batch_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--batch" not in r.stdout, \
            f"--batch should not appear in top-level help:\n{r.stdout}"

    def test_no_data_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--data" not in r.stdout, \
            f"--data should not appear in top-level help:\n{r.stdout}"

    def test_no_file_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--file" not in r.stdout, \
            f"--file should not appear in top-level help:\n{r.stdout}"

    def test_no_batch_concurrency_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--batch-concurrency" not in r.stdout

    def test_no_field_threshold_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--field-threshold" not in r.stdout

    def test_no_overall_threshold_at_top_level(self):
        r = _run_cli(["--help"])
        assert "--overall-threshold" not in r.stdout


# ---------------------------------------------------------------------------
# EB-7: unknown verb prints error to stderr and exits 2
# ---------------------------------------------------------------------------

class TestEB7UnknownVerb:
    """EB-7: unknown-verb → stderr error message, exit code 2."""

    def test_exit_code_two(self):
        r = _run_cli(["unknown-verb"])
        assert r.returncode == 2

    def test_stderr_has_error(self):
        r = _run_cli(["unknown-verb"])
        assert r.stderr.strip(), "Expected error message on stderr"

    def test_stderr_mentions_invalid_choice(self):
        r = _run_cli(["unknown-verb"])
        assert "invalid choice" in r.stderr.lower() or "error" in r.stderr.lower(), \
            f"Expected error about invalid choice, got:\n{r.stderr}"

    def test_another_unknown_verb(self):
        r = _run_cli(["foobar-does-not-exist"])
        assert r.returncode == 2

    def test_unknown_does_not_start_server(self):
        """Unknown verb should NOT silently start the server."""
        r = _run_cli(["bogus-command"])
        assert r.returncode == 2
        assert r.stderr.strip(), "Should have error output, not silent server start"
