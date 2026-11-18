"""Top-level entry point for the `reflect-mcp` console script.

Uses argparse with subparsers so that ``reflect-mcp --help`` lists available
subcommands and ``reflect-mcp --version`` prints the package version.
Bare ``reflect-mcp`` (no arguments) still starts the MCP server for backward
compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import typing

from reflect_mcp import __version__

# Provider -> API-key env var a host must set; keys mirror providers._CLIENTS
# (a test pins that), and bedrock is None because it uses the AWS credential chain.
_KEY_VARS: dict[str, str | None] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "bedrock": None}
# REFLECT_* vars the server reads, echoed in this order when set so the printed
# block reproduces the operator's current configuration.
_ECHO_VARS = (
    "REFLECT_GENERATOR_MODEL", "REFLECT_CRITIC_MODEL",
    "REFLECT_FIELD_THRESHOLD", "REFLECT_OVERALL_THRESHOLD", "REFLECT_LOG_LEVEL",
)
_KEY_PLACEHOLDER = "<your-api-key>"


def _build_top_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="reflect-mcp",
        description=(
            "Self-verifying extraction pipeline exposed over MCP. "
            "Run with no arguments to start the MCP server."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--mcp-config",
        dest="mcp_config",
        action="store_true",
        default=False,
        help="Print a paste-ready mcpServers JSON block for this install and exit.",
    )

    sub = parser.add_subparsers(dest="command")

    extract_parser = sub.add_parser(
        "extract",
        help="Extract structured data from text via the reflective pipeline.",
    )
    _populate_extract_parser(extract_parser)

    validate_parser = sub.add_parser(
        "validate",
        help="Verify pre-extracted data against source text.",
    )
    _populate_validate_parser(validate_parser)

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --file, threshold, and heuristics-only flags shared by extract and validate."""
    parser.add_argument(
        "--file",
        dest="file",
        default=None,
        help="Path to a text file (default: read stdin).",
    )
    parser.add_argument(
        "--field-threshold",
        dest="field_threshold",
        type=float,
        default=None,
        help="Per-field confidence threshold in [0.0, 1.0] (default: 0.70).",
    )
    parser.add_argument(
        "--overall-threshold",
        dest="overall_threshold",
        type=float,
        default=None,
        help="Overall confidence threshold in [0.0, 1.0] (default: 0.75).",
    )
    parser.add_argument(
        "--heuristics-only",
        dest="heuristics_only",
        action="store_true",
        default=False,
        help="Skip the Critic LLM call; use only deterministic heuristics for verification.",
    )


def _populate_extract_parser(parser: argparse.ArgumentParser) -> None:
    """Add extract-specific arguments to *parser*."""
    _add_common_args(parser)
    _add_batch_args(parser, '{"text": "..."}')


def _add_batch_args(parser: argparse.ArgumentParser, line_shape: str) -> None:
    """Add --batch/--batch-concurrency; *line_shape* is the verb's JSONL record."""
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help=f"JSONL batch mode: each stdin/file line is {line_shape}; "
        "emits one JSON result per line.",
    )
    parser.add_argument(
        "--batch-concurrency",
        dest="batch_concurrency",
        type=int,
        default=None,
        help="Number of concurrent workers for batch mode (default: 1). Requires --batch.",
    )


def _populate_validate_parser(parser: argparse.ArgumentParser) -> None:
    """Add validate-specific arguments to *parser*."""
    _add_common_args(parser)
    parser.add_argument(
        "--data",
        dest="data",
        help="JSON string of pre-extracted fields to verify (required unless --batch).",
    )
    _add_batch_args(parser, '{"text": "...", "data": {...}}')


def main() -> None:
    if not _has_subcommand():
        from reflect_mcp.server import main as server_main  # noqa: PLC0415

        server_main()
        return

    parser = _build_top_parser()
    args = parser.parse_args()
    if getattr(args, "file", None) == "-":  # POSIX convention: "-" names stdin
        args.file = None
    if args.mcp_config:
        sys.exit(_print_mcp_config())

    if args.command == "extract":
        from reflect_mcp.cli import run_extract_from_args  # noqa: PLC0415

        _exit_with_verb(run_extract_from_args, args)

    if args.command == "validate":
        from reflect_mcp.cli import run_validate_from_args  # noqa: PLC0415

        _exit_with_verb(run_validate_from_args, args)

    from reflect_mcp.server import main as server_main  # noqa: PLC0415

    server_main()


def _print_mcp_config() -> int:
    """Print an MCP client-config block naming this interpreter; return the exit code.

    Hosts such as Claude Desktop do not inherit the venv's PATH, so the block
    needs the absolute ``sys.executable``, which a user would otherwise hand-type.
    The API key is always a placeholder: a printed block gets pasted and shared,
    so a real secret must never reach stdout.
    """
    provider = os.getenv("REFLECT_PROVIDER", "anthropic").lower()  # as build_client reads it
    if provider not in _KEY_VARS:
        print(f"reflect-mcp: error: unknown REFLECT_PROVIDER={provider!r}; "
              f"choose one of: {', '.join(_KEY_VARS)}.", file=sys.stderr)
        return 2
    env = {"REFLECT_PROVIDER": provider}
    env.update({name: os.environ[name] for name in _ECHO_VARS if os.environ.get(name)})
    key_var = _KEY_VARS[provider]
    if key_var is not None:
        env[key_var] = _KEY_PLACEHOLDER
    server = {"command": sys.executable, "args": ["-m", "reflect_mcp.server"], "env": env}
    print(json.dumps({"mcpServers": {"reflect-mcp": server}}, indent=2))
    return 0


def _exit_with_verb(
    verb: typing.Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> typing.NoReturn:
    """Run a CLI verb and exit with its code, even if stdout's reader closed early.

    ``| head -1`` closes the pipe, so a write or the final flush raises
    ``BrokenPipeError``; left alone, Python prints a traceback and exits 120,
    breaking the documented 0/1/2/3 contract. Pointing fd 1 at devnull silences
    the interpreter's exit-time flush. The fd surgery lives here, not in cli.py,
    because in-process callers (pytest capture) have no real stdout fd. A verb
    that hit EPIPE before returning exits 1: fail-safe, never a silent 0.
    """
    code = 1
    try:
        code = verb(args)
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(code)


def _has_subcommand() -> bool:
    """Return True if sys.argv has any arguments at all.

    Bare ``reflect-mcp`` (no arguments) must start the MCP server without
    invoking argparse, because argparse would intercept stdin that the
    server needs. Any argument — known subcommand, flag, or unknown verb —
    goes through argparse so it can produce proper help/error output.
    """
    return len(sys.argv) > 1


if __name__ == "__main__":
    main()
