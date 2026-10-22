"""Top-level entry point for the `reflect-mcp` console script.

Uses argparse with subparsers so that ``reflect-mcp --help`` lists available
subcommands and ``reflect-mcp --version`` prints the package version.
Bare ``reflect-mcp`` (no arguments) still starts the MCP server for backward
compatibility.
"""

from __future__ import annotations

import argparse
import sys

from reflect_mcp import __version__


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
    """Add --file and threshold flags shared by extract and validate."""
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


def _populate_extract_parser(parser: argparse.ArgumentParser) -> None:
    """Add extract-specific arguments to *parser*."""
    _add_common_args(parser)
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help=(
            "JSONL batch mode: each stdin/file line is {\"text\": \"...\"}; "
            "emits one JSON result per line."
        ),
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
        required=True,
        help="JSON string of pre-extracted fields to verify.",
    )


def main() -> None:
    if not _has_subcommand():
        from reflect_mcp.server import main as server_main  # noqa: PLC0415

        server_main()
        return

    parser = _build_top_parser()
    args = parser.parse_args()

    if args.command == "extract":
        from reflect_mcp.cli import run_extract_from_args  # noqa: PLC0415

        sys.exit(run_extract_from_args(args))

    if args.command == "validate":
        from reflect_mcp.cli import run_validate_from_args  # noqa: PLC0415

        sys.exit(run_validate_from_args(args))

    from reflect_mcp.server import main as server_main  # noqa: PLC0415

    server_main()


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
