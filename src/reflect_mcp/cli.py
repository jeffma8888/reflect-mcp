"""CLI entry point: `reflect-mcp extract` for pipe-friendly one-shot extraction."""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflect-mcp extract",
        description="Extract structured data from text via the reflective pipeline.",
    )
    parser.add_argument(
        "--file",
        dest="file",
        default=None,
        help="Path to a text file to extract from (default: read stdin).",
    )
    return parser


def run_extract(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print JSON to stdout, return exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 2
    else:
        if sys.stdin.isatty():
            print("Error: no input provided. Pipe text to stdin or use --file.", file=sys.stderr)
            return 2
        text = sys.stdin.read()

    if not text.strip():
        print("Error: input is empty.", file=sys.stderr)
        return 2

    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    try:
        pipeline = build_default_pipeline()
        result = pipeline.extract(text)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_message": str(exc)}))
        return 3

    print(json.dumps(result.model_dump(), indent=2))

    if result.status == "ok":
        return 0
    return 1
