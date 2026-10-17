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

    from reflect_mcp.gate import GateConfig  # noqa: PLC0415
    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    gate_kwargs: dict[str, float] = {}
    if args.field_threshold is not None:
        gate_kwargs["field_threshold"] = args.field_threshold
    if args.overall_threshold is not None:
        gate_kwargs["overall_threshold"] = args.overall_threshold

    try:
        if gate_kwargs:
            gate_config = GateConfig(**gate_kwargs)
            pipeline = build_default_pipeline(gate_config=gate_config)
        else:
            pipeline = build_default_pipeline()
        result = pipeline.extract(text)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_message": str(exc)}))
        return 3

    print(json.dumps(result.model_dump(), indent=2))

    if result.status == "ok":
        return 0
    return 1
