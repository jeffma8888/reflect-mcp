"""CLI entry points: `reflect-mcp extract` and `reflect-mcp validate`."""

from __future__ import annotations

import argparse
import json
import sys
import typing
from concurrent.futures import ThreadPoolExecutor


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


def _build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflect-mcp extract",
        description="Extract structured data from text via the reflective pipeline.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="JSONL batch mode: each stdin/file line is {\"text\": \"...\"}; "
        "emits one JSON result per line.",
    )
    parser.add_argument(
        "--batch-concurrency",
        dest="batch_concurrency",
        type=int,
        default=None,
        help="Number of concurrent workers for batch mode (default: 1). "
        "Requires --batch.",
    )
    return parser


def _build_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflect-mcp validate",
        description="Verify pre-extracted data against source text (Critic + heuristics + gate, no Generator).",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--data",
        dest="data",
        required=True,
        help="JSON string of pre-extracted fields to verify.",
    )
    return parser


def _read_text(args: argparse.Namespace) -> str | None:
    """Read source text from --file or stdin. Returns None on error (already printed)."""
    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return None
    else:
        if sys.stdin.isatty():
            print("Error: no input provided. Pipe text to stdin or use --file.", file=sys.stderr)
            return None
        return sys.stdin.read()


def _build_gate_config(args: argparse.Namespace) -> "GateConfig | None":
    from reflect_mcp.gate import GateConfig  # noqa: PLC0415

    gate_kwargs: dict[str, float] = {}
    if args.field_threshold is not None:
        gate_kwargs["field_threshold"] = args.field_threshold
    if args.overall_threshold is not None:
        gate_kwargs["overall_threshold"] = args.overall_threshold
    return GateConfig(**gate_kwargs) if gate_kwargs else None


def _read_input_stream(args: argparse.Namespace) -> "typing.TextIO | None":
    """Open the input source as a text stream. Returns None on error (already printed)."""
    if args.file is not None:
        try:
            return open(args.file, encoding="utf-8")  # noqa: SIM115
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return None
    else:
        if sys.stdin.isatty():
            print("Error: no input provided. Pipe text to stdin or use --file.", file=sys.stderr)
            return None
        return sys.stdin


def _run_batch(
    pipeline: "ReflectivePipeline",
    stream: "typing.TextIO",
) -> int:
    """Process JSONL lines from *stream*, print one JSON result per line, return exit code."""
    worst = 0
    for raw_line in stream:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"status": "error", "error_message": f"invalid JSON: {exc}"}))
            worst = 3
            continue

        if not isinstance(record, dict) or "text" not in record:
            print(json.dumps({"status": "error", "error_message": "missing \"text\" key"}))
            worst = 3
            continue

        try:
            result = pipeline.extract(record["text"])
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"status": "error", "error_message": str(exc)}))
            worst = 3
            continue

        print(json.dumps(result.model_dump()))

        if result.status == "needs_review" and worst < 3:
            worst = max(worst, 1)
    return worst


def _process_one_line(
    pipeline: "ReflectivePipeline",
    raw_line: str,
) -> tuple[str, int]:
    """Extract from a single JSONL line. Returns (json_output, exit_code)."""
    line = raw_line.rstrip("\n\r")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "error_message": f"invalid JSON: {exc}"}), 3

    if not isinstance(record, dict) or "text" not in record:
        return json.dumps({"status": "error", "error_message": 'missing "text" key'}), 3

    try:
        result = pipeline.extract(record["text"])
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error_message": str(exc)}), 3

    code = 1 if result.status == "needs_review" else 0
    return json.dumps(result.model_dump()), code


def _run_batch_concurrent(
    build_pipeline: typing.Callable[[], "ReflectivePipeline"],
    stream: "typing.TextIO",
    concurrency: int,
) -> int:
    """Process JSONL lines concurrently, preserving output order."""
    import threading  # noqa: PLC0415

    lines = [raw for raw in stream if raw.rstrip("\n\r")]
    if not lines:
        return 0

    pipelines: dict[int, "ReflectivePipeline"] = {}
    lock = threading.Lock()

    def _worker(item: tuple[int, str]) -> tuple[str, int]:
        idx, raw_line = item
        tid = _thread_id()
        with lock:
            if tid not in pipelines:
                pipelines[tid] = build_pipeline()
            pl = pipelines[tid]
        return _process_one_line(pl, raw_line)

    worst = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for output_json, code in executor.map(_worker, enumerate(lines)):
            print(output_json)
            if code == 3:
                worst = 3
            elif code == 1 and worst < 3:
                worst = max(worst, 1)
    return worst


def _thread_id() -> int:
    """Return current thread's identity for per-thread pipeline keying."""
    import threading  # noqa: PLC0415

    return threading.get_ident()


def run_extract(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print JSON to stdout, return exit code."""
    parser = _build_extract_parser()
    args = parser.parse_args(argv)

    if args.batch_concurrency is not None and not args.batch:
        print(
            "Error: --batch-concurrency requires --batch.",
            file=sys.stderr,
        )
        return 2

    if args.batch:
        return _run_extract_batch(args)

    text = _read_text(args)
    if text is None:
        return 2

    if not text.strip():
        print("Error: input is empty.", file=sys.stderr)
        return 2

    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    try:
        gate_config = _build_gate_config(args)
        if gate_config:
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


def _run_extract_batch(args: argparse.Namespace) -> int:
    """Batch mode: build pipeline once (serial) or per-thread (concurrent), return worst exit code."""
    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    stream = _read_input_stream(args)
    if stream is None:
        return 2

    concurrency = args.batch_concurrency if args.batch_concurrency is not None else 1

    if concurrency > 1:
        gate_config = _build_gate_config(args)

        def _build() -> "ReflectivePipeline":
            if gate_config:
                return build_default_pipeline(gate_config=gate_config)
            return build_default_pipeline()

        try:
            return _run_batch_concurrent(_build, stream, concurrency)
        finally:
            if stream is not sys.stdin:
                stream.close()

    try:
        gate_config = _build_gate_config(args)
        if gate_config:
            pipeline = build_default_pipeline(gate_config=gate_config)
        else:
            pipeline = build_default_pipeline()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_message": str(exc)}))
        return 3

    try:
        return _run_batch(pipeline, stream)
    finally:
        if stream is not sys.stdin:
            stream.close()


def run_validate(argv: list[str] | None = None) -> int:
    """Parse args, validate pre-extracted data, print JSON to stdout, return exit code."""
    parser = _build_validate_parser()
    args = parser.parse_args(argv)

    text = _read_text(args)
    if text is None:
        return 2

    if not text.strip():
        print("Error: input is empty.", file=sys.stderr)
        return 2

    try:
        data_dict = json.loads(args.data)
    except json.JSONDecodeError as exc:
        print(f"Error: --data is not valid JSON: {exc}", file=sys.stderr)
        return 2

    from pydantic import ValidationError  # noqa: PLC0415

    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415
    from reflect_mcp.schema import InvoiceData  # noqa: PLC0415

    try:
        data = InvoiceData.model_validate(data_dict)
    except ValidationError as exc:
        print(f"Error: --data failed InvoiceData validation: {exc}", file=sys.stderr)
        return 2

    try:
        gate_config = _build_gate_config(args)
        if gate_config:
            pipeline = build_default_pipeline(gate_config=gate_config)
        else:
            pipeline = build_default_pipeline()
        result = pipeline.validate(text, data)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_message": str(exc)}))
        return 3

    print(json.dumps(result.model_dump(), indent=2))

    if result.status == "ok":
        return 0
    return 1
