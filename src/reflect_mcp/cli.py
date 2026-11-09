"""CLI entry points: `reflect-mcp extract` and `reflect-mcp validate`."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import typing
from concurrent.futures import Future, ThreadPoolExecutor


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


def _build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflect-mcp extract",
        description="Extract structured data from text via the reflective pipeline.",
    )
    _add_common_args(parser)
    _add_batch_args(parser, '{"text": "..."}')
    return parser


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


def _build_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflect-mcp validate",
        description="Verify pre-extracted data against source text (Critic + heuristics + gate, no Generator).",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--data",
        dest="data",
        help="JSON string of pre-extracted fields to verify (required unless --batch).",
    )
    _add_batch_args(parser, '{"text": "...", "data": {...}}')
    return parser


def _read_text(args: argparse.Namespace) -> str | None:
    """Read the whole input. Delegates to ``_read_input_stream`` so single, batch
    and validate share one set of input-error messages. Returns None on error."""
    stream = _read_input_stream(args)
    if stream is None:
        return None
    try:
        return stream.read()
    finally:
        _close_input(stream)


def _build_gate_config(args: argparse.Namespace) -> "GateConfig | None":
    from reflect_mcp.gate import GateConfig  # noqa: PLC0415

    gate_kwargs: dict[str, float] = {}
    if args.field_threshold is not None:
        gate_kwargs["field_threshold"] = args.field_threshold
    if args.overall_threshold is not None:
        gate_kwargs["overall_threshold"] = args.overall_threshold
    return GateConfig(**gate_kwargs) if gate_kwargs else None


def _build_pipeline_kwargs(args: argparse.Namespace) -> dict[str, typing.Any]:
    """Build kwargs for ``build_default_pipeline`` from parsed CLI args."""
    kwargs: dict[str, typing.Any] = {}
    gate_config = _build_gate_config(args)
    if gate_config is not None:
        kwargs["gate_config"] = gate_config
    if getattr(args, "heuristics_only", False):
        kwargs["critic_enabled"] = False
    return kwargs


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


def _close_input(stream: "typing.TextIO") -> None:
    """Close a file we opened; never close the process-wide stdin."""
    if stream is not sys.stdin:
        stream.close()


def _error_json(message: str, echo: dict[str, typing.Any] | None = None) -> str:
    """The one failure envelope every CLI path prints; *echo* (a batch id) leads so rows join."""
    return json.dumps({**(echo or {}), "status": "error", "error_message": message})


def _process_one_line(
    pipeline: "ReflectivePipeline",
    raw_line: str,
    validate: bool = False,
) -> tuple[str, int]:
    """Extract (or *validate* the ``data`` of) one JSONL line -> (json_output, exit_code)."""
    line = raw_line.rstrip("\n\r")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error_json(f"invalid JSON: {exc}"), 3
    # Echo the caller's id verbatim as the first key so each row joins to its record.
    echo = {"id": record["id"]} if isinstance(record, dict) and "id" in record else {}
    if not isinstance(record, dict) or "text" not in record:
        return _error_json('missing "text" key', echo), 3
    if validate and "data" not in record:
        return _error_json('missing "data" key', echo), 3

    try:  # ``data`` is schema-checked in here, so a bad record is that line's envelope
        if validate:
            from reflect_mcp.schema import InvoiceData  # noqa: PLC0415

            result = pipeline.validate(record["text"], InvoiceData.model_validate(record["data"], context={"caller": True}))
        else:
            result = pipeline.extract(record["text"])
    except Exception as exc:  # noqa: BLE001
        return _error_json(str(exc), echo), 3

    code = 1 if result.status == "needs_review" else 0
    return json.dumps({**echo, **result.model_dump()}), code


def _line_processor(
    build_pipeline: typing.Callable[[], "ReflectivePipeline"],
    first: "ReflectivePipeline",
    validate: bool = False,
) -> typing.Callable[[str], tuple[str, int]]:
    """Return a per-line function that gives each thread its own pipeline.

    SDK clients are not shared across threads, so every thread lazily takes a
    pipeline: the eagerly-built *first* one goes to whichever thread asks first
    (the main thread when serial, so serial builds exactly one), later threads
    build their own. A late build failure becomes that line's error envelope
    instead of escaping the executor as a traceback.
    """
    local = threading.local()
    spares = [first]
    lock = threading.Lock()

    def _process(raw_line: str) -> tuple[str, int]:
        pipeline = getattr(local, "pipeline", None)
        if pipeline is None:
            try:
                with lock:
                    pipeline = spares.pop() if spares else build_pipeline()
            except Exception as exc:  # noqa: BLE001
                return _error_json(str(exc)), 3
            local.pipeline = pipeline
        return _process_one_line(pipeline, raw_line, validate)

    return _process


def _windowed(pool: ThreadPoolExecutor, process: typing.Callable[[str], tuple[str, int]],
              lines: typing.Iterable[str], size: int) -> typing.Iterator[tuple[str, int]]:
    """Yield results in input order, <= *size* in flight (``executor.map`` reads all input first)."""
    window: collections.deque[Future[tuple[str, int]]] = collections.deque()
    for raw in lines:
        window.append(pool.submit(process, raw))
        while window and (window[0].done() or len(window) >= size):
            yield window.popleft().result()
    while window:
        yield window.popleft().result()


def _run_batch(
    stream: "typing.TextIO",
    process: typing.Callable[[str], tuple[str, int]],
    concurrency: int,
) -> int:
    """The one batch driver: print one result per non-blank line in input order.

    Serial uses builtin ``map`` so the stream is consumed lazily (line N is
    printed before line N+1 is read); concurrent keeps at most ``2 * concurrency``
    lines pending, so output streams in input order with bounded memory. Exit
    codes rank ok 0 < needs_review 1 < error 3, so the worst is simply the max.
    """
    lines = (raw for raw in stream if raw.rstrip("\n\r"))
    pool = ThreadPoolExecutor(max_workers=concurrency) if concurrency > 1 else None
    worst = 0
    try:
        results = _windowed(pool, process, lines, 2 * concurrency) if pool else map(process, lines)
        for output_json, code in results:
            worst = max(worst, code)  # before print: a row whose write fails still counts
            print(output_json)
    except BrokenPipeError:  # reader closed early (`| head`): stop pulling input
        pass
    finally:
        if pool is not None:
            pool.shutdown()
    return worst


def run_extract(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print JSON to stdout, return exit code."""
    parser = _build_extract_parser()
    args = parser.parse_args(argv)
    return run_extract_from_args(args)


def run_extract_from_args(args: argparse.Namespace) -> int:
    """Run extract from a pre-parsed *args* namespace, return exit code."""
    if args.batch_concurrency is not None and not args.batch:
        print("Error: --batch-concurrency requires --batch.", file=sys.stderr)
        return 2
    if args.batch:
        return _run_jsonl_batch(args)

    text = _read_text(args)
    if text is None:
        return 2

    if not text.strip():
        print("Error: input is empty.", file=sys.stderr)
        return 2

    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    try:
        pipeline_kwargs = _build_pipeline_kwargs(args)
        pipeline = build_default_pipeline(**pipeline_kwargs)
        result = pipeline.extract(text)
    except Exception as exc:  # noqa: BLE001
        print(_error_json(str(exc)))
        return 3

    print(json.dumps(result.model_dump(), indent=2))

    if result.status == "ok":
        return 0
    return 1


def _run_jsonl_batch(args: argparse.Namespace, validate: bool = False) -> int:
    """Batch mode for either verb at any concurrency, return the worst exit code.

    The pipeline is built eagerly in the main thread, before any line is read,
    so a provider/config failure yields one error envelope and exit 3 at every
    ``--batch-concurrency`` (and on empty input), never a traceback or exit 1.
    """
    from reflect_mcp.pipeline import build_default_pipeline  # noqa: PLC0415

    stream = _read_input_stream(args)
    if stream is None:
        return 2

    try:
        try:
            pipeline_kwargs = _build_pipeline_kwargs(args)
            if validate:  # only validate adds the key, so extract kwargs stay as before
                pipeline_kwargs["generator_enabled"] = False
            first = build_default_pipeline(**pipeline_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(_error_json(str(exc)))
            return 3
        rebuild = lambda: build_default_pipeline(**pipeline_kwargs)  # noqa: E731
        process = _line_processor(rebuild, first, validate)
        concurrency = args.batch_concurrency if args.batch_concurrency is not None else 1
        return _run_batch(stream, process, concurrency)
    finally:
        _close_input(stream)


def run_validate(argv: list[str] | None = None) -> int:
    """Parse args, validate pre-extracted data, print JSON to stdout, return exit code."""
    parser = _build_validate_parser()
    args = parser.parse_args(argv)
    return run_validate_from_args(args)


def run_validate_from_args(args: argparse.Namespace) -> int:
    """Run validate from a pre-parsed *args* namespace, return exit code."""
    if args.batch_concurrency is not None and not args.batch:
        print("Error: --batch-concurrency requires --batch.", file=sys.stderr)
        return 2
    if args.batch != (args.data is None):
        print("Error: validate needs exactly one of --data or --batch.", file=sys.stderr)
        return 2
    if args.batch:
        return _run_jsonl_batch(args, validate=True)
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
        data = InvoiceData.model_validate(data_dict, context={"caller": True})
    except ValidationError as exc:
        print(f"Error: --data failed InvoiceData validation: {exc}", file=sys.stderr)
        return 2

    try:
        pipeline_kwargs = _build_pipeline_kwargs(args)
        pipeline = build_default_pipeline(**pipeline_kwargs, generator_enabled=False)
        result = pipeline.validate(text, data)
    except Exception as exc:  # noqa: BLE001
        print(_error_json(str(exc)))
        return 3

    print(json.dumps(result.model_dump(), indent=2))

    if result.status == "ok":
        return 0
    return 1
