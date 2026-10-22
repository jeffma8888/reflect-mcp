"""Unit tests for CLI internal helpers (parser construction, dispatch routing)."""

from __future__ import annotations

from reflect_mcp.cli import _build_extract_parser


def test_parser_accepts_file_flag() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args(["--file", "invoice.txt"])
    assert ns.file == "invoice.txt"


def test_parser_defaults_file_to_none() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args([])
    assert ns.file is None


def test_parser_accepts_field_threshold() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args(["--field-threshold", "0.85"])
    assert ns.field_threshold == 0.85


def test_parser_accepts_overall_threshold() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args(["--overall-threshold", "0.80"])
    assert ns.overall_threshold == 0.80


def test_parser_accepts_both_thresholds() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args(["--field-threshold", "0.85", "--overall-threshold", "0.80"])
    assert ns.field_threshold == 0.85
    assert ns.overall_threshold == 0.80


def test_parser_defaults_thresholds_to_none() -> None:
    parser = _build_extract_parser()
    ns = parser.parse_args([])
    assert ns.field_threshold is None
    assert ns.overall_threshold is None


def test_main_dispatches_extract() -> None:
    """Bare smoke test: `__main__.main` routes 'extract --file ...' to cli."""
    import sys
    from unittest.mock import patch

    with patch("reflect_mcp.cli.run_extract_from_args") as mock_run:
        mock_run.return_value = 0
        with patch.object(sys, "argv", ["reflect-mcp", "extract", "--file", "f.txt"]):
            from reflect_mcp.__main__ import main

            try:
                main()
            except SystemExit as exc:
                assert exc.code == 0
            mock_run.assert_called_once()
            ns = mock_run.call_args[0][0]
            assert ns.file == "f.txt"
            assert ns.command == "extract"
