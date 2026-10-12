"""Unit tests for CLI internal helpers (parser construction, dispatch routing)."""

from __future__ import annotations

from reflect_mcp.cli import _build_parser


def test_parser_accepts_file_flag() -> None:
    parser = _build_parser()
    ns = parser.parse_args(["--file", "invoice.txt"])
    assert ns.file == "invoice.txt"


def test_parser_defaults_file_to_none() -> None:
    parser = _build_parser()
    ns = parser.parse_args([])
    assert ns.file is None


def test_main_dispatches_extract() -> None:
    """Bare smoke test: `__main__.main` routes 'extract --file ...' to cli."""
    import sys
    from unittest.mock import patch

    with patch("reflect_mcp.cli.run_extract", return_value=0) as mock_run:
        with patch.object(sys, "argv", ["reflect-mcp", "extract", "--file", "f.txt"]):
            from reflect_mcp.__main__ import main

            try:
                main()
            except SystemExit as exc:
                assert exc.code == 0
            mock_run.assert_called_once_with(["--file", "f.txt"])
