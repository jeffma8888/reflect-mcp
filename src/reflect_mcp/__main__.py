"""Top-level entry point for the `reflect-mcp` console script.

Dispatches to subcommands (`extract`, `validate`) or falls through to the
MCP server, preserving backward compatibility: bare `reflect-mcp` still starts
the server exactly as before.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "extract":
        from reflect_mcp.cli import run_extract  # noqa: PLC0415

        sys.exit(run_extract(sys.argv[2:]))

    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        from reflect_mcp.cli import run_validate  # noqa: PLC0415

        sys.exit(run_validate(sys.argv[2:]))

    from reflect_mcp.server import main as server_main  # noqa: PLC0415

    server_main()


if __name__ == "__main__":
    main()
