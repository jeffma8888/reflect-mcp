"""Drift test: every symbol in reflect_mcp.__all__ appears in the README."""

from pathlib import Path

import reflect_mcp


def _readme_text() -> str:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    return readme.read_text(encoding="utf-8")


def test_every_exported_symbol_appears_in_readme():
    text = _readme_text()
    missing = [
        name
        for name in reflect_mcp.__all__
        if name != "__version__" and name not in text
    ]
    assert missing == [], f"Exported symbols missing from README: {missing}"
