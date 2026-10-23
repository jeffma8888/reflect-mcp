"""Black-box behavioral tests for iter-40: README library-usage docs + drift test.

Covers Expected Behaviors from pm.md:
  1. README contains "Use as a Python library" section between CLI (§5) and Architecture.
  2. Full-pipeline import example using build_default_pipeline + ReflectivePipeline.
  3. Gate-only import example using decide + GateConfig.
  4. Every __all__ name (excluding __version__) appears in README body.
  5. Drift test (test_readme_api_completeness.py) asserts EB4 programmatically.
  6. Existing test suite passes (verified by full suite run).

Isolation contract: tests read README.md and __all__ via the public package
surface. No implementation source was read.
"""

from __future__ import annotations

import re
from pathlib import Path

import reflect_mcp


def _readme_text() -> str:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    return readme.read_text(encoding="utf-8")


def _readme_lines() -> list[str]:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    return readme.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# EB-1: README contains "Use as a Python library" section between CLI and Architecture
# ---------------------------------------------------------------------------

class TestEB1SectionExists:
    """EB-1: README has a library-usage section between CLI and Architecture."""

    def test_section_heading_present(self):
        text = _readme_text()
        assert "Use as a Python library" in text

    def test_section_is_numbered_heading(self):
        lines = _readme_lines()
        found = [l for l in lines if "Use as a Python library" in l and l.startswith("#")]
        assert len(found) >= 1, "Expected a markdown heading containing 'Use as a Python library'"

    def test_section_between_cli_and_architecture(self):
        lines = _readme_lines()
        cli_idx = None
        lib_idx = None
        arch_idx = None
        for i, line in enumerate(lines):
            if line.startswith("#") and "CLI" in line:
                cli_idx = i
            if line.startswith("#") and "Use as a Python library" in line:
                lib_idx = i
            if line.startswith("#") and "Architecture" in line:
                arch_idx = i
        assert cli_idx is not None, "CLI section heading not found"
        assert lib_idx is not None, "Library section heading not found"
        assert arch_idx is not None, "Architecture section heading not found"
        assert cli_idx < lib_idx < arch_idx, (
            f"Expected CLI ({cli_idx}) < Library ({lib_idx}) < Architecture ({arch_idx})"
        )


# ---------------------------------------------------------------------------
# EB-2: Full-pipeline import example (build_default_pipeline + ReflectivePipeline)
# ---------------------------------------------------------------------------

class TestEB2FullPipelineExample:
    """EB-2: library section shows full-pipeline example with key symbols."""

    def _library_section_text(self) -> str:
        lines = _readme_lines()
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("#") and "Use as a Python library" in line:
                start = i
            elif start is not None and line.startswith("## ") and i > start:
                end = i
                break
        assert start is not None, "Library section not found"
        if end is None:
            end = len(lines)
        return "\n".join(lines[start:end])

    def test_build_default_pipeline_in_example(self):
        section = self._library_section_text()
        assert "build_default_pipeline" in section

    def test_reflective_pipeline_in_example(self):
        section = self._library_section_text()
        assert "ReflectivePipeline" in section

    def test_extract_method_call_in_example(self):
        section = self._library_section_text()
        assert ".extract(" in section

    def test_extraction_result_in_example(self):
        section = self._library_section_text()
        assert "ExtractionResult" in section

    def test_example_shows_import_from_reflect_mcp(self):
        section = self._library_section_text()
        assert "from reflect_mcp import" in section


# ---------------------------------------------------------------------------
# EB-3: Gate-only import example (decide + GateConfig)
# ---------------------------------------------------------------------------

class TestEB3GateOnlyExample:
    """EB-3: library section shows gate-only verification example."""

    def _library_section_text(self) -> str:
        lines = _readme_lines()
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("#") and "Use as a Python library" in line:
                start = i
            elif start is not None and line.startswith("## ") and i > start:
                end = i
                break
        assert start is not None, "Library section not found"
        if end is None:
            end = len(lines)
        return "\n".join(lines[start:end])

    def test_decide_in_example(self):
        section = self._library_section_text()
        assert "decide" in section

    def test_gate_config_in_example(self):
        section = self._library_section_text()
        assert "GateConfig" in section

    def test_invoice_data_in_example(self):
        section = self._library_section_text()
        assert "InvoiceData" in section

    def test_critique_result_in_example(self):
        section = self._library_section_text()
        assert "CritiqueResult" in section

    def test_field_verdict_in_example(self):
        section = self._library_section_text()
        assert "FieldVerdict" in section


# ---------------------------------------------------------------------------
# EB-4: Every __all__ name (excluding __version__) appears in README
# ---------------------------------------------------------------------------

class TestEB4AllExportsDocumented:
    """EB-4: every exported symbol in __all__ appears at least once in README."""

    def test_no_missing_symbols(self):
        text = _readme_text()
        missing = [
            name
            for name in reflect_mcp.__all__
            if name != "__version__" and name not in text
        ]
        assert missing == [], f"Exported symbols missing from README: {missing}"

    def test_all_has_expected_minimum_count(self):
        assert len(reflect_mcp.__all__) >= 15, (
            f"Expected at least 15 exports but found {len(reflect_mcp.__all__)}"
        )

    def test_version_excluded_from_check(self):
        assert "__version__" in reflect_mcp.__all__, (
            "__version__ should be in __all__ (it is excluded from the README check, not from __all__)"
        )


# ---------------------------------------------------------------------------
# EB-5: Drift test file exists and has correct structure
# ---------------------------------------------------------------------------

class TestEB5DriftTestExists:
    """EB-5: test_readme_api_completeness.py exists and covers EB4."""

    def test_drift_test_file_exists(self):
        drift_test = Path(__file__).resolve().parent / "test_readme_api_completeness.py"
        assert drift_test.exists(), "Drift test file not found"

    def test_drift_test_reads_all(self):
        drift_test = Path(__file__).resolve().parent / "test_readme_api_completeness.py"
        content = drift_test.read_text(encoding="utf-8")
        assert "__all__" in content, "Drift test must reference __all__"

    def test_drift_test_reads_readme(self):
        drift_test = Path(__file__).resolve().parent / "test_readme_api_completeness.py"
        content = drift_test.read_text(encoding="utf-8")
        assert "README" in content or "readme" in content.lower(), (
            "Drift test must reference README"
        )

    def test_drift_test_excludes_version(self):
        drift_test = Path(__file__).resolve().parent / "test_readme_api_completeness.py"
        content = drift_test.read_text(encoding="utf-8")
        assert "__version__" in content, (
            "Drift test must exclude __version__ from the check"
        )

    def test_drift_test_has_assertion(self):
        drift_test = Path(__file__).resolve().parent / "test_readme_api_completeness.py"
        content = drift_test.read_text(encoding="utf-8")
        assert "assert" in content, "Drift test must contain at least one assertion"
