"""Black-box behavioral tests for iter-16: critic_summary documentation + drift detection.

Covers all 5 Expected Behaviors from pm.md:
  1. README §4 tool table Returns column includes critic_summary
  2. server.py extract docstring includes critic_summary in field list
  3. Drift test: docstring field names == ExtractionResult.model_fields exactly
  4. Drift test detects a missing field (simulated removal breaks equality)
  5. Full quality-check suite passes (verified by full suite run)

Isolation contract: tests drive the public API (imports + source file read).
server.py is read as text (not imported) to avoid requiring the MCP SDK.
"""

from __future__ import annotations

import ast
import pathlib
import re

from reflect_mcp.schema import ExtractionResult


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER_PY = REPO_ROOT / "src" / "reflect_mcp" / "server.py"


def _extract_docstring() -> str:
    """Parse server.py's extract() function docstring from the AST."""
    tree = ast.parse(SERVER_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "extract":
            doc = ast.get_docstring(node)
            assert doc is not None, "extract() must have a docstring"
            return doc
    raise AssertionError("extract() function not found in server.py")


def _docstring_field_names() -> set[str]:
    """Parse field names from the extract function's docstring bullet list."""
    doc = _extract_docstring()
    names: set[str] = set()
    for line in doc.splitlines():
        m = re.match(r"\s*-\s+([\w\s/]+):", line)
        if m:
            raw = m.group(1)
            for part in raw.split("/"):
                names.add(part.strip())
    return names


# ---------------------------------------------------------------------------
# EB-1: README §4 table includes critic_summary
# ---------------------------------------------------------------------------

class TestReadmeDocumentation:
    def test_readme_section4_table_contains_critic_summary(self):
        readme = (REPO_ROOT / "README.md").read_text()
        table_match = re.search(
            r"\|\s*`extract`\s*\|.*?\|.*?\|", readme
        )
        assert table_match is not None, "README must contain the extract tool table row"
        table_row = table_match.group(0)
        assert "critic_summary" in table_row, (
            "README §4 tool table Returns column must include critic_summary"
        )

    def test_readme_section4_table_lists_all_schema_fields(self):
        readme = (REPO_ROOT / "README.md").read_text()
        table_match = re.search(
            r"\|\s*`extract`\s*\|.*?\|.*?\|", readme
        )
        assert table_match is not None
        table_row = table_match.group(0)
        for field in ExtractionResult.model_fields.keys():
            assert field in table_row, (
                f"README §4 tool table must include field '{field}'"
            )


# ---------------------------------------------------------------------------
# EB-2: server.py extract docstring includes critic_summary
# ---------------------------------------------------------------------------

class TestExtractDocstring:
    def test_extract_has_docstring(self):
        doc = _extract_docstring()
        assert doc is not None

    def test_docstring_contains_critic_summary(self):
        doc = _extract_docstring()
        assert "critic_summary" in doc, (
            "extract docstring must mention critic_summary"
        )

    def test_docstring_critic_summary_in_bullet_list(self):
        doc_fields = _docstring_field_names()
        assert "critic_summary" in doc_fields, (
            "critic_summary must appear as a bullet-listed field in the docstring"
        )


# ---------------------------------------------------------------------------
# EB-3: Drift detection — docstring fields == schema fields (exact)
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_docstring_matches_schema(self):
        schema_fields = set(ExtractionResult.model_fields.keys())
        doc_fields = _docstring_field_names()
        assert doc_fields == schema_fields, (
            f"Docstring/schema drift — "
            f"in schema but not docstring: {schema_fields - doc_fields}, "
            f"in docstring but not schema: {doc_fields - schema_fields}"
        )

    def test_schema_has_expected_field_count(self):
        schema_fields = set(ExtractionResult.model_fields.keys())
        assert len(schema_fields) == 9, (
            f"Expected 9 ExtractionResult fields, got {len(schema_fields)}: {schema_fields}"
        )

    def test_docstring_has_expected_field_count(self):
        doc_fields = _docstring_field_names()
        assert len(doc_fields) == 9, (
            f"Expected 9 documented fields, got {len(doc_fields)}: {doc_fields}"
        )


# ---------------------------------------------------------------------------
# EB-4: Drift test catches a missing field (failure mode)
# ---------------------------------------------------------------------------

class TestDriftFailureMode:
    def test_removing_critic_summary_causes_mismatch(self):
        schema_fields = set(ExtractionResult.model_fields.keys())
        doc_fields = _docstring_field_names()
        simulated = doc_fields - {"critic_summary"}
        assert simulated != schema_fields, "Drift test must detect a removed field"

    def test_removing_any_single_field_causes_mismatch(self):
        schema_fields = set(ExtractionResult.model_fields.keys())
        doc_fields = _docstring_field_names()
        for field in doc_fields:
            simulated = doc_fields - {field}
            assert simulated != schema_fields, (
                f"Removing '{field}' from docstring set must cause a mismatch"
            )

    def test_adding_spurious_field_causes_mismatch(self):
        schema_fields = set(ExtractionResult.model_fields.keys())
        doc_fields = _docstring_field_names()
        simulated = doc_fields | {"nonexistent_field"}
        assert simulated != schema_fields, (
            "Adding a field not in the schema must cause a mismatch"
        )
