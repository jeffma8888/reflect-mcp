"""Black-box behavioral tests for iter-36: packaging DX — ProvenanceSpan export + PEP 561.

Covers all 6 Expected Behaviors from pm.md:
  1. `from reflect_mcp import ProvenanceSpan` succeeds without ImportError.
  2. `ProvenanceSpan` appears in `reflect_mcp.__all__`.
  3. `src/reflect_mcp/py.typed` exists and is an empty marker file (PEP 561).
  4. Every Pydantic BaseModel subclass referenced by ExtractionResult's field type
     annotations (directly or transitively) is importable from `reflect_mcp` and
     appears in `__all__`. Specifically: InvoiceData, FlaggedField, ProvenanceSpan.
  5. The existing test suite continues to pass unchanged (additive change; verified
     by full-suite run, not by a test in this module).
  6. `reflect_mcp.__all__` does not contain duplicate entries.

Isolation contract: tests drive the public import interface and inspect the
filesystem for the py.typed marker. No implementation source was read.
"""

from __future__ import annotations

import importlib
import os
import typing
from collections import Counter
from pathlib import Path

import pytest

import reflect_mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_package_dir() -> Path:
    """Return the directory containing reflect_mcp's __init__.py."""
    return Path(reflect_mcp.__file__).parent


def _discover_basemodel_types_in_annotations(cls: type) -> set[str]:
    """Recursively discover all Pydantic BaseModel subclass names referenced
    in `cls`'s field type annotations.

    Handles: direct BaseModel fields, generic containers like dict[str, list[X]],
    Optional[X], and list[X]. Returns only the class names (not the classes
    themselves), to compare against __all__ strings.
    """
    from pydantic import BaseModel

    hints = typing.get_type_hints(cls)
    found: set[str] = set()
    visited: set[type] = set()

    def _walk(tp: type) -> None:
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)

        if origin is not None:
            for arg in args:
                _walk(arg)
        elif isinstance(tp, type) and issubclass(tp, BaseModel) and tp is not cls:
            if tp not in visited:
                visited.add(tp)
                found.add(tp.__name__)
                _walk_fields(tp)

    def _walk_fields(model_cls: type) -> None:
        hints_inner = typing.get_type_hints(model_cls)
        for hint_val in hints_inner.values():
            _walk(hint_val)

    for hint_val in hints.values():
        _walk(hint_val)

    return found


# ---------------------------------------------------------------------------
# EB-1: `from reflect_mcp import ProvenanceSpan` succeeds
# ---------------------------------------------------------------------------

class TestEB1ProvenanceSpanImport:
    """EB-1: ProvenanceSpan is importable from the package root."""

    def test_import_succeeds(self):
        from reflect_mcp import ProvenanceSpan
        assert ProvenanceSpan is not None

    def test_is_a_class(self):
        from reflect_mcp import ProvenanceSpan
        assert isinstance(ProvenanceSpan, type)

    def test_has_start_and_end_fields(self):
        from reflect_mcp import ProvenanceSpan
        span = ProvenanceSpan(start=0, end=5)
        assert span.start == 0
        assert span.end == 5

    def test_importable_via_importlib(self):
        mod = importlib.import_module("reflect_mcp")
        assert hasattr(mod, "ProvenanceSpan")


# ---------------------------------------------------------------------------
# EB-2: ProvenanceSpan in __all__
# ---------------------------------------------------------------------------

class TestEB2ProvenanceSpanInAll:
    """EB-2: ProvenanceSpan appears in reflect_mcp.__all__."""

    def test_in_all(self):
        assert "ProvenanceSpan" in reflect_mcp.__all__

    def test_accessible_via_getattr(self):
        cls = getattr(reflect_mcp, "ProvenanceSpan")
        assert cls is not None


# ---------------------------------------------------------------------------
# EB-3: py.typed marker file exists and is empty
# ---------------------------------------------------------------------------

class TestEB3PyTypedMarker:
    """EB-3: src/reflect_mcp/py.typed exists and is an empty (0-byte) file."""

    def test_py_typed_exists(self):
        marker = _get_package_dir() / "py.typed"
        assert marker.exists(), f"Expected py.typed at {marker}"

    def test_py_typed_is_file(self):
        marker = _get_package_dir() / "py.typed"
        assert marker.is_file(), "py.typed should be a regular file"

    def test_py_typed_is_empty(self):
        marker = _get_package_dir() / "py.typed"
        size = marker.stat().st_size
        assert size == 0, f"py.typed should be 0 bytes, got {size}"


# ---------------------------------------------------------------------------
# EB-4: All BaseModel types in ExtractionResult annotations are in __all__
# ---------------------------------------------------------------------------

class TestEB4APICompleteness:
    """EB-4: Every BaseModel subclass transitively referenced by ExtractionResult's
    annotations is importable from reflect_mcp and appears in __all__."""

    def test_known_models_in_all(self):
        for name in ("InvoiceData", "FlaggedField", "ProvenanceSpan"):
            assert name in reflect_mcp.__all__, \
                f"{name} must be in reflect_mcp.__all__"

    def test_known_models_importable(self):
        for name in ("InvoiceData", "FlaggedField", "ProvenanceSpan"):
            cls = getattr(reflect_mcp, name, None)
            assert cls is not None, f"{name} must be importable from reflect_mcp"
            assert isinstance(cls, type), f"{name} should be a class"

    def test_transitive_discovery_matches_known_set(self):
        """Use type introspection to discover BaseModel subclasses referenced
        in ExtractionResult annotations and verify all are exported."""
        from reflect_mcp import ExtractionResult
        discovered = _discover_basemodel_types_in_annotations(ExtractionResult)
        expected = {"InvoiceData", "FlaggedField", "ProvenanceSpan"}
        assert discovered == expected, \
            f"Discovered models {discovered} != expected {expected}"
        for name in discovered:
            assert name in reflect_mcp.__all__, \
                f"Discovered model {name} missing from __all__"

    def test_each_model_is_pydantic_basemodel(self):
        from pydantic import BaseModel
        for name in ("InvoiceData", "FlaggedField", "ProvenanceSpan"):
            cls = getattr(reflect_mcp, name)
            assert issubclass(cls, BaseModel), \
                f"{name} should be a Pydantic BaseModel subclass"


# ---------------------------------------------------------------------------
# EB-5: Existing test suite still passes (verified externally; structural check)
# ---------------------------------------------------------------------------

class TestEB5NoBreakingChanges:
    """EB-5: The change is additive. We verify that key existing exports still work."""

    def test_extraction_result_still_importable(self):
        from reflect_mcp import ExtractionResult
        assert ExtractionResult is not None

    def test_invoice_data_still_importable(self):
        from reflect_mcp import InvoiceData
        assert InvoiceData is not None

    def test_flagged_field_still_importable(self):
        from reflect_mcp import FlaggedField
        assert FlaggedField is not None

    def test_all_original_exports_preserved(self):
        """All entries that were in __all__ before this iter are still there."""
        known_pre_iter36 = {
            "InvoiceData", "CritiqueResult", "FieldVerdict", "FlaggedField",
            "ExtractionResult", "EXTRACTION_FIELDS",
        }
        for name in known_pre_iter36:
            assert name in reflect_mcp.__all__, \
                f"Pre-existing export {name} missing from __all__"


# ---------------------------------------------------------------------------
# EB-6: No duplicates in __all__
# ---------------------------------------------------------------------------

class TestEB6NoDuplicatesInAll:
    """EB-6: reflect_mcp.__all__ has no duplicate entries."""

    def test_no_duplicates(self):
        counts = Counter(reflect_mcp.__all__)
        dupes = {k: v for k, v in counts.items() if v > 1}
        assert not dupes, f"Duplicate entries in __all__: {dupes}"

    def test_all_entries_are_strings(self):
        for entry in reflect_mcp.__all__:
            assert isinstance(entry, str), f"__all__ entry should be str, got {type(entry)}"

    def test_all_entries_resolvable(self):
        """Every name in __all__ should be an actual attribute of the module."""
        for name in reflect_mcp.__all__:
            assert hasattr(reflect_mcp, name), \
                f"__all__ entry {name!r} is not a module attribute"
