"""reflect-mcp: a self-verifying multi-agent extraction pipeline exposed over MCP.

Public loop:  Generator -> deterministic heuristics + second-model Critic -> HITL gate.
The gate never emits a silent low-confidence guess; below threshold it returns a
`needs_review` payload naming the specific flagged fields.

The public names are re-exported lazily (PEP 562): `import reflect_mcp` must stay
cheap because `reflect-mcp --version`, `--help` and `--mcp-config` import this
package but never touch the pipeline, and eagerly importing it would make every
one of those on-ramp invocations pay for pydantic plus the whole pipeline graph.
"""

from __future__ import annotations

import importlib
import typing

if typing.TYPE_CHECKING:
    # Real imports for type checkers and IDEs only; at runtime `__getattr__`
    # resolves the same names on first access.
    from reflect_mcp.schema import (
        InvoiceData,
        CritiqueResult,
        FieldVerdict,
        FlaggedField,
        ProvenanceSpan,
        ExtractionResult,
        EXTRACTION_FIELDS,
    )
    from reflect_mcp.pipeline import ReflectivePipeline, build_default_pipeline
    from reflect_mcp.generator import Generator, GeneratorError
    from reflect_mcp.critic import Critic, CriticError
    from reflect_mcp.gate import GateConfig, decide

__version__ = "0.1.0"

# Public name -> submodule that defines it. Insertion order is the `__all__` order.
_LAZY: dict[str, str] = {
    "InvoiceData": "schema",
    "CritiqueResult": "schema",
    "FieldVerdict": "schema",
    "FlaggedField": "schema",
    "ProvenanceSpan": "schema",
    "ExtractionResult": "schema",
    "EXTRACTION_FIELDS": "schema",
    "ReflectivePipeline": "pipeline",
    "build_default_pipeline": "pipeline",
    "Generator": "generator",
    "GeneratorError": "generator",
    "Critic": "critic",
    "CriticError": "critic",
    "GateConfig": "gate",
    "decide": "gate",
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name: str) -> typing.Any:
    """Import the defining submodule on first access and cache the name here.

    Caching in `globals()` means later lookups never reach this hook again, so
    the lazy path costs one dict miss per name per process.
    """
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the lazy names too, so `dir()` and completion see them before first use."""
    return sorted({*globals(), *__all__})
