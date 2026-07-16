"""reflect-mcp: a self-verifying multi-agent extraction pipeline exposed over MCP.

Public loop:  Generator -> deterministic heuristics + second-model Critic -> HITL gate.
The gate never emits a silent low-confidence guess; below threshold it returns a
`needs_review` payload naming the specific flagged fields.
"""

from reflect_mcp.schema import (
    InvoiceData,
    CritiqueResult,
    FieldVerdict,
    FlaggedField,
    ExtractionResult,
    EXTRACTION_FIELDS,
)
from reflect_mcp.pipeline import ReflectivePipeline, build_default_pipeline
from reflect_mcp.generator import Generator, GeneratorError
from reflect_mcp.critic import Critic, CriticError
from reflect_mcp.gate import GateConfig, decide

__version__ = "0.1.0"

__all__ = [
    "InvoiceData",
    "CritiqueResult",
    "FieldVerdict",
    "FlaggedField",
    "ExtractionResult",
    "EXTRACTION_FIELDS",
    "ReflectivePipeline",
    "build_default_pipeline",
    "Generator",
    "GeneratorError",
    "Critic",
    "CriticError",
    "GateConfig",
    "decide",
    "__version__",
]
