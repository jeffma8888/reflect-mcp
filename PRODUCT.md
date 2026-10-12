# reflect-mcp — Product Roadmap

An MCP server for LLM tasks that check their own work before surfacing it.
Generator extracts typed data, a second-model Critic + deterministic heuristics
score every field, and a confidence-thresholded gate returns the result or
escalates a `needs_review` payload. Never a silent wrong answer.

**Target user:** Engineers wiring LLM extraction into real systems who need
verification as a first-class step, not a demo-time afterthought.

**Quality bar:** Offline-testable (no network/secrets in tests or `make eval`);
model calls behind one stubbable provider seam; Critic stays decorrelated;
heuristics only lower scores; every verification/gate change ships golden-set
coverage; `make eval` keeps its threshold exit. Python >=3.10, MCP-compliant
tool surface, small deps, MIT, publication-safe.

## Roadmap

| # | Feature | Value | Risk | Status |
|---|---------|-------|------|--------|
| 1 | CLI extract verb: pipe-friendly `reflect-mcp extract` (stdin/file → JSON stdout, exit codes) | High | Low | DONE (iter 01) |
| 2 | Structured error result (`status="error"`) instead of raw exceptions on pipeline failure | High | Low | **SELECTED — iter 02** |
| 3 | Generator/Critic retry with configurable back-off on JSON parse failure | Med | Low | QUEUED |
| 4 | `make typecheck` target with mypy/pyright in strict mode over `src/` | Med | Low | QUEUED |
| 5 | Eval as a CLI verb: `reflect-mcp eval [--live] [--min-accuracy N]` | Med | Low | QUEUED |
| 6 | Configurable confidence thresholds via env vars / CLI flags | Med | Low | QUEUED |
| 7 | Schema extensibility: user-defined Pydantic models beyond InvoiceData | High | Med | QUEUED |
| 8 | Batch extraction: process multiple inputs in one invocation | Med | Med | QUEUED |

## Done ledger

- iter 01 — CLI extract verb: pipe-friendly `reflect-mcp extract` (stdin/file → JSON stdout, exit 0/1/2)
