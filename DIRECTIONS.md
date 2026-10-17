# Foundry directions

foundry directions -- reflect-mcp
  iter-19
    lenses: unknown
    - Candidate A1 — Top-level `--help` and `--version` for the CLI entry point
    - Candidate A2 — `make typecheck` target with mypy strict mode + `py.typed` marker
    - Candidate A3 — Generator/Critic retry with configurable back-off on JSON parse failure
    - Candidate B1 — `reflect-mcp eval` CLI verb wiring the shipped eval harness
    - Candidate B2 — Module-level convenience functions: `reflect_mcp.extract()` and `reflect_mcp.validate()`
    - Candidate B3 — `--data-file` flag for the `validate` verb
    winner: A3
    ship: pending (not yet decided)
  iter-18
    lenses: unknown
    - Candidate A1 — `reflect-mcp validate` verb: verify-only mode (skip Generator)
    - Candidate A2 — `reflect-mcp eval` CLI verb
    - Candidate A3 — Batch extraction: `--batch` flag for multi-document processing
    - Candidate B1 — MCP `extract` tool: structured error envelope + empty-input guard
    - Candidate B2 — `make typecheck` target with mypy strict mode
    - Candidate B3 — Top-level `--help` and `--version` for the CLI entry point
    winner: A1
    ship: PUSHED 194b1d5
  iter-17
    lenses: unknown
    - Candidate A1 — Sync PRODUCT.md Done ledger through iter 16 + add ledger-freshness test
    - Candidate A2 — Document pipeline execution order (heuristics BEFORE Critic) + order test
    - Candidate A3 — Update stale model identifiers in providers.py defaults and README example
    - Candidate B1 — CLI threshold flags (`--field-threshold`, `--overall-threshold`)
    - Candidate B2 — `reflect-mcp validate` verb: verify-only mode (skip Generator)
    - Candidate B3 — Generator/Critic retry with configurable back-off on JSON parse failure
    winner: B1
    ship: PUSHED 0a1ed7f
  iter-16
    lenses: unknown
    - Candidate A1 — Cache pipeline instance across MCP tool invocations
    - Candidate A2 — Reuse pipeline across golden-set examples in live eval
    - Candidate A3 — Precompute number candidates once in run_heuristics
    - Candidate B1 — Add missing `critic_summary` to README tool-return table + schema-docs drift test
    - Candidate B2 — Sync PRODUCT.md Done ledger through iter 15 + add a ledger-freshness assertion
    - Candidate B3 — Document the pipeline execution order (heuristics BEFORE Critic) and add an execution-order test
    winner: B1
    ship: PUSHED d4917dc
  iter-15
    lenses: unknown
    - Candidate A1 — Convert GateConfig from dataclass to Pydantic BaseModel
    - Candidate A2 — Inline `_model_for` into `build_client`
    - Candidate A3 — Delete dormant `ScriptedLLMClient.remaining` property
    - Candidate B1 — Cache pipeline instance across MCP tool invocations
    - Candidate B2 — Run heuristics concurrently with the Critic LLM call
    - Candidate B3 — Reuse pipeline across golden-set examples in live eval
    winner: A1
    ship: PUSHED d3f9f34
  iter-03
    lenses: unknown
    - Candidate A1 — Remove heuristics_enabled toggle and its dead else-branch
    - Candidate A2 — Deduplicate dev dependency declaration in pyproject.toml
    - Candidate A3 — Eliminate eval's sys.path hack by adding eval to pythonpath
    - Candidate B1 — Reuse SDK clients across calls instead of constructing per-complete()
    - Candidate B2 — Run heuristics concurrently with the Critic LLM call
    - Candidate B3 — Cache the MCP server pipeline instance across tool invocations
    winner: B1
    ship: PUSHED b5ef47b
  iter-02
    lenses: unknown
    - Candidate A1 — CLI threshold flags (`--field-threshold`, `--overall-threshold`)
    - Candidate A2 — Eval as a CLI verb (`reflect-mcp eval`)
    - Candidate A3 — Structured error envelope on pipeline failure
    - Candidate B1 — Remove `heuristics_enabled` toggle and dead disable path
    - Candidate B2 — Eliminate eval's `sys.path` hack by making eval a proper subpackage
    - Candidate B3 — Collapse near-duplicate test setup in `test_iter01_behavior.py`
    winner: A3
    ship: PUSHED b59bd87
  iter-01
    lenses: unknown
    - Candidate A1 -- Generator/Critic retry with back-off on parse failure
    - Candidate A2 -- Structured error result instead of raw exception on pipeline failure
    - Candidate A3 -- `make typecheck` target with inline type annotations verified by mypy/pyright
    - Candidate B1 -- CLI extract verb: pipe-friendly one-shot extraction
    - Candidate B2 -- Machine-readable exit codes and status field for scripting
    - Candidate B3 -- Eval as a console-script verb: `reflect-mcp eval`
    winner: B1
    ship: PUSHED 78ab8ff
8 scouted iterations
