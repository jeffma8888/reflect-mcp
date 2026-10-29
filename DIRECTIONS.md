# Foundry directions

foundry directions -- reflect-mcp
  iter-55
    lenses: integration-and-adoption
    - Candidate A1 -- A numeric `invoice_number` from the Generator crashes the run instead of being coerced to a string
    - Candidate A2 -- Non-finite amounts (`NaN`/`Infinity`) are accepted end to end and the CLI prints invalid JSON
    - Candidate A3 -- Blank strings and non-code currencies sail through as `ok`: give text fields the format contract `invoice_date` already has
    - Candidate B1 -- MCP tools reach CLI parity: `heuristics_only` arg + structured `status:"error"` envelope, testable offline
    - Candidate B2 -- `validate --batch`: verify a JSONL corpus of `{text, data}` produced by someone else's extractor
    - Candidate B3 -- Batch output carries a correlation key: echo the caller's `id` (and the 1-based input `line`) on every result, including error lines
    winner: B1
    ship: pending (not yet decided)
  iter-54
    lenses: new-capability, hardening/DX
    - Candidate A1 -- Amount-role grounding: penalize a subtotal/tax/total whose matching number sits next to a different field's label
    - Candidate A2 -- Omission detection: flag a null field when the source plainly labels a value for it
    - Candidate A3 -- Token-boundary grounding: a value must match a whole source token, not a fragment of one
    - Candidate B1 -- Gate honors the Critic's own contract: `grounded:false` caps the field, and a missing, duplicate or unknown verdict can't pass silently
    - Candidate B2 -- trace_id contract: unique per call and present on every log event (including retries)
    - Candidate B3 -- Generator output contract: an all-null extraction can't be `ok`, and dropped unknown keys are surfaced
    winner: B1
    ship: PUSHED f8c6301
  iter-53
    lenses: narrative-and-docs, new-capability
    - Candidate A1 -- Document when the Critic doesn't run (short-circuit + heuristics-only) in README §7/§10 and the MCP tool docstrings, plus a drift test
    - Candidate A2 -- PRODUCT.md is missing iter 18 (`validate` verb / MCP tool); add the row + a ledger-freshness test keyed off per-iteration behavior tests
    - Candidate A3 -- Observability event catalog in README §10 + correct the "every stage carries a shared trace_id" claim, with a two-way drift test
    - Candidate B1 -- Date-grounding heuristic: penalize an ISO-valid invoice_date that appears nowhere in the source
    - Candidate B2 -- Omission detection: flag a field the Generator left null when the source plainly contains it
    - Candidate B3 -- Per-field confidence thresholds: stricter gate for high-stakes fields
    winner: B1
    ship: PUSHED c7c28fb
  iter-52
    lenses: unknown
    - Candidate A1 — Short-circuit Critic when heuristics already doom all fields
    - Candidate A2 — Use `model_dump_json()` for Pydantic serialization on the output path
    - Candidate A3 — Cache the lowered source text across heuristic sub-checks
    - Candidate B1 -- "Three roles" → "Four stages" in README §7 Architecture + count-drift test
    - Candidate B2 -- §10 Design Decisions: document heuristic-only mode rationale + neutral-critique arithmetic
    - Candidate B3 -- §11 "What I'd productionize next" staleness sweep + shipped-item guard
    winner: A1
    ship: PUSHED 480d13f
  iter-41
    lenses: unknown
    - Candidates
    - Candidate B1 -- Schema extensibility: user-supplied Pydantic model via `--schema`
    - Candidate B2 -- Heuristic-only verification mode (`--heuristics-only` flag)
    - Candidate B3 -- `reflect-mcp eval` CLI verb (Roadmap #5)
    stubs: 1 of 4 candidate line(s) are write-early placeholders, not measured candidates
    winner: B2
    ship: PUSHED 0bb84e8
  iter-40
    lenses: unknown
    - Candidate A1 -- Cache MCP server pipeline instance across tool invocations
    - Candidate A2 -- Precompute lowered text once in run_heuristics
    - Candidate A3 -- Streaming concurrent batch to cap peak memory at O(concurrency)
    - Candidate B1 -- Python library usage section in README
    - Candidate B2 -- §9 Design Decisions for batch-mode architecture
    - Candidate B3 -- Refresh §10 "What I'd productionize next" + staleness guard
    winner: B1
    ship: PUSHED 546ce98
  iter-38
    lenses: unknown
    - Candidate A1 -- Module-level `extract()` and `validate()` convenience functions
    - Candidate A2 -- Machine-readable exit-code contract documentation and `--output-format` flag
    - Candidate A3 -- `reflect-mcp eval` CLI verb (Roadmap #5)
    - Candidate B1 -- Collapse dual-parser _add_common_args into single shared definition
    - Candidate B2 -- Collapse _run_batch serial path into _process_one_line
    - Candidate B3 -- Delete test-only run_extract/run_validate wrappers and their standalone parsers
    winner: B2
    ship: PUSHED 82cefee
  iter-37
    lenses: unknown
    - Candidate A1 -- CLI boundary validation for threshold and concurrency flags
    - Candidate A2 -- `make typecheck` target with mypy strict mode (Roadmap #4)
    - Candidate A3 -- Input-size guard to catch oversized documents before hitting the provider
    - Candidate B1 -- Module-level `extract()` and `validate()` convenience functions
    - Candidate B2 -- Top-level `--help` subcommand listing and `--version` flag
    - Candidate B3 -- `--data-file` flag for the validate verb
    winner: B2
    ship: PUSHED 46d5235
  iter-36
    lenses: unknown
    - Candidate A1 -- Heuristic-only verification mode (`--no-critic` flag)
    - Candidate A2 -- Confidence calibration verb (`reflect-mcp calibrate`)
    - Candidate A3 -- Schema introspection verb (`reflect-mcp schema`)
    - Candidate B1 -- CLI boundary validation for threshold and concurrency flags
    - Candidate B2 -- Thread-safe logger initialization (fix double-logging under concurrency)
    - Candidate B3 -- Export ProvenanceSpan from public API + py.typed marker + API completeness test
    winner: B3
    ship: PUSHED 85db17f
  iter-35
    lenses: unknown
    - Candidate A1 -- Fix "Three roles" count in README §6 and add architecture-count drift test
    - Candidate A2 -- Document retry semantics in README §9 Design Decisions + drift test
    - Candidate A3 -- Update §10 "What I'd productionize next" to remove shipped items + staleness guard
    - Candidate B1 -- Schema extensibility: accept a user-supplied Pydantic model
    - Candidate B2 -- Confidence calibration verb: tune thresholds from labeled data
    - Candidate B3 -- Field-level provenance spans in ExtractionResult
    winner: B3
    ship: PUSHED e613b05
  iter-34
    lenses: unknown
    - Candidate A1 — Cache MCP server pipeline instance across tool invocations
    - Candidate A2 — Run heuristics concurrently with the Critic LLM call
    - Candidate A3 — Concurrent batch processing: overlap LLM calls across JSONL lines
    - Candidate B1 — Fix "Three roles" count mismatch in README §6
    - Candidate B2 — Correct TL;DR ASCII diagram to show the sequential 4-stage pipeline
    - Candidate B3 — Document retry semantics in §9 Design Decisions + drift test
    winner: A3
    ship: PUSHED 8e9e658
  iter-29
    lenses: unknown
    - Candidate A1 -- Design-decisions section: document retry semantics and gate arithmetic
    - Candidate A2 -- Stale model identifiers in providers.py Bedrock defaults
    - Candidate A3 -- §10 "What I'd productionize next" list is stale after 19 iterations
    - Candidate B1 -- Eval as a CLI verb (`reflect-mcp eval`)
    - Candidate B2 -- Batch extraction via JSONL I/O
    - Candidate B3 -- `reflect-mcp diff` verb: compare two extraction results
    winner: B2
    ship: PUSHED a606c0a
  iter-19
    lenses: unknown
    - Candidate A1 — Top-level `--help` and `--version` for the CLI entry point
    - Candidate A2 — `make typecheck` target with mypy strict mode + `py.typed` marker
    - Candidate A3 — Generator/Critic retry with configurable back-off on JSON parse failure
    - Candidate B1 — `reflect-mcp eval` CLI verb wiring the shipped eval harness
    - Candidate B2 — Module-level convenience functions: `reflect_mcp.extract()` and `reflect_mcp.validate()`
    - Candidate B3 — `--data-file` flag for the `validate` verb
    winner: A3
    ship: PUSHED 400a62c
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
20 scouted iterations
