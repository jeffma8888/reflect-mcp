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
| 2 | Structured error result (`status="error"`) instead of raw exceptions on pipeline failure | High | Low | DONE (iter 02) |
| 3 | Generator/Critic retry (configurable `max_retries`, fixed 0.5s delay) on JSON parse/schema failure | Med | Low | DONE (iter 19) |
| 4 | `make typecheck` target with mypy/pyright in strict mode over `src/` | Med | Low | QUEUED |
| 5 | Eval as a CLI verb: `reflect-mcp eval [--live] [--min-accuracy N]` | Med | Low | QUEUED |
| 6 | Configurable confidence thresholds via env vars / CLI flags | Med | Low | DONE (iter 17: CLI flags; iter 61: MCP env floor) |
| 7 | Schema extensibility: user-defined Pydantic models beyond InvoiceData | High | Med | QUEUED |
| 8 | Batch extraction: process multiple inputs in one invocation | Med | Med | DONE (iter 29) |
| 10 | Concurrent batch processing: `--batch-concurrency N` thread pool | Med | Low | DONE (iter 34) |
| 11 | Field-level provenance spans in ExtractionResult | High | Low | DONE (iter 35) |
| 12 | Export ProvenanceSpan + PEP 561 py.typed marker | Med | Low | DONE (iter 36) |
| 9 | Reuse SDK clients across calls within one pipeline instance (construction in `__init__`) | Med | Low | DONE (iter 03) |
| 13 | Top-level `--help` subcommand listing and `--version` flag | Med | Low | DONE (iter 37) |
| 14 | Collapse `_run_batch` serial path into `_process_one_line` | Low | Low | DONE (iter 38) |
| 15 | README "Use as a Python library" section + API-completeness drift test | Med | Low | DONE (iter 40) |
| 16 | Heuristic-only verification mode (`--heuristics-only` flag) | Med | Low | DONE (iter 41) |
| 17 | Short-circuit Critic when heuristics already doom all fields | Med | Low | DONE (iter 52) |
| 18 | Date-grounding heuristic: penalize an ISO-valid `invoice_date` absent from the source | High | Low | DONE (iter 53) |
| 19 | Gate honors Critic contract: `grounded:false` / missing verdict caps at 0.5; duplicates use min | High | Low | DONE (iter 54) |
| 20 | All-null extraction is `needs_review`; dropped unknown Generator keys surfaced in notes | Med | Low | QUEUED |
| 21 | trace_id unique per call and present on retry log events | Med | Low | QUEUED |
| 22 | MCP tools reach CLI parity: `heuristics_only` arg + `status:"error"` envelope, offline-tested | High | Low | DONE (iter 55) |
| 23 | One batch driver: build/threshold failure is a JSON envelope + exit 3 at any `--batch-concurrency` | High | Low | DONE (iter 57) |
| 24 | `validate` rejects unknown / empty `data` keys instead of verifying nothing and returning `ok` | High | Low | QUEUED |
| 25 | Batch output joinable: echo caller `id` + 1-based `line`; per-line empty-text pre-check | Med | Low | QUEUED |
| 26 | Drop the fixed 0.5s sleep before a parse/schema retry (pure latency; transport retries live in SDKs) | Med | Low | QUEUED |
| 27 | Build each provider client once per process; validate / heuristics-only build no unused clients | Med | Low | QUEUED |
| 28 | Delete the `heuristics_enabled` knob (Critic-less + heuristics-off returns `ok` on fabricated data) | High | Low | QUEUED |
| 29 | README §3 demo is the real inv-004 replay output, pinned by a test that re-runs it | Med | Low | DONE (iter 58) |
| 30 | Retry carries the parse/schema error to the model instead of a byte-identical re-request | Med | Low | QUEUED |
| 31 | `make eval` measures the trust contract: catch rate, silent-wrong, false alarms; `--max-silent-wrong` exit | High | Low | DONE (iter 59) |
| 32 | Amount-role grounding: an amount found in the source but not the number it labels for that role is penalized | High | Low | DONE (iter 60) |
| 33 | Amount-role labels attributed per line (first label wins, skip `N%` rates): "Total tax"/"Net total" stop vouching for a wrong total | High | Low | QUEUED |
| 34 | MCP gate floor: `REFLECT_FIELD_THRESHOLD`/`REFLECT_OVERALL_THRESHOLD`; a tool arg can only raise the threshold | High | Low | DONE (iter 61) |

## Done ledger

- iter 01 — CLI extract verb: pipe-friendly `reflect-mcp extract` (stdin/file → JSON stdout, exit 0/1/2)
- iter 02 — Structured error envelope on CLI pipeline failure (exit 3, JSON on stdout)
- iter 03 — Reuse SDK clients across calls: move client construction to provider `__init__`
- iter 15 — GateConfig from dataclass to Pydantic BaseModel with boundary validation
- iter 16 — critic_summary in README tool-return table + schema-docs drift-detection test
- iter 17 — CLI `--field-threshold` / `--overall-threshold` flags for confidence-gate control
- iter 19 — Generator/Critic retry (configurable `max_retries`, fixed 0.5s delay) on JSON parse failure
- iter 29 — Batch extraction: `--batch` JSONL mode for multi-document processing
- iter 34 — Concurrent batch processing: `--batch-concurrency N` thread pool for overlapping LLM calls
- iter 35 — Field-level provenance spans: character-offset evidence linking extracted values to source text
- iter 36 — Export ProvenanceSpan from public API + PEP 561 py.typed marker for typed package consumers
- iter 37 — Top-level `--help` subcommand listing and `--version` flag via argparse dispatch
- iter 38 — Collapse `_run_batch` serial path into `_process_one_line` (single-source per-line contract)
- iter 40 — README "Use as a Python library" section with full-pipeline and gate-only examples + API-completeness drift test
- iter 41 — Heuristic-only verification mode: `--heuristics-only` flag skips Critic LLM call for zero-cost structural checks
- iter 52 — Short-circuit Critic when heuristics already doom all fields: skip provably redundant LLM call when gate outcome is predetermined
- iter 53 — Date-grounding heuristic: made-up ISO `invoice_date` now penalized (0.3) + `invoice_date` provenance span; golden row inv-006
- iter 54 — Gate honors the Critic contract: `grounded:false` or missing verdict caps a committed field at 0.5, duplicate verdicts resolve to min; golden row inv-007
- iter 55 — MCP tools at CLI parity: `heuristics_only` + `status:"error"` envelope; bodies in mcp-free `tools.py`
- iter 57 — One batch driver + one build-error rule: build/threshold failure is an envelope + exit 3 at any concurrency
- iter 58 — README §3 demo = real inv-004 replay output (583.20 source, 683.2 model), pinned by a re-running test
- iter 59 — `make eval` counts silent-wrong (wrong, unflagged) fields + catch rate; exits 1 above `--max-silent-wrong`
- iter 60 — Amount-role grounding: a found amount unequal to its role's labelled number is penalized; golden inv-008
- iter 61 — MCP gate floor from env (default 0.70/0.75): agent args can only raise thresholds; golden 0/0 pass
