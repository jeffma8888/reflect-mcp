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
| 21 | trace_id unique per call and present on retry log events | Med | Low | DONE (iter 70) |
| 22 | MCP tools reach CLI parity: `heuristics_only` arg + `status:"error"` envelope, offline-tested | High | Low | DONE (iter 55) |
| 23 | One batch driver: build/threshold failure is a JSON envelope + exit 3 at any `--batch-concurrency` | High | Low | DONE (iter 57) |
| 24 | `validate` rejects unknown / empty `data` keys instead of verifying nothing and returning `ok` (iter-74 A2: `{"totl":999}` → `ok`, all-null; MCP `data` schema names no fields) | High | Low | PARTIAL (iter 68: own result envelope unwrapped/rejected; unknown-key rule QUEUED) |
| 25 | Batch output joinable: echo caller `id` + 1-based `line`; per-line empty-text pre-check | Med | Low | PARTIAL (iter 67: id echo; 1-based line QUEUED) |
| 26 | Drop the fixed 0.5s sleep before a parse/schema retry (pure latency; transport retries live in SDKs) | Med | Low | QUEUED |
| 27 | NEXT: locked per-role client memo on the MCP path (20 calls 40 → 0 TCP conns; closes iter-69 unlocked boto3 build race) | High | Low | PARTIAL (iter 63: validate builds no Generator client; memo QUEUED) |
| 28 | Delete the `heuristics_enabled` knob (Critic-less + heuristics-off returns `ok` on fabricated data) | High | Low | DONE (iter 62) |
| 29 | README §3 demo is the real inv-004 replay output, pinned by a test that re-runs it | Med | Low | DONE (iter 58) |
| 30 | Retry carries the parse/schema error to the model instead of a byte-identical re-request | Med | Low | QUEUED |
| 31 | `make eval` measures the trust contract: catch rate, silent-wrong, false alarms; `--max-silent-wrong` exit | High | Low | DONE (iter 59) |
| 32 | Amount-role grounding: an amount found in the source but not the number it labels for that role is penalized | High | Low | DONE (iter 60) |
| 33 | Amount-role labels attributed per line (first label wins, skip `N%` rates): "Total tax"/"Net total" stop vouching for a wrong total | High | Low | QUEUED |
| 34 | MCP gate floor: `REFLECT_FIELD_THRESHOLD`/`REFLECT_OVERALL_THRESHOLD`; a tool arg can only raise the threshold | High | Low | DONE (iter 61) |
| 35 | MCP `heuristics_only` needs operator opt-in (`REFLECT_ALLOW_HEURISTICS_ONLY`) (its inv-007 motivation closed by #41) | Med | Low | QUEUED |
| 36 | One CLI flag set: delete `__main__`'s copy of `_add_common_args`/`_add_batch_args` (152→104 lines, 0 test edits; iter-74 B1) | Med | Low | QUEUED |
| 37 | `validate --batch`: verify a `{text, data}` JSONL corpus in one process (≈129 ms/doc → <1 ms/doc heuristics-only) | High | Low | DONE (iter 64) |
| 38 | Heuristics-only `ok` says no Critic ran (notes) and logs `critic.skipped` (reason), not a fabricated `critique.done` | High | Low | DONE (iter 65) |
| 39 | Omission detection: a null amount the source labels is flagged (reverses the null-abstention contract; 14-test migration) | High | Med | QUEUED |
| 40 | `needs_review` flagged fields carry deterministic `suggestions` (source-labelled value + span) | Med | Low | QUEUED |
| 41 | Party-role grounding: a `vendor_name` found only on "Bill to"/"Ship to"/"Sold to"/"Deliver to" lines is penalized (0.3) | High | Low | DONE (iter 66) |
| 42 | Date-role grounding: an `invoice_date` the source shows only on a "Due"/"Ship date" line is penalized | Med | Low | DONE (iter 71) |
| 43 | `make eval` adds a heuristics-only (Critic-off) trust readout line: catch / silent-wrong / false alarms | Med | Low | QUEUED |
| 44 | Party-role check runs on normalized grounding too: NBSP/tab/double-space/CR "Bill to" buyer is no silent `ok` (4/6 bypass; ~8 lines, eval byte-identical; needs golden row) | High | Low | QUEUED |
| 45 | `--batch \| head -1` exits 0 with no BrokenPipeError traceback (today exit 120; fix in `__main__.py`, cli.py at budget) | Med | Low | QUEUED |
| 46 | `--batch` tolerates a UTF-8 BOM and skips whitespace-only lines (today error rows + exit 3) | Med | Low | QUEUED |
| 47 | `--batch --summary PATH` writes `{lines, ok, needs_review, error, flagged_fields, exit}` for CI gates | Med | Low | QUEUED |
| 48 | Result carries `critic_ran: bool` so a consumer can require a Critic-verified `ok` (2 src lines, 30-pin digest migration) | Med | Med | QUEUED |
| 49 | MCP tools run on a worker thread (`async` + `asyncio.to_thread`): parallel agent calls overlap (4 calls 1.67s → 0.42s) | High | Low | DONE (iter 69) |
| 50 | Pin `mcp>=1.x,<2` and raise the floor to a version that imports on Python 3.13 (mcp 2.x / 1.10.1 break server.py) | Med | Low | PARTIAL (iter 73: `mcp>=1.28,<2`; lower measured floor QUEUED) |
| 51 | `validate` verb: verify-only mode on caller-supplied data (record backfilled iter 70) | High | Low | DONE (iter 18) |
| 52 | Golden row inv-009 (due date as invoice date; pre-iter-71 heuristics silent-wrong 1) + pin paydown: HEAD-identity pins read a frozen inv-001..008 subset (`tests/_golden.py`) | High | Low | DONE (iter 72) |
| 53 | NEXT: Identifier-role grounding: an `invoice_number` seen only on a PO/order-no/customer-id line is penalized (iter-72 A1: 5/5 silent ok caught; abstain unless an invoice-id label exists, else inv-002 false-alarms) | Med | Low | QUEUED |
| 54 | Token-boundary grounding for `invoice_number`/`currency`: "INV-100" in "INV-1001", "US" in "USD" stop grounding (iter-72 B2: 4/4 silent ok; ~15 lines) | High | Low | QUEUED |
| 55 | Date-role label is colon-scoped: "Ship to: … Invoice date: X" stops false-alarming (iter-72 B3: 4/4 fixed, 6/6 catches kept; REV iter71 digit-split loses 2) | Med | Low | QUEUED |
| 56 | Currency grounded by unambiguous symbol on a labelled amount ("€"→EUR, "£"→GBP; never "$") (iter-72 A2: removes a measured false alarm) | Med | Low | QUEUED |
| 57 | MCP tools publish a typed outputSchema (`status` enum required) + `structuredContent`; `status:"error"` sets `isError` (iter-73 B1) | High | Low | DONE (iter 73) |
| 58 | MCP threshold args default to `null`: inputSchema stops advertising 0.7/0.75; an omitted arg never gets a "raised from" note (iter-74 A1) | High | Low | DONE (iter 74) |
| 59 | NEXT: one autouse conftest floor reset: suite green with `REFLECT_*_THRESHOLD` exported (HEAD: 5–7 fail; tests only, net −22; iter-74 B3) | Med | Low | QUEUED |
| 60 | `validate --batch` grades rows carrying `expected` (caught/silent-wrong/false alarms) + `--max-silent-wrong` exit; grader moves to `grading.py` (iter-74 A3) | High | Med | QUEUED |
| 61 | One single-document CLI tail: `run_extract/validate_from_args` share `_run_verb` (−18 cli.py lines, 18-probe byte-identical; iter-75 A1; repays #62's budget) | Med | Low | QUEUED |
| 62 | Concurrent `--batch` streams via a bounded 2N window (row 1 after 2N pulls, not EOF; 88.9→0.2 MB on 50k lines; iter-75 B1) | High | Low | DONE (iter 75) |
| 63 | `--batch` flushes each row so a piped consumer sees it immediately (stdout is block-buffered on pipes) | Med | Low | QUEUED |
| 64 | Critic emits `rationale` only for doubted fields (30% of critic_raw bytes are discarded on `ok`; iter-75 B3; needs live evidence) | Low | Med | QUEUED |
| 65 | `ok` notes name heuristic-penalized fields that cleared a lowered threshold; all-null `ok` says nothing was verified (iter-76 B1) | High | Low | DONE (iter 76) |

## Done ledger

- iter 01 — CLI extract verb: pipe-friendly `reflect-mcp extract` (stdin/file → JSON stdout, exit 0/1/2)
- iter 02 — Structured error envelope on CLI pipeline failure (exit 3, JSON on stdout)
- iter 03 — Reuse SDK clients across calls: move client construction to provider `__init__`
- iter 15 — GateConfig from dataclass to Pydantic BaseModel with boundary validation
- iter 16 — critic_summary in README tool-return table + schema-docs drift-detection test
- iter 17 — CLI `--field-threshold` / `--overall-threshold` flags for confidence-gate control
- iter 18 — `validate` verb: verify-only mode (Critic + heuristics + gate on caller-supplied data)
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
- iter 62 — Delete `heuristics_enabled` knob + dead `_number_in_text`: every pipeline runs heuristics (0/4 → 3/1)
- iter 63 — validate builds no Generator client; heuristics-only validate builds 0 clients, works with no SDK
- iter 64 — `validate --batch`: one process verifies a {text,data} JSONL corpus (200 docs 0.13s)
- iter 65 — Heuristics-only ok notes say no Critic ran; log critic.skipped(reason), no fake critique.done
- iter 66 — Party-role grounding: vendor_name only on a bill-to/ship-to line is penalized; heuristics-only 3/1 → 4/0
- iter 67 — Batch rows echo the input record's `id` as first key (error rows too), so output joins to input
- iter 68 — validate verifies a stored result envelope's `data` (error envelope rejected), not an all-null `ok`
- iter 69 — MCP tools are async on a worker thread: 4 parallel calls overlap (0.84s → 0.21s stub)
- iter 70 — trace_id unique per call (50 parallel MCP calls: 8 → 50 ids); retry events carry it
- iter 71 — Date-role grounding: invoice_date seen only as a due/ship/delivery date is penalized (0.3)
- iter 72 — Golden row inv-009 + frozen-subset pins: a new golden row no longer breaks ~79 HEAD-identity tests
- iter 73 — MCP outputSchema requires a `status` enum; results carry structuredContent; error sets isError
- iter 74 — MCP threshold args default to null: no fake 0.7/0.75 in inputSchema, no raise note unless asked
- iter 75 — Concurrent `--batch` streams through a bounded 2N window: row 1 after ≤2N pulls, not stdin EOF
- iter 76 — `ok` notes name heuristic-penalized fields that still cleared; all-null `ok` says nothing verified
