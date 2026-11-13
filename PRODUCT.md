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
| 17 | Short-circuit Critic when heuristics already doom all fields (iter-93 B2 `any`-doomed skip declined: drops secondary-field Critic flags) | Med | Low | DONE (iter 52) |
| 18 | Date-grounding heuristic: penalize an ISO-valid `invoice_date` absent from the source | High | Low | DONE (iter 53) |
| 19 | Gate honors Critic contract: `grounded:false` / missing verdict caps at 0.5; duplicates use min | High | Low | DONE (iter 54) |
| 20 | All-null extraction is `needs_review`; dropped unknown Generator keys surfaced in notes | Med | Low | QUEUED |
| 21 | trace_id unique per call and present on retry log events | Med | Low | DONE (iter 70) |
| 22 | MCP tools reach CLI parity: `heuristics_only` arg + `status:"error"` envelope, offline-tested | High | Low | DONE (iter 55) |
| 23 | One batch driver: build/threshold failure is a JSON envelope + exit 3 at any `--batch-concurrency` | High | Low | DONE (iter 57) |
| 24 | `validate` rejects unknown / empty `data` keys instead of verifying nothing and returning `ok` (iter-74 A2: `{"totl":999}` → `ok`, all-null; MCP `data` schema names no fields) | High | Low | DONE (iter 68: own envelope unwrapped; iter 79: unknown keys rejected, naming them; `{}` still QUEUED) |
| 25 | Batch output joinable: echo caller `id` + 1-based `line`; per-line empty-text pre-check | Med | Low | PARTIAL (iter 67: id echo; 1-based line QUEUED) |
| 26 | Drop the fixed 0.5s sleep before a parse/schema retry (pure latency; transport retries live in SDKs) | Med | Low | QUEUED |
| 27 | Locked per-role client memo on the MCP path (20 calls 40 → 0 TCP conns; closes iter-69 unlocked boto3 build race) | High | Low | PARTIAL (iter 63: validate builds no Generator client; memo QUEUED) |
| 28 | Delete the `heuristics_enabled` knob (Critic-less + heuristics-off returns `ok` on fabricated data) | High | Low | DONE (iter 62) |
| 29 | README §3 demo is the real inv-004 replay output, pinned by a test that re-runs it | Med | Low | DONE (iter 58) |
| 30 | Retry carries the parse/schema error to the model instead of a byte-identical re-request | Med | Low | QUEUED |
| 31 | `make eval` measures the trust contract: catch rate, silent-wrong, false alarms; `--max-silent-wrong` exit | High | Low | DONE (iter 59) |
| 32 | Amount-role grounding: an amount found in the source but not the number it labels for that role is penalized | High | Low | DONE (iter 60) |
| 33 | Amount-role labels attributed per line (first label wins, skip `N%` rates): "Total tax"/"Net total" stop vouching for a wrong total | High | Low | QUEUED |
| 34 | MCP gate floor: `REFLECT_FIELD_THRESHOLD`/`REFLECT_OVERALL_THRESHOLD`; a tool arg can only raise the threshold | High | Low | DONE (iter 61) |
| 35 | MCP `heuristics_only` needs operator opt-in (`REFLECT_ALLOW_HEURISTICS_ONLY`) (its inv-007 motivation closed by #41) | Med | Low | QUEUED |
| 36 | One CLI argparse tree: `cli.run_extract`/`run_validate` parse via `__main__._build_top_parser`; cli.py's duplicate parser builders deleted (375→306; iter-81 A1 flipped #36's direction) | Med | Low | DONE (iter 81) |
| 37 | `validate --batch`: verify a `{text, data}` JSONL corpus in one process (≈129 ms/doc → <1 ms/doc heuristics-only) | High | Low | DONE (iter 64) |
| 38 | Heuristics-only `ok` says no Critic ran (notes) and logs `critic.skipped` (reason), not a fabricated `critique.done` | High | Low | DONE (iter 65) |
| 39 | Omission detection: an explicit `"total": null` the source labels is flagged needs_review (`model_fields_set` guard: omitted key unchanged; 13-pin migration + golden inv-013; iter 89 reverted at PRESHIP by a `git archive HEAD` fixture, re-landed pinned to a fixed base) | High | Med | DONE (iter 90) |
| 40 | `needs_review` flagged fields carry deterministic `suggestions` (source-labelled value + span) | Med | Low | QUEUED |
| 41 | Party-role grounding: a `vendor_name` found only on "Bill to"/"Ship to"/"Sold to"/"Deliver to" lines is penalized (0.3) | High | Low | DONE (iter 66) |
| 42 | Date-role grounding: an `invoice_date` the source shows only on a "Due"/"Ship date" line is penalized | Med | Low | DONE (iter 71) |
| 43 | `make eval` adds a heuristics-only (Critic-off) trust readout line: catch / silent-wrong / false alarms | Med | Low | QUEUED |
| 44 | Party-role check runs on normalized grounding too: NBSP/tab/double-space/CR "Bill to" buyer is no silent `ok` (iter-85 A2: 10/10 bypass over 5 whitespace kinds × 2 role checks; ~8 lines, 0 test edits; needs golden row) | High | Low | NEXT |
| 45 | `--batch \| head -1` keeps the 0/1/2/3 contract: worst code of rows produced, no traceback, input stops (was exit 120) | Med | Low | DONE (iter 78) |
| 46 | `--batch` tolerates a UTF-8 BOM and skips whitespace-only lines (today error rows + exit 3) | Med | Low | QUEUED |
| 47 | `--batch --summary PATH` writes `{lines, ok, needs_review, error, flagged_fields, exit}` for CI gates | Med | Low | QUEUED |
| 48 | Result carries `critic_ran: bool` so a consumer can require a Critic-verified `ok` (2 src lines, 30-pin digest migration) | Med | Med | QUEUED |
| 49 | MCP tools run on a worker thread (`async` + `asyncio.to_thread`): parallel agent calls overlap (4 calls 1.67s → 0.42s) | High | Low | DONE (iter 69) |
| 50 | Pin `mcp>=1.x,<2` and raise the floor to a version that imports on Python 3.13 (mcp 2.x / 1.10.1 break server.py) | Med | Low | PARTIAL (iter 73: `mcp>=1.28,<2`; lower measured floor QUEUED) |
| 51 | `validate` verb: verify-only mode on caller-supplied data (record backfilled iter 70) | High | Low | DONE (iter 18) |
| 52 | Golden row inv-009 (due date as invoice date; pre-iter-71 heuristics silent-wrong 1) + pin paydown: HEAD-identity pins read a frozen inv-001..008 subset (`tests/_golden.py`) | High | Low | DONE (iter 72) |
| 53 | Identifier-role grounding: an `invoice_number` seen only on a PO/order/customer/account line is penalized (0.3) when an invoice-number label exists; golden row inv-010 | Med | Low | DONE (iter 77) |
| 54 | Token-boundary grounding for `invoice_number`/`currency`: "INV-100" in "INV-1001", "US" in "USD" stop grounding (iter-72 B2: 4/4 silent ok; ~15 lines) | High | Low | QUEUED |
| 55 | Date-role label is colon-scoped: "Ship to: … Invoice date: X" stops false-alarming (iter-72 B3: 4/4 fixed, 6/6 catches kept; REV iter71 digit-split loses 2) | Med | Low | QUEUED |
| 56 | Currency grounded by unambiguous symbol on a labelled amount ("€"→EUR, "£"→GBP; never "$") (iter-72 A2: removes a measured false alarm) | Med | Low | QUEUED |
| 57 | MCP tools publish a typed outputSchema (`status` enum required) + `structuredContent`; `status:"error"` sets `isError` (iter-73 B1) | High | Low | DONE (iter 73) |
| 58 | MCP threshold args default to `null`: inputSchema stops advertising 0.7/0.75; an omitted arg never gets a "raised from" note (iter-74 A1) | High | Low | DONE (iter 74) |
| 59 | One autouse conftest floor reset: suite green with `REFLECT_*_THRESHOLD` exported (HEAD: 5–7 fail; tests only, net −22; iter-74 B3) | Med | Low | QUEUED |
| 60 | `validate --batch` grades rows carrying `expected` (caught/silent-wrong/false alarms) + `--max-silent-wrong` exit; grader moves to `grading.py` (iter-74 A3) | High | Med | QUEUED |
| 61 | One single-document CLI tail: `run_extract/validate_from_args` share `_run_verb` (−18 cli.py lines, 18-probe byte-identical; iter-75 A1; repays #62's budget) | Med | Low | QUEUED |
| 62 | Concurrent `--batch` streams via a bounded 2N window (row 1 after 2N pulls, not EOF; 88.9→0.2 MB on 50k lines; iter-75 B1) | High | Low | DONE (iter 75) |
| 63 | `--batch` flushes each row so a piped consumer sees it immediately (stdout is block-buffered on pipes; iter 87 reverted at PRESHIP by iter-86's open-ended diff pin, re-landed with the fixed-range defuse) | Med | Low | DONE (iter 88) |
| 64 | Critic emits `rationale` only for doubted fields (30% of critic_raw bytes are discarded on `ok`; iter-75 B3; needs live evidence) | Low | Med | QUEUED |
| 65 | `ok` notes name heuristic-penalized fields that cleared a lowered threshold; all-null `ok` says nothing was verified (iter-76 B1) | High | Low | DONE (iter 76) |
| 66 | CI: `.github/workflows/ci.yml` runs `uv run pytest` + `make eval` on Python 3.10 (floor) and 3.13; iter73 import sweep stops leaking package attrs (3.10: 47 red → 0) | High | Low | DONE (iter 80) |
| 67 | Month/amount-label regexes get first-letter lookaheads + unique 3-letter month prefixes: `_MONTH_FIRST_RE` 3.9× faster, `run_heuristics` −36% on 297 KB, reports byte-identical (iter-82 A1) | Med | Low | DONE (iter 82) |
| 68 | Currency-code shape check: a `currency` not matching `[A-Z]{3}` (`$`, `US Dollars`, `usd`) gets factor 0.2, not a silent grounded ok; golden inv-011 (iter-83 B1) | High | Low | DONE (iter 83) |
| 69 | `.env.example` lists every `REFLECT_*` var src reads (missing both iter-61 floor vars) + two-way src↔template test (iter-83 A1) | Med | Low | QUEUED |
| 70 | Non-ASCII hardening: `APRİL 3, 2026`/`ſep` month tokens read by the matching regex rule (KeyError → validate exit 3 / `make eval` crash), provenance spans index the original text (`İ` shifted every later span); seeded property test; golden inv-012 (iter-84 B2) | High | Low | DONE (iter 84) |
| 71 | Currency grounded by its written name, word-bounded (`US Dollars`→USD, `Pounds Sterling`→GBP; not `AUS Dollars`, not bare `dollars`); golden coverage = every golden `expected` passes heuristics clean (iter-95 B2; new row priced at 22 count-pin reds) | Med | Low | DONE (iter 95) |
| 72 | MCP `initialize` reports reflect-mcp's own `serverInfo.version` (was the mcp SDK's 1.28.1); `__version__`↔pyproject pin (iter-85 B1) | Med | Low | DONE (iter 85) |
| 73 | One parse-and-retry loop: Generator + Critic share `parsing.complete_parsed` and one `_RETRY_DELAY_SECONDS`; retries/logs/errors identical, eval byte-identical (iter-86 B1; makes #26/#30 one-place) | Med | Low | DONE (iter 86) |
| 74 | MCP hosts see tool `title` + `annotations` (readOnly, non-destructive, idempotent, openWorld) and a server `instructions` naming the `needs_review`/`flagged_fields` contract (iter-86 A1: ~12 src lines, 0 test edits; pass annotations as a dict, stub `mcp.types` lacks `ToolAnnotations`; iter-91 B1, instructions tied two-way to the status enum) | High | Low | DONE (iter 91) |
| 75 | One role-penalty rule: party/identifier/date checks share `_penalize_if_only`; `_contains_ci` folds into `_find_text_span` (iter-87 A1; 2236-row differential byte-identical) | Med | Low | QUEUED |
| 76 | One source-size budget test (`cli.py <= 306`, `gate.py <= 185`) replaces 9 scattered line-count pins + 4 test-about-test pins (iter-87 A2) | Low | Low | QUEUED |
| 77 | No-Critic results stop calling the placeholder `overall_confidence: 1.0` "the critic's": heuristics-only and `all_fields_doomed` docs say it is a placeholder and `--overall-threshold` is inert there (iter-88 B1; re-hash the iter-69 extract-docstring sha pin; #48 is the behavior fix) | High | Low | QUEUED |
| 78 | Omission detection for `subtotal`/`tax`: an explicit null the source labels is flagged (#39 follow-up; measured 29 red for all amounts vs 13 total-only) | Med | Med | QUEUED |
| 79 | Suite-wide AST guard: no test passes `HEAD`/`@` as a git archive/diff/show rev or runs a single-rev `git diff` (iter-90 B2: red on both iter-87/89 bombs, 0 hits on 65 modules; iter 90 shipped it over a frozen 12-module list -- remainder: glob `tests/test_*.py`, iter-91 A3 red on both bombs) | Med | Low | QUEUED |
| 80 | One provider-SDK import guard: `_require(module, provider, extra)` replaces 3 copied try-import blocks; `build_client` dispatches via a `_CLIENTS` dict keyed like `_DEFAULTS` (iter-92 B3; 0 red, providers.py 140→131) | Med | Low | DONE (iter 92) |
| 81 | MCP `validate.data` inputSchema names the 7 InvoiceData fields (+ stored-envelope branch) instead of `additionalProperties: true` (iter-92 A1; 1 iter-85 digest re-hash) | High | Low | QUEUED |
| 82 | MCP tool arguments publish `description`s; thresholds advertise `minimum 0`/`maximum 1` via `json_schema_extra` (not `ge`/`le`) (iter-92 A3; pairs with #81) | Med | Low | QUEUED |
| 83 | One single-document CLI tail: `extract`/`validate` share `_run_verb` (cli.py 306→279; 2 pin literals: iter68 `HEAD_CLI_LINES`, iter79 `== N`) (iter-92 B1) | Med | Med | QUEUED |
| 84 | One "Critic didn't speak" branch in `_verify`: one `_neutral_critique` site, `skip_reason` picked once (iter-92 B2; 0 red) | Low | Low | QUEUED |
| 85 | Heuristics stop re-scanning: first-match date span, date-role pass skips lines without the target year, lazy whitespace fold (iter-93 B1 narrowed; byte-identical, same-year doc −8%, other-year −42%) | Med | Low | DONE (iter 93) |
| 86 | One MCP tool tail: `extract_tool`/`validate_tool` share `_run_tool`; `_build_pipeline` deleted (iter-93 A1; 0 red) | Med | Low | QUEUED |
| 87 | One wrong-role-lines predicate `_only_on` shared by party- and identifier-role checks (iter-93 A3; 0 red, heuristics −3 lines) | Low | Low | QUEUED |
| 88 | `--batch` default concurrency 4 when a model is configured (heuristics-only stays 1) (iter-93 B3 variant; 8 serial pins; 429 risk) | Med | Med | QUEUED |
| 89 | README §2/§5 name the omitted-total check; one test pins every `heuristics.__doc__` check to all three README lists (iter-94 B1) | Med | Low | DONE (iter 94) |
| 90 | Rationale only for doubted fields: Critic output −28% on replay, gate decisions unchanged (iter-94 A1; = #64; live effect unprovable offline) | Med | Med | QUEUED |
| 91 | Role checks compile each value's pattern once (`_lines_containing`): golden −19..28%, byte-identical (iter-94 A2; 0 red) | Low | Low | QUEUED |
| 92 | Coverage badge enforced: `fail_under = 90`, stale server/providers `omit` retired (iter-94 B2; 97% either way) | Low | Low | QUEUED |
| 93 | README layout + §7 name `tools.py` and `py.typed`, two-way pinned to `src/reflect_mcp/` (iter-94 B3; 0 red) | Low | Low | QUEUED |
| 94 | README §10 says logs go to stderr and lists all 10 `stage` events, AST-pinned two-way to src (iter-95 A2; 8/10 undocumented) | Med | Low | QUEUED |
| 95 | Prompt key lists pinned to `EXTRACTION_FIELDS` / `CritiqueResult`+`FieldVerdict`; "versioned" prose made true (iter-95 A3; green on HEAD) | Med | Low | QUEUED |
| 96 | Literal currency grounding is a case-insensitive substring: `Euro Parts` grounds EUR, `USDollars` grounds USD; word-bound it (iter-95 PM probe) | Med | Low | QUEUED |

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
- iter 77 — Identifier-role grounding: invoice_number seen only on a PO/order/customer line is penalized; golden inv-010
- iter 78 — `| head -1` keeps the exit contract: worst produced row's code, no traceback, input stops (was 120)
- iter 79 — validate rejects unknown data keys (typo'd vendr_name was a silent ok); envelope keys exempt
- iter 80 — CI runs the offline suite + `make eval` on 3.10 and 3.13; iter73 sweep restores package attrs
- iter 81 — CLI test entry points parse via the console's argparse tree; cli.py's duplicate parsers deleted
- iter 82 — Month/label regexes get first-letter lookaheads: heuristics −36% on 297 KB, reports byte-identical
- iter 83 — currency must be a 3-letter ISO 4217 code shape: `$`/`US Dollars` stop passing as ok; golden inv-011
- iter 84 — `APRİL`/`ſep` month tokens stop crashing; provenance spans exact after `İ`; golden inv-012
- iter 85 — MCP `initialize` reports `serverInfo.version` = `reflect-mcp --version`, not the mcp SDK's
- iter 86 — Generator and Critic share one parse-and-retry loop + delay constant; retry behavior byte-identical
- iter 87 — REVERTED at PRESHIP: iter-86 `git diff BASE..worktree` scope pin went red once a new test was committed
- iter 88 — `--batch` flushes each row (pipe sees row 1 in 0.12 s, was never pre-EOF); iter-86 pins diff a fixed range
- iter 89 — REVERTED at PRESHIP: new test's `git archive HEAD src` baseline became the ship once committed
- iter 90 — Explicit `"total": null` on a labelled total is needs_review, not a silent ok; golden inv-013
- iter 91 — MCP `initialize` gives `instructions` (needs_review contract); tools get title + read-only annotations
- iter 92 — Provider SDK guard is one `_require` helper + `_CLIENTS` dict; same clients, same install hints
- iter 93 — Heuristics skip redundant date scans + lazy fold: same reports, 200 KB doc −8% (same-year) / −42%
- iter 94 — README §2/§5 name the omitted-total check; docstring-driven pin covers all three check lists
- iter 95 — Currency written by name (`US Dollars`, `pounds sterling`) grounds its ISO code; golden expected all clean
