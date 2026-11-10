# Foundry directions

foundry directions -- reflect-mcp
  iter-82
    lenses: unknown
    - Candidate A1 -- Regexes that can only start on a month/label letter stop being tried at every character: 297 KB `validate` heuristics 152 → 98 ms (−36%), 200-row `--batch` 3.24 → 2.13 s, with byte-identical reports
    - Candidate A2 -- `--batch` rows reach a piped consumer as each one finishes: first row after 0.08 s instead of 1.86 s (roadmap #63, marked NEXT by iter 81's PM)
    - Candidate A3 -- Date tokens are found once per document, not twice: `_find_date_span` reads its earliest match from the per-line tokenization `_date_role_labels` already does, cutting the remaining date cost about in half
    - Candidate B1 -- README's repository layout and architecture name every module the package ships: `tools.py`, the MCP tool bodies, has been missing since iter 55. A two-way layout↔`src/` test keeps them from drifting apart again
    - Candidate B2 -- PRODUCT.md's roadmap stops sending three contradictory signals: one `NEXT` row instead of three, #39's stale "14-test migration" figure replaced with the measured 34, and #20's "all-null is needs_review" reconciled with the shipped iter-76 decision
    - Candidate B3 -- README's "coverage ≥90%" badge becomes a checked claim: `fail_under = 90` in `[tool.coverage.report]`, CI's test step runs with `--cov`, and §9 says the figure excludes `server.py`/`providers.py`
    winner: A1
    ship: pending (not yet decided)
  iter-81
    lenses: simplification-and-deletion
    - Candidate A1 -- One argparse tree: `cli.run_extract`/`run_validate` parse through the console script's own `reflect-mcp <verb>` parser, and cli.py's duplicate parser builders are deleted (cli.py 375 → 304)
    - Candidate A2 -- One provider-SDK import guard: the three copied try-import/`RuntimeError` blocks in providers.py become one `_require(module, extra, provider)` helper, and `build_client`'s if-chain becomes a dict (providers.py 140 → 129)
    - Candidate A3 -- One MCP tool body: `extract_tool`/`validate_tool` share a `_run_tool(text, thresholds, run)` tail, and `tools._build_pipeline` with its "pass a kwarg only when opted out" rule is deleted (tools.py 217 → 213)
    - Candidate B1 -- Date heuristics stop scanning a long document twice in full: a 43 KB `validate` goes from 18.2 to 12.8 ms (−30%), and a 200-row `--batch` from 3.77 to 2.55 s, with byte-identical reports
    - Candidate B2 -- `--batch` rows reach a piped consumer as each one finishes: first row after 0.13 s instead of 2.21 s (roadmap #63)
    - Candidate B3 -- The Critic writes `rationale` only for fields it doubts: 29% of every Critic reply's bytes are paid output tokens the product never shows
    winner: A1
    ship: PUSHED 26de4e7
  iter-80
    lenses: integration-and-adoption, simplification-and-deletion
    - Candidate A1 -- CI runs the gates the product already ships: a `.github/workflows/ci.yml` on Python 3.10 and 3.13 runs `uv run pytest` and `make eval`, and the suite goes green on the advertised 3.10 floor (today 47 red)
    - Candidate A2 -- MCP `validate`'s published `data` schema names the InvoiceData fields and stops advertising `additionalProperties: true`, which iter 79's rejection now makes false
    - Candidate A3 -- Hosts learn the tools' nature and the `needs_review` contract before the first call: MCP tool `annotations` (title, `readOnlyHint`, `openWorldHint`) and a server `instructions` string
    - Candidate B1 -- One CLI flag set: `reflect-mcp extract|validate` and the `cli.run_*` entry points the tests drive build their flags from one function, and `__main__`'s duplicate goes (roadmap #36)
    - Candidate B2 -- One parse-and-retry loop: Generator and Critic share `parsing.complete_parsed`, so there is one retry loop and one `RETRY_DELAY_SECONDS`, not two copies
    - Candidate B3 -- One "Critic didn't speak" branch: `_verify` picks a `skip` reason once, instead of two copies of the neutral critique, `critic_ran=False` and `critic.skipped`
    winner: A1
    ship: PUSHED 2bb26a1
  iter-79
    lenses: integration-and-adoption
    - Candidate A1 -- Unreadable input (non-UTF-8 byte, directory, no permission) exits 2 with one `Error:` line instead of a traceback whose exit 1 means `needs_review`
    - Candidate A2 -- `validate` stops silently dropping a misspelt field: `{"total":10.0,"vendr_name":"Acme"}` is rejected naming `vendr_name`, not a verified `ok` (roadmap #24 remainder)
    - Candidate A3 -- The suite stays green under an operator's exported `REFLECT_FIELD_THRESHOLD`/`REFLECT_OVERALL_THRESHOLD` (roadmap #59, marked NEXT)
    - Candidate B1 -- The documented MCP launch is proven by a real MCP client: `mcp` joins the dev group, and one stdio test runs the bare `reflect-mcp` command through initialize → tools/list → tools/call
    - Candidate B2 -- `--batch` rows reach a piped consumer as each one finishes: per-row flush, so `… | jq` or a `while read` loop sees row 1 after 0.11 s, not at stdin EOF (roadmap #63)
    - Candidate B3 -- A CI gate can tell "one bad line" from "everything failed": `--batch --summary PATH` writes `{lines, ok, needs_review, error, flagged_fields, exit}` (roadmap #47)
    winner: A2
    ship: PUSHED 83ce30c
  iter-78
    lenses: new-capability, hardening/DX
    - Candidate A1 -- Token-boundary grounding: a truncated `invoice_number` ("FS-2045" inside "FS-20451") or a partial `currency` ("US" inside "USD") stops being a silent `ok` (roadmap #54, golden row inv-011)
    - Candidate A2 -- `needs_review` hands the reviewer the value to check: flagged amount and invoice-number fields carry deterministic `suggestions` (roadmap #40)
    - Candidate A3 -- Your own labelled corpus gets the trust readout: `validate --batch` rows that carry `expected` are graded (caught / silent-wrong / false alarm), with a `--max-silent-wrong N` exit (roadmap #60)
    - Candidate B1 -- Identifier-role false alarms closed: a correct `invoice_number` on a multi-label header row ("PO: 4410   Invoice #: INV-77"), after "Customer's invoice", or under an "Invoices2024" non-label stops being penalized to 0.3
    - Candidate B2 -- The suite stays green with `REFLECT_FIELD_THRESHOLD` / `REFLECT_OVERALL_THRESHOLD` exported: one autouse conftest fixture resets the operator floor (roadmap #59, marked NEXT)
    - Candidate B3 -- `validate --batch | head -1` exits within the documented 0/1/2/3 contract: no BrokenPipeError traceback, no exit 120, at any `--batch-concurrency` (roadmap #45)
    winner: B3
    ship: PUSHED 7be732b
  iter-77
    lenses: narrative-and-docs
    - Candidate A1 -- An `ok` on partial data stops saying "every field passed" / "All fields cleared" when only some fields were committed; the note counts what was checked
    - Candidate A2 -- One README configuration table lists every environment variable the package reads, with its default, plus a two-way drift test against `src/`
    - Candidate A3 -- The structured-log contract is written down and pinned: an event table per path, and the false "every stage emits generate/heuristics/critique/decide" docstring corrected
    - Candidate B1 -- Identifier-role grounding: an `invoice_number` the source shows only on a PO / order / customer / account line is flagged (roadmap #53, marked NEXT)
    - Candidate B2 -- `needs_review` hands the reviewer the value to check: each heuristic-flagged field carries deterministic `suggestions`, the value(s) the source labels for that role (roadmap #40)
    - Candidate B3 -- Omission detection: a `null` amount the source plainly labels ("Total: 108.50") is flagged, not a silent `ok` that says "every field passed" (roadmap #39)
    winner: B1
    ship: PUSHED 56871a5
  iter-76
    lenses: performance-and-throughput, narrative-and-docs
    - Candidate A1 -- `--batch` rows reach a piped consumer as each one finishes, not at EOF: the first row arrives after 0.12 s instead of 2.76 s (roadmap #63)
    - Candidate A2 -- Heuristics stop re-scanning long documents: 43 KB goes from 18.6 ms to 12.8 ms per `validate` (−31%) with identical reports, and `--batch-concurrency` can't hide this cost
    - Candidate A3 -- The MCP tool path builds each provider client once per process: 20 `extract` calls build 2 SDK clients instead of 40 (roadmap #27, NEXT)
    - Candidate B1 -- A heuristics-only `ok` stops claiming "every field passed the deterministic heuristic checks" when a check penalized a field, or when no field was checked at all
    - Candidate B2 -- README documents each heuristic's penalty factor and what a lowered threshold silences, pinned two ways to `heuristics.py`
    - Candidate B3 -- The MCP tool description stops calling `overall_confidence` "the critic's overall confidence" when no Critic ran; the advertised contract is pinned to measured behavior
    winner: B1
    ship: PUSHED 0b7a53a
  iter-75
    lenses: simplification-and-deletion, performance-and-throughput
    - Candidate A1 -- One single-document CLI path: `run_extract_from_args` and `run_validate_from_args` become one-line wrappers over `_run_verb(args, validate)`, taking cli.py from 360 to 342 lines with byte-identical console behavior
    - Candidate A2 -- One CLI flag set: `__main__` stops carrying its own byte-identical copy of `cli`'s `_add_common_args`/`_add_batch_args` (152 → 104 lines, 0 test edits; roadmap #36)
    - Candidate A3 -- One "the Critic didn't speak" branch: `_verify`'s two copies of neutral critique + `critic_ran=False` + `critic.skipped` collapse into one `skip` reason (pipeline.py 163 → 159, 0 test edits)
    - Candidate B1 -- `--batch --batch-concurrency N` streams again: the first result row comes out after ~0.2 s instead of after stdin EOF, and memory stays flat instead of holding the whole input
    - Candidate B2 -- MCP tool path builds each provider client once per process (locked per-role memo; roadmap #27, marked NEXT): 20 `extract` calls open 2 connections instead of 40, and parallel calls stop racing the SDK client build
    - Candidate B3 -- The Critic writes `rationale` only for fields it doubts: about 28% of every Critic reply's bytes are paid output tokens the product never shows
    winner: B1
    ship: PUSHED e490269
  iter-74
    lenses: integration-and-adoption
    - Candidate A1 -- MCP threshold args default to the operator floor: the published inputSchema stops advertising a `0.7`/`0.75` default the server never applies, and a call that omits them stops getting "field_threshold raised from 0.7" notes
    - Candidate A2 -- An agent can learn what `validate` checks: the MCP `data` argument publishes the invoice field names, and a `data` whose keys are all unknown is no longer a silent `ok` that verified nothing
    - Candidate A3 -- Your own labelled corpus gets the trust readout: `validate --batch` rows that carry `expected` are graded (caught / silent-wrong / false alarms), with a `--max-silent-wrong N` exit for CI
    - Candidate B1 -- One CLI flag set: `__main__` stops carrying its own copy of `cli`'s argparse helpers (152 → 104 lines, zero test edits)
    - Candidate B2 -- One parse-and-retry loop: Generator and Critic share `parsing.complete_parsed` (src −19 lines, two `_RETRY_DELAY_SECONDS` → one)
    - Candidate B3 -- One operator-floor reset for the whole suite: a single autouse `conftest` fixture replaces 6 per-file env-clearing copies, and the suite stops failing under an exported `REFLECT_*_THRESHOLD`
    winner: A1
    ship: PUSHED ac7df7e
  iter-73
    lenses: hardening/DX, integration-and-adoption
    - Candidate A1 -- Token-boundary grounding: a truncated `invoice_number` ("INV-100" inside "INV-1001", "2045" inside "20451") or a partial `currency` ("US" inside "USD") stops being a silent `ok` (roadmap #54)
    - Candidate A2 -- `reflect-mcp … --batch | head -1` exits within the documented contract: no BrokenPipeError traceback, no exit 120 (roadmap #45)
    - Candidate A3 -- A `--batch` JSONL that starts with a UTF-8 BOM, or contains whitespace-only lines, verifies instead of emitting anonymous error rows and exit 3 (roadmap #46)
    - Candidate B1 -- MCP tools publish a real outputSchema: agents and hosts get a typed `status` enum (`ok`/`needs_review`/`error`) instead of "any object", and `status:"error"` results set `isError`
    - Candidate B2 -- The CLI composes in pipelines: `--file -` means stdin, and `--data @PATH` reads the data JSON from a file, so "verify my extractor's output" is a two-file Makefile rule rather than shell quoting
    - Candidate B3 -- `--batch` rows echo the 1-based input `line`, so error rows and id-less rows still join back to their input (roadmap #25 remainder)
    winner: B1
    ship: PUSHED c466b8c
  iter-72
    lenses: new-capability, hardening/DX
    - Candidate A1 -- Identifier-role grounding: an `invoice_number` the source shows only on a PO / order / customer / account line is penalized (roadmap #53)
    - Candidate A2 -- Currency grounding from unambiguous money symbols: "€" grounds `EUR`, "£" grounds `GBP` (also ₹ → INR, ₩ → KRW)
    - Candidate A3 -- `needs_review` hands the reviewer the answer to check: a flagged field carries deterministic `suggestions` (the value the source labels for that role) (roadmap #40)
    - Candidate B1 -- Golden-set pin paydown (roadmap #52, NEXT): 55 of the 79 failures from adding a golden row come from loaders that hard-assert "exactly 8 rows"; one shared frozen-id helper retires them, so inv-009 can land
    - Candidate B2 -- Token-boundary grounding: a truncated `invoice_number` ("INV-100" inside "INV-1001") or a partial `currency` ("US" inside "USD") stops being silently grounded
    - Candidate B3 -- Date-role label boundary: a correct invoice date that shares its line with an earlier "Amount Due:" / "Ship to:" field stops false-alarming, and no catch is lost
    winner: B1
    ship: PUSHED 7e300d5
  iter-71
    lenses: new-capability
    - A1 -- A heuristics-only result reports `overall_confidence: 1.0`, which the tool docs call "the critic's overall confidence", and its `needs_review` notes never say that no Critic ran
    - A2 -- README §7 Architecture and Repository layout describe the post-iter-69 module tree, plus a two-way layout↔`src/` drift test
    - A3 -- PRODUCT.md reports two different speedups for the same iter-69 ship, and nothing checks that roadmap DONE rows and ledger lines agree
    - Candidate B1 -- Date-role grounding: an `invoice_date` the source shows only on a "Due"/"Ship date" line is penalized (roadmap #42)
    - Candidate B2 -- Currency grounding from the money symbol: "€1487.50" grounds `EUR`, "£" grounds `GBP`, and a code contradicted by the symbols on the labelled amounts is penalized
    - Candidate B3 -- Identifier-role grounding: an `invoice_number` seen only on a "PO"/"Order no"/"Customer ID"/"Account" line is penalized
    winner: B1
    ship: PUSHED 823cdfb
  iter-70
    lenses: unknown
    - Candidate A1 -- MCP tools reuse provider clients across calls: 20 `extract` calls open 0 new connections instead of 40, and the client build stops racing between the threads iter 69 made concurrent
    - Candidate A2 -- A parse/schema retry fires immediately and tells the model what failed: the one-bad-reply path drops from 512 ms to about 1 ms, and the paid retry is no longer a byte-identical re-roll
    - Candidate A3 -- Heuristics on long documents run about 2× faster with identical verdicts: a 68 KB source drops from 19.6 ms to under 10 ms per `validate`
    - Candidate B1 -- "One shared trace_id, greppable end to end" becomes true again after iter 69 made MCP calls concurrent: 50 parallel calls get 50 ids instead of 8, retry events stop logging `trace_id=""`, and the prose is pinned by a behavior test
    - Candidate B2 -- README §7 Architecture and Repository layout describe the post-iter-69 module tree: `tools.py` listed, "the `extract` tool (`server.py`)" and "Three roles" corrected, plus a two-way layout↔`src/` drift test
    - Candidate B3 -- PRODUCT.md's roadmap and ledger stop contradicting the repo: iter 18's `validate` verb gets its missing ledger + roadmap row, stale figures are corrected, and a check fails when a shipped iteration's behavior test has no ledger row
    winner: B1
    ship: PUSHED 1a09a74
  iter-69
    lenses: simplification-and-deletion, performance-and-throughput
    - Candidate A1 -- One single-document CLI tail: `run_extract_from_args` and `run_validate_from_args` become one-line wrappers over a shared `_run_verb(args, validate)`, taking cli.py from 360 to 337 lines with byte-identical console behavior
    - Candidate A2 -- One parse-and-retry loop: the Generator and Critic retry loops (identical apart from the target model, prompt and stage name) become `parsing.complete_parsed(...)`, and the two `_RETRY_DELAY_SECONDS` constants become one
    - Candidate A3 -- One MCP tool body: `extract_tool` and `validate_tool` share `_run_tool`, and `tools._build_pipeline` with its "pass a kwarg only when opted out" rule is deleted
    - Candidate B1 -- The MCP server stops serializing tool calls: `extract`/`validate` run on a worker thread, so 4 parallel agent calls take 0.42 s instead of 1.67 s and the event loop keeps serving
    - Candidate B2 -- A parse/schema retry fires immediately and carries the rejection reason, so the failure path drops from 510 ms to 0.1 ms and the paid retry is no longer a byte-identical re-roll (roadmap #26 + #30)
    - Candidate B3 -- The MCP tool path builds each provider client once per process: 10 `extract` calls build 2 SDK clients (and 2 connection pools), not 20 (roadmap #27 memo half)
    winner: B1
    ship: PUSHED 9bd522c
  iter-68
    lenses: integration-and-adoption, simplification-and-deletion
    - Candidate A1 -- Re-verifying a stored reflect-mcp result works: `validate` given the product's own result envelope as `data` checks its `data` instead of returning a vacuous `ok`
    - Candidate A2 -- Library users are shown the shipped verify path: README §6 gains a "Verify your own extractor's output" example using `build_default_pipeline(generator_enabled=False, critic_enabled=False).validate(text, data)`, and a test runs it
    - Candidate A3 -- `--file -` means stdin, so `reflect-mcp` composes in pipelines and Makefiles like other CLIs
    - Candidate B1 -- One CLI parser: delete cli.py's second argparse stack so `__main__` owns every flag (roadmap #36, parser half)
    - Candidate B2 -- One parse-and-retry loop: the Generator's and Critic's twin retry loops become `parsing.complete_with_retry`
    - Candidate B3 -- Delete `tools._build_pipeline` and its "pass a kwarg only when opted out" rule: the two MCP tools call `build_default_pipeline` directly
    winner: A1
    ship: PUSHED 6068d69
  iter-67
    lenses: hardening/DX, integration-and-adoption
    - Candidate A1 -- Whitespace variants can't bypass party-role grounding: an NBSP, tab, double-spaced or CR-only "Bill to" line still flags a buyer extracted as the vendor
    - Candidate A2 -- `reflect-mcp … --batch | head -1` exits cleanly: no BrokenPipeError traceback and no exit 120 outside the documented 0/1/2/3 contract
    - Candidate A3 -- A `--batch` JSONL that starts with a UTF-8 BOM, or has whitespace-only lines, verifies instead of emitting error lines and exit 3
    - Candidate B1 -- `--batch` output is joinable: every row echoes the caller's `id` (and optionally the 1-based input `line`), error rows included
    - Candidate B2 -- A batch run can end with one machine-readable verdict: `--summary PATH` writes `{lines, ok, needs_review, error, flagged_fields:{field:count}, exit}`
    - Candidate B3 -- Every result says machine-readably whether a Critic graded it (`critic_ran: bool`), so an auto-commit consumer can require a Critic-verified `ok`
    winner: B1
    ship: PUSHED 4326d92
  iter-66
    lenses: new-capability, hardening/DX
    - Candidate A1 -- Party-role grounding: a `vendor_name` the source shows only on a "Bill to"/"Ship to"/"Sold to" line is penalized (0.3), so heuristics-only inv-007 stops being a silent `ok`
    - Candidate A2 -- Every committed field says which checks actually vouched for it: a machine-readable `field_checks` map (`pass` / `penalized` / `abstained` per check: grounding, amount_role, date, arithmetic, critic)
    - Candidate A3 -- Schema extensibility, first slice: `validate` (library + CLI `--schema module:Model`) verifies a user-supplied flat Pydantic model with generic grounding over its str/float fields
    - Candidate B1 -- trace_id contract: unique per call, and the retry events carry it (roadmap #21)
    - Candidate B2 -- A verification that committed zero fields can't be `ok`: all-null extraction, `validate` with `{}`, or only-unknown keys (`{"totl": 108.5}`) return `needs_review` (roadmap #20 + #24)
    - Candidate B3 -- Non-finite amounts are rejected at the boundary, so stdout is always strict RFC 8259 JSON
    winner: A1
    ship: PUSHED 618c480
  iter-65
    lenses: narrative-and-docs, new-capability
    - Candidate A1 -- A heuristics-only result stops saying "All fields cleared" and stops logging a `critique.done` that never happened: `notes`, `critic_summary` and the log name what actually verified it
    - Candidate A2 -- "One shared trace_id, greppable end to end" becomes true: a unique id per call (26/200 → 200/200 on a batch), carried onto retry events, with the three prose sites pinned
    - Candidate A3 -- The repo's maps match the territory: README §7 + Repository layout list `tools.py` and both tools, PRODUCT.md gets its missing iter-18 `validate` row, and a two-way check fails when a module or a shipped iteration goes unrecorded
    - Candidate B1 -- Omission detection: a null amount the source labels ("Total: $108.50" → `total: null`) is flagged in `needs_review`, not passed as `ok`
    - Candidate B2 -- `needs_review` gives the reviewer the answer to check: each flagged field carries deterministic `suggestions` (source-labelled value + span)
    - Candidate B3 -- `validate --batch` rows may carry `expected`: grade the gate against your own extractor's labelled output offline, with a `--max-silent-wrong N` exit
    winner: A1
    ship: PUSHED acb7832
  iter-64
    lenses: performance-and-throughput, narrative-and-docs
    - Candidate A1 -- `validate --batch`: verify a JSONL corpus of `{text, data}` in one process, so CI drops from 129 ms/doc to under 1 ms/doc
    - Candidate A2 -- MCP tools reuse provider clients across calls: 10 extract calls build 2 clients, not 20 (roadmap #27 remainder)
    - Candidate A3 -- Retry immediately and tell the model what failed (roadmap #26 + #30): the failure path drops from 510 ms to about 0 and stops resending a byte-identical prompt
    - Candidate B1 -- README §5 says heuristics-only works "including batch mode", but `validate --batch` exits 2: fix the sentence and add a two-way check that README verb+flag claims match the real argparse tree
    - Candidate B2 -- README §7 Architecture and Repository layout match the module tree: add `tools.py`, fix "Three roles" and "The `extract` tool (`server.py`)", say when the Generator or Critic doesn't run, plus a two-way layout drift test
    - Candidate B3 -- A heuristics-only result stops claiming "All fields cleared" and a confidence of 1.0 that no Critic produced: `notes` and `critic_summary` say what actually verified the result
    winner: A1
    ship: PUSHED 3bcb037
  iter-63
    lenses: simplification-and-deletion, performance-and-throughput
    - Candidate A1 -- One CLI parser and one single-document tail: delete cli.py's copy of the argparse stack and fold the two `*_from_args` tails into `_run_single` (roadmap #36)
    - Candidate A2 -- One "the Critic didn't speak" branch: `_verify` synthesizes the neutral critique in one place and logs `critique.done` once
    - Candidate A3 -- Delete the "pass `critic_enabled` only when opted out" rule: `tools._build_pipeline` goes away, and optionally cli's `_build_gate_config` folds into `_build_pipeline_kwargs`
    - Candidate B1 -- Drop the fixed 0.5 s sleep before a parse/schema retry (roadmap #26)
    - Candidate B2 -- Build provider clients only when used, and once per process (roadmap #27)
    - Candidate B3 -- A Critic failure keeps the paid Generator output: the error envelope carries `unverified_data`, so recovery costs one Critic call instead of a full re-extract
    winner: B2
    ship: PUSHED c511bac
  iter-62
    lenses: integration-and-adoption, simplification-and-deletion
    - Candidate A1 -- Operator lock for `heuristics_only` over MCP: unless the server's `env` sets `REFLECT_ALLOW_HEURISTICS_ONLY=1`, an agent's `heuristics_only=True` is overridden (the Critic still runs) and `notes` says so
    - Candidate A2 -- Every result says how it was verified: a machine-readable `verification` field (`"critic+heuristics"` | `"heuristics_only"` | `"critic_skipped_doomed"`), so a consumer that auto-commits `ok` results can require a Critic-verified `ok`
    - Candidate A3 -- The decorrelation rule is consulted where the operator configures models: `build_default_pipeline` refuses a Critic whose resolved model id equals the Generator's (error envelope / exit 3), unless `REFLECT_ALLOW_SAME_MODEL=1`
    - Candidate B1 -- One CLI parser and one single-document run path: delete cli.py's duplicate argparse stack and fold `run_extract_from_args`/`run_validate_from_args` into one `_run_single`
    - Candidate B2 -- Delete the `heuristics_enabled` constructor knob and the dead `_number_in_text` helper (roadmap #28)
    - Candidate B3 -- One parse-and-retry loop: replace the twin Generator/Critic retry loops with `parsing.complete_with_retry`
    winner: B2
    ship: PUSHED dd3368d
  iter-61
    lenses: hardening/DX, integration-and-adoption
    - Candidate A1 -- Amount-role grounding attributes each amount to its own line's label, so "Sales tax (8.25%): 16.50" stops false-alarming and "Total tax: 20.00" stops vouching for a wrong total
    - Candidate A2 -- `make typecheck`: locked strict mypy over src/ backs the README's `py.typed` "full type-checker support" claim (roadmap #4), and fixes the 5 real errors it finds today
    - Candidate A3 -- Non-finite amounts are rejected at the input boundary, so CLI stdout is always strict RFC 8259 JSON
    - Candidate B1 -- Operator-pinned gate floor: `REFLECT_FIELD_THRESHOLD` / `REFLECT_OVERALL_THRESHOLD` in the MCP server's `env` set the thresholds, and a tool-call argument can only tighten them
    - Candidate B2 -- `validate --heuristics-only` (CLI and MCP) runs on a base `pip install reflect-mcp` with no provider SDK and no credentials
    - Candidate B3 -- `reflect-mcp eval --golden FILE`: an installed user grades their own labelled corpus with the iter-59 trust-contract readout (accuracy, catch rate, silent-wrong) and `--max-silent-wrong` exit
    winner: B1
    ship: PUSHED 4829966
  iter-60
    lenses: new-capability, hardening/DX
    - Candidate A1 -- Omission detection: a null amount the source plainly labels ("Total: $108.50" → `total: null`) is flagged in `needs_review` instead of returning `ok`
    - Candidate A2 -- Amount-role grounding: a subtotal/tax/total whose value differs from the number the source labels for that role is penalized (swapped subtotal↔tax no longer passes)
    - Candidate A3 -- Token-boundary grounding for text fields: `invoice_number`/`vendor_name`/`currency` must match a whole source token, not a fragment ("QM-993" inside "QM-9930" is no longer grounded)
    - Candidate B1 -- A verification that checked zero fields can't be `ok`: all-null extraction, `validate` with `{}`, or only-unknown keys ("totl") return `needs_review`, with dropped keys named in `notes` (roadmap #20 + #24)
    - Candidate B2 -- Non-finite amounts are rejected at the boundary: `NaN`/`Infinity` from the Generator or `validate --data` is a schema error, and stdout is always strict RFC 8259 JSON
    - Candidate B3 -- trace_id contract: unique per call and present on every log event, retries included (roadmap #21)
    winner: A2
    ship: PUSHED 0f80a9f
  iter-59
    lenses: narrative-and-docs, new-capability
    - Candidate A -- The result explains what was actually verified: `notes` stop saying "All fields cleared" when nothing was committed or no Critic ran
    - Candidate B -- README "What verification does NOT catch" table, each row pinned by a strict-xfail test that forces the doc update when the gap closes
    - Candidate C -- README §5's CLI block is executed by the suite: every documented command reaches its documented exit code, and `--help` carries the 0/1/2/3 contract
    - Candidate B1 -- `needs_review` hands the reviewer evidence: each flagged field carries deterministic `suggestions` (source spans it could be), e.g. inv-004's total gets the "583.20" token
    - Candidate B2 -- Omission detection: a null field the source plainly labels ("Total: $108.50" → `total: null`) is flagged, not silently dropped
    - Candidate B3 -- `make eval` measures the trust contract: a silent-wrong count (wrong field not flagged) and a catch rate, with `--max-silent-wrong N` exit
    winner: B3
    ship: PUSHED cd55f64
  iter-58
    lenses: performance-and-throughput, narrative-and-docs
    - Candidate A1 -- Make the one paid retry count: retry immediately (no 0.5s sleep) and tell the model what failed, instead of resending a byte-identical prompt
    - Candidate A2 -- Build only the clients a call will use: validate builds no Generator client, heuristics-only builds none, and the MCP tool path reuses clients across calls
    - Candidate A3 -- `extract --batch` pays once per distinct document: identical `text` lines in one run reuse the first line's result instead of re-paying 2 LLM calls each
    - Candidate A -- README §3 Demo shows the run the product actually makes: replace the made-up Globex block with the real inv-004 replay output, pinned by a test that re-runs it
    - Candidate B -- §7 Architecture + Repository layout match the module tree: add `tools.py` (iter 55), fix "Three roles" / "The `extract` tool (`server.py`)", say when the Critic doesn't run, plus a two-way file-tree drift test
    - Candidate C -- The log tells the truth about the Critic: one event name for "Critic didn't run", and the README §10 / `logging_config` "every stage carries the shared trace_id" claim corrected, with a catalog test over the `log_stage` call sites
    winner: A1
    ship: PUSHED 9183f52
  iter-57
    lenses: performance-and-throughput
    - Candidate A1 -- Re-land iter-56's "one batch path" (the winner was aborted for infra reasons, not on merit), which also fixes a bad-threshold traceback that returns exit 1
    - Candidate A2 -- Delete the `heuristics_enabled` constructor knob: the exported library API can build a pipeline that returns `ok` on fabricated data
    - Candidate A3 -- One threshold/pipeline-build rule: delete cli.py's `_build_gate_config` + `_build_pipeline_kwargs` and route single-mode `extract`/`validate` through `tools.extract_tool`/`validate_tool`
    - Candidate B1 -- Drop the fixed 0.5s sleep before a parse/schema retry: it is pure latency on every retry
    - Candidate B2 -- Build each provider client once per process, not once per call (and never build an unused one)
    - Candidate B3 -- Don't throw away the paid Generator output when the Critic fails: return it in the error envelope so recovery costs one Critic call, not a full re-extract
    winner: A1
    ship: PUSHED ac9f9c7
  iter-56
    lenses: integration-and-adoption, simplification-and-deletion
    - Candidate A1 -- `validate` rejects `data` keys it doesn't know (and empty `data`) instead of verifying nothing and returning `ok`
    - Candidate A2 -- Batch output is joinable: echo the caller's `id` and the 1-based input `line` on every result, and apply the empty-input pre-check per line
    - Candidate A3 -- `validate --heuristics-only` works with no provider SDK or credentials, delivering the "zero-cost, CI" on-ramp the README promises
    - Candidate B1 -- One CLI parser: delete cli.py's duplicate argparse stack (and the dormant `_number_in_text`)
    - Candidate B2 -- Collapse the Generator and Critic retry loops into one `parse_with_retry` helper
    - Candidate B3 -- One batch path: fold `_run_batch` and `_run_batch_concurrent` (and `_read_text` and `_read_input_stream`) into single implementations
    winner: B3
    ship: unknown
  iter-55
    lenses: integration-and-adoption
    - Candidate A1 -- A numeric `invoice_number` from the Generator crashes the run instead of being coerced to a string
    - Candidate A2 -- Non-finite amounts (`NaN`/`Infinity`) are accepted end to end and the CLI prints invalid JSON
    - Candidate A3 -- Blank strings and non-code currencies sail through as `ok`: give text fields the format contract `invoice_date` already has
    - Candidate B1 -- MCP tools reach CLI parity: `heuristics_only` arg + structured `status:"error"` envelope, testable offline
    - Candidate B2 -- `validate --batch`: verify a JSONL corpus of `{text, data}` produced by someone else's extractor
    - Candidate B3 -- Batch output carries a correlation key: echo the caller's `id` (and the 1-based input `line`) on every result, including error lines
    winner: B1
    ship: PUSHED 9c540a3
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
47 scouted iterations
