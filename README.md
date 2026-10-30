# reflect-mcp

> An MCP server for LLM tasks that verify themselves: **Generator → second-model Critic → human-in-the-loop gate**. Never a silent wrong answer.

![tests](https://img.shields.io/badge/tests-passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## 1. TL;DR

`reflect-mcp` is a small multi-agent pipeline that **checks its own work before surfacing it**, exposed as an [MCP](https://modelcontextprotocol.io) server so it plugs into Claude Desktop, Cursor, or any MCP client. A **Generator** extracts a typed object from messy text; a **second-model Critic** scores every field and the whole extraction; a **confidence-thresholded gate** either returns the result or, if it isn't confident enough, returns a `needs_review` payload naming the exact fields a human should confirm.

```
                 ┌───────────┐      ┌──────────────┐      ┌────────┐      ┌──────────────┐
  messy text ──▶ │ Generator │ ───▶ │ heuristics + │ ───▶ │  Gate  │ ───▶ │  ok  /       │
                 │ (model A) │      │ Critic (B)   │      │ (HITL) │      │  needs_review│
                 └───────────┘      └──────────────┘      └────────┘      └──────────────┘
                   typed JSON        per-field + overall     threshold      flagged fields
                                       confidence            decision       + rationale
```

## 2. Why — the trust problem

LLMs are **confidently wrong**. The failure mode that actually hurts in production isn't a model that says "I don't know" — it's a model that returns a clean, plausible, wrong value with no signal that anything is off. `reflect-mcp` makes verification a **first-class step**: a *different* model critiques the output, deterministic structural checks run alongside it, and anything below a confidence threshold is escalated instead of emitted.

The pattern is well established: naive extraction pipelines look great in a demo and then quietly degrade on real inputs, and adding a multi-model verification + human-review gate is a proven way to claw accuracy back. The point `reflect-mcp` makes concrete: *the verification discipline is the product*, not the extraction call.

### In research terms — an evaluation harness with a verifiable reward

Stripped of the invoice example, `reflect-mcp` is a small **evaluation harness with a verifiable reward**, wired into the inference path instead of bolted on after the fact. The Critic is a **decorrelated grader** — a *different* model from the Generator, because same-model self-critique rubber-stamps its own errors. The deterministic heuristics are the **programmatic pass criterion**: grounding, amount-role grounding (a total must match the number the source labels as the total), arithmetic reconciliation, date validity and date grounding are checkable without a model, and they can only *lower* the score, so a structurally impossible value can't be talked through on a grader's confidence alone. The gate is the **escalation path for the cases the reward signal can't settle** — which is what turns a low-confidence result into useful output instead of a failure. And `make eval` grades the harness itself: a labeled golden set, field-level accuracy, a catch rate for wrong fields, and a non-zero exit on low accuracy or any silent-wrong field, so the whole loop is measurable offline in CI with no secrets.

## 3. Demo

Messy input → typed JSON with per-field confidence → one low-confidence field flagged for review.
This is golden row `inv-004`, replayed offline through the real pipeline (`make eval` runs the same replay).
The source says the amount due is **583.20**. The scripted Generator wrote **683.2**, an invented total.
The gate caught it three ways: the arithmetic check (540.00 + 43.20 ≠ 683.2), the heuristic source search, and the Critic's `grounded: false` verdict.

Input:

```text
Hi team, forwarding the bill from Globex Ltd below.
Ref: GLX/2026/778
Billing date 2026-02-28.
Items came to 540.00 before tax, tax was 43.20, and the amount due is 583.20. Please pay in GBP. Thanks!
```

Output (`trace_id` is per-run and shown here as a placeholder):

```json
{
  "status": "needs_review",
  "data": {
    "vendor_name": "Globex Ltd",
    "invoice_number": "GLX/2026/778",
    "invoice_date": "2026-02-28",
    "subtotal": 540.0,
    "tax": 43.2,
    "total": 683.2,
    "currency": "GBP"
  },
  "field_confidence": {
    "vendor_name": 0.95,
    "invoice_number": 0.95,
    "invoice_date": 0.95,
    "subtotal": 0.95,
    "tax": 0.95,
    "total": 0.12,
    "currency": 0.95
  },
  "overall_confidence": 0.7,
  "flagged_fields": [
    {
      "field": "total",
      "value": 683.2,
      "confidence": 0.12,
      "reason": "683.20 does not reconcile with 540.00 + 43.20 and is not in the source; critic marked the value not grounded in the source; amount 683.2 not found in source text; subtotal(540.0) + tax(43.2) != total(683.2)"
    }
  ],
  "field_provenance": {
    "vendor_name": [
      {
        "start": 34,
        "end": 44
      }
    ],
    "invoice_number": [
      {
        "start": 57,
        "end": 69
      }
    ],
    "invoice_date": [
      {
        "start": 83,
        "end": 93
      }
    ],
    "subtotal": [
      {
        "start": 109,
        "end": 115
      }
    ],
    "tax": [
      {
        "start": 136,
        "end": 141
      }
    ],
    "total": [],
    "currency": [
      {
        "start": 187,
        "end": 190
      }
    ]
  },
  "trace_id": "trace-…",
  "critic_summary": "Total appears hallucinated; escalate.",
  "notes": "1 field(s) below the 70% confidence threshold require human review before use."
}
```

*(A ~20s GIF of this in Claude Desktop goes here in the published repo.)*

## 4. Use it in Claude Desktop

Install and point Claude Desktop at the server. Add this to your `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "reflect-mcp": {
      "command": "/ABSOLUTE/PATH/reflect-mcp/.venv/bin/python",
      "args": ["-m", "reflect_mcp.server"],
      "env": {
        "REFLECT_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "REFLECT_GENERATOR_MODEL": "claude-3-5-sonnet-latest",
        "REFLECT_CRITIC_MODEL": "claude-3-5-haiku-latest"
      }
    }
  }
}
```

Then restart Claude Desktop and ask it to *"extract the invoice fields from this text …"* — it will call the `extract` tool. You can also ask it to *"validate this data against the source"* to run verification without re-extracting.

**Tools exposed**

| Tool | Args | Returns |
| --- | --- | --- |
| `extract` | `text: str`, `field_threshold=0.70`, `overall_threshold=0.75`, `heuristics_only=False` | `{ status, data, field_confidence, overall_confidence, flagged_fields, field_provenance, critic_summary, trace_id, notes }` |
| `validate` | `text: str`, `data: dict`, `field_threshold=0.70`, `overall_threshold=0.75`, `heuristics_only=False` | Same `ExtractionResult` envelope as `extract` |

When `status == "needs_review"`, do **not** use `data` blindly — confirm the fields in `flagged_fields` first.

The tools never raise: failures (empty input, out-of-range thresholds, `data` that fails `InvoiceData` validation, provider or parse errors) return `{ status: "error", error_message }` instead of raising, mirroring the CLI's error envelope. `heuristics_only=True` skips the Critic LLM call, like the CLI's `--heuristics-only`.

**Operator threshold floor.** Over MCP, the threshold arguments are chosen by the calling agent, so they can only be raised above the operator's floor, never lowered below it. Set the floor in the server's `env` block with `REFLECT_FIELD_THRESHOLD` and `REFLECT_OVERALL_THRESHOLD` (defaults 0.70 / 0.75 when unset). A lower argument is raised to the floor, and `notes` records it (e.g. `field_threshold raised from 0.0 to 0.7.`). An agent may always ask for stricter thresholds; only the operator can lower them, e.g. `"REFLECT_FIELD_THRESHOLD": "0"`. (The agent-controlled `heuristics_only=True` argument still drops the Critic, so it is not covered by this floor.) A malformed value (not a number, including nan/inf, or outside [0, 1]) makes every call return `{ status: "error", error_message }` rather than falling back to a looser gate. The CLI does not read these variables: its `--field-threshold` / `--overall-threshold` flags stay authoritative.

## 5. CLI

For scripting, CI pipelines, or batch processing without an MCP client:

```bash
# discover subcommands and flags
reflect-mcp --help
reflect-mcp --version
reflect-mcp extract --help

# extract: generate + verify
echo "ACME CORP, Invoice #INV-1001, Total: \$108.50" | reflect-mcp extract
reflect-mcp extract --file invoice.txt
reflect-mcp extract --field-threshold 0.85 --overall-threshold 0.80 --file invoice.txt

# batch extract: process a corpus as JSONL (one {"text": "..."} per line)
cat invoices.jsonl | reflect-mcp extract --batch
reflect-mcp extract --batch --file invoices.jsonl
reflect-mcp extract --batch --field-threshold 0.85 --overall-threshold 0.80

# concurrent batch: overlap LLM calls across documents (4 workers)
cat invoices.jsonl | reflect-mcp extract --batch --batch-concurrency 4

# heuristics-only: skip the Critic LLM call, use only deterministic checks
echo "ACME CORP, Invoice #INV-1001, Total: \$108.50" | reflect-mcp extract --heuristics-only

# validate: verify pre-extracted data (no Generator, Critic + heuristics + gate only)
reflect-mcp validate --file invoice.txt --data '{"vendor_name":"ACME CORP","total":108.5}'
reflect-mcp validate --file invoice.txt --data '{"vendor_name":"ACME CORP"}' --field-threshold 0.5
reflect-mcp validate --file invoice.txt --data '{"vendor_name":"ACME CORP"}' --heuristics-only
```

Exit codes (both verbs): **0** = `ok`, **1** = `needs_review`, **2** = usage/input error, **3** = pipeline error (JSON error envelope on stdout).

**Heuristics-only mode** (`--heuristics-only`): skips the Critic LLM call entirely. Verification relies solely on deterministic structural checks (text grounding, amount-role grounding, arithmetic reconciliation, date validation and grounding). Useful for CI pipelines, pre-screening, or zero-cost structural-only validation. Works with both `extract` and `validate`, including batch mode.

**Batch mode** (`--batch`): reads JSONL from stdin or `--file`, emits one JSON result per line to stdout, reuses the pipeline across all lines. Exit code is the worst severity across all lines (error 3 > needs_review 1 > ok 0). Malformed lines and per-line extraction failures emit `{"status":"error","error_message":"..."}` without aborting the batch. Add `--batch-concurrency N` (N > 1) to process documents concurrently via a thread pool; each worker builds its own pipeline instance so SDK clients are never shared across threads. Output order is always preserved. The pipeline is built once up front, so a pipeline/provider build failure or an invalid threshold flag yields a single `status:"error"` line and exit 3 at any `--batch-concurrency`.

The bare `reflect-mcp` command (no subcommand) still starts the MCP server.

## 6. Use as a Python library

Install `reflect-mcp` as a dependency and import directly — the package ships a `py.typed` marker for full type-checker support.

### Full extraction pipeline

```python
from reflect_mcp import build_default_pipeline, GateConfig, ExtractionResult

pipeline = build_default_pipeline(
    gate_config=GateConfig(field_threshold=0.80, overall_threshold=0.85),
)
result: ExtractionResult = pipeline.extract("ACME Corp, Invoice #A-99, total $250.00")

if result.status == "needs_review":
    for flag in result.flagged_fields:        # list[FlaggedField]
        print(f"{flag.field}: {flag.reason}")
else:
    print(result.data.vendor_name)            # InvoiceData fields
```

`build_default_pipeline` reads `REFLECT_PROVIDER` / `REFLECT_GENERATOR_MODEL` / `REFLECT_CRITIC_MODEL` from the environment and returns a `ReflectivePipeline`. Pass `critic_enabled=False` to skip the Critic LLM call and use only deterministic heuristics for verification. You can also construct one manually with your own `Generator` and `Critic` instances (useful for testing with `ScriptedLLMClient`).

### Gate-only verification

If you already have extracted data and just need the confidence gate:

```python
from reflect_mcp import (
    decide,
    GateConfig,
    InvoiceData,
    CritiqueResult,
    FieldVerdict,
)
from reflect_mcp.heuristics import run_heuristics

data = InvoiceData(vendor_name="ACME Corp", total=250.0, currency="USD")
critique = CritiqueResult(
    field_verdicts=[
        FieldVerdict(field="vendor_name", confidence=0.95, grounded=True),
        FieldVerdict(field="total", confidence=0.95, grounded=True),
        FieldVerdict(field="currency", confidence=0.95, grounded=True),
    ],
    overall_confidence=0.92,
)
heuristics = run_heuristics("ACME Corp, total $250.00 USD", data)
result = decide(data, critique, heuristics, trace_id="demo")
# result.status == "ok"
```

### Exported symbols

The public API (`reflect_mcp.__all__`) exposes:

| Category | Symbols |
| --- | --- |
| Schema models | `InvoiceData`, `CritiqueResult`, `FieldVerdict`, `FlaggedField`, `ProvenanceSpan`, `ExtractionResult`, `EXTRACTION_FIELDS` |
| Pipeline | `ReflectivePipeline`, `build_default_pipeline` |
| Generator | `Generator`, `GeneratorError` |
| Critic | `Critic`, `CriticError` |
| Gate | `GateConfig`, `decide` |

All models are Pydantic `BaseModel` subclasses. Errors (`GeneratorError`, `CriticError`) are raised on parse failures after retries are exhausted.

## 7. Architecture

Three roles, one shared trace id, clean layering (`resolver`-free by design — the pipeline has no infra dependency):

- **Generator** (`generator.py`) — LLM call (with configurable retry on parse failure) → validated `InvoiceData` (Pydantic). Absent fields come back `null`; the model is instructed never to invent a value.
- **Critic** (`critic.py`) — a **different model** (`REFLECT_CRITIC_MODEL`) is shown the source *and* the candidate extraction and must ground each field, returning per-field + overall confidence (with configurable retry on parse failure). A different model matters: self-critique with the same model tends to rubber-stamp its own errors.
- **Deterministic heuristics** (`heuristics.py`) — a free, non-LLM signal: grounding (is a committed value actually present in the source?), amount-role grounding (a grounded `subtotal`/`tax`/`total` must match a number the source labels for that role, e.g. `Total: 75.00`, so a swapped amount or a stray item-line number is penalized), arithmetic reconciliation (`subtotal + tax == total`), ISO-date validation, and date grounding (an ISO date must appear in the source as ISO, `MM/DD/YYYY`/`DD/MM/YYYY`/`DD.MM.YYYY`, or an English month-name form such as `March 14, 2026` / `14 Mar 2026`, otherwise it is penalized). These *cap* the Critic — a structurally impossible value can't be waved through on the Critic's word alone.
- **Gate** (`gate.py`) — combines the two signals multiplicatively (`combined = critic_confidence × heuristic_factor`), flags any non-null field below `field_threshold`, and escalates the whole result if any field is flagged **or** overall confidence is below `overall_threshold`. It also enforces the Critic's contract: a committed (non-null) field the Critic marks `grounded:false`, or returns no verdict for, has its critic confidence capped at 0.5, and duplicate verdicts for a field use the minimum confidence (on a tie, the `grounded:false` verdict wins, so verdict order never matters).

The `extract` tool (`server.py`) is the only module that imports the `mcp` SDK, and `providers.py` is the only module that imports a provider SDK — so the core pipeline and the entire test suite have **zero** hard dependency on either.

## 8. Evaluation

A tiny golden-file eval treats LLM quality as **measurable, not vibes**. `eval/golden/dataset.jsonl` holds labeled messy-text → expected-fields examples; `eval/run_eval.py` runs the full pipeline over them and reports field accuracy, exact-match accuracy, needs-review rate, and *exactly where it fails*. It also grades the trust contract itself: **catch rate** (wrong fields the gate named in `flagged_fields`), **silent-wrong** (a wrong field not named in `flagged_fields` — returned as if it were fine), and **false alarms** (correct fields flagged anyway; reported, never gated). Each failure line is tagged `(flagged)` or `(SILENT)`.

```bash
make eval          # OFFLINE: deterministic, uses recorded model outputs (no API keys)
make eval-live     # LIVE: runs against REFLECT_PROVIDER (needs credentials)
```

Offline run (recorded fixtures; four examples carry an injected hallucination — a wrong total in `inv-004`, in `inv-006` a made-up date that the date-grounding heuristic catches, in `inv-007` the bill-to customer extracted as the vendor, which the Critic marks `grounded:false`, and in `inv-008` a stray item-line amount taken as the total, which amount-role grounding catches even though the Critic is fooled — to prove the harness catches misses):

```
==============================================================
reflect-mcp eval  |  mode: OFFLINE (replay fixtures)
==============================================================
examples          : 8
field accuracy    : 92.9%  (52/56)
exact-match acc   : 50.0%  (4/8)
needs-review rate : 50.0%  (4/8)
catch rate        : 100.0%  (4/4)
silent-wrong      : 0
false alarms      : 0
--------------------------------------------------------------
field-level failures (where it breaks):
  [inv-004] total: expected 583.2 but got 683.2  (flagged)
  [inv-006] invoice_date: expected '2026-03-14' but got '2026-04-14'  (flagged)
  [inv-007] vendor_name: expected 'NORTHWIND TRADERS' but got 'Contoso Ltd'  (flagged)
  [inv-008] total: expected 75.0 but got 45.0  (flagged)
==============================================================
PASS: field accuracy 92.9% >= required 90.0%
PASS: silent-wrong fields 0 <= allowed 0
```

`make eval` exits non-zero when field accuracy is below `--min-accuracy` (default 0.90) **or** silent-wrong is above `--max-silent-wrong` (default 0), so it works as an **offline eval-CI gate** with no secrets — and a regression that drops a field without flagging it fails the gate even while accuracy stays above threshold.

## 9. Testing

Deterministic tests around a stochastic core. The single LLM boundary (`LLMClient`) is mocked with a `ScriptedLLMClient` that returns canned JSON and raises if it's ever asked for more than expected — so `pytest` runs **fully offline, with no API keys, and can't leak a real call**.

```bash
make install       # .venv + package + dev/server deps
make test          # deterministic offline suite
make cov           # with coverage
```

The flagship test is `tests/test_critic_catches_error.py`: the Generator is fed a deliberately wrong value (`vendor_name = "Invoice"` — a word that *does* appear in the source, so the deterministic grounding check can't catch it), and the test asserts the **second-model Critic** catches it and the gate routes to `needs_review`. It also flips only the Critic's verdict and shows the decision flip to `ok` — proving the verification step is doing real work, not the plumbing.

## 10. Design decisions

- **Second-model Critic (multi-model reflection), not self-critique.** The Critic defaults to a *different, cheaper* model than the Generator (`sonnet` generates, `haiku` critiques). Cross-model disagreement is a real signal; same-model self-review is mostly agreement theater.
- **Provider switching via env var.** `REFLECT_PROVIDER` selects `anthropic` | `openai` | `bedrock`, so a reviewer runs it against whatever key they have. The **Bedrock (Claude)** path is there for AWS users; the public-API paths (Anthropic/OpenAI) let anyone try it without AWS. Provider SDKs are optional extras (`pip install 'reflect-mcp[anthropic]'`) and imported lazily.
- **Confidence is multiplicative.** `critic_confidence × heuristic_factor` — evidence can only *lower* confidence. A structurally impossible value (bad arithmetic, ungrounded amount) is capped regardless of how sure the Critic sounded.
- **Nulls are honest.** A field absent from the source returns `null` and is never flagged — abstention is correct behavior, not a low-confidence guess.
- **Prompt versioning + injection awareness.** Prompts live in versioned files (`src/reflect_mcp/prompts/`). Both instruct the model to treat document content as **data, not commands** — the source text is untrusted input, a basic prompt-injection guard.
- **Observability.** Every stage emits a structured JSON log line carrying a shared `trace_id`, so one extraction is greppable end to end.

## 11. What I'd productionize next

- **Caching** of Generator/Critic calls; per-request cost/latency budgets.
- A real **human-review queue** (persist `needs_review` payloads, route to a reviewer UI, feed corrections back as new golden examples).
- **Offline eval in CI** on every PR (the `make eval` gate is already CI-shaped) plus periodic **live** eval to catch model drift.
- **Confidence calibration** — replace hand-set thresholds with thresholds tuned on the labeled set to hit a target precision.
- Streaming + partial results, and a richer schema library (invoices, receipts, resumes, log lines) behind the same verify-then-gate loop.

---

## Repository layout

```
reflect-mcp/
├── src/reflect_mcp/
│   ├── schema.py          # Pydantic extraction + verdict/decision models
│   ├── providers.py       # provider switching (Anthropic/OpenAI/Bedrock), lazy SDKs
│   ├── scripted.py        # ScriptedLLMClient — the mock at the LLM boundary
│   ├── generator.py       # Generator stage
│   ├── critic.py          # second-model Critic stage
│   ├── heuristics.py      # deterministic structural checks
│   ├── gate.py            # confidence gate / HITL decision
│   ├── pipeline.py        # orchestration (generate → heuristics → critique → decide)
│   ├── logging_config.py  # structured JSON-line logging
│   ├── parsing.py         # robust JSON extraction + prompt loader
│   ├── prompts/           # versioned generator.txt / critic.txt
│   ├── server.py          # FastMCP server exposing `extract` and `validate` tools
│   ├── cli.py             # CLI entry points: `reflect-mcp extract` / `validate`
│   └── __main__.py        # argparse dispatch: --help/--version/subcommands → cli, bare → server
├── eval/
│   ├── golden/dataset.jsonl   # labeled golden examples
│   ├── fixtures/replay.json   # recorded model outputs for offline eval
│   └── run_eval.py            # eval harness
├── tests/                 # deterministic, fully offline
├── Makefile               # install / test / cov / eval targets
├── pyproject.toml
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
