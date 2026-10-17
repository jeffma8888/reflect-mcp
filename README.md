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

Stripped of the invoice example, `reflect-mcp` is a small **evaluation harness with a verifiable reward**, wired into the inference path instead of bolted on after the fact. The Critic is a **decorrelated grader** — a *different* model from the Generator, because same-model self-critique rubber-stamps its own errors. The deterministic heuristics are the **programmatic pass criterion**: grounding, arithmetic reconciliation, and date validity are checkable without a model, and they can only *lower* the score, so a structurally impossible value can't be talked through on a grader's confidence alone. The gate is the **escalation path for the cases the reward signal can't settle** — which is what turns a low-confidence result into useful output instead of a failure. And `make eval` grades the harness itself: a labeled golden set, field-level accuracy, and a non-zero exit below a threshold, so the whole loop is measurable offline in CI with no secrets.

## 3. Demo

Messy input → typed JSON with per-field confidence → one low-confidence field flagged for review:

```jsonc
// input
"Hi team, bill from Globex Ltd. Ref GLX/2026/778, dated 2026-02-28.
 540.00 before tax, tax 43.20, amount due 683.20. Pay in GBP."

// output  (the model hallucinated the total; the critic + arithmetic check caught it)
{
  "status": "needs_review",
  "data": { "vendor_name": "Globex Ltd", "invoice_number": "GLX/2026/778",
            "invoice_date": "2026-02-28", "subtotal": 540.0, "tax": 43.2,
            "total": 683.2, "currency": "GBP" },
  "field_confidence": { "total": 0.12, "subtotal": 0.95, "tax": 0.95, "...": 0.95 },
  "overall_confidence": 0.70,
  "flagged_fields": [
    { "field": "total", "value": 683.2, "confidence": 0.12,
      "reason": "683.20 does not reconcile with 540.00 + 43.20; not found in source" }
  ],
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
| `extract` | `text: str`, `field_threshold=0.70`, `overall_threshold=0.75` | `{ status, data, field_confidence, overall_confidence, flagged_fields, critic_summary, trace_id, notes }` |
| `validate` | `text: str`, `data: dict`, `field_threshold=0.70`, `overall_threshold=0.75` | Same `ExtractionResult` envelope as `extract` |

When `status == "needs_review"`, do **not** use `data` blindly — confirm the fields in `flagged_fields` first.

## 5. CLI

For scripting, CI pipelines, or batch processing without an MCP client:

```bash
# extract: generate + verify
echo "ACME CORP, Invoice #INV-1001, Total: \$108.50" | reflect-mcp extract
reflect-mcp extract --file invoice.txt
reflect-mcp extract --field-threshold 0.85 --overall-threshold 0.80 --file invoice.txt

# validate: verify pre-extracted data (no Generator, Critic + heuristics + gate only)
reflect-mcp validate --file invoice.txt --data '{"vendor_name":"ACME CORP","total":108.5}'
reflect-mcp validate --file invoice.txt --data '{"vendor_name":"ACME CORP"}' --field-threshold 0.5
```

Exit codes (both verbs): **0** = `ok`, **1** = `needs_review`, **2** = usage/input error, **3** = pipeline error (JSON error envelope on stdout).

The bare `reflect-mcp` command (no subcommand) still starts the MCP server.

## 6. Architecture

Three roles, one shared trace id, clean layering (`resolver`-free by design — the pipeline has no infra dependency):

- **Generator** (`generator.py`) — one LLM call → validated `InvoiceData` (Pydantic). Absent fields come back `null`; the model is instructed never to invent a value.
- **Critic** (`critic.py`) — a **different model** (`REFLECT_CRITIC_MODEL`) is shown the source *and* the candidate extraction and must ground each field, returning per-field + overall confidence. A different model matters: self-critique with the same model tends to rubber-stamp its own errors.
- **Deterministic heuristics** (`heuristics.py`) — a free, non-LLM signal: grounding (is a committed value actually present in the source?), arithmetic reconciliation (`subtotal + tax == total`), and ISO-date validation. These *cap* the Critic — a structurally impossible value can't be waved through on the Critic's word alone.
- **Gate** (`gate.py`) — combines the two signals multiplicatively (`combined = critic_confidence × heuristic_factor`), flags any non-null field below `field_threshold`, and escalates the whole result if any field is flagged **or** overall confidence is below `overall_threshold`.

The `extract` tool (`server.py`) is the only module that imports the `mcp` SDK, and `providers.py` is the only module that imports a provider SDK — so the core pipeline and the entire test suite have **zero** hard dependency on either.

## 7. Evaluation

A tiny golden-file eval treats LLM quality as **measurable, not vibes**. `eval/golden/dataset.jsonl` holds labeled messy-text → expected-fields examples; `eval/run_eval.py` runs the full pipeline over them and reports field accuracy, exact-match accuracy, needs-review rate, and *exactly where it fails*.

```bash
make eval          # OFFLINE: deterministic, uses recorded model outputs (no API keys)
make eval-live     # LIVE: runs against REFLECT_PROVIDER (needs credentials)
```

Offline run (recorded fixtures, one example carries an injected hallucination to prove the harness catches misses):

```
============================================================
reflect-mcp eval  |  mode: OFFLINE (replay fixtures)
============================================================
examples          : 5
field accuracy    : 97.1%  (34/35)
exact-match acc   : 80.0%  (4/5)
needs-review rate : 20.0%  (1/5)
------------------------------------------------------------
field-level failures (where it breaks):
  [inv-004] total: expected 583.2 but got 683.2
============================================================
PASS: field accuracy 97.1% >= required 90.0%
```

`make eval` exits non-zero below `--min-accuracy` (default 0.90), so it works as an **offline eval-CI gate** with no secrets.

## 8. Testing

Deterministic tests around a stochastic core. The single LLM boundary (`LLMClient`) is mocked with a `ScriptedLLMClient` that returns canned JSON and raises if it's ever asked for more than expected — so `pytest` runs **fully offline, with no API keys, and can't leak a real call**.

```bash
make install       # .venv + package + dev/server deps
make test          # deterministic offline suite
make cov           # with coverage
```

The flagship test is `tests/test_critic_catches_error.py`: the Generator is fed a deliberately wrong value (`vendor_name = "Invoice"` — a word that *does* appear in the source, so the deterministic grounding check can't catch it), and the test asserts the **second-model Critic** catches it and the gate routes to `needs_review`. It also flips only the Critic's verdict and shows the decision flip to `ok` — proving the verification step is doing real work, not the plumbing.

## 9. Design decisions

- **Second-model Critic (multi-model reflection), not self-critique.** The Critic defaults to a *different, cheaper* model than the Generator (`sonnet` generates, `haiku` critiques). Cross-model disagreement is a real signal; same-model self-review is mostly agreement theater.
- **Provider switching via env var.** `REFLECT_PROVIDER` selects `anthropic` | `openai` | `bedrock`, so a reviewer runs it against whatever key they have. The **Bedrock (Claude)** path is there for AWS users; the public-API paths (Anthropic/OpenAI) let anyone try it without AWS. Provider SDKs are optional extras (`pip install 'reflect-mcp[anthropic]'`) and imported lazily.
- **Confidence is multiplicative.** `critic_confidence × heuristic_factor` — evidence can only *lower* confidence. A structurally impossible value (bad arithmetic, ungrounded amount) is capped regardless of how sure the Critic sounded.
- **Nulls are honest.** A field absent from the source returns `null` and is never flagged — abstention is correct behavior, not a low-confidence guess.
- **Prompt versioning + injection awareness.** Prompts live in versioned files (`src/reflect_mcp/prompts/`). Both instruct the model to treat document content as **data, not commands** — the source text is untrusted input, a basic prompt-injection guard.
- **Observability.** Every stage emits a structured JSON log line carrying a shared `trace_id`, so one extraction is greppable end to end.

## 10. What I'd productionize next

- **Batching + caching** of Generator/Critic calls; per-request cost/latency budgets.
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
│   └── __main__.py        # dispatch: extract/validate → cli, bare → server
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
