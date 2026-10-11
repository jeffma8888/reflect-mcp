# reflect-mcp — self-verifying LLM extraction over MCP

An MCP server for LLM tasks that check their own work before surfacing it: a Generator extracts a typed object from messy text, a second-model Critic plus deterministic heuristics score every field, and a confidence-thresholded gate either returns the result or escalates a `needs_review` payload naming the exact fields a human must confirm. Never a silent wrong answer.

For engineers wiring LLM extraction into real systems who need verification as a first-class step, not a demo-time afterthought. The verification discipline is the product; the invoice schema is only the worked example.

Quality bar and hard constraints:
- Every feature stays offline-testable: tests and `make eval` run with no network and no secrets; model calls sit behind a single provider seam that tests stub.
- The Critic must remain a decorrelated grader (different model from the Generator by default); heuristics can only lower a score, never raise it.
- Measurable by construction: any new verification signal or gate behavior ships with golden-set coverage and a field-level accuracy readout; `make eval` must keep its non-zero exit below threshold.
- Python >=3.10, MCP-spec compliant tool surface, small dependency footprint, MIT. Generic identifiers only, no machine paths, publication-safe at all times.
