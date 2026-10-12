# Foundry directions

foundry directions -- reflect-mcp
  iter-02
    lenses: unknown
    - Candidate A1 — CLI threshold flags (`--field-threshold`, `--overall-threshold`)
    - Candidate A2 — Eval as a CLI verb (`reflect-mcp eval`)
    - Candidate A3 — Structured error envelope on pipeline failure
    - Candidate B1 — Remove `heuristics_enabled` toggle and dead disable path
    - Candidate B2 — Eliminate eval's `sys.path` hack by making eval a proper subpackage
    - Candidate B3 — Collapse near-duplicate test setup in `test_iter01_behavior.py`
    winner: A3
    ship: pending (not yet decided)
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
2 scouted iterations
