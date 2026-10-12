# Foundry directions

foundry directions -- reflect-mcp
  iter-01
    lenses: unknown
    - Candidate A1 -- Generator/Critic retry with back-off on parse failure
    - Candidate A2 -- Structured error result instead of raw exception on pipeline failure
    - Candidate A3 -- `make typecheck` target with inline type annotations verified by mypy/pyright
    - Candidate B1 -- CLI extract verb: pipe-friendly one-shot extraction
    - Candidate B2 -- Machine-readable exit codes and status field for scripting
    - Candidate B3 -- Eval as a console-script verb: `reflect-mcp eval`
    winner: B1
    ship: pending (not yet decided)
1 scouted iterations
