# QDK Chemistry Playbook — Index

Workflow recipes, decision guides, worked examples, and Python reference for running quantum chemistry simulations with QDK Chemistry MCP tools and the `qdk_chemistry` Python package.

## Playbook Index

### Overview (`qdk-chemistry-overview/references/`)
- `how-qdk-chemistry-works.md` — How the Python package and MCP tools work — the `create()` factory, file naming, tool return formats, auto-behaviors. The foundation everything else builds on.
- `workflow-patterns.md` — The three quantum chemistry workflow patterns (resource estimation, energy computation, classical-only) with step sequences and decision logic.
- `playbook-index.md` — This index of workflow recipes, decision guides, worked examples, and Python reference material.

### MCP Workflow Guides (`qdk-chemistry-mcp/references/`)
- `active-space-guide.md` — How to choose an active space strategy — valence vs SCI-driven vs combined, with decision reasoning and orbital localization.
- `qpe-and-state-prep.md` — QPE configuration, evolution time computation, state preparation strategies, multi-trial voting, and the sparse isometry advantage.
- `quantum-resource-compression.md` — Strategies for minimizing qubit count, circuit depth, and gate count — active space minimization, wavefunction sparsification, Hamiltonian trimming, Trotter tuning, QPE precision trade-offs.
- `things-that-go-wrong.md` — Coordinate units, convergence failures, invalid QPE defaults, resource-vs-energy confusion, and other real failure modes with fixes.

### Coding Examples (`qdk-chemistry-coding/references/`)
- `example-n2-stretched.md` — Complete worked example: stretched N₂ with orbital localization, AutoCAS refinement, iQPE multi-trial voting. Shows decision reasoning.
- `example-benzene-state-prep.md` — Complete worked example: benzene diradical state preparation, sparse isometry, Hamiltonian filtering, energy measurement.
- `python-sdk-reference.md` — All valid `create()` calls, utility functions, file I/O patterns — everything needed to write a reproducible Python script using `qdk_chemistry`.
