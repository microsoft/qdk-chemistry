---
name: qdk-chemistry-overview
version: '{{QDK_CHEMISTRY_VERSION}}'
description: 'Top-level guide for QDK Chemistry usage modes. Use when: deciding how to approach a quantum chemistry task, choosing between Python SDK coding and MCP tools, planning a workflow, or orienting to the QDK Chemistry ecosystem. Covers decision trees for interface selection, workflow patterns, and when to invoke each specialized skill.'
---

# QDK Chemistry Overview

## When to Use This Skill

- You need to decide **how** to approach a quantum chemistry task (code vs MCP)
- You want to understand the QDK Chemistry ecosystem and available interfaces
- You are planning a multi-step workflow and need to pick the right mode
- You are unsure which specialized skill to load next

## Three Interfaces — One Pipeline

QDK Chemistry provides two ways to run the same quantum chemistry pipeline. Each has distinct strengths:

| Interface | Best For | Skill |
|-----------|----------|-------|
| **Python SDK** | Custom scripts, notebooks, loops, parameter sweeps, programmatic control, integration with other Python libraries | `qdk-chemistry-coding` |
| **MCP Tools** | Interactive agent-driven workflows in VS Code/Claude, no-code execution, visualization, step-by-step exploration | `qdk-chemistry-mcp` |

## Decision Tree

```
User wants to do quantum chemistry
│
├─ "Run this molecule / analyze this system" ───► Load qdk-chemistry-mcp skill
│   Interactive, step-by-step, visual exploration
│   (Requires MCP server configured)
│
├─ "Hubbard / Ising / lattice model" ──────────► Load qdk-chemistry-mcp skill
│   Use create_model_hamiltonian or create_spin_model_hamiltonian
│   Agent determines model parameters from the physics
│
├─ "Write me a script / notebook" ──────────────► Load qdk-chemistry-coding skill
│   Python code, parameter sweeps, automation
│
└─ "I need both code and interactivity" ────────► Load qdk-chemistry-coding + qdk-chemistry-mcp
    Write script, then run steps interactively
```

## The Chemistry Pipeline

Regardless of interface, all workflows follow the same conceptual pipeline:

**Stage 1 — Classical Preparation & Compression**

*Molecular systems:*
1. Define molecular structure (geometry + elements, coordinates in Bohr)
2. Run SCF (Hartree-Fock or DFT)
3. Check wavefunction stability (never skip)
4. Select active space (valence, AutoCAS, or combined)
5. Extract orbitals → build Hamiltonian
6. Optional: dynamical correlation (MP2, CCSD, CCSD(T))

*Model Hamiltonians (alternative entry — no molecular structure needed):*
1. `create_model_hamiltonian` (Hückel, Hubbard, PPP) → fermionic Hamiltonian
   OR `create_spin_model_hamiltonian` (Heisenberg, Ising) → qubit Hamiltonian directly
2. Skip to Stage 2 (fermionic models need qubit mapping; spin models skip it too)

The agent must determine model parameters from the user's description of the physical system.

**Stage 2 — Quantum Mapping & State Preparation**
7. Create a fermion-to-qubit mapping file — skip for spin models
8. Map fermions to qubits with the mapping file — skip for spin models
9. Prepare quantum state from classical wavefunction (sparse isometry)

**Stage 3 — Quantum Circuit & Execution** (three distinct endpoints)
- **Classical energy only** → stop after Stage 1
- **Circuit resource analysis** → build QPE circuit components, extract gate counts / qubit counts / T-count, run `run_resource_estimation` for full logical + physical resource profile
- **QPE eigenvalue** → run full quantum phase estimation for energy

## Key Principles

- **Coordinates in Bohr** — always convert from Ångströms before uploading structures
- **Always check SCF stability** — unstable solutions produce wrong downstream results
- **AutoCAS requires RDMs** — run SCI with `calculate_one_rdm=True`, `calculate_two_rdm=True`, and `calculate_mutual_information=True` first
- **Compress first, compute second** — minimize active space before building quantum circuits
- **Logical + physical resources** — `get_circuit_stats` provides circuit-level logical metrics; `run_resource_estimation` adds physical qubit counts, runtime, and error budgets
- **Mode A ≠ Mode B** — circuit resource analysis and QPE eigenvalue answer different questions; never switch without approval

## Combining Skills

Skills can be combined for complex tasks:

| Scenario | Skills to Load |
|----------|---------------|
| Interactively explore a molecule, then generate a reproducible script | `qdk-chemistry-mcp` + `qdk-chemistry-coding` |
| Full QPE analysis with report generation | `qdk-chemistry-mcp` (agents handle this end-to-end) |

## Project Structure

All data is stored in `/scratch/projects/{project_name}/` with typed filenames:
- `*.structure.json` — molecular geometry
- `*.wavefunction.json` — electronic wavefunction
- `*.hamiltonian.json` — fermionic Hamiltonian
- `*.qubit_hamiltonian.json` — qubit-mapped Hamiltonian
- `*.circuit.json` — quantum circuit
- `*.orbitals.json` — orbital data
- `*.ansatz.json` — ansatz (wavefunction + Hamiltonian pair)
- `*.config.json` — algorithm configuration (individual QPE step tools)
- `*.resource_estimator_data.json` — resource estimation results (logical + physical counts)

## Reference Documents

These files contain detailed workflow recipes, decision trees, and foundational knowledge. Load them when you need deeper guidance:

- [Playbook Index](./references/playbook-index.md) — directory of all playbook content and how files relate
- [How QDK Chemistry Works](./references/how-qdk-chemistry-works.md) — the `create()` factory, MCP tool patterns, file naming, auto-behaviors
- [Workflow Patterns](./references/workflow-patterns.md) — the three workflow endpoints (resource estimation, energy, classical-only) with step sequences and decision logic

## External References

- Repository: [microsoft/qdk-chemistry](https://github.com/microsoft/qdk-chemistry)
- Paper: [arXiv:2601.15253](https://arxiv.org/abs/2601.15253)
