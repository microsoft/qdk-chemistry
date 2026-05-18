---
name: domain-reference
description: "Use when: you need domain context about quantum chemistry conventions, data structure semantics, plugin conversion details, algorithm registry mechanics, or design rationale. Provides on-demand reference material — load specific reference files relevant to your task."
---

# Domain Reference Skill

This is a **reference skill** — it provides on-demand domain knowledge about the qdk-chemistry quantum chemistry package. Load the specific reference file(s) relevant to your current task rather than reading everything at once.

## Available Reference Files

| File | When to Use |
|------|-------------|
| `references/data-flow.md` | Understanding the core pipeline (molecule → qubits → energy), what each stage produces, which algorithm types connect stages |
| `references/algorithm-registry.md` | Working with `create()`, registering new algorithms, understanding algorithm types/variants/settings, the factory pattern |
| `references/data-structures.md` | Understanding C++ core types (Structure, Orbitals, Hamiltonian, Wavefunction, Settings), Python data classes, container variants |
| `references/plugin-conversions.md` | Working with PySCF/Qiskit/OpenFermion plugins, data conversion details, spin-orbital ordering, Pauli encoding differences |
| `references/notation-conventions.md` | Unit conventions (Bohr/Hartree), integral indexing, spin-orbital ordering, Pauli string encoding, 2-RDM conventions |
| `references/design-rationale.md` | Understanding why things are designed the way they are — Settings system, pybind11 patterns, build system, documentation pipeline, key architectural decisions |

**Load only what you need. Each reference file is self-contained.**
