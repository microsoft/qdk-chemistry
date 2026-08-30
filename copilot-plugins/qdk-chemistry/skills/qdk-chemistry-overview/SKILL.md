---
name: qdk-chemistry-overview
version: 'v2.0.0'
description: 'Describes the QDK Chemistry interfaces and the artifact types they read and write.'
---

# QDK Chemistry Overview

QDK Chemistry exposes algorithms and data types through a Python SDK and an
MCP server. Remote backends can execute supported algorithm calls.

| Interface | Function |
|---|---|
| Python SDK | Creates, configures, and executes algorithms from Python |
| MCP server | Exposes project, algorithm, calculation, inspection, and visualization tools |
| Remote execution | Serializes an algorithm call for execution by a configured backend |

Tools exchange typed artifacts, including structures, wavefunctions, orbitals,
Hamiltonians, qubit Hamiltonians, circuits, and algorithm results. Tool input
schemas identify the artifact required by each argument.

Runtime tool schemas and algorithm-discovery calls define the available
arguments, implementations, settings, and defaults.

## Algorithm Model

An algorithm has a type, an implementation name, and a settings object. The
registry resolves an omitted implementation name to the installed default.
Settings are validated by the selected implementation. Nested algorithms are
represented by algorithm references containing their type, implementation,
and settings.

## Artifact Flow

- A `Structure` stores elements, coordinates, and optional atomic metadata.
- A `Wavefunction` stores orbitals and electronic-state data produced by an
	electronic-structure calculation.
- `Orbitals` can be extracted from supported electronic-structure artifacts.
- A `Hamiltonian` stores fermionic one- and two-body terms.
- A `MajoranaMapping` stores a fermion-to-qubit mapping.
- A `QubitHamiltonian` stores weighted Pauli terms.
- A `Circuit` stores quantum operations.
- Result objects store outputs specific to an algorithm, such as phase or
	stability data.

File-producing MCP calls persist one of these artifacts and return its
filename. Inspection tools read persisted artifacts without running a new
calculation.
