---
name: qdk-chemistry-mcp
version: 'v2.0.0'
description: 'Use QDK Chemistry MCP tools for interactive quantum chemistry workflows. Use when: running molecules through the MCP server, calling MCP tools directly, building no-code agent-driven pipelines, visualizing structures/circuits/orbitals, or working with the quantum-agent multi-agent system. Covers the MCP tools, return format, visualization, remote execution, backend management, and agent orchestration patterns.'
---

# QDK Chemistry MCP Tools

## When to Use

- Running quantum chemistry workflows interactively via MCP tools
- Using the quantum-agent multi-agent system (researcher → reviewer → chemist → reporter)
- Calling individual MCP tools for step-by-step exploration
- Visualizing molecules, orbitals, circuits, and entanglement diagrams
- No-code pipeline execution — no Python scripting needed

## Prerequisites

Install and enable the QDK Chemistry agent plugin, or configure the
`qdk_chemistry` MCP server manually. Call `bind_workspace` before every other
QDK Chemistry tool. Prefer client workspace discovery; if unavailable, pass the
active workspace as an absolute `workspace_root`.

## Tool Catalog And Guidance

The MCP catalog intentionally exposes only a short summary for each tool so the
model context remains usable. Input schemas define call syntax. This skill and
its references document tool dependencies, artifact handoff, and interface
constraints; scientific method and parameter choices remain caller supplied.

Load this skill before planning or executing a multi-step MCP workflow. Load
the relevant reference when the task involves active-space selection, QPE,
state preparation, resource reporting, or failure diagnostics. Use
`list_algorithms`, `describe_algorithm`, and the default-setting tools for the
runtime implementations and settings available in the active installation.
The full Python docstrings remain developer reference material and can be
exposed for diagnostics with `--no-compact-tool-descriptions`; normal agents
must not depend on that expanded startup catalog.

## Tool Discovery

MCP tools are deferred — discover them before first use:

```
tool_search_tool_regex(pattern="mcp_qdk_chemistry")
```

After discovery, tool names follow the pattern `mcp_qdk_chemistry_<action>`.

## Return Format

All tools return a JSON envelope:

| Status | Shape |
|--------|-------|
| `"ok"` | `{"status": "ok", "result": <value>}` |
| `"error"` | `{"status": "error", "message": "...", "error_type": "..."}` |
| `"exists"` | `{"status": "exists", "message": "..."}` — output file already exists; pass `overwrite=True` to replace it |

Always check `status` before using `result`. All `run_*` tools accept `overwrite: bool = False` — set it to `True` to skip the existing-file check.

## File Naming Convention

All files use typed markers: `{name}.{data_type}.{extension}`

Examples: `h2.structure.json`, `h2.wavefunction.json`, `h2.circuit.json`

## Tool Routing

Use `list_tools(category=...)` to discover the active catalog and use each input
schema for exact call syntax. Route by task rather than loading a duplicate
parameter reference:

| Need | Category or entry tools |
|------|-------------------------|
| Projects and files | `project`, `data_inspection` |
| Runtime algorithms and settings | `list_algorithms`, `describe_algorithm`, default-setting tools |
| Units and molecular input | `utility`, `input_construction` |
| SCF, derivatives, optimization, correlation, active spaces | `classical_calculation` |
| Fermion mapping, state preparation, energy and resource evidence | `quantum_preparation` |
| Stepwise resource circuits or eigenvalue execution | `qpe` |
| Cached or asynchronous execution | `remote_execution` |
| Interactive VS Code output | `visualization` |

Descriptions carry the blocking call invariants. Use input schemas and runtime
discovery to determine the supported call shape.

### Algorithm Configuration Policy

- Call `list_algorithms` before choosing an implementation, then call
  `describe_algorithm` to discover its accepted settings and method values.
- Start from the selected algorithm's defaults and override only settings needed
  for the requested call.
- Apply each override to the deepest nested algorithm that owns that behavior.
  Do not replace an algorithm implementation merely to reach one of its child
  settings, and preserve unspecified nested defaults.
- A direct ``run_*`` tool and a nested algorithm setting are independent
  access paths. The same algorithm type can be used both ways; use the tool
  schema for direct execution and ``describe_algorithm`` for nested settings.
- For geometry optimization, `algorithm_name` selects the optimization driver
  (normally `geometric`). Select the energy and gradient implementation through
  `settings["derivative_calculator"]` while retaining the requested optimization
  driver.
- Treat nested algorithm settings recursively and preserve unspecified nested
  defaults.

### Model Hamiltonians (no molecular structure needed)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_model_hamiltonian` | Fermionic lattice Hamiltonian (Hückel, Hubbard, PPP) | `project_name`, `model`, `out_hamiltonian_filename`, `lattice_type`, `lattice_params`, `epsilon?`, `t?`, `U?`, `V?` |
| `create_spin_model_hamiltonian` | Qubit spin Hamiltonian (Heisenberg, Ising) | `project_name`, `model`, `out_qubit_hamiltonian_filename`, `lattice_type`, `lattice_params`, `jx?`, `jy?`, `jz?`, `j?`, `h?` |

These bypass the molecular workflow. Fermionic models produce a `Hamiltonian`
(then `create_majorana_mapping` and `run_qubit_mapper`). Spin models produce a
`QubitHamiltonian` directly. The input schema identifies the model parameters.

### Quantum Preparation

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_majorana_mapping` | Create fermion-to-qubit mapping file | `project_name`, `out_mapping_filename`, `encoding?`, `num_modes?`, `hamiltonian_filename?` |
| `run_qubit_mapper` | Apply mapping file to fermionic Hamiltonian | `project_name`, `hamiltonian_filename`, `mapping_filename`, `out_qubit_hamiltonian_filename` |
| `run_state_preparation` | Build state-prep circuit | `project_name`, `wavefunction_filename`, `out_circuit_filename` |
| `run_amplitude_amplification` | Build an amplified circuit from state-preparation and good-state oracle circuits | `project_name`, `state_prep_oracle_filename`, `good_state_oracle_filename`, `out_circuit_filename` |
| `run_term_grouper` | Group the Pauli terms of a qubit operator | `project_name`, `qubit_hamiltonian_filename`, `out_qubit_hamiltonian_filename` |
| `run_qubit_hamiltonian_solver` | Exact diagonalization | `project_name`, `qubit_hamiltonian_filename` |
| `run_energy_estimator` | Shot-based energy | `project_name`, `circuit_filename`, `qubit_hamiltonian_filenames`, `total_shots` |
| `run_resource_estimation` | QDK QRE physical-qubit/runtime Pareto frontier | `project_name`, `circuit_filename`, `architecture?`, `physical_error_rate?`, `max_error?`, `gate_time_ns?`, `measurement_time_ns?`, `use_graph?` |

`run_resource_estimation` describes only the exact `circuit_filename` supplied
to it. A state-preparation-circuit estimate is not a QPE estimate. Hamiltonian
mapping statistics and `get_circuit_stats` outputs are also distinct from
physical-resource estimates.

The tool directly invokes QDK QRE and returns estimation assumptions plus a
structured `pareto_front` inline. Each point contains `physical_qubits`,
`runtime_ns`, and `error`; no resource-estimator data file is written. Use
`get_circuit_stats` separately for logical circuit metrics. Code distance,
factory breakdowns, and logical error-budget details are not exposed by this
MCP response and must be reported as unavailable.

### QPE — Mode A: Circuit Resource Analysis

Build and inspect QPE circuit components for resource estimates. No energy computed.

| Tool | Purpose |
|------|---------|
| `run_time_evolution_builder` | Build U = exp(-iHt) |
| `run_controlled_evolution_circuit_mapper` | Map to controlled-U circuit |
| `run_circuit_executor` | Execute circuit with shots |
| `run_evolution_circuit_builder` | Build a circuit for a driven Hamiltonian |
| `run_hamiltonian_simulation` | Evolve a driven Hamiltonian and measure observables |
| `run_hadamard_test` | Measure a unitary with a state-preparation circuit |

### QPE — Mode B: Full Eigenvalue

| Tool | Purpose |
|------|---------|
| `run_phase_estimation` | Complete QPE for energy eigenvalue. Sub-algorithms (evolution builder, circuit mapper, circuit executor) are configured inline via `settings` dicts. |

**Mode A ≠ Mode B.** Never switch without explicit user approval.

### QPE Settings

Before selecting QPE settings or building resource circuits, load
[QPE and State Preparation](./references/qpe-and-state-prep.md) for the nested
settings shape and required explicit values. Load
[Quantum Resource Inputs](./references/quantum-resource-compression.md) for
artifact relationships and reporting boundaries.

`run_phase_estimation` rejects its invalid sentinel defaults for phase bits and
evolution time. In Mode A, configure powered evolution in the time-evolution
builder before mapping the resulting circuit.

### Remote / Async Job Execution

Any `run_*` tool can execute remotely by passing `remote` and `remote_config`. Jobs are persisted to disk and can be checked/resumed later. Use `remote_timeout=0` for fire-and-forget submission. Discover available backends with `list_remote_backends` before configuring remote execution.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `check_remote_job` | Poll job status and logs | `project_name`, `job_id` |
| `retrieve_remote_results` | Download results into project directory | `project_name`, `job_id` |
| `list_remote_jobs` | List jobs (optionally filtered by status) | `project_name?`, `status?` |
| `cancel_remote_job` | Cancel a running job | `project_name`, `job_id` |

See the `remote-execution` skill for full async workflow details.

### Visualization (VS Code only)

These interactive widget tools require `qsharp_widgets` and only work in VS Code with MCP Apps.

| Tool | Output | Key Parameters |
|------|--------|----------------|
| `visualize_molecule` | Interactive 3D viewer | `project_name`, `structure_filename` |
| `visualize_orbitals` | Interactive orbital viewer | `project_name`, `wavefunction_filename`, `orbital_indices?` |
| `visualize_orbital_entanglement` | Chord diagram | `project_name`, `wavefunction_filename`, `selected_indices?` (**absolute** orbital indices — auto-converted) |
| `visualize_circuit` | Interactive circuit diagram | `project_name`, `circuit_filename` |
| `visualize_scatter_plot` | Interactive SVG scatter plot with optional log axes | `title`, `x_label`, `y_label`, `series` (list of data series), `log_x?`, `log_y?` |

## Workflow Routing

Load only the reference required by the claimed endpoint:

- [Workflow Patterns](../qdk-chemistry-overview/references/workflow-patterns.md)
  for molecular versus model-Hamiltonian entry points and classical, resource,
  or eigenvalue endpoints.
- [Active Space Tool Reference](./references/active-space-guide.md) for input
  dependencies and orbital-index handoff.
- [QPE and State Preparation](./references/qpe-and-state-prep.md) for state
  preparation and QPE settings shape.
- [Quantum Resource Inputs](./references/quantum-resource-compression.md) for
  circuit artifacts and resource-reporting boundaries.
- [MCP Diagnostics Reference](./references/things-that-go-wrong.md) after a
  failed call.
- The separate `remote-execution` skill for asynchronous backend operation.

Use one `project_name` throughout a workflow and pass actual returned filenames
between tools. The work contract chooses the endpoint; never substitute a
different endpoint because it is easier to execute.

## Multi-Agent Architecture

The MCP setup supports a multi-agent workflow in VS Code:

| Agent | Role |
|-------|------|
| `quantum-agent` | Top-level orchestrator, visualization, user interaction |
| `researcher` | Focused Q&A, consults Playbook Copilot Space |
| `reviewer` | Critiques plans before execution |
| `chemist` | Validates and executes MCP tool workflows |
| `reporter` | Generates Markdown report + Python script |

Workflow: **Research → Plan → Critique → Present → Validate → Execute → Report**

## Reference Documents

These files contain mechanical guidance for MCP-driven workflows. Load them as needed:

- [Active Space Tool Reference](./references/active-space-guide.md) — selector inputs and output handoff
- [QPE and State Preparation](./references/qpe-and-state-prep.md) — required nested settings and endpoint calls
- [Quantum Resource Inputs](./references/quantum-resource-compression.md) — circuit artifacts and estimate types
- [MCP Diagnostics Reference](./references/things-that-go-wrong.md) — interface constraints and error routing

## Critical Rules

1. **Coordinates in Bohr** for `create_structure`
2. **Check `status`** before using a result
3. **Query defaults and schemas** before sending algorithm overrides
4. **Pass returned filenames** to dependent calls
5. **Use restricted HF inputs** for active-space paths that require shared
   spatial orbitals
6. **Supply RDM outputs** before entropy-based AutoCAS selection
7. **Keep artifact evidence distinct**: mapping data, logical circuit metrics,
   and physical resource estimates come from different tools
