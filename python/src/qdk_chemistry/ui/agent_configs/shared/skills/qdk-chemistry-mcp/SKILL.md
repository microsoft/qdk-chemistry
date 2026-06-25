---
name: qdk-chemistry-mcp
version: '{{QDK_CHEMISTRY_VERSION}}'
description: 'Use QDK Chemistry MCP tools for interactive quantum chemistry workflows. Use when: running molecules through the MCP server, calling MCP tools directly, building no-code agent-driven pipelines, visualizing structures/circuits/orbitals, or working with the quantum-agent multi-agent system. Covers all 42 MCP tools, return format, visualization, remote execution, backend management, and agent orchestration patterns.'
---

# QDK Chemistry MCP Tools

## When to Use

- Running quantum chemistry workflows interactively via MCP tools
- Using the quantum-agent multi-agent system (researcher → reviewer → chemist → reporter)
- Calling individual MCP tools for step-by-step exploration
- Visualizing molecules, orbitals, circuits, and entanglement diagrams
- No-code pipeline execution — no Python scripting needed

## Prerequisites

The MCP server must be configured in `.vscode/mcp.json`:

```json
{
  "servers": {
    "qdk_chemistry": {
      "command": "bash",
      "args": ["-c", "source /home/vscode/qdk_chemistry_venv/bin/activate && qdk_chem_mcp"]
    }
  }
}
```

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

## MCP Tool Reference

### Project Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_projects` | List all projects | — |
| `create_project` | Create a new project directory | `project_name` |
| `list_project_files` | List files in a project with inferred types | `project_name` |
| `list_tools` | List available tools by category | `category?` |

### Backend & Cache Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_cache_backends` | List available local caching backends | — |
| `list_remote_backends` | List available remote execution backends | — |
| `describe_backend` | Get details about a specific backend (local or remote) | `backend_name` |

Use these tools to discover available resources before submitting remote jobs or configuring caching strategies.

### Structure & Helpers

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_structure` | Create molecular geometry | `project_name`, `coordinates_json` (Bohr!), `symbols`, `overwrite?` |
| `get_summary` | Inspect any data file (auto-detects type) | `project_name`, `filename` |
| `get_algorithm_default_type` | Get default algorithm name | `algorithm_type` |
| `get_algorithm_default_settings` | Get default settings | `algorithm_type`, `algorithm_name?` |
| `get_orbitals_from_input` | Extract orbitals | `project_name`, `input_filename`, `out_orbitals_filename` |
| `get_active_space_indices` | Get orbital index partitioning | `project_name`, `input_filename` |
| `get_ansatz` | Build ansatz from wfn + Hamiltonian | `project_name`, `wavefunction_filename`, `hamiltonian_filename`, `out_ansatz_filename` |
| `get_top_configurations` | Highest-weight CI determinants | `project_name`, `wavefunction_filename`, `max_determinants?` |
| `get_circuit_stats` | Circuit resource metrics | `project_name`, `circuit_filename` |
| `convert_coordinates` | Convert Bohr ↔ Angstrom | `coordinates_json`, `to_unit` |
| `convert_energy` | Convert energy units | `value`, `from_unit`, `to_unit` |

### Classical Calculations

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_scf` | Hartree-Fock / DFT | `project_name`, `structure_filename`, `out_wavefunction_filename`, `charge`, `spin_multiplicity`, `basis_set` |
| `run_stability_checker` | Verify SCF stability | `project_name`, `wavefunction_filename`, `out_stability_result_filename` |
| `run_active_space_selector` | Select active orbitals | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `charge?`, `algorithm_name?`, `settings?` |
| `run_orbital_localization` | Localize orbitals | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `loc_indices_alpha` |
| `run_multi_configuration_calculation` | CASCI / SCI | `project_name`, `hamiltonian_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons`, `n_active_beta_electrons?`, `settings?` |
| `run_multi_configuration_scf` | CASSCF | `project_name`, `orbitals_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons` |
| `run_projected_multi_configuration_calculation` | Sparse CI | `project_name`, `hamiltonian_filename`, `configurations_json`, `out_wavefunction_filename` |
| `run_dynamical_correlation_calculator` | MP2 / CCSD / CCSD(T) | `project_name`, `ansatz_filename`, `out_wavefunction_filename` |
| `run_hamiltonian_constructor` | Build fermionic Hamiltonian | `project_name`, `orbitals_filename`, `out_hamiltonian_filename` |

### Model Hamiltonians (no molecular structure needed)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_model_hamiltonian` | Fermionic lattice Hamiltonian (Hückel, Hubbard, PPP) | `project_name`, `model`, `out_hamiltonian_filename`, `lattice_type`, `lattice_params`, `epsilon?`, `t?`, `U?`, `V?` |
| `create_spin_model_hamiltonian` | Qubit spin Hamiltonian (Heisenberg, Ising) | `project_name`, `model`, `out_qubit_hamiltonian_filename`, `lattice_type`, `lattice_params`, `jx?`, `jy?`, `jz?`, `j?`, `h?` |

These bypass the molecular workflow. Fermionic models produce a `Hamiltonian` (needs `run_qubit_mapper`). Spin models produce a `QubitHamiltonian` directly. The agent must determine model parameters from the user's description.

### Quantum Preparation

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_qubit_mapper` | Jordan-Wigner encoding | `project_name`, `hamiltonian_filename`, `out_qubit_hamiltonian_filename` |
| `run_state_preparation` | Build state-prep circuit | `project_name`, `wavefunction_filename`, `out_circuit_filename` |
| `run_qubit_hamiltonian_solver` | Exact diagonalization | `project_name`, `qubit_hamiltonian_filename` |
| `run_energy_estimator` | Shot-based energy | `project_name`, `circuit_filename`, `qubit_hamiltonian_filenames`, `total_shots` |
| `run_resource_estimation` | Quantum resource estimation (logical + physical) | `project_name`, `circuit_filename`, `out_resource_estimator_data_filename` |
| `run_resource_estimation` (with `algorithm_name='qdk_qre_v3'`) | Pareto-frontier resource estimation (qubits vs runtime tradeoffs) | Same params; returns JSON array of results |

### QPE — Mode A: Circuit Resource Analysis

Build and inspect QPE circuit components for resource estimates. No energy computed.

| Tool | Purpose |
|------|---------|
| `run_time_evolution_builder` | Build U = exp(-iHt) |
| `run_controlled_evolution_circuit_mapper` | Map to controlled-U circuit |
| `run_circuit_executor` | Execute circuit with shots |

### QPE — Mode B: Full Eigenvalue

| Tool | Purpose |
|------|---------|
| `run_phase_estimation` | Complete QPE for energy eigenvalue. Sub-algorithms (evolution builder, circuit mapper, circuit executor) are configured inline via `settings` dicts. |

**Mode A ≠ Mode B.** Never switch without explicit user approval.

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
| `visualize_scatter_plot` | Interactive Plotly scatter plot with optional log axes | `title`, `x_label`, `y_label`, `series` (list of data series), `log_x?`, `log_y?` |

## Standard MCP Workflow

Use a consistent `project_name` across all steps:

```
Molecular workflow:
1. create_structure        → "h2.structure.json"
2. run_scf                 → "h2.wavefunction.json"
3. run_stability_checker   → "h2.stability.json"
4. run_active_space_selector → "h2_as.wavefunction.json"
5. get_orbitals_from_input → "h2_as.orbitals.json"
6. run_hamiltonian_constructor → "h2.hamiltonian.json"
7. run_multi_configuration_calculation → "h2_mc.wavefunction.json"
8. run_qubit_mapper        → "h2.qubithamiltonian.json"
9. run_state_preparation   → "h2.circuit.json"
10. (Mode A) run_time_evolution_builder → run_controlled_evolution_circuit_mapper → run_resource_estimation
    (Mode B) run_phase_estimation (sub-algorithms configured inline via settings)

Model Hamiltonian shortcut (no molecular structure):
1. create_model_hamiltonian  → "hubbard.hamiltonian.json"     (fermionic: Hückel, Hubbard, PPP)
   OR create_spin_model_hamiltonian → "ising.qubit_hamiltonian.json"  (spin: Heisenberg, Ising — already a qubit Hamiltonian)
2. run_qubit_mapper          → "hubbard.qubithamiltonian.json" (fermionic models only)
3. Continue with quantum steps (state prep, QPE, energy estimation, resource estimation)
```

The agent must determine appropriate model parameters from the user's description of the physical system.

## Active Space Selection Strategies

| Strategy | When | Tool Call |
|----------|------|-----------|
| Valence | Simple molecules, valence chemistry | `run_active_space_selector` with `algorithm_name="qdk_valence"` and `charge` |
| AutoCAS | Strongly correlated, rigorous | Run SCI first (with RDMs), then `run_active_space_selector` with `algorithm_name="qdk_autocas_eos"` |
| Combined | Large molecules | Valence first to reduce, then SCI + AutoCAS on reduced space |

**Critical:** AutoCAS requires prior SCI with `calculate_one_rdm=True`, `calculate_two_rdm=True`, and `calculate_mutual_information=True` in settings. It cannot run on bare SCF output.

## Entanglement Diagram Index Convention

The `selected_indices` parameter in `visualize_orbital_entanglement` accepts **absolute orbital indices** — the same indices shown as arc labels in the diagram (e.g., [8, 9, 10]). The tool converts these to diagram-relative positions automatically. No manual index arithmetic is needed.

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

These files contain domain-specific guidance for MCP-driven workflows. Load them when you need deeper knowledge:

- [Active Space Guide](./references/active-space-guide.md) — AutoCAS, valence selection, SCI pre-filtering, orbital localization, visualization
- [QPE and State Preparation](./references/qpe-and-state-prep.md) — sparse isometry, wavefunction truncation, evolution time, multi-trial strategies
- [Quantum Resource Compression](./references/quantum-resource-compression.md) — strategies for minimizing qubits, depth, and gate count
- [Things That Go Wrong](./references/things-that-go-wrong.md) — real failure modes with symptoms, causes, and fixes

## Critical Rules

1. **Coordinates in Bohr** for `create_structure`
2. **Always run `run_stability_checker`** after SCF
3. **Check `status` field** in every response
4. **Query defaults** with `get_algorithm_default_settings` before overriding
5. **Pass actual output filenames** between steps — don't assume
6. **Stop on failure** — report errors, don't improvise fixes
7. **Autocas needs RDMs** — cannot run directly on SCF wavefunction
8. **Use `run_resource_estimation` for full resource profiles** — provides both logical and physical counts, runtime, code distance, and error budget
