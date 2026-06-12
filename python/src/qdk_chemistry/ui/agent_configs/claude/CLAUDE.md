<!-- qdk-chemistry-agent-config {{QDK_CHEMISTRY_VERSION}} -->
# QDK Chemistry — Claude Code Instructions

You are working with **QDK/Chemistry**, an end-to-end quantum chemistry package by Microsoft. It has a C++/Python core and a CLI + MCP server UI layer.

## MCP Server (Preferred)

Claude Code supports MCP natively. Configure the QDK Chemistry MCP server in your `.mcp.json` or project settings:

```json
{
  "mcpServers": {
    "qdk_chemistry": {
      "command": "qdk_chem_mcp"
    }
  }
}
```

> **Note:** If `qdk_chem_mcp` is installed in a virtual environment, activate the environment first or wrap the command: `"command": "bash", "args": ["-c", "source <venv>/bin/activate && qdk_chem_mcp"]`.

Once configured, the MCP server exposes tools for no-code quantum chemistry workflows.

### MCP Tool Reference

> **Canonical reference:** The authoritative tool documentation is in `shared/skills/qdk-chemistry-mcp/SKILL.md`. The tables below are a summary — consult the skill file for full details and worked examples.

#### Project Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_projects` | List all projects | — |
| `create_project` | Create a new project directory | `project_name` |
| `list_project_files` | List files in a project with inferred types | `project_name` |
| `list_tools` | List available tools by category | `category?` |

#### Structure & Helpers

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_structure` | Create molecular geometry (coordinates **must be in Bohr**) | `project_name`, `coordinates_json`, `symbols`, `filename_to_save?`, `overwrite?` |
| `get_summary` | Inspect any data file (auto-detects type) | `project_name`, `filename` |
| `get_algorithm_default_type` | Get default algorithm name for a given type | `algorithm_type` |
| `get_algorithm_default_settings` | Get default settings for an algorithm | `algorithm_type`, `algorithm_name?` |
| `get_orbitals_from_input` | Extract orbitals from a wavefunction/hamiltonian | `project_name`, `input_filename`, `out_orbitals_filename` |
| `get_active_space_indices` | Get active/inactive/virtual orbital index partitioning | `project_name`, `input_filename` |
| `get_ansatz` | Build an Ansatz from wavefunction + Hamiltonian | `project_name`, `wavefunction_filename`, `hamiltonian_filename`, `out_ansatz_filename` |
| `get_top_configurations` | Get highest-weight CI determinants | `project_name`, `wavefunction_filename`, `max_determinants?` |
| `get_circuit_stats` | Circuit resource metrics (qubits, depth, gates, T-count) | `project_name`, `circuit_filename` |
| `convert_coordinates` | Convert coordinates between Bohr and Angstrom | `coordinates_json`, `to_unit` (`"bohr"` or `"angstrom"`) |
| `convert_energy` | Convert energy units | `value`, `from_unit`, `to_unit` (hartree, ev, kcal/mol, kj/mol) |

#### Classical Calculations

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_scf` | Hartree-Fock / DFT self-consistent field calculation | `project_name`, `structure_filename`, `out_wavefunction_filename`, `charge`, `spin_multiplicity`, `basis_set` |
| `run_stability_checker` | Verify SCF solution stability (always run after SCF) | `project_name`, `wavefunction_filename`, `out_stability_result_filename` |
| `run_active_space_selector` | Select active orbitals (`qdk_valence`, `qdk_autocas`, etc.) | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `charge?`, `algorithm_name?` |
| `run_orbital_localization` | Localize orbitals (Pipek-Mezey, MP2 natural orbitals) | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `loc_indices_alpha`, `loc_indices_beta?` |
| `run_multi_configuration_calculation` | CASCI / selected CI in active space | `project_name`, `hamiltonian_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons`, `n_active_beta_electrons?` |
| `run_multi_configuration_scf` | CASSCF orbital optimization | `project_name`, `orbitals_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons`, `n_active_beta_electrons?` |
| `run_projected_multi_configuration_calculation` | Sparse/projected CI for specific determinants | `project_name`, `hamiltonian_filename`, `configurations_json`, `out_wavefunction_filename` |
| `run_dynamical_correlation_calculator` | MP2 / CCSD / CCSD(T) correlation | `project_name`, `ansatz_filename`, `out_wavefunction_filename` |
| `run_hamiltonian_constructor` | Build fermionic Hamiltonian from orbitals | `project_name`, `orbitals_filename`, `out_hamiltonian_filename` |

#### Model Hamiltonians (no molecular structure needed)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_model_hamiltonian` | Build fermionic lattice Hamiltonian (Hückel, Hubbard, PPP) | `project_name`, `model`, `out_hamiltonian_filename`, `lattice_type`, `lattice_params`, `epsilon?`, `t?`, `U?`, `V?` |
| `create_spin_model_hamiltonian` | Build qubit spin Hamiltonian (Heisenberg, Ising) | `project_name`, `model`, `out_qubit_hamiltonian_filename`, `lattice_type`, `lattice_params`, `jx?`, `jy?`, `jz?`, `j?`, `h?` |

These tools bypass the molecular workflow (SCF, active space, etc.) and produce Hamiltonians directly from lattice parameters. The agent must determine appropriate model parameters from the user's description of the physical system.

#### Quantum Preparation

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_qubit_mapper` | Jordan-Wigner fermion-to-qubit mapping | `project_name`, `hamiltonian_filename`, `out_qubit_hamiltonian_filename` |
| `run_state_preparation` | Build state-prep circuit from classical wavefunction | `project_name`, `wavefunction_filename`, `out_circuit_filename` |
| `run_qubit_hamiltonian_solver` | Exact diagonalization (small systems only) | `project_name`, `qubit_hamiltonian_filename` |
| `run_energy_estimator` | Shot-based energy estimation from circuit measurements | `project_name`, `circuit_filename`, `qubit_hamiltonian_filename`, `out_energy_result_filename`, `out_measurement_data_filename`, `total_shots` |
| `run_resource_estimation` | Quantum resource estimation (logical + physical counts) | `project_name`, `circuit_filename`, `out_resource_estimator_data_filename` |

#### QPE — Circuit Construction (Mode A: Resource Analysis)

Use this mode when the user asks about resource estimates, circuit costs, or "what would it take to run this?" This does NOT produce an energy estimate.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_time_evolution_builder` | Build U = exp(-iHt) from qubit Hamiltonian | `project_name`, `qubit_hamiltonian_filename`, `evolution_time`, `out_time_evolution_unitary_filename` |
| `run_controlled_evolution_circuit_mapper` | Map to controlled circuit for phase kickback | `project_name`, `time_evolution_unitary_filename`, `out_circuit_filename`, `control_indices?`, `power?` |
| `run_circuit_executor` | Execute circuit with measurement shots | `project_name`, `circuit_filename`, `shots`, `out_executor_data_filename` |

#### QPE — Full Execution (Mode B: Eigenvalue)

Use this mode only when the user explicitly wants a computed energy from QPE.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_phase_estimation` | Run complete QPE to obtain energy eigenvalue | `project_name`, `state_prep_circuit_filename`, `qubit_hamiltonian_filename`, `out_qpe_result_filename`. Sub-algorithms configured inline via `settings`: `evolution_builder`, `circuit_mapper`, `circuit_executor` (each a dict with `algorithm_name` + setting overrides). |

**Mode A ≠ Mode B.** They answer different questions. Never switch from one to the other without explicit user approval.

#### Visualization (VS Code only)

These interactive widget tools require `qsharp_widgets` and only work in VS Code with MCP Apps.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `visualize_molecule` | Interactive 3D ball-and-stick molecular viewer | `project_name`, `structure_filename` |
| `visualize_orbitals` | Interactive 3D orbital isosurface viewer | `project_name`, `wavefunction_filename`, `orbital_indices?` (which orbitals to render), `isoval?` (default 0.02), `grid_size?` (default 40) |
| `visualize_orbital_entanglement` | Orbital entropy/mutual-information chord diagram | `project_name`, `wavefunction_filename`, `selected_indices?` (**absolute** orbital indices — auto-converted to diagram positions), `group_selected?` (reorder arcs so highlighted orbitals sit adjacent), `mi_threshold?` (minimum mutual-information value to draw a chord) |
| `visualize_circuit` | Interactive quantum circuit diagram | `project_name`, `circuit_filename` |

> **Note on `visualize_orbital_entanglement`:** Requires a wavefunction from a multi-configurational calculation run with `calculate_mutual_information=True`. Setting only `calculate_one_rdm`/`calculate_two_rdm` is **not sufficient** — the mutual information matrix must be explicitly requested. The `selected_indices` parameter accepts **absolute orbital indices** (e.g., [8, 9, 10]) matching the arc labels — the tool converts these to diagram positions automatically.

### MCP Tool Return Format

All MCP tools return a structured JSON envelope:

| Status | Meaning | Shape |
|--------|---------|-------|
| `"ok"` | Success | `{"status": "ok", "result": <value>}` |
| `"error"` | Failure | `{"status": "error", "message": "..."}` |
| `"exists"` | Output file already exists | `{"status": "exists", "message": "..."}` |
| `"submitted"` | Remote job submitted (async) | `{"status": "submitted", "result": {"job_id": "...", ...}}` |

**Always check `result.status`** before using `result.result`. When `status` is `"exists"`, the tool is asking whether to overwrite — you can re-run to overwrite or use the existing file. When `status` is `"submitted"`, the job is running remotely — use `check_remote_job` with the returned `job_id` to monitor progress.

### Standard Workflow via MCP

A typical workflow chains tools in order with a consistent `project_name`.

1. **Create structure** → `create_structure` (coordinates in Bohr)
2. **Run SCF** → `run_scf` (basis set, charge, spin multiplicity)
3. **Check stability** → `run_stability_checker` (never skip this)
4. **Select active space** → `run_active_space_selector` (choose strategy based on system — see below)
5. **Extract orbitals** → `get_orbitals_from_input`
6. **Build Hamiltonian** → `run_hamiltonian_constructor`
7. **Map to qubits** → `run_qubit_mapper`
8. **Prepare state** → `run_state_preparation`
9. **Choose endpoint:**
   - **Resource analysis (Mode A):** `run_time_evolution_builder` → `run_controlled_evolution_circuit_mapper` → `run_resource_estimation` for full logical + physical resource profile
   - **Energy (Mode B):** `run_phase_estimation` with sub-algorithms configured inline via `settings`
   - **Shot-based energy:** `run_energy_estimator` (grouping is handled internally)

### Alternative: Model Hamiltonian Workflow (no molecular structure)

For lattice models (Hubbard, Ising, Heisenberg, etc.), skip the molecular pipeline entirely:

- **Fermionic models** (Hückel, Hubbard, PPP): `create_model_hamiltonian` → `run_qubit_mapper` → continue with quantum steps
- **Spin models** (Heisenberg, Ising): `create_spin_model_hamiltonian` → produces a qubit Hamiltonian directly (no qubit mapping needed)

The agent must determine appropriate model parameters (couplings, lattice geometry, etc.) from the user's description of the physical system.

### Active Space Selection Strategies

Active space selection is problem-dependent. Choose the strategy that fits:

- **Valence:** `run_active_space_selector` with `algorithm_name="qdk_valence"` and `charge`. Works directly from SCF. Best when chemistry is restricted to valence orbitals.
- **SCI-driven (autocas):** Run `run_multi_configuration_calculation` with `calculate_one_rdm=True`, `calculate_two_rdm=True`, **and `calculate_mutual_information=True`**, then `run_active_space_selector` with `algorithm_name="qdk_autocas_eos"`. Identifies strongly correlated orbitals via entropy analysis. Most rigorous.
- **Combined:** Valence first for initial space, SCI on the reduced space to generate RDMs, then autocas to refine. Useful when the full orbital space is too large for an initial SCI.
- **Orbital localization:** Use `run_orbital_localization` before selection if localized orbitals improve the chemical picture (e.g., transition metals, localized bonds).

**Important:** The default autocas requires RDMs — it will **not** work directly on an SCF wavefunction. Either use `qdk_valence` first, or run SCI to generate RDMs first.

### Critical Rules (MCP)

- **Coordinates must be in Bohr** for `create_structure`.
- **Always check stability** after SCF — never skip `run_stability_checker`.
- **Always query defaults** with `get_algorithm_default_settings` before overriding parameters.
- **Check `status` field** in every tool response — `"ok"`, `"error"`, or `"exists"`.
- **Use `overwrite=True`** to re-run a tool when the output file already exists (all `run_*` tools support this).
- **Stop on failure** — report errors rather than guessing fixes. Consult the error recovery guidance in the skill files.
- **Pass actual output filenames** between steps — don't assume filenames.
- **Mode A ≠ Mode B** — resource analysis and energy computation answer different questions. Never switch without user approval.
- **Autocas requires RDMs** — cannot run directly on SCF wavefunction. Use `qdk_valence` or run SCI first.

---

## CLI Fallback

If MCP is not configured, use the CLI directly. Entry point: `qc`.

Ensure `qc` is on your `PATH`. If the package is installed in a virtual environment, activate it first:

```bash
source <venv>/bin/activate
```

### CLI Structure

Commands are organised into groups: `run`, `data`, `config`, `project`, `util`, `setup`.

```text
qc <group> <command> [options]
```

```bash
qc --help                # list all groups
qc run scf --help        # help for a specific command
qc --dry-run run scf ... # preview parameters without executing
```

### Quick Reference — Algorithm Commands

```bash
# SCF calculation
qc run scf --project-name h2 --structure-filename h2.structure.json \
  --out-wavefunction-filename h2.wavefunction.json \
  --charge 0 --spin-multiplicity 1 --basis-set sto-3g

# Stability check
qc run stability --project-name h2 --wavefunction-filename h2.wavefunction.json \
  --out-stability-result-filename h2.stability.json

# Active space (valence)
qc run active-space --project-name h2 --wavefunction-filename h2.wavefunction.json \
  --out-wavefunction-filename h2_as.wavefunction.json --algorithm-name qdk_valence --charge 0

# Orbital localization
qc run localize --project-name h2 --wavefunction-filename h2.wavefunction.json \
  --out-wavefunction-filename h2_loc.wavefunction.json --loc-indices-alpha '[0,1,2,3]'

# Dynamical correlation (MP2 / CCSD / CCSD(T))
qc run correlate --project-name h2 --ansatz-filename h2.ansatz.json \
  --out-wavefunction-filename h2_corr.wavefunction.json

# Hamiltonian
qc run hamiltonian --project-name h2 --orbitals-filename h2_as.orbitals.json \
  --out-hamiltonian-filename h2.hamiltonian.json

# Qubit mapping
qc run qubit-map --project-name h2 --hamiltonian-filename h2.hamiltonian.json \
  --out-qubit-hamiltonian-filename h2.qubit_hamiltonian.json

# State preparation
qc run state-prep --project-name h2 --wavefunction-filename h2_as.wavefunction.json \
  --out-circuit-filename h2.circuit.json

# Exact qubit solver (small systems)
qc run qubit-solve --project-name h2 --qubit-hamiltonian-filename h2.qubit_hamiltonian.json

# Resource estimation (logical + physical counts)
qc run resource-estimation --project-name h2 --circuit-filename h2.circuit.json \
  --out-resource-estimator-data-filename h2.resource_estimator_data.json

# Model Hamiltonian (lattice models — no molecular structure needed)
qc run model-hamiltonian --project-name hub --model hubbard \
  --lattice-type chain --lattice-params '{"n": 6, "periodic": true}' \
  --epsilon 0 --t 1.0 --U 4.0 --out-hamiltonian-filename hubbard.hamiltonian.json

# Spin model (produces qubit Hamiltonian directly)
qc run spin-model --project-name ising --model ising \
  --lattice-type square --lattice-params '{"nx": 3, "ny": 3}' \
  --j 1.0 --h 0.5 --out-qubit-hamiltonian-filename ising.qubit_hamiltonian.json

# CASCI / Selected CI
qc run casci --project-name h2 --hamiltonian-filename h2.hamiltonian.json \
  --out-wavefunction-filename h2_casci.wavefunction.json --n-active-alpha-electrons 1

# Sparse / Projected CI
qc run sparse-ci --project-name h2 --hamiltonian-filename h2.hamiltonian.json \
  --out-wavefunction-filename h2_sparse.wavefunction.json --configurations-json '[[1,1,0,0]]'

# MCSCF (compound — use --config)
qc config defaults --type mcscf > mcscf_config.json
qc run mcscf --project-name h2 --orbitals-filename h2.orbitals.json \
  --out-wavefunction-filename h2_mcscf.wavefunction.json --n-active-alpha-electrons 1 \
  --config mcscf_config.json

# QPE: step-by-step circuit construction (Mode A — resource analysis)
qc run qpe-build-evolution --project-name h2 --qubit-hamiltonian-filename h2.qubit_hamiltonian.json \
  --evolution-time 1.0 --out-time-evolution-unitary-filename h2.unitary.json
qc run qpe-map-circuit --project-name h2 --time-evolution-unitary-filename h2.unitary.json \
  --out-circuit-filename h2_qpe.circuit.json

# QPE: circuit execution with shots
qc run qpe-execute --project-name h2 --circuit-filename h2.circuit.json \
  --shots 1000 --out-executor-data-filename h2.executor_data.json

# QPE: full pipeline (Mode B — eigenvalue)
qc run qpe --project-name h2 --state-prep-circuit-filename h2.circuit.json \
  --qubit-hamiltonian-filename h2.qubit_hamiltonian.json --out-qpe-result-filename h2.qpe_result.json \
  --config qpe_config.json
# Where qpe_config.json contains:
# {"qpe": {"settings": {"num_bits": 10, "evolution_time": 1.0,
#   "evolution_builder": {"algorithm_name": "trotter"},
#   "circuit_mapper": {"algorithm_name": "pauli_sequence"},
#   "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"}}}}

# Discover algorithms
qc config algorithms
qc config defaults --algorithm-type scf_solver
qc config defaults --algorithm-type scf_solver --algorithm-name pyscf

```

### Quick Reference — Data, Utils

```bash
# Data inspection
qc data summary --project-name h2 --filename h2.wavefunction.json
qc data get-energy --project-name h2 --filename h2.wavefunction.json
qc data get-active-space-indices --project-name h2 --input-filename h2.wavefunction.json
qc data get-structure-xyz --project-name h2 --filename h2.structure.json
qc data get-circuit-qasm --project-name h2 --filename h2.circuit.json
qc data get-qubit-hamiltonian-info --project-name h2 --filename h2.qubit_hamiltonian.json
qc data get-stability-result --project-name h2 --filename h2.stability.json
qc data get-qpe-result --project-name h2 --filename h2.qpe_result.json
qc data get-top-configurations --project-name h2 --wavefunction-filename h2_casci.wavefunction.json --max-determinants 10
qc data convert --project-name h2 --filename h2.wavefunction.json --out-filename h2.wavefunction.hdf5
qc data upload-structure --project-name h2 --coordinates-json '[[0,0,0],[0,0,1.4]]' --symbols H H

# Utilities
qc project list
qc project create --project-name my_molecule
qc project files --project-name my_molecule
qc util convert-coordinates --coordinates '[[0,0,0],[1.5,0,0]]' --to-bohr
qc util convert-energy --value -1.137 --from-unit hartree --to-unit ev
qc util compute-valence-params --project-name h2 --wavefunction-filename h2.wavefunction.json --charge 0
qc util resolve-phase-energy --phase-fraction 0.25 --evolution-time 1.0 --reference-energy -1.0

# Multi-step workflow from config (supports $prev / $step.N references)
qc workflow --config pipeline.json --project-name h2
# Validate workflow chain without executing
qc --dry-run workflow --config pipeline.json --project-name h2

# Agent discovery
qc list-commands                   # JSON list of all commands
qc describe "run scf"              # JSON schema with parameter types/roles
```

### Compound Algorithm Config

```bash
# Generate template, then pass with optional overrides
qc config defaults --type mcscf > mcscf_config.json
qc run mcscf ... --config mcscf_config.json \
  --set mc_calculator.settings.calculate_one_rdm=true \
  --set mc_calculator.settings.calculate_two_rdm=true \
  --set mc_calculator.settings.calculate_mutual_information=true
```

---

## Resource Estimation Concepts

QDK Chemistry includes a built-in resource estimator (`run_resource_estimation` MCP tool / `qc run resource-estimation` CLI) that produces both logical and physical resource profiles from any circuit file.

When reporting QPE or circuit results, include the full resource profile — not just qubit count:

| Metric | Description |
|--------|-------------|
| Logical qubits | Error-corrected qubits the algorithm operates on |
| Physical qubits | Total hardware qubits after error correction overhead |
| Circuit depth | Number of sequential gate layers (deeper = slower, noisier) |
| Gate count | Total gates, broken down by type |
| T-count / T-depth | Number and depth of T gates — the dominant fault-tolerant cost |
| Runtime | Estimated wall-clock time for execution |
| Code distance | QEC surface code distance |
| Error budget | Breakdown across logical, rotation, and T-state errors |

Use `run_resource_estimation` on any circuit file (state preparation, QPE, controlled-U) to get the full profile. The default estimator (`qdk_qre_v3`) uses the Q# resource estimator backend and returns a Pareto frontier of optimal (qubit, runtime) tradeoffs.

**Gate types:**

- **Clifford gates** (H, S, CNOT, Paulis) — cheap in fault-tolerant schemes.
- **Non-Clifford / T gates** (π/8 gate) — expensive, each needs magic state distillation.
- **Arbitrary rotations** (Rz(θ), Rx(θ), Ry(θ)) — decomposed into Clifford + T sequences; precision ε trades off against T-count.

---

## Data File Types

| Extension | Type |
|-----------|------|
| `.structure.json` | Molecular geometry |
| `.wavefunction.json` | Wavefunction (orbitals, coefficients, energies) |
| `.hamiltonian.json` | Fermionic Hamiltonian |
| `.qubit_hamiltonian.json` | Qubit Hamiltonian (Pauli operator sum) |
| `.circuit.json` | Quantum circuit |
| `.orbitals.json` | Orbital data |
| `.ansatz.json` | Ansatz (wavefunction + Hamiltonian) |
| `.qpe_result.json` | QPE result (energy, phase, bits) |
| `.stability.json` | Stability analysis result |
| `.config.json` | Algorithm configuration (individual QPE step tools) |
| `.energy_result.json` | Energy estimation result |
| `.measurement.json` | Measurement data from circuit execution |
| `.resource_estimator_data.json` | Resource estimation results (logical + physical counts) |

---

## References

- Repository: [microsoft/qdk-chemistry](https://github.com/microsoft/qdk-chemistry)
- Reference data: [microsoft/qdk-chemistry-data](https://github.com/microsoft/qdk-chemistry-data)
- Paper: [arXiv:2601.15253](https://arxiv.org/abs/2601.15253)
