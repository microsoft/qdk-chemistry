<!-- qdk-chemistry-agent-config {{QDK_CHEMISTRY_VERSION}} -->
# QDK Chemistry — GitHub Copilot Instructions

You are working with **QDK/Chemistry**, an end-to-end quantum chemistry package by Microsoft. It has two interfaces: an **MCP server** (preferred in VS Code) and a grouped **CLI** (`qc`).

**Prefer visuals** — whenever a VS Code visualization tool can show something (molecule, orbitals, circuit, entanglement, Pareto frontier), use it instead of describing it in text.

For small molecular systems (up to ~16 spatial orbitals / ~20 qubits), the full orbital space is often tractable — don't default to active space selection unless the system requires it.

---

## MCP Server (Preferred in VS Code)

GitHub Copilot in VS Code supports MCP natively. Configure the QDK Chemistry MCP server in `.vscode/mcp.json`:

```json
{
  "servers": {
    "qdk_chemistry": {
      "command": "bash",
      "args": ["-c", "source <venv>/bin/activate && qdk_chem_mcp"]
    }
  }
}
```

Once configured, the MCP tools are available for no-code quantum chemistry workflows.

### MCP Tool Reference

> **Canonical reference:** The authoritative tool documentation is in `shared/skills/qdk-chemistry-mcp/SKILL.md`. The tables below are a summary — consult the skill file for full details and worked examples.

#### Project Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_projects` | List all projects | — |
| `create_project` | Create a new project directory | `project_name` |
| `list_project_files` | List files in a project with inferred types | `project_name` |
| `list_tools` | List available tools by category | `category?` |

#### Backend & Cache Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_cache_backends` | List available local caching backends | — |
| `describe_backend` | Get details about a specific cache backend | `backend_type`, `name` |

Use these tools to discover available resources before configuring caching strategies.

#### Structure & Helpers

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `create_structure` | Create molecular geometry | `project_name`, `coordinates_json` (Bohr), `symbols`, `overwrite?` |
| `get_summary` | Inspect any data file (auto-detects type) | `project_name`, `filename` |
| `get_algorithm_default_type` | Get default algorithm for a type | `algorithm_type` |
| `get_algorithm_default_settings` | Get default settings dict | `algorithm_type`, `algorithm_name?` |
| `get_orbitals_from_input` | Extract orbitals from wavefunction/hamiltonian | `project_name`, `input_filename`, `out_orbitals_filename` |
| `get_active_space_indices` | Get active/inactive/virtual index partitioning | `project_name`, `input_filename` |
| `get_ansatz` | Build Ansatz from wavefunction + Hamiltonian | `project_name`, `wavefunction_filename`, `hamiltonian_filename`, `out_ansatz_filename` |
| `get_top_configurations` | Get highest-weight CI determinants | `project_name`, `wavefunction_filename`, `max_determinants?` |
| `get_circuit_stats` | Circuit resource metrics (qubits, depth, gates) | `project_name`, `circuit_filename` |
| `convert_coordinates` | Convert coordinates between Bohr and Angstrom | `coordinates_json`, `to_unit` |
| `convert_energy` | Convert energy units (Hartree, eV, kcal/mol, kJ/mol) | `value`, `from_unit`, `to_unit` |

#### Classical Calculations

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_scf` | Hartree-Fock / DFT | `project_name`, `structure_filename`, `out_wavefunction_filename`, `charge`, `spin_multiplicity`, `basis_set` |
| `run_stability_checker` | Verify SCF solution stability | `project_name`, `wavefunction_filename`, `out_stability_result_filename` |
| `run_active_space_selector` | Select active orbitals | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `charge?`, `algorithm_name?` |
| `run_orbital_localization` | Localize orbitals (Pipek-Mezey, MP2 NOs) | `project_name`, `wavefunction_filename`, `out_wavefunction_filename`, `loc_indices_alpha` |
| `run_multi_configuration_calculation` | CASCI / selected CI | `project_name`, `hamiltonian_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons` |
| `run_multi_configuration_scf` | CASSCF orbital optimization | `project_name`, `orbitals_filename`, `out_wavefunction_filename`, `n_active_alpha_electrons` |
| `run_projected_multi_configuration_calculation` | Sparse/projected CI on specific determinants | `project_name`, `hamiltonian_filename`, `configurations_json`, `out_wavefunction_filename` |
| `run_dynamical_correlation_calculator` | MP2 / CCSD / CCSD(T) | `project_name`, `ansatz_filename`, `out_wavefunction_filename` |
| `run_hamiltonian_constructor` | Build fermionic Hamiltonian | `project_name`, `orbitals_filename`, `out_hamiltonian_filename` |

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
| `run_state_preparation` | Build state-prep circuit from wavefunction | `project_name`, `wavefunction_filename`, `out_circuit_filename` |
| `run_qubit_hamiltonian_solver` | Exact diagonalization (small systems) | `project_name`, `qubit_hamiltonian_filename` |
| `run_energy_estimator` | Shot-based energy estimation | `project_name`, `circuit_filename`, `qubit_hamiltonian_filename`, `total_shots`, `out_energy_result_filename`, `out_measurement_data_filename` |
| `run_resource_estimation` | Quantum resource estimation (logical + physical) | `project_name`, `circuit_filename`, `out_resource_estimator_data_filename` |

#### QPE — Circuit Construction (Mode A: Resource Analysis)

| Tool | Purpose |
|------|---------|
| `run_time_evolution_builder` | Build U = exp(-iHt) from qubit Hamiltonian |
| `run_controlled_evolution_circuit_mapper` | Map to controlled circuit for phase kickback |
| `run_circuit_executor` | Execute circuit with measurement shots |

#### QPE — Full Execution (Mode B: Eigenvalue)

| Tool | Purpose |
|------|---------|
| `run_phase_estimation` | Run complete QPE to obtain energy eigenvalue |

#### Visualization (VS Code only)

These interactive widget tools require `qsharp_widgets` and only work in VS Code with MCP Apps.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `visualize_molecule` | Interactive 3D ball-and-stick viewer | `project_name`, `structure_filename` |
| `visualize_orbitals` | Interactive 3D orbital isosurface viewer | `project_name`, `wavefunction_filename`, `orbital_indices?`, `isoval?`, `grid_size?` |
| `visualize_orbital_entanglement` | Orbital entropy/mutual-information chord diagram | `project_name`, `wavefunction_filename`, `selected_indices?` (**absolute** orbital indices — auto-converted), `group_selected?`, `mi_threshold?` |
| `visualize_circuit` | Interactive quantum circuit diagram | `project_name`, `circuit_filename` |

### MCP Tool Return Format

All tools return a JSON envelope:

| Status | Meaning | Shape |
|--------|---------|-------|
| `"ok"` | Success | `{"status": "ok", "result": <value>}` |
| `"error"` | Failure | `{"status": "error", "message": "..."}` |
| `"exists"` | Output file already exists | `{"status": "exists", "message": "..."}` |

Always check `result.status` before using `result.result`. When `status` is `"exists"`, the tool is asking whether to overwrite.

---

## CLI Reference

The CLI is the alternative interface. Entry point: `qc`.

### Setup

Ensure `qc` is on your `PATH`. If installed in a virtual environment, activate it first:

```bash
source <venv>/bin/activate
```

### CLI Overview

Commands are organised into groups: `run`, `data`, `config`, `project`, `util`, `setup`.

```text
qc <group> <command> [options]
```

```bash
qc --help                # list all groups
qc run --help            # list algorithm commands
qc run scf --help        # help for a specific command
qc --dry-run run scf ... # preview parameters without executing
```

---

## Standard Workflow

Use a consistent `--project-name` (CLI) or `project_name` (MCP) across all steps.

### Stage 1: Project Setup & Structure

```bash
# Create a project
qc project create --project-name myproj

# Upload molecular structure (coordinates MUST be in Bohr)
qc data upload-structure \
  --project-name myproj \
  --coordinates-json '[[0,0,0],[0,0,1.4]]' \
  --symbols H H

# If you have Angstrom coordinates, convert first
qc util convert-coordinates --coordinates '[[0,0,0],[0,0,0.74]]' --to-bohr
```

### Stage 2: Classical Preparation & Compression

```bash
# 1. Run SCF (Hartree-Fock / DFT)
qc run scf \
  --project-name myproj \
  --structure-filename structure.structure.json \
  --out-wavefunction-filename scf.wavefunction.json \
  --charge 0 --spin-multiplicity 1 --basis-set cc-pvdz

# 2. ALWAYS check stability after SCF
qc run stability \
  --project-name myproj \
  --wavefunction-filename scf.wavefunction.json \
  --out-stability-result-filename scf.stability.json

qc data get-stability-result --project-name myproj --filename scf.stability.json

# 3. Select active space (choose strategy — see Active Space Strategies below)
#    Option A: Valence space (fast, works directly from SCF)
qc run active-space \
  --project-name myproj \
  --wavefunction-filename scf.wavefunction.json \
  --out-wavefunction-filename as.wavefunction.json \
  --algorithm-name qdk_valence --charge 0

#    Option B: SCI-driven (autocas, requires RDMs first)
qc run casci \
  --project-name myproj \
  --hamiltonian-filename ham.hamiltonian.json \
  --out-wavefunction-filename sci.wavefunction.json \
  --n-active-alpha-electrons 3 \
  --settings '{"calculate_one_rdm": true, "calculate_two_rdm": true, "calculate_mutual_information": true}'

qc run active-space \
  --project-name myproj \
  --wavefunction-filename sci.wavefunction.json \
  --out-wavefunction-filename as_refined.wavefunction.json \
  --algorithm-name qdk_autocas_eos

#    Option C: Orbital localization before selection
qc run localize \
  --project-name myproj \
  --wavefunction-filename scf.wavefunction.json \
  --out-wavefunction-filename loc.wavefunction.json \
  --loc-indices-alpha '[0,1,2,3]'

# 4. Extract orbitals
qc data get-orbitals \
  --project-name myproj \
  --input-filename as.wavefunction.json \
  --out-orbitals-filename as.orbitals.json

# 5. Build fermionic Hamiltonian
qc run hamiltonian \
  --project-name myproj \
  --orbitals-filename as.orbitals.json \
  --out-hamiltonian-filename as.hamiltonian.json

# 6. (Optional) Dynamical correlation — MP2 / CCSD / CCSD(T)
qc run correlate \
  --project-name myproj \
  --ansatz-filename as.ansatz.json \
  --out-wavefunction-filename corr.wavefunction.json
```

### Alternative: Model Hamiltonian Workflow (no molecular structure)

For lattice models (Hubbard, Ising, Heisenberg, etc.), skip the entire molecular pipeline and go directly to a Hamiltonian. The agent must determine appropriate model parameters (couplings, lattice size, etc.) from the user's description of the physical system.

```bash
# Fermionic model → use create_model_hamiltonian, then qubit-map
qc run model-hamiltonian \
  --project-name myproj \
  --model hubbard \
  --lattice-type chain --lattice-params '{"n": 6, "periodic": true}' \
  --epsilon 0 --t 1.0 --U 4.0 \
  --out-hamiltonian-filename hubbard.hamiltonian.json

# Spin model → produces qubit Hamiltonian directly (no qubit-map needed)
qc run spin-model \
  --project-name myproj \
  --model ising \
  --lattice-type square --lattice-params '{"nx": 3, "ny": 3}' \
  --j 1.0 --h 0.5 \
  --out-qubit-hamiltonian-filename ising.qubit_hamiltonian.json
```

Then continue with Stage 3 (qubit mapping for fermionic models) or Stage 4 (quantum execution) directly.

### Stage 3: Quantum Preparation & Compression

```bash
# 7. Map fermions to qubits (Jordan-Wigner)
qc run qubit-map \
  --project-name myproj \
  --hamiltonian-filename as.hamiltonian.json \
  --out-qubit-hamiltonian-filename as.qubit_hamiltonian.json

# 8. Build state preparation circuit
qc run state-prep \
  --project-name myproj \
  --wavefunction-filename as.wavefunction.json \
  --out-circuit-filename as.circuit.json
```

### Stage 4: Quantum Circuit Construction & Execution

These two modes answer different questions. **Never switch from one to the other without explicit user approval.**

#### Mode A: Resource Analysis (circuit costs, no energy)

Build the QPE circuit and extract resource estimates (qubit count, depth, gate counts, T-count/T-depth). Use when the user asks "what would it take to run this?"

```bash
qc run qpe-build-evolution \
  --project-name myproj \
  --qubit-hamiltonian-filename as.qubit_hamiltonian.json \
  --evolution-time 1.0 \
  --out-time-evolution-unitary-filename as.unitary.json

qc run qpe-map-circuit \
  --project-name myproj \
  --time-evolution-unitary-filename as.unitary.json \
  --out-circuit-filename as_qpe.circuit.json

# Extract resource profile
qc data summary --project-name myproj --filename as_qpe.circuit.json

# Full resource estimation (logical + physical counts, error budget)
qc run resource-estimation \
  --project-name myproj \
  --circuit-filename as_qpe.circuit.json \
  --out-resource-estimator-data-filename as_qpe.resource_estimator_data.json
```

#### Mode B: Full QPE (compute energy eigenvalue)

Run the complete QPE algorithm. Use only when the user explicitly wants an energy from QPE.
Sub-algorithms are configured inline via `--config`:

```bash
# Create a QPE config file (or use defaults --type qpe to generate a template)
cat > qpe_config.json << 'EOF'
{"qpe": {"settings": {"num_bits": 10, "evolution_time": 1.0,
  "evolution_builder": {"algorithm_name": "trotter"},
  "circuit_mapper": {"algorithm_name": "pauli_sequence"},
  "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"}}}}
EOF

qc run qpe \
  --project-name myproj \
  --state-prep-circuit-filename as.circuit.json \
  --qubit-hamiltonian-filename as.qubit_hamiltonian.json \
  --out-qpe-result-filename as.qpe_result.json \
  --config qpe_config.json

qc data get-qpe-result --project-name myproj --filename as.qpe_result.json
```

#### Alternative: Shot-Based Energy Estimation

Pauli term grouping is handled internally by `qc run energy`; there is no separate `filter-pauli` CLI step.

```bash
qc config defaults --type energy > energy_config.json
qc run energy \
  --project-name myproj \
  --circuit-filename as.circuit.json \
  --qubit-hamiltonian-filename as.qubit_hamiltonian.json \
  --out-energy-result-filename as.energy_result.json \
  --out-measurement-data-filename as.measurement.json \
  --total-shots 10000 --config energy_config.json
```

#### Standalone Circuit Execution

```bash
qc run qpe-execute \
  --project-name myproj \
  --circuit-filename as.circuit.json \
  --shots 1000 \
  --out-executor-data-filename as.executor_data.json
```

---

## Active Space Selection Strategies

Active space selection is **problem-dependent**. Choose the strategy that fits the chemistry:

- **Valence space:** `run active-space --algorithm-name qdk_valence --charge <charge>`. Works directly from SCF. Best when chemistry is restricted to valence orbitals.
- **SCI-driven (autocas):** Run `run casci` with `calculate_one_rdm`, `calculate_two_rdm`, and `calculate_mutual_information` all set to `true`, then `run active-space --algorithm-name qdk_autocas_eos`. Identifies strongly correlated orbitals via entropy analysis. Most rigorous.
- **Combined:** Valence first for initial space, then SCI on the reduced space, then autocas refinement. Useful when the full orbital space is too large for an initial SCI.
- **Orbital localization:** `run localize` before selection if localized orbitals improve the chemical picture (e.g., transition metals, localized bonds).

**Important:** The default autocas requires RDMs — it will NOT work directly on an SCF wavefunction. Either use `qdk_valence` first, or run SCI to generate RDMs first.

---

## Multi-Step Workflow

Run an entire pipeline from a single JSON config:

```bash
qc workflow --config pipeline.json --project-name myproj
```

```json
{
  "steps": [
    {"command": "upload-structure", "args": {"coordinates_json": "[[0,0,0],[0,0,1.4]]", "symbols": ["H","H"], "filename_to_save": "h2.structure.json"}},
    {"command": "scf", "args": {"structure_filename": "$prev", "out_wavefunction_filename": "scf.wavefunction.json", "charge": 0, "spin_multiplicity": 1, "basis_set": "sto-3g"}},
    {"command": "get-orbitals", "args": {"input_filename": "$prev", "out_orbitals_filename": "scf.orbitals.json"}},
    {"command": "hamiltonian", "args": {"orbitals_filename": "$prev", "out_hamiltonian_filename": "h2.hamiltonian.json"}},
    {"command": "qubit-map", "args": {"hamiltonian_filename": "$prev", "out_qubit_hamiltonian_filename": "h2.qubit_hamiltonian.json"}},
    {"command": "state-prep", "args": {"wavefunction_filename": "$step.2", "out_circuit_filename": "h2.circuit.json"}}
  ]
}
```

- `--project-name` is injected into every step automatically.
- Use `$prev` to reference the first output of the previous step.
- Use `$step.N` to reference the first output of step N (1-indexed).
- Use `$prev.1` or `$step.N.1` to reference the second output.
- Stops on first error.

**Validate before executing:**

```bash
qc --dry-run workflow --config pipeline.json --project-name myproj
```

Dry-run checks: all commands valid, required parameters present, `$prev`/`$step.N` references resolve, input/output filename chaining verified.

---

## Data Inspection

```bash
qc data summary --project-name myproj --filename scf.wavefunction.json
qc data get-energy --project-name myproj --filename scf.wavefunction.json
qc data get-active-space-indices --project-name myproj --input-filename as.wavefunction.json
qc data get-structure-xyz --project-name myproj --filename structure.structure.json
qc data get-circuit-qasm --project-name myproj --filename as.circuit.json
qc data get-qubit-hamiltonian-info --project-name myproj --filename as.qubit_hamiltonian.json
qc data get-stability-result --project-name myproj --filename scf.stability.json
qc data get-qpe-result --project-name myproj --filename as.qpe_result.json
qc data convert --project-name myproj --filename scf.wavefunction.json --out-filename scf.wavefunction.hdf5
qc data get-top-configurations --project-name myproj --wavefunction-filename sci.wavefunction.json --max-determinants 10
```

---

## Utilities

```bash
qc project list
qc project create --project-name newproj
qc project files --project-name myproj
qc util convert-coordinates --coordinates '[[0,0,0],[1.5,0,0]]' --to-bohr
qc util convert-energy --value -1.137 --from-unit hartree --to-unit ev
qc util compute-valence-params --project-name myproj --wavefunction-filename scf.wavefunction.json --charge 0
qc util resolve-phase-energy --phase-fraction 0.25 --evolution-time 1.0 --reference-energy -1.0
```

---

## Compound Algorithms & Discovery

```bash
# Generate config templates for compound algorithms
qc config defaults --type mcscf
qc config defaults --type qpe
qc config defaults --type energy

# Use config with inline overrides
qc run mcscf --project-name myproj --orbitals-filename as.orbitals.json \
  --out-wavefunction-filename mcscf.wavefunction.json --n-active-alpha-electrons 3 \
  --config mcscf_config.json \
  --set mc_calculator.settings.calculate_one_rdm=true \
  --set mc_calculator.settings.calculate_two_rdm=true \
  --set mc_calculator.settings.calculate_mutual_information=true

# Discover available algorithms
qc config algorithms
qc config algorithms --algorithm-type scf_solver
qc config defaults --algorithm-type scf_solver
qc config defaults --algorithm-type scf_solver --algorithm-name pyscf
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

Use `run_resource_estimation` on any circuit file (state preparation, QPE, controlled-U) to get the full profile. The tool uses the circuit's built-in `estimate()` method and forwards any settings as estimator parameters.

**Gate types:**

- **Clifford gates** (H, S, CNOT, Paulis) — cheap in fault-tolerant schemes.
- **Non-Clifford / T gates** (π/8 gate) — expensive, each needs magic state distillation.
- **Arbitrary rotations** (Rz(θ), Rx(θ), Ry(θ)) — decomposed into Clifford + T sequences; precision ε trades off against T-count.

---

## Critical Rules

1. **Coordinates must be in Bohr** — convert with `qc util convert-coordinates --to-bohr`.
2. **Always run stability check** after SCF — never skip `run stability`.
3. **Query defaults** before overriding: `qc config defaults --algorithm-type <type>`.
4. **Use actual output filenames** from previous steps — don't assume names.
5. **Mode A ≠ Mode B** — resource analysis and energy computation answer different questions. Never switch between them without user approval.
6. All output is JSON — pipe through `jq`: `qc ... | jq .`
7. **Preview before executing** — use `qc --dry-run run <command> ...` to see parameters without running.
8. **Autocas requires RDMs** — cannot run directly on SCF wavefunction. Use `qdk_valence` or run SCI first.
9. **Check `status` field** in every MCP response — `"ok"`, `"error"`, or `"exists"`.

---

## Agent Discovery

Programmatic command introspection for agents:

```bash
# List all commands with categories and workflow compatibility
qc list-commands

# Get JSON schema for any command (parameters, types, roles)
qc describe "run scf"
qc describe "data upload-structure"

# Preview any command without executing
qc --dry-run run scf --project-name h2 --structure-filename h2.structure.json ...
```

`describe` output includes `role: "input"` / `role: "output"` annotations and `workflow_compatible: true/false`.

---

## Complete CLI Command List

### Algorithm Commands (`qc run ...`)

| Command | Purpose |
|---------|---------|
| `run scf` | SCF (Hartree-Fock / DFT) |
| `run stability` | Check wavefunction stability |
| `run active-space` | Select active orbital space |
| `run localize` | Localize orbitals |
| `run correlate` | Dynamical correlation (MP2, CCSD, CCSD(T)) |
| `run hamiltonian` | Build fermionic Hamiltonian |
| `run model-hamiltonian` | Build lattice model Hamiltonian (Hückel, Hubbard, PPP) |
| `run spin-model` | Build spin model Hamiltonian (Heisenberg, Ising) |
| `run casci` | CASCI / selected CI |
| `run mcscf` | MCSCF / CASSCF (compound, `--config`) |
| `run sparse-ci` | Projected/sparse multi-configuration CI |
| `run qubit-map` | Map fermionic → qubit Hamiltonian |
| `run state-prep` | Generate state preparation circuit |
| `run qubit-solve` | Exact diagonalization of qubit Hamiltonian |
| `run resource-estimation` | Quantum resource estimation (logical + physical) |
| `run energy` | Shot-based energy estimation (compound, `--config`; groups commuting Pauli terms internally) |
| `run qpe-build-evolution` | Build time evolution unitary |
| `run qpe-map-circuit` | Map unitary to controlled circuit |
| `run qpe-execute` | Execute circuit with shots |
| `run qpe` | Full QPE pipeline (compound) |

### Data (`qc data ...`), Project (`qc project ...`), Config (`qc config ...`), Util (`qc util ...`)

| Command | Purpose |
|---------|---------|
| `data summary` | Human-readable summary of any data file |
| `data convert` | Convert between JSON and HDF5 |
| `data get-orbitals` | Extract orbitals |
| `data get-active-space-indices` | Get orbital index partitioning |
| `data get-ansatz` | Build Ansatz |
| `data get-top-configurations` | Get top CI determinants |
| `data upload-structure` | Upload molecular structure |
| `data get-energy` | Get energy from wavefunction/result |
| `data get-structure-xyz` | Export as XYZ format |
| `data get-circuit-qasm` | Export circuit as OpenQASM |
| `data get-qubit-hamiltonian-info` | Inspect qubit Hamiltonian |
| `data get-stability-result` | Inspect stability result |
| `data get-qpe-result` | Inspect QPE result |
| `project list` | List all projects |
| `project create` | Create new project |
| `project files` | List project data files |
| `util convert-coordinates` | Bohr ↔ Angstrom |
| `util convert-energy` | Convert energy units |
| `util compute-valence-params` | Compute valence space parameters |
| `util resolve-phase-energy` | Resolve QPE phase to energy |
| `config defaults` | Show/generate algorithm default configs |
| `config algorithms` | List available algorithms |
| `workflow` | Run multi-step pipeline from JSON config |
| `describe` | JSON schema for any command |
| `list-commands` | List all commands (JSON) |

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
