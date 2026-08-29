# Workflow Patterns

QDK Chemistry workflows have two entry points and three endpoints.

## Entry Point A: Molecular Systems (the common stem)

For real molecules, every workflow begins with these steps:

1. **Upload structure** — `create_structure` (coordinates in Bohr)
2. **SCF** — `run_scf` with the requested method, basis, charge, and spin multiplicity. An open-shell wavefunction sent to valence selection, ASCI, or AutoCAS needs HF with `scf_type="restricted"`.
3. **Stability check** — `run_stability_checker` when requested by the workflow
4. **Active space analysis** — `run_active_space_selector` when the requested workflow uses an active space. Entropy-based selection needs RDM output from a preceding multi-configuration calculation.
5. **Get orbitals** — `get_orbitals_from_input`
6. **Build Hamiltonian** — `run_hamiltonian_constructor`

After step 6, you have a fermionic Hamiltonian on the active space. What happens next depends on the endpoint.

## Entry Point B: Model Hamiltonians (skip the molecular pipeline)

For lattice models (Hubbard, Hückel, PPP, Ising, Heisenberg), skip SCF and active space entirely:

**Fermionic models** (Hückel, Hubbard, PPP):
1. `create_model_hamiltonian` — specify model type, lattice geometry, and coupling parameters
2. `create_majorana_mapping` — create a mapping file sized from the Hamiltonian
3. `run_qubit_mapper` — apply the mapping file to produce a qubit Hamiltonian
4. Continue with the chosen endpoint below

**Spin models** (Heisenberg, Ising):
1. `create_spin_model_hamiltonian` — produces a qubit Hamiltonian directly (no qubit mapping needed)
2. Continue with the chosen endpoint below

The creation-tool schema identifies the required model parameters.

## Endpoint 1: "What would it cost to run this on quantum hardware?"

The user wants a resource profile — qubit count, circuit depth, gate counts, T-count. They do NOT want an energy.

Continue from the common stem:

7. **Create mapping** — `create_majorana_mapping` (Jordan-Wigner by default)
8. **Map to qubits** — `run_qubit_mapper`
9. **State preparation** — `run_state_preparation`
10. **Build time evolution** — `run_time_evolution_builder` (constructs U = exp(-iHt))
11. **Build controlled circuit** — `run_controlled_evolution_circuit_mapper`
12. **Resource estimation** — `run_resource_estimation` on the circuit(s)

Present logical qubits, depth, and gate metrics from `get_circuit_stats`. Present each `run_resource_estimation` Pareto point with its physical qubits, runtime, achieved error, and estimation assumptions. Report fields absent from either response as unavailable.

**If circuit construction fails:** Report the error. Don't fall back to computing an energy or invent a resource estimate — those answer different questions.

## Endpoint 2: "Compute the ground state energy"

The user wants an actual energy number from QPE.

Continue from the common stem:

7. **Create mapping** — `create_majorana_mapping`
8. **Map to qubits** — `run_qubit_mapper`
9. **State preparation** — `run_state_preparation`
10. **Run phase estimation** — `run_phase_estimation` with sub-algorithm overrides in `settings`

`run_phase_estimation` has intentionally invalid defaults under
`settings.qpe_circuit_builder`: `num_bits=-1` and `unitary_builder.time=0.0`.
Supply both explicit nested values. See the MCP QPE reference for the settings
shape and runtime discovery calls.

**If QPE fails:** Report the error. Don't switch to resource analysis — they answer different questions.

## Endpoint 3: "What is the classical energy?"

The user only wants SCF, post-HF, or multi-reference energy. No quantum circuits.

Stop after the common stem. Optionally add:

- `run_dynamical_correlation_calculator` — MP2/CCSD/CCSD(T)
- `run_multi_configuration_calculation` — CASCI/SCI
- `run_multi_configuration_scf` — MCSCF/CASSCF
- `run_qubit_hamiltonian_solver` — exact diagonalization

Do NOT proceed to qubit mapping or circuit construction.

## How to Decide Entry Point and Endpoint

**Entry point** — listen to what the user describes:

| User says... | Entry point |
|---|---|
| molecule name, chemical formula, geometry, XYZ coordinates | Molecular (Entry Point A) |
| "Hubbard model", "Ising model", "lattice", "chain", "square lattice" | Model Hamiltonian (Entry Point B) |

**Endpoint** — listen to what the user wants to know:

| User says... | Endpoint |
|---|---|
| "how many qubits", "what resources", "circuit cost", "could this run on hardware" | Resource profile |
| "compute the energy", "run QPE", "ground state energy", "eigenvalue" | Energy computation |
| "SCF energy", "classical calculation", "CCSD energy" | Classical only |

These are fundamentally different questions. Never switch between endpoints without asking the user.

## Visualization During Execution

Show results as they happen, not all at the end:

| After step | Show |
|---|---|
| `create_structure` | 3D molecule viewer |
| Active space selection | Orbital isosurface viewer (selected orbitals) |
| SCI with `calculate_mutual_information=True` | Entanglement chord diagram |
| State preparation | Circuit diagram |
| Circuit construction (resource analysis) | Circuit diagram + resource table |
| Phase estimation | Circuit diagram + energy result |
