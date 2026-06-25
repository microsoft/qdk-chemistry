# Workflow Patterns

QDK Chemistry workflows have two entry points and three endpoints.

## Entry Point A: Molecular Systems (the common stem)

For real molecules, every workflow begins with these steps:

1. **Upload structure** — `create_structure` (coordinates in Bohr)
2. **SCF** — `run_scf` (HF or DFT, specify basis set, charge, spin multiplicity)
3. **Stability check** — `run_stability_checker` (verify SCF found the true minimum, not a saddle point)
4. **Active space analysis** — For systems with >16 spatial orbitals or strong correlation, run SCI + AutoCAS (`qdk_autocas_eos`) to identify and compress the active space (see active-space-guide.md). For small systems (≤16 spatial orbitals), full-space simulation is often tractable — ask the user or assess system size before defaulting to active space compression. For large systems, use valence selection as a pre-filter before SCI
5. **Get orbitals** — `get_orbitals_from_input`
6. **Build Hamiltonian** — `run_hamiltonian_constructor`

After step 6, you have a fermionic Hamiltonian on the active space. What happens next depends on the endpoint.

## Entry Point B: Model Hamiltonians (skip the molecular pipeline)

For lattice models (Hubbard, Hückel, PPP, Ising, Heisenberg), skip SCF and active space entirely:

**Fermionic models** (Hückel, Hubbard, PPP):
1. `create_model_hamiltonian` — specify model type, lattice geometry, and coupling parameters
2. `run_qubit_mapper` — map to qubit Hamiltonian
3. Continue with the chosen endpoint below

**Spin models** (Heisenberg, Ising):
1. `create_spin_model_hamiltonian` — produces a qubit Hamiltonian directly (no qubit mapping needed)
2. Continue with the chosen endpoint below

The agent must determine appropriate model parameters (coupling constants, lattice size, boundary conditions) from the user's description of the physical system — do not ask the user for parameter values that can be inferred from the physics.

## Endpoint 1: "What would it cost to run this on quantum hardware?"

The user wants a resource profile — qubit count, circuit depth, gate counts, T-count. They do NOT want an energy.

Continue from the common stem:

7. **Map to qubits** — `run_qubit_mapper` (Jordan-Wigner)
8. **State preparation** — `run_state_preparation`
9. **Build time evolution** — `run_time_evolution_builder` (constructs U = exp(-iHt))
10. **Build controlled circuit** — `run_controlled_evolution_circuit_mapper`
11. **Resource estimation** — `run_resource_estimation` on the circuit(s)

Extract the resource profile from steps 10–11 and present it: logical qubits, circuit depth, total gates, Clifford gates, T-count, T-depth, physical qubits, runtime, code distance, and error budget.

**If circuit construction fails:** Report the error. Don't fall back to computing an energy — that's a different question. You can provide analytical estimates based on qubit count and Hamiltonian size, but label them as estimates.

## Endpoint 2: "Compute the ground state energy"

The user wants an actual energy number from QPE.

Continue from the common stem:

7. **Map to qubits** — `run_qubit_mapper`
8. **State preparation** — `run_state_preparation`
9. **Run phase estimation** — `run_phase_estimation` with sub-algorithm overrides in `settings`

**Critical:** `run_phase_estimation` has intentionally invalid defaults: `num_bits=-1` and `evolution_time=0.0`. You MUST override these in settings. Typical values: `num_bits=10` (≈1 mHa precision), `evolution_time` computed from the Hamiltonian's spectral norm.

**Multi-trial strategy (from the real examples):** Run 20 trials with different random seeds, then use majority voting on the energy results. This is more robust than a single high-precision run.

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
