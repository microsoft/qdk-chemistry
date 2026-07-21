# QPE and State Preparation

## State Preparation: The Sparse Isometry Advantage

QDK Chemistry offers two state preparation methods. The difference is dramatic:

| Method | Algorithm name | Fine rotations | Circuit depth | When to use |
|---|---|---|---|---|
| **Sparse isometry** | `sparse_isometry_gf2x` | ~1 | ~50 gates | Production — always prefer this |
| **General isometry** | `qiskit_regular_isometry` | ~thousands | ~1000+ gates | Only for comparison / understanding cost |

The real benzene diradical example shows this: a 2-determinant wavefunction needs **1 fine rotation with sparse isometry** vs **thousands with general isometry**, for the same quantum state. Sparse isometry exploits the structure of chemistry wavefunctions (Slater determinants over Gaussian orbitals) to dramatically reduce circuit depth.

**Always use `sparse_isometry_gf2x` unless the user specifically asks for a comparison.**

## Wavefunction Truncation for State Prep

You don't need the full wavefunction — the top few determinants usually capture almost all the physics:

- Benzene diradical: top 2 determinants capture 98.5% of the wavefunction
- Stretched N₂: top 2 determinants achieve 97% fidelity with the full CASCI solution

The pattern from real examples:

1. Run CASCI or SCI to get the full wavefunction
2. Extract top determinants: `get_top_determinants(max_determinants=2)`
3. Recompute with only those determinants: `run_projected_multi_configuration_calculation` using the determinant list
4. Use the sparse wavefunction for state preparation

This is not just an optimization — it's the standard approach in the real examples.

## Hamiltonian Filtering for Energy Measurement

After state preparation, the qubit Hamiltonian may have 1000+ Pauli terms. The energy estimator (`run_energy_estimator`) automatically groups commuting Pauli terms internally via qubit-wise abelian grouping, so no separate filtering step is needed.

## QPE Configuration

### Required Settings

`run_phase_estimation` has **intentionally invalid defaults** that force you to think about the values:

- `num_bits` defaults to **-1** → MUST be set. Determines precision: 10 bits ≈ 1 mHa, 15 bits ≈ 1 µHa
- `evolution_time` defaults to **0.0** → MUST be set. Compute it, don't guess

If you forget to set these, the tool returns a helpful error message telling you to set them.

### Computing Evolution Time

The evolution time `t` for U = exp(-iHt) should be computed, not guessed:

**Via MCP:** Use the Hamiltonian's spectral properties (Schatten norm). The base formula is `t = π / ||H||`, refined by phase discretization for the chosen number of bits.

**Via Python package:** `compute_evolution_time(qubit_hamiltonian, num_bits=10, target_energy_precision=1e-3)`

### Multi-Trial Strategy

The real N₂ QPE example doesn't run QPE once — it runs **20 trials** with different random seeds and uses **majority voting** to pick the most frequent energy result. This is more robust than a single high-precision run because:

- Individual trials can hit phase aliasing
- Low-precision + many trials is cheaper than high-precision + one trial
- The majority vote naturally rejects outliers

Pattern: `num_bits=10`, `shots_per_bit=3`, `num_trials=20`, seed incremented per trial.

## QPE Sub-Algorithm Configuration

QPE needs three sub-algorithms configured before execution:

1. **Time evolution builder** — how to construct U = exp(-iHt). Options:
   - `trotter` — Trotterized product formula (standard, introduces Trotter error)
   - `matrix_exponential` — exact matrix exponentiation (no Trotter error, but limited to small systems)

2. **Controlled evolution circuit mapper** — how to map U to a controlled circuit:
   - `pauli_sequence` — standard approach

3. **Circuit executor** — how to simulate/execute the circuit:
   - `qdk_full_state_simulator` — full statevector simulation (exact but memory-limited)
   - Settings: `type="cpu"`, `seed=42`

For resource analysis (building circuit without executing): you only need steps 1 and 2 — `run_time_evolution_builder` and `run_controlled_evolution_circuit_mapper`. No executor needed.

For full QPE: call `run_phase_estimation` with sub-algorithms configured inline via `settings`:
```json
{"evolution_builder": {"algorithm_name": "trotter"}, "circuit_mapper": {"algorithm_name": "pauli_sequence"}, "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"}}
```

## Iterative vs Standard QPE

The real examples use **iterative QPE** (Kitaev-style) — one ancilla qubit, repeated measurements. This is more hardware-friendly than standard QFT-based QPE which requires many ancilla qubits.

The `phase_estimation` algorithm name `"iterative"` selects this. If not specified, check the default with `get_algorithm_default_type("phase_estimation")`.
