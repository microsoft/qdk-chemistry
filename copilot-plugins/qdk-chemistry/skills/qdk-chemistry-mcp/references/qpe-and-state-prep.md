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
2. Extract top determinants with `get_top_configurations`
3. Recompute with only those determinants: `run_projected_multi_configuration_calculation` using the determinant list
4. Use the sparse wavefunction for state preparation

This is not just an optimization — it's the standard approach in the real examples.

For a QRE upper-bound workflow where direct preparation of the full-valence
ASCI/CI state is infeasible, the default is projected CI using exactly the top
10 available ASCI determinants. Record the cumulative squared amplitude or
overlap retained by those 10 determinants and flag the state-preparation
truncation. Do not increase the default determinant count merely to reach a
target overlap unless the user or campaign explicitly overrides it.

## Hamiltonian Filtering for Energy Measurement

After state preparation, the qubit Hamiltonian may have 1000+ Pauli terms. The energy estimator (`run_energy_estimator`) automatically groups commuting Pauli terms internally via qubit-wise abelian grouping, so no separate filtering step is needed.

## QPE Configuration

### Required Settings

`run_phase_estimation` has **intentionally invalid defaults** that force you to set the QPE policy explicitly:

- `settings.qpe_circuit_builder.num_bits` defaults to **-1** and MUST be set.
- `settings.qpe_circuit_builder.unitary_builder.time` defaults to **0.0** and MUST be set.
- Controlled powers are set on `settings.qpe_circuit_builder.unitary_builder.power` by the QPE circuit builder, or on `run_time_evolution_builder(..., settings={"power": ...})` for Mode A circuit resource analysis.

If you forget to set these, the tool returns a helpful error message telling you to set them.

### Default QPE Policy

Unless the user or campaign specifies a different policy, use this default and record the derived values in the result provenance:

- `target_precision_ha = 0.000797` (0.5 kcal/mol).
- `energy_window_ha = E_max - E_min` from approved energy bounds. If no bounds are supplied, compute the mapped qubit-Hamiltonian coefficient 1-norm `lambda_1 = sum_j(abs(c_j))` for `H = sum_j c_j P_j` and use the conservative window `energy_window_ha = 2 * lambda_1`.
- `evolution_time = 2π / energy_window_ha`. This is equivalent to `π / lambda_1` when using the conservative coefficient-1-norm window.
- `num_bits = ceil(log2(energy_window_ha / target_precision_ha)) + 1`; the extra bit is a guard bit. Do not use the invalid default `-1`.
- Controlled-QPE powers are `1, 2, 4, ..., 2^(num_bits - 1)`. For an upper-bound resource circuit, use `max_power = 2^(num_bits - 1)` and record that it represents the largest scheduled controlled-U.
- For Trotterized time evolution, use `algorithm_name="trotter"`, `power_strategy="repeat"`, `error_bound="commutator"`, and set `target_accuracy` to one tenth of the target energy precision unless a campaign-specific Trotter error budget is provided. Let the builder choose `num_divisions` automatically from `target_accuracy` unless an explicit division count is approved.

If the mapped Hamiltonian coefficient 1-norm or energy bounds are unavailable,
stop with a missing QPE policy decision. Do not blame missing CAS/mapping when
CAS, Hamiltonian, qubit mapping, qubit Hamiltonian, and state-prep artifacts
already exist.

### Computing Evolution Time

The evolution time `t` for U = exp(-iHt) should be computed, not guessed:

**Via MCP:** Use approved energy bounds when available. Otherwise compute the
mapped qubit-Hamiltonian coefficient 1-norm and use the conservative default
`t = π / lambda_1`.

**Via Python package:** There is no public `compute_evolution_time()` helper.
Estimate energy bounds for the target Hamiltonian and set `evolution_time`
explicitly, for example `2π / (E_max - E_min)`, or compute
`lambda_1 = sum(abs(coefficients))` from the mapped Pauli expansion.

### Multi-Trial Strategy

The real N₂ QPE example doesn't run QPE once — it runs **20 trials** with different random seeds and uses **majority voting** to pick the most frequent energy result. This is more robust than a single high-precision run because:

- Individual trials can hit phase aliasing
- Low-precision + many trials is cheaper than high-precision + one trial
- The majority vote naturally rejects outliers

Pattern: use the default `num_bits` rule above, `shots_per_bit=3`, `num_trials=20`, seed incremented per trial.

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

For resource analysis (building circuit without executing): you only need steps 1 and 2 — `run_time_evolution_builder` and `run_controlled_evolution_circuit_mapper`. No executor needed. Build either each scheduled power or the `max_power` upper-bound circuit by passing the power in the time-evolution builder settings:

```json
{
   "evolution_time": 1.0,
   "settings": {
      "algorithm_name": "trotter",
      "power": 16,
      "power_strategy": "repeat",
      "target_accuracy": 0.0000797,
      "error_bound": "commutator"
   }
}
```

For full QPE: call `run_phase_estimation` with sub-algorithms configured inline via `settings`:
```json
{
   "qpe_circuit_builder": {
      "algorithm_name": "qdk_iterative",
      "num_bits": 12,
      "unitary_builder": {
         "algorithm_name": "trotter",
         "time": 1.0,
         "power_strategy": "repeat",
         "target_accuracy": 0.0000797,
         "error_bound": "commutator"
      },
      "controlled_circuit_mapper": {"algorithm_name": "pauli_sequence"}
   },
   "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"},
   "shots_per_bit": 3
}
```

## Iterative vs Standard QPE

The real examples use **iterative QPE** (Kitaev-style) — one ancilla qubit, repeated measurements. This is more hardware-friendly than standard QFT-based QPE which requires many ancilla qubits.

The `phase_estimation` algorithm name `"iterative"` selects this. If not specified, check the default with `get_algorithm_default_type("phase_estimation")`.
