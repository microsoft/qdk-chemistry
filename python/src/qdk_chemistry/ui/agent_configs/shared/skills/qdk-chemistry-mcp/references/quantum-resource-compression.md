# Quantum Resource Compression Strategies

Practical strategies for reducing qubit count, circuit depth, and gate count in QDK Chemistry workflows while preserving chemical accuracy.

## Classical Compression (before any quantum step)

### 1. Active Space Minimization

- Start with the smallest defensible active space — typically `qdk_valence`.
- Only enlarge if orbital entanglement analysis reveals strong correlations outside the initial space.
- Fewer active orbitals → linearly fewer qubits and an exponentially smaller Hilbert space (Jordan-Wigner maps one spin-orbital to one qubit, i.e. two qubits per spatial orbital).
- Rule of thumb: ~12 active orbitals (24 spin-orbitals → 24 qubits) is near the practical limit for full-circuit simulation on current hardware.

### 2. Orbital Localization

- Run orbital localization (Pipek-Mezey or MP2 natural orbitals) before active space selection.
- Particularly effective for molecules with spatially separable regions (e.g., stretched bonds, weakly interacting fragments).
- Localized orbitals often allow a smaller active space for the same accuracy because correlation becomes more local.

### 3. Entropy-Driven Active Space Refinement

- After an initial SCI/CASCI with `calculate_one_rdm=True`, `calculate_two_rdm=True`, and `calculate_mutual_information=True`, use `qdk_autocas_eos` to prune weakly entangled orbitals.
- Inspect the entanglement chord diagram: orbitals with negligible mutual information to all others are safe candidates for removal.
- This is an iterative process — refine, re-run SCI, check entanglement again.

### 4. Wavefunction Sparsification

- Use `get_top_configurations` to identify dominant determinants by CI coefficient magnitude.
- Feed only the top 10–50 determinants into `run_projected_multi_configuration_calculation`.
- This often captures >99% of the correlation energy while producing dramatically shallower state-preparation circuits.
- **This is the single most impactful reduction for state-preparation circuit depth.** Always sparsify before `run_state_preparation`.

## Quantum Compression (circuit-level)

### 5. Hamiltonian Trimming

- The energy estimator groups commuting Pauli terms internally. For time evolution circuits, the number of Trotter terms is determined by the Hamiltonian size.
- If the qubit Hamiltonian has many negligible terms, consider reducing the active space or using a tighter Hamiltonian construction threshold to keep only significant integrals.

### 6. Qubit Mapping Awareness

- Jordan-Wigner encoding: 1 spin-orbital → 1 qubit (2 qubits per spatial orbital).
- Gate count scales superlinearly with qubit count — removing even one orbital from the active space has an outsized impact.
- Consider qubit reduction techniques if available (e.g., symmetry-based tapering).

### 7. Trotter Parameter Tuning

- Start conservative: `order=2, num_trotter_steps=1`.
- Increase Trotter steps or order only if the estimated Trotter error exceeds the target accuracy.
- Higher order or more steps multiplies circuit depth linearly — use the minimum that meets accuracy requirements.
- `evolution_time` is derived from the Hamiltonian's spectral range (t = π / ‖H‖) — smaller time → fewer repetitions needed for the same precision.

### 8. QPE Precision Trade-offs

- Each additional precision bit roughly doubles the total controlled-U cost (total invocations ≈ 2^B − 1).
- Start with fewer bits (e.g., 5) for initial exploration; increase only if the energy resolution is insufficient.
- Iterative QPE (1 ancilla qubit) is more hardware-friendly than QFT-based QPE (B ancilla qubits).

## When to Suspect Over-Specification

- Circuit depth exceeds ~10,000 gates → likely needs tighter compression
- More than ~30 logical qubits → beyond current simulator practical limits
- State preparation circuit dominates total depth → wavefunction needs more sparsification
- Time evolution unitary has thousands of Pauli terms → increase Hamiltonian trimming tolerance
- QPE with >10 precision bits on a >15-qubit system → likely infeasible, reduce active space or precision
