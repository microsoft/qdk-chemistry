# Worked Example: Stretched N₂ with iQPE

Based on the real `qpe_stretched_n2.ipynb` notebook in the QDK Chemistry repo.

## The Question

"Compute the ground state energy of N₂ at a stretched bond distance using iterative QPE."

This is an energy computation — run full QPE, get an eigenvalue. Stretched N₂ is multi-reference (the σ bond is partially broken), which makes it a genuine test of the quantum workflow.

## Reasoning Through the Decisions

### Why this is interesting

N₂ at equilibrium is well-described by a single Slater determinant (RHF works fine). But at stretched distances (~1.27 Å), the σ and σ* orbitals become near-degenerate, and the wavefunction becomes a superposition of multiple configurations. This is exactly where classical single-reference methods fail and quantum approaches add value.

### Choosing basis set: cc-pVDZ

STO-3G would be faster but only qualitative. cc-pVDZ gives quantitatively meaningful results without being prohibitively expensive. For a benchmark comparison, cc-pVTZ would be better but significantly slower.

### Choosing SCF method: RHF → check stability

Start with RHF (simpler). If the stability check reveals an instability (likely for stretched bonds), switch to UHF. In the real example, the stretched N₂ is handled correctly, but the stability check is the safety net.

### Active space strategy: Valence → Localize → SCI → AutoCAS

This is the most rigorous approach because stretched N₂ is multi-reference:

1. **Valence selection first** — gets a reasonable starting space from the frontier orbitals. For N₂ with charge=0, `compute_valence_space_parameters` returns (10, 10).
2. **Orbital localization** — use `qdk_mp2_natural_orbitals` to produce orbitals that are chemically meaningful (localized σ, σ*, π, π* etc.) rather than delocalized canonical MOs. This improves the quality of the subsequent SCI.
3. **ASCI (Selected CI)** — run on the localized space with `calculate_one_rdm=True`, `calculate_two_rdm=True`, `calculate_mutual_information=True`. Uses `macis_asci` with `core_selection_strategy="fixed"` (works best when starting from an HF reference).
4. **AutoCAS-EOS refinement** — automatically identifies which orbitals are strongly entangled. For stretched N₂, this typically selects 4 out of the 10 valence orbitals — the σ/σ* pair.

**Why not just use valence selection alone?** Because all 10 valence orbitals aren't equally important. The σ bond breaking involves mainly 4 orbitals. AutoCAS finds exactly these, reducing the quantum circuit size without losing the essential physics.

### State preparation: top 2 determinants + sparse isometry

The SCI wavefunction has many determinants, but the top 2 capture ~97% of the state. Using only 2 determinants with `sparse_isometry_gf2x` produces a circuit with ~1 fine rotation — orders of magnitude cheaper than encoding the full wavefunction.

### QPE strategy: iterative, 10 bits, 20 trials × 3 shots/bit

- **Iterative (Kitaev-style):** One ancilla qubit, repeated measurements. More hardware-friendly than QFT-based QPE.
- **10 bits:** Gives ~1 mHa precision — enough to resolve the energy meaningfully.
- **3 shots/bit:** Low shot count per measurement, compensated by multiple independent trials.
- **20 trials with majority voting:** Each trial uses a different random seed. The most frequently observed energy is the final answer. This is more robust than a single high-shot trial because it naturally rejects aliased phases.

## MCP Tool Sequence

```python
1. create_structure(project_name="n2_stretched", symbols=["N","N"],
     coordinates_json="[[0,0,0],[0,0,2.4]]",  ← Bohr
     filename_to_save="n2.structure.json")

2. run_scf(project_name="n2_stretched",
     structure_filename="n2.structure.json",
     out_wavefunction_filename="n2_hf.wavefunction.json",
     charge=0, spin_multiplicity=1, basis_set="cc-pvdz")
   → VISUALIZE: molecule

3. run_stability_checker(project_name="n2_stretched",
     wavefunction_filename="n2_hf.wavefunction.json")

4. run_active_space_selector(project_name="n2_stretched",
     wavefunction_filename="n2_hf.wavefunction.json",
     out_wavefunction_filename="n2_valence.wavefunction.json",
     algorithm_name="qdk_valence", charge=0)
   → VISUALIZE: orbitals

5. run_orbital_localization(project_name="n2_stretched",
     wavefunction_filename="n2_valence.wavefunction.json",
     out_wavefunction_filename="n2_localized.wavefunction.json")

6. run_hamiltonian_constructor(project_name="n2_stretched",
     orbitals_filename="n2_localized.wavefunction.json",
     out_hamiltonian_filename="n2.hamiltonian.json")

7. run_multi_configuration_calculation(project_name="n2_stretched",
     hamiltonian_filename="n2.hamiltonian.json",
     out_wavefunction_filename="n2_sci.wavefunction.json",
     settings={"calculate_one_rdm": true, "calculate_two_rdm": true,
               "calculate_mutual_information": true,
               "core_selection_strategy": "fixed"})
   → VISUALIZE: orbital entanglement

8. run_active_space_selector(project_name="n2_stretched",
     wavefunction_filename="n2_sci.wavefunction.json",
     out_wavefunction_filename="n2_autocas.wavefunction.json",
     algorithm_name="qdk_autocas_eos")
   → VISUALIZE: orbitals (refined selection)
   → VISUALIZE: entanglement with selected_indices highlighting

9-10. Rebuild Hamiltonian on refined space, run qubit mapper

11. run_state_preparation(...)
    → VISUALIZE: circuit

12. run_phase_estimation(settings={"num_bits": 10, "evolution_time": <computed>,
       "evolution_builder": {...}, "circuit_mapper": {...}, "circuit_executor": {...}})
    → 20 trials with different seeds
    → majority voting for final energy
    → VISUALIZE: circuit
```

## What Would Be Different For

**N₂ at equilibrium (not stretched):**

- RHF would be fine (stable)
- Still run SCI + AutoCAS — it would confirm the valence space is correct and might even find a smaller sufficient space
- Skip orbital localization (canonical MOs are fine at equilibrium)
- Fewer QPE trials needed (cleaner spectrum)

**A transition metal complex:**

- Would use orbital localization (absolutely critical for d-orbitals)
- Might need UHF or ROHF for SCF
- Larger active space potentially needed
- Convergence more challenging — may need multiple attempts

**Only wanting resource estimates (not energy):**

- Same steps 1-11
- Then `run_time_evolution_builder` + `run_controlled_evolution_circuit_mapper`
- Extract resource profile (qubits, depth, T-count) — don't run QPE
- No multi-trial needed — it's a single circuit analysis
