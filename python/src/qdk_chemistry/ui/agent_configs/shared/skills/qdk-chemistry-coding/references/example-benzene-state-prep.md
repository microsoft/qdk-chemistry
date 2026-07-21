# Worked Example: Benzene Diradical State Preparation

Based on the real `state_prep_energy.ipynb` notebook in the QDK Chemistry repo.

## The Question

"Prepare a quantum state for the benzene diradical and estimate its energy."

This is about state preparation and energy measurement — not QPE. The benzene diradical is a multi-reference system (two unpaired electrons in the π system) where Hartree-Fock fails, making it a good test for quantum state preparation.

## Reasoning Through the Decisions

### Why benzene diradical

It's produced by Bergman cyclization — a real reaction in organic chemistry. The diradical has two nearly degenerate configurations (the two unpaired electrons can be on different carbon pairs), so the wavefunction is fundamentally a superposition. HF gives the wrong answer because it can only represent one configuration. This is exactly the kind of system where quantum computing offers an advantage.

### Choosing active space: (6, 6) valence → confirmed by AutoCAS

The interesting chemistry lives in the 6 π electrons across 6 frontier orbitals. The molecule has ~100 total orbitals, but only these 6 matter for the diradical character. Valence selection with `num_active_electrons=6, num_active_orbitals=6` gives a good initial space.

Running SCI + AutoCAS on this space would confirm the selection — the π orbitals will show high entropy, and the σ framework will show low entropy. AutoCAS would keep the same 6 orbitals. The analysis is quick and validates the chemical intuition, so there's no reason to skip it even when you're confident in the valence space.

### State prep: sparse isometry vs general isometry

The real example explicitly compares both methods to demonstrate the efficiency advantage:

- **General isometry** (`qiskit_regular_isometry`): Encodes any state vector. Produces thousands of fine rotation gates. Circuit depth ~1000+.
- **Sparse isometry** (`sparse_isometry_gf2x`): Exploits the Slater determinant structure of chemistry wavefunctions. Produces **1 fine rotation**. Circuit depth ~50.

Same quantum state, dramatically different cost. This is why `sparse_isometry_gf2x` is the production choice.

### Top-2 determinant truncation

The CASCI wavefunction has many determinants, but the top 2 capture 98.5% of the norm. Using `projected_multi_configuration_calculator` with just those 2 determinants gives a sparse wavefunction that produces extremely compact circuits.

This isn't an approximation hack — it's the physically correct insight that the diradical character is dominated by two configurations (the two radical arrangements).

### Hamiltonian filtering

The full qubit Hamiltonian has 1247 Pauli terms. After filtering based on the prepared state (using `filter_and_group_pauli_ops_from_wavefunction`), only 2 measurement groups remain. That's a 99.8% reduction in measurement cost.

## MCP Tool Sequence

```python
1. create_structure(project_name="benzene_dirad",
     symbols=["C","C","C","C","C","C","H","H","H","H"],
     coordinates_json="[...]",  ← Bohr coordinates
     filename_to_save="bd.structure.json")
   → VISUALIZE: molecule

2. run_scf(project_name="benzene_dirad",
     structure_filename="bd.structure.json",
     out_wavefunction_filename="bd_hf.wavefunction.json",
     charge=0, spin_multiplicity=1, basis_set="cc-pvdz")

3. run_stability_checker(project_name="benzene_dirad",
     wavefunction_filename="bd_hf.wavefunction.json")

4. run_active_space_selector(project_name="benzene_dirad",
     wavefunction_filename="bd_hf.wavefunction.json",
     out_wavefunction_filename="bd_active.wavefunction.json",
     algorithm_name="qdk_valence", charge=0)
   → VISUALIZE: orbitals (verify the π orbitals are selected)

5. run_hamiltonian_constructor(project_name="benzene_dirad",
     orbitals_filename="bd_active.wavefunction.json",
     out_hamiltonian_filename="bd.hamiltonian.json")

6. run_multi_configuration_calculation(project_name="benzene_dirad",
     hamiltonian_filename="bd.hamiltonian.json",
     out_wavefunction_filename="bd_casci.wavefunction.json",
     settings={"calculate_one_rdm": true, "calculate_two_rdm": true,
               "calculate_mutual_information": true,
               "core_selection_strategy": "fixed"})
   → Report CASCI energy

7. get_top_configurations(project_name="benzene_dirad",
     wavefunction_filename="bd_casci.wavefunction.json",
     max_determinants=2)
   → Report that top 2 dets capture 98.5%

8. run_projected_multi_configuration_calculation(project_name="benzene_dirad",
     hamiltonian_filename="bd.hamiltonian.json",
     out_wavefunction_filename="bd_sparse.wavefunction.json",
     configurations_json=<top_2_from_step_7>)

9. run_state_preparation(project_name="benzene_dirad",
     wavefunction_filename="bd_sparse.wavefunction.json",
     out_circuit_filename="bd_sparse.circuit.json",
     algorithm_name="sparse_isometry_gf2x")
   → VISUALIZE: circuit (note: ~1 fine rotation)

10. create_majorana_mapping(project_name="benzene_dirad",
     hamiltonian_filename="bd.hamiltonian.json",
     out_mapping_filename="bd.majorana_mapping.json")

11. run_qubit_mapper(project_name="benzene_dirad",
      hamiltonian_filename="bd.hamiltonian.json",
     mapping_filename="bd.majorana_mapping.json",
      out_qubit_hamiltonian_filename="bd.qubit_hamiltonian.json")

12. run_energy_estimator(project_name="benzene_dirad",
      circuit_filename="bd_sparse.circuit.json",
      qubit_hamiltonian_filename="bd.qubit_hamiltonian.json",
      # Pauli term grouping is handled internally by the energy estimator
      out_energy_result_filename="bd.energy_result.json",
      total_shots=250000)
    → Report estimated energy ± uncertainty
    → Compare vs CASCI reference
```

## What Would Be Different For

**A simpler molecule (H₂O at equilibrium):**

- Still use valence selection, but the system is single-reference — CASCI might not be necessary
- State prep produces even simpler circuits
- No diradical character → top 1 determinant might capture 99%+

**Wanting resource estimates instead of energy:**

- Same steps 1-9 for state preparation
- Then `run_time_evolution_builder` + `run_controlled_evolution_circuit_mapper`
- Extract circuit resource profile
- Skip energy estimation entirely

**Wanting QPE instead of shot-based energy:**

- Same steps 1-9
- Then configure QPE sub-algorithms + `run_phase_estimation`
- QPE gives a single eigenvalue; energy estimator gives expectation value ± standard deviation
- QPE is more expensive but gives a sharper answer
