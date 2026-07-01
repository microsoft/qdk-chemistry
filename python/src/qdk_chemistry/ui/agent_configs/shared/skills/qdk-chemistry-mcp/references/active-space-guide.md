# Active Space Selection Guide

Active space selection determines which orbitals get simulated on the quantum computer. Too few orbitals misses important physics. Too many makes the calculation intractable. The goal is compression — reducing a large orbital space down to the orbitals that actually matter for the chemistry.

## Key Concept: AutoCAS Determines the Active Space From Orbital Entanglement

The active space is **not something the user needs to specify**. The standard workflow uses `qdk_autocas_eos` to analyze orbital entanglement entropies from a preliminary SCI calculation and automatically identify which orbitals are strongly correlated. This is the primary mechanism for active space selection — the orbital entanglement analysis *is* the active space decision.

The workflow is: run SCI (with RDMs) → feed the result to AutoCAS → AutoCAS reads the single-orbital entropies → outputs the active orbital indices. The user's role is to verify the result (via the entanglement diagram), not to provide the indices.

## When to Use Active Space Selection

Active space selection is valuable when the full orbital space is too large for quantum simulation. **Ask the user** whether they want the full orbital space or a compressed active space.

**Skip active space selection** when:
- The system is small (up to ~16 spatial orbitals / ~20 qubits after encoding)
- The user wants an exact full-space treatment without approximation
- The system is a model Hamiltonian (Hubbard, PPP, etc.) where the space is already defined

**Use active space selection** when:
- The full orbital space exceeds ~16 spatial orbitals and would produce too many qubits
- The user wants to compress the problem for near-term hardware feasibility
- The system has a clear separation between strongly and weakly correlated orbitals

When using active space selection, AutoCAS (entropy-based orbital entanglement analysis) is the recommended method. It identifies which orbitals are strongly correlated and drops the rest, directly reducing the size of the quantum circuit.

## The Standard Sequence

1. **Get an initial active space with RDMs** — AutoCAS needs reduced density matrices. You can't feed it a bare SCF wavefunction. Two ways to get RDMs:
   - **Direct SCI on a broad space:** If the orbital space is small enough, run `run_multi_configuration_calculation` (SCI) directly on the SCF wavefunction's valence space. This is the most straightforward path.
   - **Valence pre-filter + SCI:** If the full orbital space is too large for SCI, use `run_active_space_selector` with `qdk_valence` first to reduce it, then run SCI on the reduced space.

2. **Run SCI with all three flags:**
   - `calculate_one_rdm=True` — needed for AutoCAS
   - `calculate_two_rdm=True` — needed for AutoCAS
   - `calculate_mutual_information=True` — needed for the entanglement visualization

   All three are important. RDMs alone power AutoCAS, but `calculate_mutual_information` is what produces the mutual information matrix for `visualize_orbital_entanglement`. Without it, the visualization won't work.

3. **Run AutoCAS:** `run_active_space_selector` with `algorithm_name="qdk_autocas_eos"` on the SCI result. This reads the orbital entropies and identifies the strongly correlated subset.

4. **Visualize and verify:** Call `visualize_orbital_entanglement` on the full wavefunction, with the AutoCAS-selected indices as `selected_indices`, to confirm the selection makes chemical sense. Then `visualize_orbitals` to show the selected orbital shapes.

5. **Proceed with the refined space** — rebuild the Hamiltonian on the compressed active space.

## When Valence Selection Is Enough (Rare)

You can skip AutoCAS and use valence selection alone only when:

- The user explicitly requests it, OR
- You have strong domain knowledge that the valence space is the correct active space (e.g., a simple main-group molecule at equilibrium where the frontier orbitals are well-separated)

Even then, running AutoCAS is not harmful — it will likely confirm the valence selection. The cost is one SCI calculation, which is usually cheap compared to the quantum stage.

## Valence Selection as Pre-Filter

```python
run_active_space_selector(algorithm_name="qdk_valence", charge=<charge>)
```

When you use `qdk_valence` with `charge`, the tool automatically computes `num_active_electrons` and `num_active_orbitals` using `compute_valence_space_parameters()`. You don't set them manually.

This is useful as the **first step** when the full orbital space is too large for a direct SCI. It reduces the space to frontier orbitals, making the subsequent SCI + AutoCAS tractable. It's a pre-filter, not the final answer.

## Index Convention for Visualization

AutoCAS returns absolute MO indices (e.g., orbitals 11 and 12 out of the full set). The `visualize_orbital_entanglement` tool accepts these **absolute indices** directly in `selected_indices` — it converts them to diagram positions automatically. No manual offset arithmetic is needed.

Example: If AutoCAS returns orbitals [11, 12], pass `selected_indices=[11, 12]` to the tool.

Best practice: Visualize the FULL wavefunction but highlight the selected orbitals via `selected_indices`.

## Orbital Localization

Use `run_orbital_localization` before the SCI step when localized orbitals improve the chemical picture. The real N₂ example uses MP2 natural orbitals (`qdk_mp2_natural_orbitals`) to localize before SCI — this produces chemically meaningful orbitals that AutoCAS can select more reliably.

Most useful for: transition metal complexes, systems with strongly localized bonds, metal-ligand interactions. Can also help when canonical MOs give a muddled picture of the active space.

## Active Space Sizing

From real examples:

- Benzene diradical: (6 electrons, 6 orbitals) — frontier π system
- Stretched N₂: starts with (10, 10) from valence, refined by AutoCAS to (4, 4) of the strongly correlated σ/σ* orbitals
- H₂: (2, 2) — minimal

General guidance:

- Below (4e, 4o): probably missing important correlation
- Above (16e, 16o): SCI/CASCI becomes very expensive
- The sweet spot depends entirely on the chemistry — that's what AutoCAS helps you find

## Always Visualize After Selection

After ANY active space selection, visualize the selected orbitals with `visualize_orbitals` BEFORE proceeding to Hamiltonian construction. This is the user's chance to verify that the right orbitals were selected for the physics they care about.
