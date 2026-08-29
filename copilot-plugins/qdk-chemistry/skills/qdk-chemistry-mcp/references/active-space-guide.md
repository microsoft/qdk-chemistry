# Active Space Tool Reference

`run_active_space_selector` creates a wavefunction with a selected orbital
space. The algorithm, selection criteria, and whether to use an active space
are caller-provided scientific choices. Use `list_algorithms` and
`describe_algorithm` for algorithms available at runtime.

## Inputs and Dependencies

- `qdk_valence` requires `charge` when it is selected.
- Entropy-based AutoCAS selection consumes reduced-density-matrix data from a
  prior multi-configuration result; a bare SCF wavefunction does not provide
  that data.
- For an active-space workflow that requires shared spatial orbitals, use a
  restricted HF reference. For open-shell input, pass
  `settings={"method": "hf", "scf_type": "restricted"}` to `run_scf` before
  the selector.
- Pass the returned wavefunction filename to downstream orbital extraction or
  Hamiltonian-construction tools.

When mutual-information visualization is requested, the preceding
multi-configuration calculation must request
`calculate_mutual_information=True`. `calculate_one_rdm=True` and
`calculate_two_rdm=True` provide the RDM outputs consumed by entropy-based
selection.

## Index Handoff

Selectors return orbital indices in absolute molecular-orbital numbering.
`visualize_orbital_entanglement(selected_indices=...)` accepts the same absolute
indices directly; no offset conversion is required.

An empty selector result is a valid tool result. Preserve the returned indices
and diagnostics for the caller's scientific interpretation.

## Related Calls

| Task | Tool |
|---|---|
| Inspect orbital data | `get_orbitals_from_input` |
| Choose an active space | `run_active_space_selector` |
| Produce multi-configuration RDM data | `run_multi_configuration_calculation` |
| View selected orbitals | `visualize_orbitals` |
| View mutual information | `visualize_orbital_entanglement` |