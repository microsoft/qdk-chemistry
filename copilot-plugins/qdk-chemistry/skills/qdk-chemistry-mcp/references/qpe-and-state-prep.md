# QPE and State Preparation Reference

## State Preparation

`run_state_preparation` converts a stored wavefunction into a circuit. The
selected state-preparation algorithm and any wavefunction reduction are caller
choices. Use `list_algorithms` and `describe_algorithm` to inspect algorithms
and their settings in the active installation.

`get_top_configurations` returns determinant data that can be supplied to
`run_projected_multi_configuration_calculation` when a caller needs a reduced
wavefunction. Record the returned configuration data and selected inputs so the
generated circuit can be interpreted later.

`run_energy_estimator` accepts a circuit and qubit-Hamiltonian filenames. It
performs its own commuting-term grouping; callers do not supply a grouping.
Use `run_term_grouper` only when a grouped QubitOperator is needed as a
separate artifact.

`run_amplitude_amplification` builds a Circuit from saved state-preparation and
good-state oracle Circuits. It is a separate circuit-construction operation;
the oracle algorithm used internally by amplitude amplification is configured
through its nested settings rather than a distinct MCP call.

## QPE Settings

`run_phase_estimation` requires explicit nested values for:

- `settings.qpe_circuit_builder.num_bits`; its default is `-1`.
- `settings.qpe_circuit_builder.unitary_builder.time`; its default is `0.0`.

These values define phase resolution and the Hamiltonian-to-phase mapping for
the requested calculation. Obtain them from the user, an approved workflow, or
a separately documented scientific policy. The tool returns an error until both
are supplied.

QPE sub-algorithms are configured under `settings`:

```json
{
  "qpe_circuit_builder": {
    "algorithm_name": "qdk_iterative",
    "num_bits": 12,
    "unitary_builder": {
      "algorithm_name": "trotter",
      "time": 1.0
    },
    "controlled_circuit_mapper": {
      "algorithm_name": "pauli_sequence"
    }
  },
  "circuit_executor": {
    "algorithm_name": "qdk_sparse_state_simulator"
  }
}
```

Algorithm names and optional settings in this shape are installation dependent.
Query them with `list_algorithms`, `describe_algorithm`, and the default-setting
tools before sending overrides.

## QPE Endpoints

Circuit resource analysis and phase estimation are different calls:

| Endpoint | Tools |
|---|---|
| Circuit construction | `run_time_evolution_builder` then `run_controlled_evolution_circuit_mapper` |
| Circuit execution | `run_circuit_executor` |
| Phase estimation | `run_phase_estimation` with nested sub-algorithm settings |
| Driven evolution circuit | `run_evolution_circuit_builder` |
| Driven evolution and measurement | `run_hamiltonian_simulation` |
| Hadamard-test measurement | `run_hadamard_test` |

`run_time_evolution_builder` builds $U = \exp(-iHt)$. For a powered circuit,
pass the chosen `power` in that builder's `settings`. The controlled-evolution
mapper receives the resulting circuit. `get_circuit_stats` reports logical
circuit metrics; `run_resource_estimation` reports a physical-resource Pareto
frontier for the supplied circuit.

## Parameter Discovery

Use the input schema for required filenames and `settings` structure. Use
`get_algorithm_default_type("phase_estimation")` and
`get_algorithm_default_settings(...)` to inspect active defaults. Tool results,
not this reference, are the source for runtime-supported algorithms and settings.
