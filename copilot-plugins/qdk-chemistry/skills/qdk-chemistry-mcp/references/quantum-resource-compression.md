# Quantum Resource Inputs Reference

Circuit cost depends on the Hamiltonian, wavefunction, mapping, circuit
algorithm, and algorithm settings. This reference describes MCP artifacts that
expose those inputs; it does not select accuracy targets, active-space sizes,
determinant counts, or resource budgets.

## Artifact Relationships

| Input or result | Tool |
|---|---|
| Selected-orbital wavefunction | `run_active_space_selector` |
| Reduced wavefunction from chosen configurations | `run_projected_multi_configuration_calculation` |
| Fermion-to-qubit mapping | `create_majorana_mapping` |
| Qubit Hamiltonian | `run_qubit_mapper` |
| State-preparation circuit | `run_state_preparation` |
| Time-evolution circuit | `run_time_evolution_builder` |
| Controlled-evolution circuit | `run_controlled_evolution_circuit_mapper` |
| Logical circuit metrics | `get_circuit_stats` |
| Physical resource Pareto points | `run_resource_estimation` |

Jordan-Wigner maps one spin orbital to one qubit. Other mappings and settings
are discovered through the mapping schema and algorithm catalog.

## Reporting Boundaries

`get_circuit_stats` reports properties of one supplied circuit.
`run_resource_estimation` returns Pareto points for one supplied circuit under
returned estimator assumptions. Mapping statistics, logical circuit statistics,
state-preparation estimates, and QPE circuit estimates describe different
artifacts and should be reported separately.

For time evolution, `settings` belongs to
`run_time_evolution_builder`; inspect the selected implementation's accepted
keys with `describe_algorithm`. For phase estimation, pass phase-resolution and
evolution-time choices through nested `run_phase_estimation` settings.
