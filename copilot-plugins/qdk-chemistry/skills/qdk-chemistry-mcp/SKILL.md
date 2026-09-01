---
name: qdk-chemistry-mcp
version: 'v2.1.0'
description: 'Describes the QDK Chemistry MCP tool interface, return envelope, and tool categories.'
---

# QDK Chemistry MCP Tools

Call `bind_workspace` before tools that access projects or files. Tool input
schemas define required arguments, accepted values, and defaults.

Tools return a status and either a result or an error message. A status of
`exists` indicates that an output path already contains a valid artifact;
`overwrite` controls replacement by file-producing tools.

## Discovery

- `list_tools` returns tools grouped by category.
- `list_algorithms` returns registered algorithm implementations.
- `describe_algorithm` returns an algorithm's settings schema.
- `get_algorithm_default_type` returns a registered default implementation.
- `get_algorithm_default_settings` returns registered default settings.

## Categories

| Category | Operations |
|---|---|
| `project` | Project and file management |
| `data_inspection` | Stored-artifact inspection |
| `utility` | Unit conversion and backend inspection |
| `input_construction` | Structure and model-Hamiltonian construction |
| `classical_calculation` | Electronic-structure and orbital algorithms |
| `quantum_preparation` | Mapping, state preparation, solvers, and estimators |
| `qpe` | Evolution, controlled-circuit, execution, and phase-estimation algorithms |
| `visualization` | Interactive views of supported artifacts |
| `remote_execution` | Remote-job inspection, retrieval, listing, and cancellation |

File-producing tools return the stored filename. That filename identifies an
input artifact when another tool schema requests the corresponding type.

For the operation, inputs, and result produced by each tool, read
[MCP Tool Reference](./references/tool-reference.md). The reference describes
interface mechanics; it does not select scientific methods or parameter values.

## Projects and Files

`create_project` creates the project directory used by file-producing tools.
`list_projects` and `list_project_files` enumerate stored data. `get_summary`
loads a supported artifact and returns its summary. Output filenames are
validated against the expected data type and project directory.

## Algorithm Calls

`run_*` tools load the artifacts named by their filename arguments, create the
selected algorithm, apply `settings`, execute it, and persist declared output
artifacts. `algorithm_name` selects an installed implementation. An omitted
name uses the registry default. `cache`, `remote`, `remote_config`, and
`remote_timeout` control execution infrastructure rather than scientific
settings.

## Construction and Conversion

Input-construction tools create structures, fermionic model Hamiltonians, spin
Hamiltonians, and mapping objects. Conversion tools convert coordinate and
energy units. Coordinates passed to `create_structure` are in Bohr;
`convert_coordinates` converts supported source units.

| Tool | Operation |
|---|---|
| `create_structure` | Validates symbols and Bohr coordinates and stores a `Structure` |
| `create_model_hamiltonian` | Constructs a fermionic lattice-model `Hamiltonian` from the named model, lattice, and supplied model parameters |
| `create_spin_model_hamiltonian` | Constructs a lattice spin `QubitHamiltonian` from the named model, lattice, couplings, and fields |
| `create_majorana_mapping` | Stores a `MajoranaMapping` for a supplied mode count or Hamiltonian |
| `convert_coordinates` | Converts a coordinate array between Bohr and Angstrom |
| `convert_energy` | Converts a scalar among the energy units accepted by its schema |

## Classical Calculation Tools

| Tool | Operation and result |
|---|---|
| `run_scf` | Runs the selected HF or DFT SCF implementation for a structure, charge, multiplicity, and basis; returns energy and stores a `Wavefunction` |
| `run_stability_checker` | Evaluates orbital-rotation stability for a stored wavefunction and returns a stability result |
| `run_active_space_selector` | Applies the selected orbital-selection implementation and stores the resulting wavefunction |
| `run_orbital_localization` | Applies an orbital localizer to supplied orbital index ranges and stores the resulting wavefunction |
| `run_hamiltonian_constructor` | Builds and stores a fermionic Hamiltonian from stored orbitals |
| `run_dynamical_correlation_calculator` | Applies the selected correlation calculator to an ansatz and returns its energy and wavefunction |
| `run_multi_configuration_calculation` | Runs the selected multi-configuration solver for a Hamiltonian and electron counts; returns energy and stores a wavefunction |
| `run_multi_configuration_scf` | Optimizes orbitals and configuration coefficients from active-space orbitals and stores a wavefunction |
| `run_projected_multi_configuration_calculation` | Solves in the subspace defined by supplied determinants and stores a wavefunction |
| `run_population_analysis` | Computes per-center populations from a structure or wavefunction |
| `run_nuclear_derivative_calculator` | Computes nuclear derivatives and stores derivative outputs |
| `run_geometry_optimization` | Runs the selected geometry optimizer and stores the optimized structure |

`get_orbitals_from_input`, `get_active_space_indices`, `get_ansatz`,
`get_top_determinants`, and `get_top_configurations` transform or inspect
stored electronic-structure artifacts without selecting a scientific method.

## Quantum Artifacts

`run_qubit_mapper` applies a stored mapping to a fermionic Hamiltonian and
returns the saved filename together with the Hamiltonian's `core_energy`.
The saved `QubitHamiltonian` intentionally excludes that constant. Energies
returned by the qubit solver, energy estimator, and phase estimation are mapped
energies, so compute total molecular energies as
`mapped_energy + core_energy`.
`run_state_preparation` compiles wavefunction data into a circuit.
Time-evolution and controlled-circuit tools persist their intermediate
artifacts. `run_phase_estimation` returns a phase-estimation result. Circuit
statistics and resource estimation inspect a supplied circuit and return
different result structures.

| Tool | Operation and result |
|---|---|
| `run_qubit_mapper` | Applies a stored `MajoranaMapping` to a fermionic Hamiltonian, stores a `QubitHamiltonian`, and returns its filename and the excluded `core_energy` |
| `run_term_grouper` | Partitions Pauli terms with the selected grouping algorithm and stores the grouped operator |
| `run_qubit_hamiltonian_solver` | Diagonalizes a qubit Hamiltonian and returns a mapped eigenvalue, excluding `core_energy`, and eigenstate data |
| `run_state_preparation` | Compiles a wavefunction with the selected implementation and stores a circuit |
| `run_amplitude_amplification` | Combines state-preparation and good-state oracle circuits into an amplitude-amplified circuit |
| `run_energy_estimator` | Executes the selected expectation estimator for a circuit and one or more Hamiltonians; estimated energies exclude `core_energy` |
| `run_hadamard_test` | Executes a Hadamard-test implementation for a state-preparation circuit and unitary representation |
| `run_evolution_circuit_builder` | Builds a circuit for a base Hamiltonian plus a piecewise-linear driven term |
| `run_hamiltonian_simulation` | Evolves the supplied driven Hamiltonian and returns measured observable results |
| `run_time_evolution_builder` | Builds and stores a `TimeEvolutionUnitary` for a Hamiltonian and evolution time |
| `run_controlled_evolution_circuit_mapper` | Maps a stored time-evolution unitary to a controlled circuit |
| `run_circuit_executor` | Executes a stored circuit and stores executor data |
| `run_phase_estimation` | Runs the selected phase-estimation implementation and stores a `QpeResult` whose energies exclude `core_energy` |
| `get_circuit_stats` | Returns logical register, operation-count, and circuit-depth data |
| `estimate_circuit` | Applies QDK estimator parameters to a stored circuit and returns the estimate inline |

Nested algorithm settings are dictionaries whose `algorithm_name` selects the
nested implementation. The remaining keys configure that implementation.
`describe_algorithm` reports the accepted keys and types for each installed
implementation.

## Errors and Existing Outputs

Input loading, filename validation, settings validation, algorithm execution,
and serialization failures return `status: "error"` with a message and, for
raised exceptions, an error type. A submitted remote call returns
`status: "submitted"` and job metadata. Existing valid output artifacts return
`status: "exists"` unless replacement was requested.
