# MCP Tool Reference

This reference describes the operation performed by each QDK Chemistry MCP
tool. The active tool schema is the source for argument types, required fields,
and default values. Algorithm implementations and settings are reported by the
algorithm-discovery tools.

## Workspace and projects

| Tool | Operation and result |
|---|---|
| `bind_workspace` | Binds the server process to an absolute workspace root used to resolve project storage. |
| `list_projects` | Returns project directories in the bound workspace. |
| `create_project` | Creates a project directory and returns its project metadata. |
| `list_project_files` | Returns files in a project with inferred QDK Chemistry data types. |
| `get_summary` | Loads a supported data file and returns its human-readable summary. |

Project file arguments are resolved inside the named project. File-producing
tools validate the output filename against the expected data type. If a valid
output already exists and replacement is disabled, the result status is
`exists`.

## Runtime discovery

| Tool | Operation and result |
|---|---|
| `list_tools` | Returns MCP tool names grouped by functional category. |
| `list_algorithms` | Returns registered algorithm types, implementations, aliases, and defaults. |
| `describe_algorithm` | Returns the selected implementation's settings, types, defaults, and constraints. |
| `get_algorithm_default_type` | Returns the registered default implementation name for an algorithm type. |
| `get_algorithm_default_settings` | Returns the selected implementation's default settings. |
| `list_cache_backends` | Returns registered cache backend names. |
| `list_remote_backends` | Returns registered remote backend names. |
| `describe_backend` | Returns configuration fields accepted by a cache or remote backend. |

An `algorithm_name` argument selects a registered implementation. A `settings`
argument applies setting overrides to that implementation. Nested algorithm
settings are represented as dictionaries containing an algorithm name and its
settings.

## Units and input construction

| Tool | Operation and result |
|---|---|
| `convert_coordinates` | Converts an array of Cartesian coordinates between Bohr and Angstrom and returns the converted array and unit. |
| `convert_energy` | Converts a scalar energy between supported energy units and returns the input and output quantities. |
| `create_structure` | Creates and persists a `Structure` from symbols and coordinates in Bohr, with optional nuclear charges and masses. |
| `create_model_hamiltonian` | Constructs and persists a fermionic Hückel, Hubbard, or PPP Hamiltonian from a lattice and supplied model parameters. |
| `create_spin_model_hamiltonian` | Constructs and persists an Ising or Heisenberg `QubitHamiltonian` from a lattice and supplied couplings and fields. |
| `create_majorana_mapping` | Creates and persists a `MajoranaMapping` for a supplied mode count or a stored Hamiltonian. |

Lattice construction arguments identify the lattice topology and its dimensions
or explicit edges. Model parameter arguments supply on-site, pair, coupling,
field, charge, or potential terms used by the selected model constructor.

## Data extraction and composition

| Tool | Operation and result |
|---|---|
| `get_orbitals_from_input` | Extracts `Orbitals` from a supported stored electronic-structure object and persists them. |
| `get_active_space_indices` | Returns inactive, active, and virtual orbital indices stored on a supported object. |
| `get_ansatz` | Combines a stored `Hamiltonian` and `Wavefunction` into a persisted `Ansatz`. |
| `get_top_determinants` | Returns determinant labels and coefficients from a stored wavefunction. |
| `get_top_configurations` | Returns configurations ranked by the absolute magnitude of their CI coefficients. |

## Electronic-structure calculations

| Tool | Operation and result |
|---|---|
| `run_scf` | Runs the selected HF or DFT self-consistent-field implementation for a structure, charge, multiplicity, and basis, then saves its `Wavefunction`. |
| `run_stability_checker` | Evaluates orbital-rotation stability for a stored wavefunction and returns a stability result. |
| `run_active_space_selector` | Runs the selected active-space selector on a stored wavefunction and saves the resulting wavefunction. |
| `run_orbital_localization` | Runs the selected localizer for supplied alpha and beta orbital index sets and saves the resulting wavefunction. |
| `run_hamiltonian_constructor` | Builds one- and two-electron fermionic Hamiltonian terms from stored orbitals and saves the `Hamiltonian`. |
| `run_dynamical_correlation_calculator` | Runs the selected dynamical-correlation calculator for an ansatz and returns its energy with a saved wavefunction. |
| `run_multi_configuration_calculation` | Runs the selected multi-configuration solver for a Hamiltonian and electron counts, returning an energy and saved wavefunction. |
| `run_multi_configuration_scf` | Jointly optimizes orbitals and CI coefficients from active-space orbitals and saves the resulting wavefunction. |
| `run_projected_multi_configuration_calculation` | Solves a Hamiltonian in the subspace defined by supplied configurations and saves the resulting wavefunction. |
| `run_population_analysis` | Computes per-center populations from a stored structure or wavefunction and returns the analysis data. |
| `run_nuclear_derivative_calculator` | Computes nuclear derivatives for a structure and persists the derivative outputs. |
| `run_geometry_optimization` | Runs a geometry optimizer using its configured derivative calculator and saves the optimized structure. |

Some implementations require data fields produced by another algorithm. These
requirements are properties of the selected implementation and are reported by
its schema or validation errors. Examples include active-space metadata,
restricted orbital spaces, reduced density matrices, or mutual-information
data.

For `run_geometry_optimization`, `algorithm_name` selects the geometry
optimizer, while the derivative implementation is represented by the nested
`derivative_calculator` setting. Derivative and optimization tools save optional
wavefunction or Hessian results only when the corresponding output filename is
supplied and the calculation returns that data.

## Qubit mapping and state construction

| Tool | Operation and result |
|---|---|
| `run_qubit_mapper` | Applies a stored `MajoranaMapping` to a stored fermionic Hamiltonian and saves the resulting `QubitHamiltonian`. |
| `run_term_grouper` | Partitions Pauli terms according to the selected grouping implementation and saves the grouped operator. |
| `run_qubit_hamiltonian_solver` | Diagonalizes a stored `QubitHamiltonian` and returns its ground-state energy and eigenstate vector. |
| `run_state_preparation` | Compiles a stored wavefunction into a quantum `Circuit` and saves it. |
| `run_amplitude_amplification` | Combines stored state-preparation and good-state oracle circuits into an amplitude-amplified circuit and saves it. |
| `run_energy_estimator` | Executes expectation-value estimation for a circuit and one or more qubit Hamiltonians, returning estimated values and variances. |
| `run_hadamard_test` | Executes a Hadamard test for a state-preparation circuit and unitary representation, then saves executor data. |

## Evolution and phase estimation

| Tool | Operation and result |
|---|---|
| `run_time_evolution_builder` | Builds a `TimeEvolutionUnitary` representing $U=\exp(-iHt)$ for a stored qubit Hamiltonian and supplied evolution time, then saves it. |
| `run_controlled_evolution_circuit_mapper` | Maps a stored time-evolution unitary to a controlled quantum circuit and saves the circuit. |
| `run_circuit_executor` | Executes a stored circuit with the selected executor and saves executor data. |
| `run_phase_estimation` | Runs the selected phase-estimation implementation for a state-preparation circuit and qubit Hamiltonian, then saves a `QpeResult`. |
| `run_evolution_circuit_builder` | Builds a circuit for a driven Hamiltonian $H(t)=H_0+f(t)H_1$ from the supplied piecewise-linear drive schedule. |
| `run_hamiltonian_simulation` | Evolves a driven Hamiltonian, measures supplied observables, and saves the returned result pairs. |

Phase-estimation implementations contain nested circuit-builder,
time-evolution, controlled-mapping, and executor settings. Their schemas define
which nested fields are required and the valid values. The stepwise evolution,
controlled-circuit, and executor tools expose those component operations
separately.

## Circuit inspection and resource estimation

| Tool | Operation and result |
|---|---|
| `get_circuit_stats` | Reads a stored circuit and returns logical qubit count, operation counts, and circuit-depth metrics. |
| `estimate_circuit` | Applies optional QDK estimator parameter objects to a stored circuit and returns one estimate or a batch inline. |

These tools inspect the exact circuit supplied by `circuit_filename`.
`get_circuit_stats` reports properties of the logical circuit.
`estimate_circuit` reports logical and physical resources from the circuit's
estimator and does not persist a separate QDK Chemistry result object. The
`resource-estimation` skill documents its parameter objects and the separate
programmatic `qdk.qre` architecture and ISA interface.

## Remote jobs

| Tool | Operation and result |
|---|---|
| `check_remote_job` | Queries backend status and updates the persisted job record. |
| `retrieve_remote_results` | Downloads and deserializes outputs from a completed job into its project. |
| `list_remote_jobs` | Returns persisted jobs, optionally filtered by status. |
| `cancel_remote_job` | Sends a cancellation request for a running job and updates its record. |

All supported `run_*` tools accept execution-infrastructure arguments for
caching and remote execution. A remote call that has not completed within its
return window yields submitted-job metadata instead of the algorithm result.
Job-management operations resolve persisted records within the requesting
workspace and project. Result retrieval deserializes completed job outputs into
that project directory.

## Visualization

| Tool | Operation and result |
|---|---|
| `visualize_circuit` | Loads a stored `Circuit` and returns an interactive circuit view. |
| `visualize_orbital_entanglement` | Loads RDM and mutual-information data from a stored `Wavefunction` and returns an entanglement view. |
| `visualize_molecule` | Loads a stored `Structure` and returns an interactive molecular view. |
| `visualize_orbitals` | Loads a stored `Wavefunction`, generates orbital cube data, and returns an interactive isosurface view. |
| `visualize_scatter_plot` | Converts supplied numeric series into an interactive SVG scatter plot. |

Visualization mechanics and detailed input requirements are documented by the
`visualization` skill. Visualization tools return MCP App content without
changing source artifacts. Orbital-entanglement selections use absolute orbital
indices and are converted to diagram-relative positions for rendering.
