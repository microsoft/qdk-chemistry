# Python Reference (`qdk_chemistry`)

All valid `create()` calls, utility functions, and file I/O patterns for writing reproducible Python scripts.

## The `create()` Factory

```python
from qdk_chemistry.algorithms import create
```

Everything flows through `create(algorithm_type, algorithm_name=None, **kwargs)`.

## Algorithm Types and Names

### SCF Solver

```python
scf = create("scf_solver")  # default: "pyscf"
E_hf, wfn_hf = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
```

Settings: `method` ("hf", "dft"), `functional` (for DFT), `max_iterations`

### Active Space Selector

```python
# Valence (auto-computes electrons/orbitals from charge)
selector = create("active_space_selector", "qdk_valence",
                  num_active_electrons=6, num_active_orbitals=6)
wfn_active = selector.run(wfn_hf)

# Entropy-based refinement (needs RDMs from prior SCI)
autocas = create("active_space_selector", "qdk_autocas_eos")
wfn_refined = autocas.run(wfn_sci)
```

### Orbital Localizer

```python
localizer = create("orbital_localizer", "qdk_mp2_natural_orbitals")
wfn_loc = localizer.run(wfn_active, *active_indices)
```

### Hamiltonian Constructor

```python
ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(orbitals)
```

### Multi-Configuration Calculator

```python
# Exact CASCI (small active spaces)
mc = create("multi_configuration_calculator", "macis_cas")
E_cas, wfn_cas = mc.run(hamiltonian, n_alpha=3, n_beta=3)

# Selected CI (larger active spaces)
sci = create("multi_configuration_calculator", "macis_asci",
             calculate_one_rdm=True, calculate_two_rdm=True,
             calculate_mutual_information=True)
E_sci, wfn_sci = sci.run(hamiltonian, n_alpha, n_beta)
```

ASCI settings: `core_selection_strategy="fixed"` (when starting from HF), `max_roots`

### Projected Multi-Configuration Calculator

```python
pmc = create("projected_multi_configuration_calculator", "macis_pmc")
E_sparse, wfn_sparse = pmc.run(hamiltonian, list(top_determinants.keys()))
```

Used for wavefunction truncation — recomputes energy for a subset of determinants.

### State Preparation

```python
# Chemistry-optimized (ALWAYS prefer this)
state_prep = create("state_prep", "sparse_isometry_gf2x")
circuit = state_prep.run(wfn_sparse)

# General (for comparison only)
state_prep = create("state_prep", "qiskit_regular_isometry")
circuit = state_prep.run(wfn_sparse)
```

### Qubit Mapper

```python
mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
qubit_H = mapper.run(hamiltonian)
```

Encodings: `"jordan-wigner"` (default), `"bravyi-kitaev"`, `"parity"`

### Dynamical Correlation Calculator

```python
mp2 = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
E_mp2, wfn_mp2 = mp2.run(wfn_hf)
```

### Qubit Hamiltonian Solver

```python
solver = create("qubit_hamiltonian_solver", "qdk_sparse_matrix_solver")
E_exact = solver.run(qubit_H)
```

Exact classical diagonalization of the qubit Hamiltonian.

### Time Evolution Builder

```python
evolution_builder = create("time_evolution_builder", "trotter")
```

Options: `"trotter"` (standard), `"matrix_exponential"` (exact, small systems only)

### Controlled Evolution Circuit Mapper

```python
circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
```

### Circuit Executor

```python
executor = create("circuit_executor", "qdk_full_state_simulator", type="cpu", seed=42)
```

### Phase Estimation

```python
iqpe = create("phase_estimation", "iterative",
              num_bits=10, evolution_time=0.5, shots_per_bit=3)
# The default nested algorithms are: trotter + pauli_sequence +
# qdk_sparse_state_simulator. Update iqpe.settings() before run() if needed.
result = iqpe.run(
    state_preparation=circuit,
    qubit_hamiltonian=qubit_H,
)
```

### Energy Estimator

```python
estimator = create("energy_estimator", "qdk")
energy_result, measurement_data = estimator.run(
    circuit=circuit,
    qubit_hamiltonian=qubit_H,
    total_shots=250000,
)
```

### Resource Estimator

```python
re = create("resource_estimator")  # default: "qdk_qre_v3"
# re.settings().set("error_budget", 0.001)  # optional: default is 0.001
resource_data = re.run(circuit)
print(resource_data.get_summary())
# Access fields: resource_data.logical_counts.num_qubits, .t_count
#                resource_data.physical_counts.physical_qubits, .runtime
#                resource_data.logical_qubit.code_distance
#                resource_data.error_budget.logical, .rotations, .tstates
```

## Utility Functions

```python
from qdk_chemistry.utils import compute_valence_space_parameters
num_e, num_o = compute_valence_space_parameters(wfn_hf, charge=0)
# Auto-determines frontier orbital count from the SCF wavefunction
```

```python
import numpy as np

# qdk_chemistry does not expose a public compute_evolution_time() helper.
# Set evolution_time explicitly from estimated energy bounds.
E_min, E_max = -1.0, 1.0  # user-supplied bounds for the target Hamiltonian
if E_max <= E_min:
    raise ValueError("E_max must be greater than E_min")
evolution_time = 2 * np.pi / (E_max - E_min)
```

```python
# qdk_chemistry does not expose a public prepare_top_dets_trial_state() helper.
# Build a sparse trial state from the leading determinants, then compile it.
top_determinants = wfn_full.get_top_determinants(max_determinants=2)
pmc = create("projected_multi_configuration_calculator", "macis_pmc")
E_trial, wfn_trial = pmc.run(hamiltonian, list(top_determinants.keys()))
state_prep = create("state_prep", "sparse_isometry_gf2x")
trial_circuit = state_prep.run(wfn_trial)
```

## File I/O

```python
# Wavefunction
wfn.to_json_file("wfn.wavefunction.json")
wfn = Wavefunction.from_json_file("wfn.wavefunction.json")

# Structure
structure = Structure.from_xyz_file("mol.xyz")

# QPE Result
result.to_json_file("result.qpe_result.json")
result = QpeResult.from_json_file("result.qpe_result.json")

# Generic
obj.to_file(path, "json")  # or "hdf5"
```

## Hamiltonian Grouping

There is no public `filter_and_group_pauli_ops_from_wavefunction` helper in the SDK.
For shot-based energy estimation, pass the full `qubit_H` to `estimator.run(...)`.
The estimator groups commuting Pauli terms internally via `qubit_hamiltonian.group_commuting(qubit_wise=True)`.

## Multi-Trial QPE Pattern

```python
from collections import Counter
NUM_TRIALS = 20
energies = Counter()
for trial in range(NUM_TRIALS):
    executor = create("circuit_executor", "qdk_full_state_simulator",
                     type="cpu", seed=42 + trial)
    result = iqpe.run(...)
    energy = result.raw_energy + hamiltonian.get_core_energy()
    energies[round(energy, 6)] += 1

best_energy, count = energies.most_common(1)[0]
```

## Core Energy Offset

QPE and energy estimator results are typically relative to the active space. To get the total molecular energy, add the core (frozen orbital) energy:

```python
total_energy = result.raw_energy + hamiltonian.get_core_energy()
```
