"""Hadamard test usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default Hadamard test algorithm
hadamard_test = create("hadamard_test", "qdk")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
from qdk_chemistry.data import AlgorithmRef

# Configure Hadamard test settings
hadamard_test = create("hadamard_test", "qdk", test_basis="X")
hadamard_test.settings().set(
    "controlled_circuit_mapper",
    AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
)
hadamard_test.settings().set(
    "circuit_executor",
    AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
)

# Change measurement basis on the same algorithm implementation if needed
# hadamard_test.settings().set("test_basis", "Y")
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import MajoranaMapping, Structure

# 1. Setup molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# 2. SCF
scf_solver = create("scf_solver")
_, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# 3. Hamiltonian construction
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# 4. Multi-configuration calculation (reference state)
cas_solver = create("multi_configuration_calculator")
_, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# 5. Qubit mapping
n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
qubit_mapper = create("qubit_mapper")
qubit_hamiltonian = qubit_mapper.run(
    hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals)
)

# 6. State preparation
state_prep = create("state_prep", "sparse_isometry_gf2x")
state_preparation = state_prep.run(wfn_cas)

# 7. Build target unitary U
unitary_builder = create(
    "hamiltonian_unitary_builder",
    "trotter",
    time=float(np.pi / 48.0),
    power=10,
)
unitary = unitary_builder.run(qubit_hamiltonian)

# 8. Run Hadamard test and estimate expectation from counts
shots = 100
result = hadamard_test.run(state_preparation, unitary, shots=shots)
counts = result.bitstring_counts
observable_value = (counts.get("0", 0) - counts.get("1", 0)) / sum(counts.values())

print("bitstring_counts:", counts)
print("estimated expectation:", observable_value)
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered Hadamard test implementations
implementations = registry.available("hadamard_test")
print(implementations)  # e.g. ['qdk']
# end-cell-list-implementations
################################################################################
