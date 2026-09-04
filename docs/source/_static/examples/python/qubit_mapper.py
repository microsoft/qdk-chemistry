"""Qubit mapper usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import MajoranaMapping

# Create a QubitMapper instance
qubit_mapper = create("qubit_mapper")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Optional: configure numerical thresholds
qubit_mapper.settings().set("threshold", 1e-12)
qubit_mapper.settings().set("integral_threshold", 1e-12)
# end-cell-configure
################################################################################

################################################################################
# docs:xyz ../data/water.structure.xyz
# start-cell-run

from qdk_chemistry.data import Structure

# Read a molecular structure from inline XYZ file
structure = Structure.from_xyz("""\
3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
""")

# Perform an SCF calculation to generate initial orbitals
scf_solver = create("scf_solver")
_, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

# Select an active space
num_active_orbitals = 6
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
    num_active_electrons=4,
    num_active_orbitals=num_active_orbitals,
)
active_wfn = active_space_selector.run(wfn_hf)
active_orbitals = active_wfn.get_orbitals()

# Construct Hamiltonian in the active space
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(active_orbitals)

# Determine the number of spin-orbitals in the active space
n_spin_orbitals = 2 * num_active_orbitals

# Choose an encoding
mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)

# Map the fermionic Hamiltonian to a qubit Hamiltonian
qubit_hamiltonian = qubit_mapper.run(hamiltonian, mapping)
print(f"Qubit Hamiltonian has {qubit_hamiltonian.num_qubits} qubits")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("qubit_mapper"))
# ['qdk', 'qiskit']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-qdk-mapper
from qdk_chemistry.algorithms import create as create_algorithm
from qdk_chemistry.data import MajoranaMapping

# Create a native QDK QubitMapper instance
qdk_mapper = create_algorithm("qubit_mapper", "qdk")

# Optional: configure thresholds for numerical precision
qdk_mapper.settings().set("threshold", 1e-12)
qdk_mapper.settings().set("integral_threshold", 1e-12)

# Choose an encoding (Jordan-Wigner, Bravyi-Kitaev, or Parity)
mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)

# Map the fermionic Hamiltonian to a qubit Hamiltonian
qdk_qubit_hamiltonian = qdk_mapper.run(hamiltonian, mapping)
print(f"QDK mapper produced {len(qdk_qubit_hamiltonian.pauli_strings)} Pauli terms")
# end-cell-qdk-mapper
################################################################################
