"""Controlled circuit mapper usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default mapper (pauli_sequence)
mapper = create("controlled_circuit_mapper")
# end-cell-create
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# 1. Setup molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# 2. SCF
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# 3. Hamiltonian and qubit mapping
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())
from qdk_chemistry.data import MajoranaMapping

n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
qubit_mapper = create("qubit_mapper")
qubit_ham = qubit_mapper.run(
    hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals)
)

# 4. Build time evolution unitary
trotter = create("hamiltonian_unitary_builder", "trotter", order=2, time=0.1)
evolution = trotter.run(qubit_ham)

# 5. Create a controlled version and map to a circuit
mapper = create("controlled_circuit_mapper", "pauli_sequence", control_indices=[0])
circuit = mapper.run(evolution)
print("Controlled evolution circuit generated")
# end-cell-run
################################################################################

################################################################################
# start-cell-cswap
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import MajoranaMapping, Structure

# 1. Molecule and mean-field reference
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, symbols=["H", "H"])
E_scf, wfn_scf = create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# 2. Fermionic Hamiltonian, then a qubit Hamiltonian (core energy excluded,
#    so the all-zero state satisfies H|0...0> = 0)
hamiltonian = create("hamiltonian_constructor").run(wfn_scf.get_orbitals())
n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
qubit_ham = create("qubit_mapper").run(
    hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals)
)

# 3. Group Pauli strings that flip the same qubits. These are the strings coming
#    from the same fermionic term, whose amplitudes cancel on |0...0>. Without
#    this step Trotterization interleaves them and the vacuum leaks.
grouped_ham = create("term_grouper", "qubit_flip").run(qubit_ham)

# 4. Trotterize. The builder honours the grouping, so each group is exponentiated
#    as one contiguous block and the product formula still fixes |0...0>.
evolution = create("hamiltonian_unitary_builder", "trotter", order=1, time=0.1).run(
    grouped_ham
)

# 5. Control it with the CSWAP sandwich. The mapper validates the ordering and
#    raises if the product formula would leak the vacuum. Qubit 0 is the control
#    ancilla (the convention the QPE circuit builders use) and the remaining
#    n_spin_orbitals qubits are auto-assigned as the system register.
cswap_mapper = create(
    "controlled_circuit_mapper",
    "cswap_pauli_sequence",
    control_indices=[0],
)
circuit = cswap_mapper.run(evolution)
print("Controlled evolution circuit generated via the CSWAP sandwich")
# end-cell-cswap
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered controlled circuit mapper implementations
implementations = registry.available("controlled_circuit_mapper")
print(implementations)  # e.g. ['pauli_sequence']
# end-cell-list-implementations
################################################################################
