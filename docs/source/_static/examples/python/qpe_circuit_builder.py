"""QpeCircuitBuilder usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-configure-iqpe
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create iterative QPE with circuit builder configuration
controlled_circuit_mapper = AlgorithmRef("controlled_circuit_mapper", "pauli_sequence")
unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=0.1)
iqpe_circuit_builder = create("qpe_circuit_builder", "qdk_iterative", num_bits=10)
iqpe_circuit_builder.settings().set(
    "controlled_circuit_mapper", controlled_circuit_mapper
)
iqpe_circuit_builder.settings().set("unitary_builder", unitary_builder)
# end-cell-configure-iqpe
################################################################################

################################################################################
# start-cell-configure-standard
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create standard QPE circuit builder
controlled_circuit_mapper = AlgorithmRef("controlled_circuit_mapper", "pauli_sequence")
unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=0.1)
qpe_circuit_builder = create("qpe_circuit_builder", "qdk_standard", num_bits=10)
qpe_circuit_builder.settings().set(
    "controlled_circuit_mapper", controlled_circuit_mapper
)
qpe_circuit_builder.settings().set("unitary_builder", unitary_builder)
# end-cell-configure-standard
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

# 3. Hamiltonian construction
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# 4. Multi-configuration calculation (reference state)
cas_solver = create("multi_configuration_calculator")
E_cas, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# 5. Qubit mapping
from qdk_chemistry.data import MajoranaMapping

n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
qubit_mapper = create("qubit_mapper")
qubit_ham = qubit_mapper.run(
    hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals)
)

# 6. State preparation
state_prep = create("state_prep", "sparse_isometry")
circuit = state_prep.run(wfn_cas)

# 7. Create and run IQPE circuit builder with nested algorithm settings
from qdk_chemistry.data import AlgorithmRef

controlled_circuit_mapper = AlgorithmRef("controlled_circuit_mapper", "pauli_sequence")
unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=0.1)
iqpe_circuit_builder = create("qpe_circuit_builder", "qdk_iterative", num_bits=4)
iqpe_circuit_builder.settings().set(
    "controlled_circuit_mapper", controlled_circuit_mapper
)
iqpe_circuit_builder.settings().set("unitary_builder", unitary_builder)

iqpe_circuits = iqpe_circuit_builder.run(
    state_preparation=circuit,
    qubit_hamiltonian=qubit_ham,
)

# 8. Print the generated circuits for each bit (iterative)
for idx, circ in enumerate(iqpe_circuits):
    print(f"Bit {idx}:")
    print(circ.get_qsharp_circuit())

# 9. Create and run standard QPE circuit builder
standard_builder = create("qpe_circuit_builder", "qdk_standard", num_bits=4)
standard_builder.settings().set("controlled_circuit_mapper", controlled_circuit_mapper)
standard_builder.settings().set("unitary_builder", unitary_builder)

standard_circuits = standard_builder.run(
    state_preparation=circuit,
    qubit_hamiltonian=qubit_ham,
)

# 10. Print the standard QPE circuit (single circuit with all ancillas)
print("Standard QPE circuit:")
print(standard_circuits[0].get_qsharp_circuit())
# end-cell-run
################################################################################
