"""Controlled evolution circuit mapper usage examples."""

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
# start-cell-configure
# Configure the power of the controlled unitary
mapper = create("controlled_circuit_mapper", "pauli_sequence")
mapper.settings().set("power", 4)
# end-cell-configure
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
qubit_mapper = create("qubit_mapper", encoding="jordan-wigner")
qubit_ham = qubit_mapper.run(hamiltonian)

# 4. Build time evolution unitary
trotter = create("unitary_builder", "trotter", order=2)
evolution = trotter.run(qubit_ham, time=0.1)

# 5. Create a controlled version and map to a circuit
from qdk_chemistry.data import ControlledUnitary

controlled = ControlledUnitary(evolution, control_indices=[0])
mapper = create("controlled_circuit_mapper", "pauli_sequence")
circuit = mapper.run(controlled)
print("Controlled evolution circuit generated")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered controlled evolution circuit mapper implementations
implementations = registry.available("controlled_circuit_mapper")
print(implementations)  # e.g. ['pauli_sequence']
# end-cell-list-implementations
################################################################################
