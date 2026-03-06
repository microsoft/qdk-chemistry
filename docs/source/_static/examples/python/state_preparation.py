"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a StatePreparation instance
sparse_prep = create("state_prep", "sparse_isometry_gf2x")
regular_prep = create("state_prep", "qiskit_regular_isometry")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure transpilation settings
sparse_prep.settings().set("transpile", True)
sparse_prep.settings().set("basis_gates", ["rz", "cz", "sdg", "h"])
sparse_prep.settings().set("transpile_optimization_level", 3)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Specify a structure
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run scf
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Compute the Hamiltonian
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# Compute CAS wavefunction
cas_solver = create("multi_configuration_calculator", "macis_cas")
E_cas, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# Construct the circuit
regular_circuit = regular_prep.run(wfn_cas)
sparse_circuit = sparse_prep.run(wfn_cas)
print(f"Regular isometry QASM:\n{regular_circuit.get_qasm()}")
print(f"Sparse isometry QASM:\n{sparse_circuit.get_qasm()}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("state_prep"))
# ['sparse_isometry_gf2x', 'qiskit_regular_isometry']
# end-cell-list-implementations
################################################################################
