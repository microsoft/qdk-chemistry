"""Multi-configuration calculator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# First run an SCF calculation and build a Hamiltonian
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(wfn.get_orbitals())

# List available multi-configuration calculator implementations
available_mc = available("multi_configuration_calculator")
print(f"Available MC calculators: {available_mc}")

# Create a CAS (Complete Active Space) calculator
mc_calculator = create("multi_configuration_calculator", "macis_cas")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the MC calculator
mc_calculator.settings().set("ci_residual_tolerance", 1.0e-6)
mc_calculator.settings().set("davidson_iterations", 200)
mc_calculator.settings().set("calculate_one_rdm", True)
mc_calculator.settings().set("calculate_two_rdm", False)

# View available settings
print(f"MC calculator settings: {mc_calculator.settings()}")
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Create a structure (H2 molecule)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Run SCF to get orbitals
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Build Hamiltonian from orbitals
ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(wfn.get_orbitals())

# Run the CI calculation
# For H2, we have 2 electrons (1 alpha, 1 beta)
n_alpha = 1
n_beta = 1
E_ci, ci_wavefunction = mc_calculator.run(hamiltonian, n_alpha, n_beta)

print(f"SCF Energy: {E_scf:.10f} Hartree")
print(f"CI Energy:  {E_ci:.10f} Hartree")
print(f"Correlation energy: {E_ci - E_scf:.10f} Hartree")
print(ci_wavefunction.get_summary())
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("multi_configuration_calculator"))
# ['macis_asci', 'macis_cas']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-asci-example
mc = create("multi_configuration_calculator", "macis_asci")
mc.settings().set("ntdets_max", 50000)
mc.settings().set("calculate_one_rdm", True)
# end-cell-asci-example
################################################################################
