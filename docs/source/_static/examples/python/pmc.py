"""Projected Multi-Configuration Calculator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a MACIS PMC calculator instance
pmc_calculator = create("projected_multi_configuration_calculator", "macis_pmc")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the PMC calculator using the settings interface
pmc_calculator.settings().set("ci_residual_tolerance", 1.0e-6)
pmc_calculator.settings().set("davidson_res_tol", 1.0e-8)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np  # noqa: E402
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Configuration, Structure  # noqa: E402

# Create a structure (H2 molecule)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols)

# Run SCF to get orbitals
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Build Hamiltonian from orbitals
ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(wfn.get_orbitals())

# Define configurations
configurations = [
    Configuration("20"),  # Ground state (both electrons in lowest orbital)
    Configuration("02"),  # Excited state (both electrons in higher orbital)
]

# Run the PMC calculation
E_pmc, pmc_wavefunction = pmc_calculator.run(hamiltonian, configurations)

print(f"SCF Energy: {E_scf:.10f} Hartree")
print(f"PMC Energy: {E_pmc:.10f} Hartree")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

available = registry.available("projected_multi_configuration_calculator")
print(available)
# end-cell-list-implementations
################################################################################
