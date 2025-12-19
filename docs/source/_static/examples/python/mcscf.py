"""Multi-configuration scf usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.utils import compute_valence_space_parameters
from qdk_chemistry.data import Structure

# First run an SCF calculation
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.4]])
charge = 0
structure = Structure(coords, ["N", "N"])
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=charge, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

# Create a HamiltonianConstructor
ham_constructor = create("hamiltonian_constructor")

# Create a CAS (Complete Active Space) calculator
mc_calculator = create("multi_configuration_calculator", "macis_cas")

# List available multi-configuration scf implementations
available_mc = available("multi_configuration_scf")
print(f"Available MCSCFs: {available_mc}")

# Create a MCSCF
mcscf = create("multi_configuration_scf", "pyscf")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the MC calculator
mc_calculator.settings().set("ci_residual_tolerance", 1.0e-6)
# MCSCF changes these settings to true anyways.
mc_calculator.settings().set("calculate_one_rdm", True)
mc_calculator.settings().set("calculate_two_rdm", True)

# Configure the Hamiltonian constructor
ham_constructor.settings().set("eri_method", "direct")

# Configure the MCSCF calculator
mcscf.settings().set("max_cycle_macro", 50)

# View available settings
print(f"MC calculator settings: {mc_calculator.settings()}")
print(f"Hamiltonian constructor settings: {ham_constructor.settings()}")
print(f"MCSCF settings: {mcscf.settings()}")
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
# Select active space based on valence orbitals
valence_selector = create("active_space_selector", "qdk_valence")
nele, norb = compute_valence_space_parameters(wfn, charge)
valence_selector.settings().set("num_active_electrons", nele)
valence_selector.settings().set("num_active_orbitals", norb)
val_wfn = valence_selector.run(wfn)

# Run mcscf
nalpha, nbeta = val_wfn.get_active_num_electrons()
E_mcscf, mcscf_wfn = mcscf.run(
    val_wfn.get_orbitals(), ham_constructor, mc_calculator, nalpha, nbeta
)

print(f"SCF Energy: {E_scf:.10f} Hartree")
print(f"MCSCF Energy:  {E_mcscf:.10f} Hartree")
print(f"Correlation energy: {E_mcscf - E_scf:.10f} Hartree")
print(mcscf_wfn.get_summary())
# end-cell-run
################################################################################
