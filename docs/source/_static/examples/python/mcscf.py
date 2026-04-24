"""Multi-configuration SCF usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create the default MultiConfigurationScf instance
mcscf = create("multi_configuration_scf", "pyscf")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure nested algorithms via AlgorithmRef settings
mcscf.settings().set(
    "hamiltonian_constructor",
    AlgorithmRef("hamiltonian_constructor", "qdk"),
)
mcscf.settings().set(
    "multi_configuration_calculator",
    AlgorithmRef("multi_configuration_calculator", "macis_cas"),
)

# Configure the MCSCF solver
mcscf.settings().set("max_cycle_macro", 50)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
from pathlib import Path
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import compute_valence_space_parameters

# Load nitrogen molecule structure
structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/n2_stretched.structure.xyz"
)
charge = 0

# First, run SCF to get molecular orbitals
scf_solver = create("scf_solver")
E_scf, scf_wavefunction = scf_solver.run(
    structure, charge=charge, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

# Select active space based on valence orbitals
valence_selector = create("active_space_selector", "qdk_valence")
nele, norb = compute_valence_space_parameters(scf_wavefunction, charge)
valence_selector.settings().set("num_active_electrons", nele)
valence_selector.settings().set("num_active_orbitals", norb)
active_wavefunction = valence_selector.run(scf_wavefunction)

# Run MCSCF calculation
nalpha, nbeta = active_wavefunction.get_active_num_electrons()
E_mcscf, mcscf_wfn = mcscf.run(active_wavefunction.get_orbitals(), nalpha, nbeta)

print(f"SCF Energy: {E_scf:.10f} Hartree")
print(f"MCSCF Energy: {E_mcscf:.10f} Hartree")
print(f"Correlation energy: {E_mcscf - E_scf:.10f} Hartree")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("multi_configuration_scf"))
# ['pyscf']
# end-cell-list-implementations
################################################################################
