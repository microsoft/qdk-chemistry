"""Multi-configuration calculator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a CAS MultiConfigurationCalculator instance
mc_calculator = create("multi_configuration_calculator", "macis_cas")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the MC calculator using the settings interface
mc_calculator.settings().set("ci_residual_tolerance", 1.0e-6)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
from pathlib import Path  # noqa: E402
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Load H2 structure from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Run SCF to get orbitals
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

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
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("multi_configuration_calculator"))
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-asci-example
mc = create("multi_configuration_calculator", "macis_asci")
mc.settings().set("ntdets_max", 50000)
mc.settings().set("calculate_one_rdm", True)
# end-cell-asci-example
################################################################################
