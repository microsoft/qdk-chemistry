"""Example showing design principles: immutability and data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf-create
from pathlib import Path
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

scf_solver = create("scf_solver")
# end-cell-scf-create
################################################################################

################################################################################
# start-cell-scf-settings
print(f"Available settings: {scf_solver.settings().get_summary()}")
scf_solver.settings().set("max_iterations", 100)
# end-cell-scf-settings
################################################################################

################################################################################
# start-cell-data-flow
# Load a Structure from file (data classes in QDK/Chemistry are immutable by design)
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Configure and run SCF calculation
scf_solver = create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

# Select active space orbitals
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
)
active_space_selector.settings().set("num_active_orbitals", 2)
active_space_selector.settings().set("num_active_electrons", 2)
active_wfn = active_space_selector.run(scf_wavefunction)
active_orbitals = active_wfn.get_orbitals()

# Create Hamiltonian with active space
ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(active_orbitals)

mc = create("multi_configuration_calculator")
mc.settings().set("davidson_iterations", 300)
E_cas, wfn_cas = mc.run(
    hamiltonian, n_active_alpha_electrons=1, n_active_beta_electrons=1
)
# end-cell-data-flow
################################################################################
