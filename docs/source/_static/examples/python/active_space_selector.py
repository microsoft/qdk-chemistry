"""Active space selection examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default ActiveSpaceSelector instance
active_space_selector = create("active_space_selector", "qdk_valence")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the selector using the settings interface
# Set the number of electrons and orbitals for the active space
active_space_selector.settings().set("num_active_electrons", 4)
active_space_selector.settings().set("num_active_orbitals", 4)

# end-cell-configure
################################################################################

################################################################################
# start-cell-run
from pathlib import Path  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Load a molecular structure (water molecule) from XYZ file
structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/water.structure.xyz"
)
charge = 0

# First, run SCF to get molecular orbitals
scf_solver = create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=charge, spin_multiplicity=1, basis_or_guess="6-31g"
)

# Run active space selection
active_wavefunction = active_space_selector.run(scf_wavefunction)
active_orbitals = active_wavefunction.get_orbitals()

print(f"SCF Energy: {scf_energy:.10f} Hartree")
print(f"Active orbitals summary:\n{active_orbitals.get_summary()}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("active_space_selector"))
# ['pyscf_avas', 'qdk_occupation', 'qdk_autocas_eos', 'qdk_autocas', 'qdk_valence']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-autocas
from qdk_chemistry.utils import compute_valence_space_parameters  # noqa: E402

# Create a valence space active space selector
valence_selector = create("active_space_selector", "qdk_valence")
# Automatically select valence parameters based on the input structure
num_electrons, num_orbitals = compute_valence_space_parameters(scf_wavefunction, charge)
valence_selector.settings().set("num_active_electrons", num_electrons)
valence_selector.settings().set("num_active_orbitals", num_orbitals)
active_valence_wfn = valence_selector.run(scf_wavefunction)

# Create active Hamiltonian
active_hamiltonian_generator = create("hamiltonian_constructor")
active_hamiltonian = active_hamiltonian_generator.run(active_valence_wfn.get_orbitals())

# Run Active Space Calculation with Selected CI
mc_calculator = create("multi_configuration_calculator", "macis_asci")
mc_calculator.settings().set("ntdets_max", 50000)
mc_calculator.settings().set("calculate_one_rdm", True)
mc_calculator.settings().set("calculate_two_rdm", True)
mc_energy, mc_wavefunction = mc_calculator.run(
    active_hamiltonian, num_electrons // 2, num_electrons // 2
)

# Print single orbital entropies which are used by autoCAS to determine the active space
entropies = mc_wavefunction.get_single_orbital_entropies()
print("Single orbital entropies:")
for idx, entropy in enumerate(entropies):
    print(f"Orbital {idx + 1}: {entropy:.6f}")

# Select active space using autoCAS
autocas_selector = create("active_space_selector", "qdk_autocas_eos")
active_autocas_wfn = autocas_selector.run(mc_wavefunction)
print("autoCAS selected active orbitals summary:")
print(active_autocas_wfn.get_orbitals().get_summary())
# end-cell-autocas
################################################################################

################################################################################
# start-cell-avas-example
avas = create("active_space_selector", "pyscf_avas")
avas.settings().set("ao_labels", ["O 2p", "O 2s"])
avas.settings().set("canonicalize", True)

active_wavefunction = avas.run(scf_wavefunction)
# end-cell-avas-example
################################################################################
