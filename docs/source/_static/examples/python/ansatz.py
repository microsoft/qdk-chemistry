"""Basic Ansatz creation and usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from pathlib import Path
from qdk_chemistry.data import Ansatz, Structure
from qdk_chemistry.algorithms import create

# Load H2 molecule structure from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# SCF
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Create Hamiltonian from SCF orbitals
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# Create ansatz
ansatz_scf = Ansatz(hamiltonian, wfn_scf)
# end-cell-create
################################################################################

################################################################################
# start-cell-access
# Access ansatz components
hamiltonian_ref = ansatz_scf.get_hamiltonian()
wavefunction_ref = ansatz_scf.get_wavefunction()
orbitals_ref = ansatz_scf.get_orbitals()

# Check component availability
has_hamiltonian = ansatz_scf.has_hamiltonian()
has_wavefunction = ansatz_scf.has_wavefunction()
has_orbitals = ansatz_scf.has_orbitals()

# Calculate energy expectation value
energy = ansatz_scf.calculate_energy()
print(f"Energy expectation value: {energy:.8f} Hartree")

# Get summary information
summary = ansatz_scf.get_summary()
print(f"Ansatz summary: {summary}")
# end-cell-access
################################################################################
