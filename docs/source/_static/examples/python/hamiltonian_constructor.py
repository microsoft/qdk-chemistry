"""Hamiltonian constructor usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

################################################################################
# start-cell-create
# List available Hamiltonian constructor implementations
available_constructors = available("hamiltonian_constructor")
print(f"Available Hamiltonian constructors: {available_constructors}")

# Create the default HamiltonianConstructor instance
hamiltonian_constructor = create("hamiltonian_constructor")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure settings (check available options)
print(f"Available settings: {hamiltonian_constructor.settings().keys()}")

# Set ERI method if needed
hamiltonian_constructor.settings().set("eri_method", "direct")
# end-cell-configure
################################################################################

################################################################################
# start-cell-construct
# Load a structure from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Run a SCF to get orbitals
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
orbitals = wfn.get_orbitals()

# Construct the Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(orbitals)

# Access the resulting integrals
h1_a, h1_b = hamiltonian.get_one_body_integrals()
h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
core_energy = hamiltonian.get_core_energy()

print(f"One-body integrals shape: {h1_a.shape}")
print(f"Two-body integrals shape: {h2_aaaa.shape}")
print(f"Core energy: {core_energy:.10f} Hartree")
print(hamiltonian.get_summary())
# end-cell-construct
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("hamiltonian_constructor"))
# ['qdk']
# end-cell-list-implementations
################################################################################
