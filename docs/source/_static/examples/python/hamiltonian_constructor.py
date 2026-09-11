"""Hamiltonian constructor usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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
# docs:xyz ../data/h2.structure.xyz
# start-cell-construct
# Load a structure from inline XYZ file
structure = Structure.from_xyz("""\
2
H2 molecule
H    0.000000    0.000000    0.000000
H    0.000000    0.000000    0.740848
""")

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
from qdk_chemistry.algorithms import registry

print(registry.available("hamiltonian_constructor"))
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-cholesky
# Reuse the SCF orbitals; Cholesky needs no auxiliary basis.
cholesky_constructor = create("hamiltonian_constructor", "qdk_cholesky")
cholesky_constructor.settings().set("cholesky_tolerance", 1e-8)
cholesky_hamiltonian = cholesky_constructor.run(orbitals)
print(f"Cholesky container: {cholesky_hamiltonian.get_container_type()}")
# end-cell-cholesky
################################################################################

################################################################################
# start-cell-density-fitted
from qdk_chemistry.data import (
    AuxiliaryBasis,
    AuxiliaryBasisCollection,
    AuxiliaryBasisRole,
    BasisSet,
)

# SCF supplies MOs in the primary basis; RIFIT is used for the Hamiltonian.
df_basis = BasisSet.from_basis_name("cc-pvdz", structure)
df_scf_solver = create("scf_solver", "qdk")
_, df_wavefunction = df_scf_solver.run(structure, 0, 1, df_basis)
df_orbitals = df_wavefunction.get_orbitals()

rifit = AuxiliaryBasis.from_basis_name("cc-pvdz-rifit", structure)
auxiliary_bases = AuxiliaryBasisCollection({AuxiliaryBasisRole.RIFIT: rifit})
df_constructor = create("hamiltonian_constructor", "qdk_density_fitted_hamiltonian")
df_hamiltonian = df_constructor.run(df_orbitals, auxiliary_bases)
print(f"Density-fitted container: {df_hamiltonian.get_container_type()}")
# end-cell-density-fitted
################################################################################
