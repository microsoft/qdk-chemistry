"""Reuse Cholesky integrals after an active-space natural-orbital rotation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# start-cell-transform
import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes

structure = Structure.from_xyz("""\
2
LiH
Li 0 0 0
H  0 0 1.60
""")
_, scf_wavefunction = create("scf_solver", "qdk").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
active_wavefunction = create(
    "active_space_selector",
    "qdk_valence",
    num_active_electrons=2,
    num_active_orbitals=3,
).run(scf_wavefunction)
source_orbitals = active_wavefunction.get_orbitals()
active_indices = list(
    source_orbitals.active_indices().indices(SymmetryLabel([axes.alpha()]))
)
nalpha, nbeta = active_wavefunction.get_active_num_electrons()

# The default "qdk" constructor returns canonical, not Cholesky, integrals.
source_hamiltonian = create(
    "hamiltonian_constructor", "qdk_cholesky", cholesky_tolerance=1e-10
).run(source_orbitals)
casci = create(
    "multi_configuration_calculator",
    "macis_cas",
    calculate_one_rdm=True,
    ci_residual_tolerance=1e-10,
)
source_energy, correlated_wavefunction = casci.run(source_hamiltonian, nalpha, nbeta)

# Restrict localization to the active orbitals; leave the frozen core unchanged.
natural_wavefunction = create("orbital_localizer", "qdk_natural_orbitals").run(
    correlated_wavefunction, active_indices, active_indices
)
transformed_hamiltonian = create("hamiltonian_basis_transformer").run(
    source_hamiltonian, natural_wavefunction.get_orbitals()
)

# Re-solve for a wavefunction expressed in the new orbital basis.
transformed_energy, transformed_wavefunction = casci.run(
    transformed_hamiltonian, nalpha, nbeta
)
np.testing.assert_allclose(
    transformed_energy, source_energy, atol=1e-9, rtol=0, equal_nan=False
)
print(f"Active orbitals: {active_indices}")
print(f"Original CASCI energy:    {source_energy:.12f} Hartree")
print(f"Transformed CASCI energy: {transformed_energy:.12f} Hartree")
# end-cell-transform
