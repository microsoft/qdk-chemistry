"""Define stretched N2 and compare Hartree-Fock basis-set energies."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create
from qdk_chemistry.constants import HARTREE_TO_KJ_PER_MOL
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import Logger

Logger.set_global_level(Logger.LogLevel.off)

################################################################################
# docs:xyz ../data/tutorial_stretched_n2.structure.xyz
# start-cell-molecule
structure = Structure.from_xyz("""\
2
Stretched N2 molecule for the ground-state QPE tutorial
N    0.000000    0.000000    0.000000
N    0.000000    0.000000    1.270025
""")
charge = 0
spin_multiplicity = 1
# end-cell-molecule
################################################################################

################################################################################
# start-cell-hartree-fock
basis_sets = ("cc-pvdz", "cc-pvtz")
energies = {}
wavefunctions = {}

for basis_set in basis_sets:
    solver = create("scf_solver", "qdk")
    energy, wavefunction = solver.run(
        structure,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        basis_or_guess=basis_set,
    )
    energies[basis_set] = energy
    wavefunctions[basis_set] = wavefunction
    print(f"{basis_set}: {energy:.12f} Hartree")

num_cc_pvdz_orbitals = wavefunctions["cc-pvdz"].get_orbitals().get_num_molecular_orbitals()
print(f"cc-pvdz wavefunction: {num_cc_pvdz_orbitals} molecular orbitals")
# end-cell-hartree-fock
################################################################################

################################################################################
# start-cell-compare
signed_difference = energies["cc-pvtz"] - energies["cc-pvdz"]
absolute_difference_millihartree = abs(signed_difference) * 1000
absolute_difference_kj_mol = abs(signed_difference) * HARTREE_TO_KJ_PER_MOL

print(f"Signed difference (cc-pVTZ - cc-pVDZ): {signed_difference:.12f} Hartree")
print(f"Absolute difference: {absolute_difference_millihartree:.6f} milliHartree")
print(f"Absolute difference: {absolute_difference_kj_mol:.6f} kJ/mol")
# end-cell-compare
################################################################################

# Numerical guards run in the documentation example test but are not displayed.
assert abs(energies["cc-pvdz"] - (-108.866810916955)) < 1e-8
assert abs(energies["cc-pvtz"] - (-108.891507935778)) < 1e-8
assert abs(signed_difference - (-0.024697018823)) < 2e-8
assert num_cc_pvdz_orbitals == 28
