"""Define stretched N2 and compare Hartree--Fock basis-set energies.

This linear first example changes only the orbital basis while holding geometry,
charge, spin, and solver fixed. The resulting energy difference measures basis-
set sensitivity rather than exact error.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create
from qdk_chemistry.constants import HARTREE_TO_KJ_PER_MOL
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import Logger

if __name__ == "__main__":
    # Change ``off`` to ``info`` to see detailed QDK/Chemistry calculation logs.
    Logger.set_global_level(Logger.LogLevel.off)

################################################################################
# start-cell-molecule
# Molecular system
structure = Structure.from_xyz("""\
2
Stretched N2 molecule for the ground-state QPE tutorial
N    0.000000    0.000000    0.000000
N    0.000000    0.000000    1.850000
""")
# The 1.85-Angstrom bond is substantially longer than the 1.097685-Angstrom
# equilibrium distance, increasing multiconfigurational character.
# The target is neutral N2 in its singlet ground state, where all electrons are
# paired and the spin multiplicity 2S + 1 equals one.
charge = 0
spin_multiplicity = 1
# end-cell-molecule
################################################################################

################################################################################
# start-cell-hartree-fock
# Hartree-Fock basis-set comparison
# Correlation-consistent double- and triple-zeta bases form a controlled sequence:
# cc-pVTZ adds radial/angular flexibility beyond cc-pVDZ at higher cost.
basis_sets = ("cc-pvdz", "cc-pvtz")
# Store each result under its basis-set name so the values from the shared loop
# can be compared afterward without repeating either calculation.
energies = {}
wavefunctions = {}

# Change only the basis set so the energy difference measures basis-set sensitivity.
for basis_set in basis_sets:
    solver = create("scf_solver", "qdk")
    # basis_or_guess accepts either a basis name or a reusable orbital guess.
    # Supplying a string asks the solver to build that basis from the structure.
    # run() returns the converged total energy and its Hartree--Fock wavefunction.
    energy, wavefunction = solver.run(
        structure,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        basis_or_guess=basis_set,
    )
    energies[basis_set] = energy
    wavefunctions[basis_set] = wavefunction
    print(f"{basis_set}: {energy:.12f} Hartree")

# Record the continued workflow's orbital count. The active-space workflow later
# selects a subset, and each retained spatial orbital contributes two spin modes.
num_cc_pvdz_orbitals = (
    wavefunctions["cc-pvdz"].get_orbitals().get_num_molecular_orbitals()
)
print(f"cc-pvdz wavefunction: {num_cc_pvdz_orbitals} molecular orbitals")
# end-cell-hartree-fock
################################################################################

# Compare basis-set energies
signed_difference = energies["cc-pvtz"] - energies["cc-pvdz"]

# Report the same absolute sensitivity in milliHartree and kJ/mol, a common
# chemical-energy unit obtained with the library's Hartree conversion constant.
absolute_difference_millihartree = abs(signed_difference) * 1000
absolute_difference_kj_mol = abs(signed_difference) * HARTREE_TO_KJ_PER_MOL

print(f"Signed difference (cc-pVTZ - cc-pVDZ): {signed_difference:.12f} Hartree")
print(f"Absolute difference: {absolute_difference_millihartree:.6f} milliHartree")
print(f"Absolute difference: {absolute_difference_kj_mol:.6f} kJ/mol")


################################################################################
# Students can stop reading here. The assertions below guard the documented
# numerical results when this example runs in automated tests.
################################################################################


# Keep this tag: test_docs_xyz_consistency.py matches the inline geometry above
# byte-for-byte against the canonical XYZ file.
# docs:xyz ../data/tutorial_stretched_n2.structure.xyz

assert abs(energies["cc-pvdz"] - (-108.418633697214)) < 1e-8
assert abs(energies["cc-pvtz"] - (-108.445215657498)) < 1e-8
assert abs(signed_difference - (-0.026581960284)) < 2e-8
assert num_cc_pvdz_orbitals == 28
