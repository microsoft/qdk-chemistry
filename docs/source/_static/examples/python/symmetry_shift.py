"""Symmetry shift (fermionic low-rank BLISS) usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Hamiltonian, Structure

################################################################################
# start-cell-create
# List available symmetry shifter implementations
print(f"Available symmetry shifters: {available('symmetry_shifter')}")

# Create the fermionic low-rank BLISS shifter
shifter = create("symmetry_shifter", "fermionic_low_rank")

# The shifter itself has no tunable settings; truncation is a property of the
# double factorization it consumes.
print(f"Shifter settings: {shifter.settings().keys()}")
# end-cell-create
################################################################################

################################################################################
# docs:xyz ../data/water.structure.xyz
# start-cell-factorize
# Load H2O molecule from inline XYZ file
structure = Structure.from_xyz("""\
3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
""")

# Obtain orbitals from SCF and build the molecular Hamiltonian
E_scf, wfn = create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
hamiltonian = create("hamiltonian_constructor").run(wfn.get_orbitals())

# The shifter consumes a double-factorized Hamiltonian
factorizer = create("hamiltonian_factorization", "double_factorization")
factorizer.settings().set("truncation_threshold", 1.0e-8)
factorized = factorizer.run(hamiltonian)

container = factorized.get_container()
print(f"Orbitals: {container.get_num_orbitals()}, ranks: {container.get_num_ranks()}")
# end-cell-factorize
################################################################################

################################################################################
# start-cell-shift
# Target electron-number sector: neutral water has 10 electrons
n_alpha, n_beta = 5, 5

lambda_before = factorized.get_container().get_lambda()
shifted = shifter.run(factorized, n_alpha, n_beta)
lambda_after = shifted.get_container().get_lambda()

print(f"lambda before shift: {lambda_before:.8f}")
print(f"lambda after  shift: {lambda_after:.8f}")
print(f"reduction          : {100 * (1 - lambda_after / lambda_before):.2f}%")
# end-cell-shift
################################################################################

################################################################################
# start-cell-inspect-shift
# Inspect the (mu1, mu2, xi) parameters that the last run applied
shift = shifter.last_shift()
print(f"mu1: {shift.mu1:.8f}")
print(f"mu2: {shift.mu2:.8f}")
print(f"xi shape: {shift.xi.shape}")
# end-cell-inspect-shift
################################################################################

################################################################################
# start-cell-persist
# The shifted Hamiltonian stays factorized, so it round-trips without
# re-factorization and can be block-encoded directly.
shifted.to_hdf5_file("water_shifted.hamiltonian.h5")

reloaded = Hamiltonian.from_hdf5_file("water_shifted.hamiltonian.h5")
print(f"lambda on reload: {reloaded.get_container().get_lambda():.8f}")
os.remove("water_shifted.hamiltonian.h5")
# end-cell-persist
################################################################################
