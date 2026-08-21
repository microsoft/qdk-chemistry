"""Effective Hamiltonian constructor usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("effective_hamiltonian_constructor"))
# Example output:
# ['qdk_swpt2']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a second-order Schrieffer-Wolff downfolder
downfolder = create("effective_hamiltonian_constructor", "qdk_swpt2")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# View available settings
print(f"Downfolder settings: {downfolder.settings().keys()}")
# end-cell-configure
################################################################################

################################################################################
# docs:xyz ../data/water.structure.xyz
# start-cell-downfold
from qdk_chemistry.data import Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes

structure = Structure.from_xyz("""\
3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
""")

E_scf, wfn = create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Assign an active space to the mean-field reference
selector = create("active_space_selector", "qdk_valence")
selector.settings().set("num_active_electrons", 6)
selector.settings().set("num_active_orbitals", 5)
reference = selector.run(wfn)

# Build the window Hamiltonian from the pre-selection orbitals, so that every
# orbital of W is active.
window_hamiltonian = create("hamiltonian_constructor").run(wfn.get_orbitals())

# Keep the reference active space as P and fold the rest of the window into it
p_indices = reference.get_orbitals().active_indices()

effective_hamiltonian = downfolder.run(reference, window_hamiltonian, p_indices)

alpha = SymmetryLabel([axes.alpha()])
emitted = effective_hamiltonian.get_orbitals()
print(f"kept orbitals: {list(emitted.active_indices().indices(alpha))}")
print(f"folded to inactive: {list(emitted.inactive_indices().indices(alpha))}")
# end-cell-downfold
################################################################################
