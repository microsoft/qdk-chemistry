"""Code snippets for v2.2.0 release notes.

Each cell is a runnable snippet included in the Sphinx release notes through
``literalinclude`` markers. The file is executed end-to-end by the
``test_docs_examples.py`` test harness against an installed 2.2.x package.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure


################################################################################
# start-cell-remote-execution
structure = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])

energy, wavefunction = create("scf_solver").run(
    structure,
    charge=0,
    spin_multiplicity=1,
    basis_or_guess="sto-3g",
    remote="local",
    cache="./cache",
)
# end-cell-remote-execution
################################################################################

print(f"Remote SCF energy: {energy:.10f} Hartree")
print(f"Wavefunction type: {wavefunction.get_type()}")
