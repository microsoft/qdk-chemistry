"""Code snippets for v2.1.0 release notes.

Each cell is a runnable snippet included in the Sphinx release notes via
``literalinclude`` with ``start-after`` / ``end-before`` markers.  The file
is executed end-to-end by the ``test_docs_examples.py`` test harness, which
gates it to an installed ``2.1.x`` library (see the release-notes version pin
in that test) and to an available geomeTRIC installation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.constants import BOHR_TO_ANGSTROM
from qdk_chemistry.data import AlgorithmRef, Structure

# ===========================================================================
# Geometry optimization
# ===========================================================================

################################################################################
# start-cell-geometry-optimizer
# Structure coordinates are in Bohr
structure = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])

optimizer = create("geometry_optimizer", "geometric", max_iterations=20)
# end-cell-geometry-optimizer
################################################################################

################################################################################
# start-cell-geometry-derivative
derivative_ref = AlgorithmRef("nuclear_derivative_calculator", "qdk_finite_difference")
derivative_ref.set("finite_difference_step", 1.0e-2)
optimizer.settings().set("derivative_calculator", derivative_ref)
# end-cell-geometry-derivative
################################################################################

################################################################################
# start-cell-geometry-run
energy, optimized_structure, hessian, wavefunction = optimizer.run(
    structure, charge=0, spin_multiplicity=1, input="sto-3g"
)
# end-cell-geometry-run
################################################################################

bond_length = np.linalg.norm(np.diff(optimized_structure.get_coordinates(), axis=0))
print(f"Optimized energy: {energy:.10f} Hartree")
print(f"Optimized H-H distance: {bond_length * BOHR_TO_ANGSTROM:.4f} Angstrom")
