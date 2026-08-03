# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem plugin for QDK/Chemistry.

Provides a CLI-based integration with `ExaChem <https://github.com/ExaChem/exachem>`_
for Double Unitary Coupled Cluster (DUCC) effective-Hamiltonian construction. ExaChem
runs as an external MPI process and communicates via FCIDUMP files.

Prerequisites:
    - ExaChem binary built and available on ``PATH`` (or set ``EXACHEM_PATH``)
    - MPI runtime (``mpirun`` or ``srun``) for parallel execution
"""

_loaded = False


def load():
    """Register ExaChem algorithm implementations with the QDK/Chemistry registry."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.ducc_solver import ExachemDuccSolver  # noqa: PLC0415

    register(lambda: ExachemDuccSolver())
