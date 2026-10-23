"""ExaChem plugin for QDK/Chemistry.

Provides a CLI-based integration with `ExaChem <https://github.com/ExaChem/exachem>`_
for CCSD calculations that can return the converged T1/T2 cluster amplitudes
(:class:`~qdk_chemistry.plugins.exachem.ccsd_calculator.ExachemCcsdCalculator`).

ExaChem runs as an external MPI process; qdk-chemistry supplies pre-computed SCF
orbitals via ExaChem's serial-IO restart format and parses the results. The
calculator is registered only if both prerequisites are on ``PATH`` when the plugin loads.

Prerequisites:
    - ExaChem binary discoverable as ``ExaChem`` on ``PATH``
    - MPI runtime (``mpirun`` or ``srun``) for parallel execution
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

_loaded = False


def load():
    """Register the ExaChem calculator if ExaChem and an MPI launcher are on ``PATH``."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.plugins.exachem.cli import (  # noqa: PLC0415
        ExachemNotFoundError,
        find_exachem_binary,
        find_mpi_launcher,
    )

    try:
        find_exachem_binary()
        find_mpi_launcher()
    except ExachemNotFoundError:
        return

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.ccsd_calculator import ExachemCcsdCalculator  # noqa: PLC0415

    register(lambda: ExachemCcsdCalculator())
