# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem plugin for QDK/Chemistry.

Provides a CLI-based integration with `ExaChem <https://github.com/ExaChem/exachem>`_
for CCSD calculations that return the converged T1/T2 cluster amplitudes
(:class:`~qdk_chemistry.plugins.exachem.ccsd_calculator.ExachemCcsdCalculator`).

ExaChem runs as an external MPI process; qdk-chemistry supplies pre-computed SCF
orbitals via ExaChem's serial-IO restart format and parses the results.

Prerequisites:
    - ExaChem binary, configured via the calculator's ``exachem_binary`` setting
      or discoverable as ``ExaChem`` on ``PATH``
    - MPI runtime (``mpirun`` or ``srun``) for parallel execution
"""

_loaded = False
QDK_CHEMISTRY_HAS_MPI = False


def load():
    """Register ExaChem algorithm implementations with the QDK/Chemistry registry."""
    global _loaded  # noqa: PLW0603
    global QDK_CHEMISTRY_HAS_MPI  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.ccsd_calculator import ExachemCcsdCalculator  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.cli import ExachemNotFoundError, find_mpi_launcher  # noqa: PLC0415

    try:
        find_mpi_launcher()
        QDK_CHEMISTRY_HAS_MPI = True
    except ExachemNotFoundError:
        QDK_CHEMISTRY_HAS_MPI = False

    register(lambda: ExachemCcsdCalculator())
