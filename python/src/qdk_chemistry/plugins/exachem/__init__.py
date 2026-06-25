# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem plugin for QDK/Chemistry.

Provides a CLI-based integration with `ExaChem <https://github.com/ExaChem/exachem>`_
for two methods:

- Double Unitary Coupled Cluster (DUCC) Hamiltonian downfolding
  (:class:`~qdk_chemistry.plugins.exachem.ducc_solver.ExachemDuccSolver`).
- CCSD(T) total-energy calculations
  (:class:`~qdk_chemistry.plugins.exachem.ccsdt_calculator.ExachemCcsdtCalculator`).

ExaChem runs as an external MPI process; qdk-chemistry supplies pre-computed SCF
orbitals via ExaChem's serial-IO restart format and parses the results.

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
    from qdk_chemistry.algorithms.registry import register_factory  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.ccsdt_calculator import ExachemCcsdtCalculator  # noqa: PLC0415
    from qdk_chemistry.plugins.exachem.ducc_solver import (  # noqa: PLC0415
        ExachemDuccSolver,
        HamiltonianDownfolderFactory,
    )

    register_factory(HamiltonianDownfolderFactory())
    register(lambda: ExachemDuccSolver())
    register(lambda: ExachemCcsdtCalculator())
