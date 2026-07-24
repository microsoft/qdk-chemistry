# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Hamiltonian downfolding algorithms for QDK/Chemistry.

Provides DUCC (Double Unitary Coupled Cluster) Hamiltonian downfolding methods
that produce effective active-space Hamiltonians incorporating dynamical correlation
from external orbitals.

Available implementations:

- ``"wicked_ducc"``: Spin-orbital symbolic BCH downfolding via the ``wicked`` library.
- ``"wicked_ducc_si_ambit"``: Spin-integrated variant with the ambit tensor backend.
- ``"exachem_ducc"``: CLI-based integration with ExaChem's MPI DUCC solver.
  Requires ExaChem binary and MPI runtime.
"""

from qdk_chemistry.algorithms.base import AlgorithmFactory


class HamiltonianDownfolderFactory(AlgorithmFactory):
    """Factory establishing the ``hamiltonian_downfolder`` algorithm type."""

    def algorithm_type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def default_algorithm_name(self) -> str:
        """Return ``"wicked_ducc"``."""
        return "wicked_ducc"


_loaded = False


def load():
    """Register Hamiltonian downfolding algorithms with the QDK/Chemistry registry."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.algorithms import register
    from qdk_chemistry.algorithms.registry import register_factory

    # Establish the algorithm type. If the ExaChem plugin already registered a
    # hamiltonian_downfolder factory, this raises ValueError — ignore it.
    try:
        register_factory(HamiltonianDownfolderFactory())
    except ValueError:
        pass  # Factory already registered by the ExaChem plugin

    # Wicked-based DUCC (requires wicked library)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc import (
            WickedDuccSolver,
        )

        register(lambda: WickedDuccSolver())
    except ImportError:
        pass  # wicked not installed

    # Spin-integrated wicked DUCC (requires wicked library)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_si import (
            WickedDuccSISolver,
        )

        register(lambda: WickedDuccSISolver())
    except ImportError:
        pass  # wicked not installed

    # Pre-sliced spin-integrated wicked DUCC (requires wicked library)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_si_presliced import (
            WickedDuccSIPreslicedSolver,
        )

        register(lambda: WickedDuccSIPreslicedSolver())
    except ImportError:
        pass  # wicked not installed

    # Ambit-backed spin-integrated wicked DUCC (requires wicked + ambit)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_si_ambit import (
            WickedDuccSIAmbitSolver,
        )

        register(lambda: WickedDuccSIAmbitSolver())
    except ImportError:
        pass  # wicked or ambit not installed

    # 4-space wicked DUCC (requires wicked library)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_4space import (
            WickedDucc4SpaceSolver,
        )

        register(lambda: WickedDucc4SpaceSolver())
    except ImportError:
        pass  # wicked not installed

    # Hybrid wicked DUCC: gen_op H + 4-space T (requires wicked library)
    try:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_hybrid import (
            WickedDuccHybridSolver,
        )

        register(lambda: WickedDuccHybridSolver())
    except ImportError:
        pass  # wicked not installed
