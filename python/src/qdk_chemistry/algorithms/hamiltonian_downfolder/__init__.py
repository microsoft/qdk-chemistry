# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Hamiltonian downfolding algorithms for QDK/Chemistry.

Provides DUCC (Double Unitary Coupled Cluster) Hamiltonian downfolding methods
that produce effective active-space Hamiltonians incorporating dynamical correlation
from external orbitals.

Available implementations:

- ``"native_ducc"``: Pure Python/NumPy implementation using PySCF for CCSD amplitudes.
  No external binary required.
- ``"reference_ducc"``: OpenFermion-based symbolic BCH implementation using PySCF for CCSD amplitudes.
  Algebraically exact but limited to small systems (~20 spatial MOs) due to OpenFermion memory scaling.
- ``"exachem_ducc"``: CLI-based integration with ExaChem's MPI DUCC solver.
  Requires ExaChem binary and MPI runtime.
"""

_loaded = False


def load():
    """Register Hamiltonian downfolding algorithms with the QDK/Chemistry registry."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.algorithms import register
    from qdk_chemistry.algorithms.hamiltonian_downfolder.native_ducc import (
        NativeDuccFactory,
        NativeDuccSolver,
    )
    from qdk_chemistry.algorithms.hamiltonian_downfolder.reference_ducc import (
        ReferenceDuccFactory,
        ReferenceDuccSolver,
    )
    from qdk_chemistry.algorithms.registry import register_factory

    # Register the factory. If ExaChem plugin already registered one, this will
    # raise ValueError — catch it and just register the algorithm.
    try:
        register_factory(NativeDuccFactory())
    except ValueError:
        pass  # Factory already registered by ExaChem plugin

    try:
        register_factory(ReferenceDuccFactory())
    except ValueError:
        pass

    register(lambda: NativeDuccSolver())
    register(lambda: ReferenceDuccSolver())

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
