"""Public entry point for the F12-Hartree-Fock solver algorithms.

This module re-exports the core :class:`F12HartreeFockSolver` and concrete
implementations so that consumers can import them directly from
``qdk_chemistry.algorithms`` without depending on internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import (
    F12HartreeFockSolver,  # noqa: F401 - re-export
    QdkCtF12HartreeFockSolver,  # noqa: F401 - re-export
)
