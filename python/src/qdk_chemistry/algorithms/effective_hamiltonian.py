"""Public entry point for the effective-Hamiltonian (downfolding) algorithms.

This module re-exports the core :class:`EffectiveHamiltonian` so that consumers
can import it directly from ``qdk_chemistry.algorithms`` without depending on
internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import EffectiveHamiltonian  # noqa: F401 - re-export
