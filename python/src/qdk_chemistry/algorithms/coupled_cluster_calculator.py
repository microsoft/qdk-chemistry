"""Public entry point for the Coupled Cluster calculator algorithms.

This module re-exports the core :class:`CoupledClusterCalculator` so that consumers can
import it directly from ``qdk_chemistry.algorithms`` without depending on
internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import CoupledClusterCalculator  # noqa: F401 - re-export
