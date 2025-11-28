"""Public entry point for the multi-configuration calculator algorithms.

This module re-exports the core :class:`MultiConfigurationCalculator` and concrete
implementations so that consumers can import them directly from
``qdk_chemistry.algorithms`` without depending on internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import (
    MultiConfigurationCalculator,  # noqa: F401 - re-export
    QdkMacisAsci,  # noqa: F401 - re-export
    QdkMacisCas,  # noqa: F401 - re-export
)
