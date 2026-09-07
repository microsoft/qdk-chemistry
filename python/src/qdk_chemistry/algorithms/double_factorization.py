"""Public entry point for the double factorization algorithms.

This module re-exports the core double factorizers so that consumers can
import them directly from ``qdk_chemistry.algorithms`` without depending on
internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import (
    CholeskyDoubleFactorizer,  # noqa: F401 - re-export
    DoubleFactorizer,  # noqa: F401 - re-export
)
