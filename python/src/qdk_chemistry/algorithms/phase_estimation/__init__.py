"""QDK/Chemistry phase estimation algorithms module.

This module provides quantum phase estimation algorithms for estimating
the eigenvalues of time evolution unitary operators.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from .base import PhaseEstimationFactory

__all__: list[str] = ["PhaseEstimationFactory"]
