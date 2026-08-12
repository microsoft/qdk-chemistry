"""QDK/Chemistry expectation estimator module.

This module provides algorithms that estimate expectation values of qubit
operators (observables) from measurement circuits.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .expectation_estimator import ExpectationEstimatorFactory

__all__ = ["ExpectationEstimatorFactory"]
