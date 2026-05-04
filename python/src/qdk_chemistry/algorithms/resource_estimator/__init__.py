"""QDK/Chemistry resource estimator module.

This module provides algorithm support for quantum resource estimation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from .base import ResourceEstimatorFactory

__all__: list[str] = ["ResourceEstimatorFactory"]
