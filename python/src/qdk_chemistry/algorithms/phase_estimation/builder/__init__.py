"""QDK/Chemistry phase estimation builder algorithms module.

This module provides the circuit-building component of quantum phase estimation,
separated from the execution logic to allow standalone circuit generation
for resource estimation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from .base import PhaseEstimationBuilderFactory

__all__: list[str] = ["PhaseEstimationBuilderFactory"]
