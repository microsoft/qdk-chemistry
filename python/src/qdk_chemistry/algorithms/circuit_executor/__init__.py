"""QDK/Chemistry circuit executor module.

This module provides backend support for executing quantum circuits.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from .base import CircuitExecutor, CircuitExecutorFactory

__all__ = ["CircuitExecutor", "CircuitExecutorFactory"]
