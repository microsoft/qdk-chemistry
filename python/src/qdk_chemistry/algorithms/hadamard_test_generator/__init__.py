"""QDK/Chemistry Hadamard test algorithms module.

This module provides factories for constructing Hadamard test circuit
generators.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from .base import HadamardTestFactory

__all__: list[str] = ["HadamardTestFactory"]
