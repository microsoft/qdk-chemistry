"""Example demonstrating algorithm interface usage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

# All algorithms in QDK/Chemistry follow a common interface pattern

# 1. Create algorithm instance
scf_solver = create("scf_solver")

# 2. Access settings
settings = scf_solver.settings()
print(f"Algorithm type: {scf_solver.type_name()}")
print(f"Settings available: {settings.has('max_iterations')}")

# 3. Configure via settings
settings.set("max_iterations", 50)

# The run() method executes the algorithm
# (requires appropriate input data, not shown here for interface demo)
