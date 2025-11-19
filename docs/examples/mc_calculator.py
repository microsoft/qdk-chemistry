"""Multi-configuration calculator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

# Create the default MCCalculator instance (MACIS implementation)
mc_calculator = create("multi_configuration_calculator")

# Create a specific type of CI calculator
selected_ci = create("multi_configuration_calculator", "macis_cas")
