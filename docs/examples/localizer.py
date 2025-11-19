"""Orbital localizer usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

# Create an MP2 natural orbital localizer
mp2_localizer = create("orbital_localizer", "qdk_mp2_natural_orbitals")

# Set the convergence threshold
# Configure settings
# mp2_localizer.settings().set("tolerance", 1.0e-6)
