# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# Type-checker view of the deprecated alias; at runtime the package's __init__.py maps it
# onto qdk_chemistry.algorithms.unitary_builder.

from qdk_chemistry.algorithms.unitary_builder.base import *  # noqa: F403
from qdk_chemistry.algorithms.unitary_builder.base import UnitaryBuilderFactory, UnitaryBuilderSettings

HamiltonianUnitaryBuilderFactory = UnitaryBuilderFactory
HamiltonianUnitaryBuilderSettings = UnitaryBuilderSettings
