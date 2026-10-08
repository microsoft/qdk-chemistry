"""Deprecated alias of :mod:`qdk_chemistry.algorithms.unitary_builder`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib
import pkgutil
import sys
import warnings
from typing import Any

from qdk_chemistry.algorithms import unitary_builder
from qdk_chemistry.algorithms.unitary_builder.base import UnitaryBuilderFactory as HamiltonianUnitaryBuilderFactory

__all__: list[str] = ["HamiltonianUnitaryBuilderFactory"]

warnings.warn(
    f"'{__name__}' is deprecated and will be removed in a future release; use '{unitary_builder.__name__}' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias every submodule.
for _module in pkgutil.walk_packages(unitary_builder.__path__, f"{unitary_builder.__name__}."):
    sys.modules[__name__ + _module.name.removeprefix(unitary_builder.__name__)] = importlib.import_module(_module.name)


def __getattr__(name: str) -> Any:
    """Resolve any other name from :mod:`qdk_chemistry.algorithms.unitary_builder`."""
    return getattr(unitary_builder, name)
