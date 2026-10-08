"""The type of a Hamiltonian description a unitary builder can take."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import ABC

from qdk_chemistry._core.data import LatticeGeometry
from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = ["UnitaryBuilderInput"]


class UnitaryBuilderInput(ABC):  # noqa: B024
    """Abstract type of every Hamiltonian description a unitary builder can take.

    The built-in types are registered below; a plugin registers its own with
    ``UnitaryBuilderInput.register``. Registering a type lets phase estimation pass it through;
    a :class:`~qdk_chemistry.algorithms.UnitaryBuilder` that names the one type it evolves
    rejects the rest.
    """


UnitaryBuilderInput.register(QubitOperator)
UnitaryBuilderInput.register(LatticeGeometry)
