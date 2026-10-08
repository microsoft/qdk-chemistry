"""The type of a Hamiltonian description a unitary builder can take."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import ABC

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry
from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = ["UnitaryBuilderInput"]


class UnitaryBuilderInput(ABC):  # noqa: B024
    """Abstract type of every Hamiltonian description a unitary builder can take.

    The built-in types are registered below; a plugin registers its own with
    ``UnitaryBuilderInput.register``. Registering a type lets phase estimation pass it through;
    each :class:`~qdk_chemistry.algorithms.UnitaryBuilder` still evolves only the one type it
    names, and rejects the rest.
    """


UnitaryBuilderInput.register(QubitOperator)
UnitaryBuilderInput.register(Hamiltonian)
UnitaryBuilderInput.register(LatticeGeometry)
