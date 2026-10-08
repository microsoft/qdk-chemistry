"""The abstract type of the Hamiltonian descriptions that Hamiltonian unitary builders take."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import ABC

from qdk_chemistry.data.model_hamiltonian_description import ModelHamiltonianDescription
from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = ["HamiltonianDescription"]


class HamiltonianDescription(ABC):  # noqa: B024
    """Abstract type of every Hamiltonian description a Hamiltonian unitary builder can take.

    :class:`~qdk_chemistry.data.QubitOperator` and :class:`~qdk_chemistry.data.ModelHamiltonianDescription`
    are registered below; a plugin registers its own type with ``HamiltonianDescription.register``.
    Phase estimation passes any registered type through to its unitary builder; a
    :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` that names the one type it evolves
    rejects the rest.
    """


HamiltonianDescription.register(QubitOperator)
HamiltonianDescription.register(ModelHamiltonianDescription)
