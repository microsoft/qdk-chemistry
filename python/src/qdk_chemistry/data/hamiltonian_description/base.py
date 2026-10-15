"""QDK/Chemistry Hamiltonian description base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data.base import DataClass

__all__: list[str] = ["HamiltonianDescription"]


class HamiltonianDescription(DataClass):
    """Abstract base for data classes that describe a Hamiltonian.

    :class:`~qdk_chemistry.data.QubitOperator` and :class:`~qdk_chemistry.data.ModelHamiltonianDescription`
    are examples; a plugin subclasses this class to add its own. Algorithms such as
    :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` and
    :class:`~qdk_chemistry.algorithms.PhaseEstimation` take one. Each subclass declares its own wire format,
    so this class has none.
    """
