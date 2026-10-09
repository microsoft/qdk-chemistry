"""QDK/Chemistry Hamiltonian description base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data.base import DataClass

__all__: list[str] = ["HamiltonianDescription"]


class HamiltonianDescription(DataClass):
    """Abstract class for a Hamiltonian description that a Hamiltonian unitary builder takes.

    :class:`~qdk_chemistry.data.QubitOperator` and :class:`~qdk_chemistry.data.ModelHamiltonianDescription`
    are Hamiltonian descriptions; a plugin subclasses this class to add its own.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for Hamiltonian descriptions.

        Returns:
            ``"hamiltonian_description"``.

        """
        return "hamiltonian_description"

    # Serialization version for this class
    _serialization_version = "0.1.0"
