"""QDK/Chemistry time evolution base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import ABC, abstractmethod

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.base import DataClass


class TimeEvolutionUnitaryContainer(ABC):
    """Abstract class for a time evolution unitary container."""

    @property
    @abstractmethod
    def type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on.

        Returns:
            The number of qubits.

        """

    @abstractmethod
    def apply(self, state: Wavefunction) -> Wavefunction:
        """Apply the time evolution unitary to a given state.

        Args:
            state: The state to which the unitary is applied.

        Returns:
            The new state after applying the unitary.

        """


class TimeEvolutionUnitary(DataClass):
    """Data class for a time evolution unitary.

    Attributes:
        container (TimeEvolutionUnitaryContainer): The container for representing the time evolution unitary.

    """

    # Class attribute for filename validation
    _data_type_name = "time_evolution_unitary"

    # Serialization version for this class
    # TODO: Add serialization support in TimeEvolutionUnitaryContainer
    _serialization_version = "0.1.0"

    # Use keyword arguments to be future-proof
    def __init__(self, container: TimeEvolutionUnitaryContainer) -> None:
        """Initialize a TimeEvolutionUnitary."""
        super().__init__()
        self._container = container

    def apply(self, state: Wavefunction) -> Wavefunction:
        """Apply the time evolution unitary to a given state.

        Args:
            state: The quantum state to which the unitary is applied.

        Returns:
            The new quantum state after applying the unitary.

        """
        return self._container.apply(state)

    def get_container_type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary.

        """
        return self._container.type

    def get_num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on.

        Returns:
            The number of qubits.

        """
        return self._container.num_qubits
