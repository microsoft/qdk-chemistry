"""QDK/Chemistry controlled time evolution module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data.base import DataClass

from .base import TimeEvolutionUnitary


class ControlledTimeEvolutionUnitary(DataClass):
    """Data class for a controlled time evolution unitary."""

    # Class attribute for filename validation
    _data_type_name = "controlled_time_evolution_unitary"

    # Serialization version for this class
    # TODO: Add serialization support in ControlledTimeEvolutionUnitary
    _serialization_version = "0.1.0"

    def __init__(
        self, time_evolution_unitary: TimeEvolutionUnitary, control_bit: int, system_qubits: list[int]
    ) -> None:
        """Initialize a ControlledTimeEvolutionUnitary.

        Args:
            time_evolution_unitary: The time evolution unitary to be controlled.
            control_bit: The control bit index.
            system_qubits: The list of system qubits indices.

        """
        super().__init__()
        self.time_evolution_unitary = time_evolution_unitary
        self.control_bit = control_bit
        self.system_qubits = system_qubits

    def get_unitary_container_type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """
        return self.time_evolution_unitary.get_container_type()
