"""QDK/Chemistry circuit mapper for controlled-unitary abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.controlled_unitary import ControlledUnitary

__all__: list[str] = ["ControlledCircuitMapper", "ControlledCircuitMapperFactory"]


class ControlledCircuitMapper(Algorithm):
    """Base class for circuit mapper for controlled-unitary in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the ControlledCircuitMapper."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        """Construct a Circuit representing the controlled unitary for the given ControlledUnitary.

        Args:
            controlled_unitary: The controlled unitary.

        Returns:
            Circuit: A Circuit representing the controlled unitary for the given ControlledUnitary.

        """


class ControlledCircuitMapperFactory(AlgorithmFactory):
    """Factory class for creating ControlledCircuitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def default_algorithm_name(self) -> str:
        """Return pauli_sequence as the default algorithm name."""
        return "pauli_sequence"
