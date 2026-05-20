"""QDK/Chemistry unitary circuit mapper abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

__all__: list[str] = ["CircuitMapper", "CircuitMapperFactory"]


class CircuitMapper(Algorithm):
    """Base class for unitary circuit mappers in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the CircuitMapper."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, evolution: UnitaryRepresentation) -> Circuit:
        """Construct a Circuit representing the given UnitaryRepresentation.

        Args:
            evolution: The unitary representation.

        Returns:
            Circuit: A Circuit representing the given UnitaryRepresentation.

        """


class CircuitMapperFactory(AlgorithmFactory):
    """Factory class for creating CircuitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return circuit_mapper as the algorithm type name."""
        return "circuit_mapper"

    def default_algorithm_name(self) -> str:
        """Return pauli_sequence as the default algorithm name."""
        return "pauli_sequence"
