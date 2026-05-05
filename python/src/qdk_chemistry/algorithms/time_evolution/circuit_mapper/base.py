"""QDK/Chemistry time evolution unitary circuit mapper abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

__all__: list[str] = ["EvolutionCircuitMapper", "EvolutionCircuitMapperFactory"]


class EvolutionCircuitMapper(Algorithm):
    """Base class for time evolution circuit mapper in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the EvolutionCircuitMapper."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, evolution: UnitaryRepresentation, *args, **kwargs) -> Circuit:
        """Construct a Circuit representing the unitary for the given UnitaryRepresentation.

        Args:
            evolution: The time evolution unitary.
            *args: Positional arguments, where the first argument is expected to be the
                time evolution unitary.
            **kwargs: Additional keyword arguments for concrete implementation.

        Returns:
            Circuit: A Circuit representing the unitary for the given UnitaryRepresentation.

        """


class EvolutionCircuitMapperFactory(AlgorithmFactory):
    """Factory class for creating EvolutionCircuitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return evolution_circuit_mapper as the algorithm type name."""
        return "evolution_circuit_mapper"

    def default_algorithm_name(self) -> str:
        """Return pauli_sequence as the default algorithm name."""
        return "pauli_sequence"
