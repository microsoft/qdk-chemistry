"""QDK/Chemistry time evolution unitary builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary

__all__: list[str] = ["TimeEvolutionBuilder", "TimeEvolutionBuilderFactory"]


class TimeEvolutionBuilder(Algorithm):
    """Base class for time evolution Builders in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the TimeEvolutionBuilder."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct a TimeEvolutionUnitary representing the time evolution unitary for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.
            time: The evolution time.

        Returns:
            TimeEvolutionUnitary: A TimeEvolutionUnitary representing the evolution of the given QubitHamiltonian.

        """


class TimeEvolutionBuilderFactory(AlgorithmFactory):
    """Factory class for creating TimeEvolutionBuilder instances."""

    def algorithm_type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"

    def default_algorithm_name(self) -> str:
        """Return Trotter as the default algorithm name."""
        return "trotter"
