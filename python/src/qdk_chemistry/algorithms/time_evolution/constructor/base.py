"""QDK/Chemistry time evolution unitary constructor abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary

__all__: list[str] = []


class TimeEvolutionConstructor(Algorithm):
    """Base class for time evolution constructors in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the TimeEvolutionConstructor."""
        super().__init__()

    def type_name(self) -> str:
        """Return time_evolution_constructor as the algorithm type name."""
        return "time_evolution_constructor"

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct a QuantumCircuit representing the controlled unitary for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian (QubitHamiltonian): The qubit Hamiltonian.
            time (float): The evolution time.

        Returns:
            TimeEvolutionUnitary: A TimeEvolutionUnitary representing the evolution of the given QubitHamiltonian.

        """


class TimeEvolutionConstructorFactory(AlgorithmFactory):
    """Factory class for creating TimeEvolutionConstructor instances."""

    def algorithm_type_name(self) -> str:
        """Return time_evolution_constructor as the algorithm type name."""
        return "time_evolution_constructor"

    def default_algorithm_name(self) -> str:
        """Return first_order_trotter as the default algorithm name."""
        return "first_order_trotter"
