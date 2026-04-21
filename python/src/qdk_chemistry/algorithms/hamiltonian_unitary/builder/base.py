"""QDK/Chemistry Hamiltonian unitary builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation

__all__: list[str] = [
    "HamiltonianUnitaryBuilder",
    "HamiltonianUnitaryBuilderFactory",
    "TimeEvolutionBuilder",
]


class HamiltonianUnitaryBuilder(Algorithm):
    """Base class for Hamiltonian unitary builders in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the HamiltonianUnitaryBuilder."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct a UnitaryRepresentation for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.

        Returns:
            UnitaryRepresentation: A UnitaryRepresentation for the given QubitHamiltonian.

        """

    # ------------------------------------------------------------------
    # Shared helpers used by Trotter, qDRIFT, and partially-randomized
    # builders.
    # ------------------------------------------------------------------

    @staticmethod
    def _pauli_label_to_map(label: str) -> dict[int, str]:
        """Translate a Pauli label to a mapping ``qubit -> {X, Y, Z}``.

        Args:
            label: Pauli string label in little-endian ordering.

        Returns:
            Dictionary assigning each non-identity qubit index to its Pauli axis.

        """
        mapping: dict[int, str] = {}
        for index, char in enumerate(reversed(label)):  # reversed: right-most char -> qubit 0
            if char != "I":
                mapping[index] = char
        return mapping


class TimeEvolutionBuilder(HamiltonianUnitaryBuilder):
    """Base class for time evolution Builders in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the TimeEvolutionBuilder."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float = 0.0) -> UnitaryRepresentation:
        """Construct a UnitaryRepresentation representing the time evolution unitary for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.
            time: The evolution time.

        Returns:
            UnitaryRepresentation: A UnitaryRepresentation representing the evolution of the given QubitHamiltonian.

        """


class HamiltonianUnitaryBuilderFactory(AlgorithmFactory):
    """Factory class for creating HamiltonianUnitaryBuilder instances."""

    def algorithm_type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"

    def default_algorithm_name(self) -> str:
        """Return Trotter as the default algorithm name."""
        return "trotter"
