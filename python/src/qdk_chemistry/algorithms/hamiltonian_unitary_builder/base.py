"""QDK/Chemistry Hamiltonian unitary builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, Settings, UnitaryRepresentation

__all__: list[str] = [
    "HamiltonianUnitaryBuilder",
    "HamiltonianUnitaryBuilderFactory",
    "TimeEvolutionBuilder",
    "TimeEvolutionSettings",
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


class TimeEvolutionSettings(Settings):
    """Base settings for time evolution builders."""

    def __init__(self):
        """Initialize TimeEvolutionSettings with default values.

        Attributes:
            time: The evolution time.
            power: The power to raise the unitary to (e.g. 2k for U^{2k}).
            power_strategy: How to realize U^power. ``"rescale"`` multiplies
                the time by the power; ``"repeat"`` repeats the base circuit.

        """
        super().__init__()
        self._set_default("time", "float", 0.0, "The evolution time.")
        self._set_default("power", "int", 1, "The power to raise the unitary to.")
        self._set_default(
            "power_strategy",
            "string",
            "repeat",
            "Strategy for U^power: 'rescale' scales time, 'repeat' repeats the circuit.",
            ["rescale", "repeat"],
        )


class TimeEvolutionBuilder(HamiltonianUnitaryBuilder):
    """Base class for time evolution Builders in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the TimeEvolutionBuilder."""
        super().__init__()

    def _resolve_power(self) -> tuple[float, int]:
        """Resolve the power setting into effective time scale and power repetitions.

        Based on the ``power`` and ``power_strategy`` settings, returns:
        - For ``"rescale"``: (time * power, 1) — scales the evolution time.
        - For ``"repeat"``: (time, power) — repeats the base circuit.

        Returns:
            A tuple (effective_time, power_repetitions).

        """
        time: float = self._settings.get("time")
        power: int = self._settings.get("power")
        strategy: str = self._settings.get("power_strategy")
        if strategy == "rescale":
            return time * power, 1
        return time, power

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct a UnitaryRepresentation representing the time evolution unitary for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.

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
