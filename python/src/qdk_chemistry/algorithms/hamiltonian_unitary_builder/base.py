"""QDK/Chemistry Hamiltonian unitary builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    FlatPartition,
    LayeredPartition,
    QubitHamiltonian,
    Settings,
    TermPartition,
    UnitaryRepresentation,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = [
    "HamiltonianUnitaryBuilder",
    "HamiltonianUnitaryBuilderFactory",
    "HamiltonianUnitaryBuilderSettings",
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


class HamiltonianUnitaryBuilderSettings(Settings):
    """Base settings for Hamiltonian unitary builders."""

    def __init__(self):
        """Initialize HamiltonianUnitaryBuilderSettings with default values.

        Attributes:
            power: The exponent to which the unitary is raised.

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power to raise the unitary to.")


class TimeEvolutionSettings(HamiltonianUnitaryBuilderSettings):
    """Base settings for time evolution builders."""

    def __init__(self):
        r"""Initialize TimeEvolutionSettings with default values.

        Attributes:
            time: The evolution time.
            power_strategy: The strategy to construct :math:`U^{\\text{power}}`:

                * ``"rescale"``: produce a single step with effective time :math:`t \\cdot \\text{power}`.
                * ``"repeat"``: repeat the base :math:`U(t)` ``power`` times.

        """
        super().__init__()
        self._set_default("time", "float", 0.0, "The evolution time.")
        self._set_default(
            "power_strategy",
            "string",
            "repeat",
            "The strategy to construct U^power: 'rescale' multiplies evolution time by power; "
            "'repeat' repeats the base U power times.",
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

    def _group_terms(
        self,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[list[QubitHamiltonian]]:
        """Group Hamiltonian terms for decomposition."""
        partition = qubit_hamiltonian.term_partition
        if partition is not None:
            Logger.debug(
                f"{self.name().capitalize()}: consuming QubitHamiltonian.term_partition "
                f"(strategy={partition.strategy!r}, num_groups={partition.num_groups})."
            )
            return self._groups_from_partition(qubit_hamiltonian, partition)

        Logger.debug(
            f"{self.name().capitalize()}: no term_partition present; treating each Pauli term as its own group."
        )
        return [
            [
                QubitHamiltonian(
                    pauli_strings=[label],
                    coefficients=[coeff],
                    encoding=qubit_hamiltonian.encoding,
                    fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
                )
            ]
            for label, coeff in zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True)
        ]

    def _groups_from_partition(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        partition: TermPartition,
    ) -> list[list[QubitHamiltonian]]:
        """Materialise a :class:`TermPartition` into sub-groups."""
        labels = qubit_hamiltonian.pauli_strings
        coeffs = qubit_hamiltonian.coefficients
        encoding = qubit_hamiltonian.encoding
        fmo = qubit_hamiltonian.fermion_mode_order

        def _make(indices: tuple[int, ...]) -> QubitHamiltonian:
            return QubitHamiltonian(
                pauli_strings=[labels[i] for i in indices],
                coefficients=np.asarray([coeffs[i] for i in indices]),
                encoding=encoding,
                fermion_mode_order=fmo,
            )

        if isinstance(partition, LayeredPartition):
            layered_groups = partition.groups
        elif isinstance(partition, FlatPartition):
            layered_groups = tuple((g,) for g in partition.groups)
        else:
            raise TypeError(
                f"Unsupported TermPartition subtype: {type(partition).__name__}. "
                "Expected FlatPartition or LayeredPartition."
            )

        groups: list[list[QubitHamiltonian]] = [
            [_make(layer) for layer in group_layers if layer] for group_layers in layered_groups
        ]

        groups = [g for g in groups if g]
        groups.sort(key=len)
        return groups


class HamiltonianUnitaryBuilderFactory(AlgorithmFactory):
    """Factory class for creating HamiltonianUnitaryBuilder instances."""

    def algorithm_type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"

    def default_algorithm_name(self) -> str:
        """Return Trotter as the default algorithm name."""
        return "trotter"
