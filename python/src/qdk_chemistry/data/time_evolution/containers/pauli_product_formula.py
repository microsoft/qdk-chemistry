"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import h5py

from .base import TimeEvolutionUnitaryContainer

__all__ = ["EvolutionOrdering", "ExponentiatedPauliTerm", "PauliProductFormulaContainer"]


@dataclass(frozen=True)
class ExponentiatedPauliTerm:
    """Dataclass for an exponentiated Pauli term.

    Attributes:
        pauli_term: A dictionary mapping qubit indices to Pauli operators ('X', 'Y', 'Z').
        angle: The rotation angle for the exponentiation.

    Example:
        >>> term = ExponentiatedPauliTerm(pauli_term={0: 'X', 2: 'Z'}, angle=1.5708)

    """

    pauli_term: dict[int, str]
    angle: float


@dataclass(frozen=True)
class EvolutionOrdering:
    """Data class for evolution ordering.

    Attributes:
        indices: List of indices representing the order of evolution.

    Example:
        >>> # Apply the third term first, then the first, and the second lastly
        >>> ordering = EvolutionOrdering(indices=[2, 0, 1])

    """

    indices: list[int]

    def validate_ordering(self, num_terms: int) -> None:
        """Validate the evolution ordering.

        Args:
            num_terms: The total number of terms to be ordered.

        Raises:
            ValueError: If the ordering is invalid.

        """
        if len(self.indices) != num_terms:
            raise ValueError("Evolution ordering length must match the number of terms.")

        if sorted(self.indices) != list(range(num_terms)):
            raise ValueError("Evolution ordering must be a permutation of term indices.")


class PauliProductFormulaContainer(TimeEvolutionUnitaryContainer):
    """Data class for a Pauli product formula container."""

    # Class attribute for filename validation
    _data_type_name = "pauli_product_formula_container"

    def __init__(
        self,
        step_terms: list[ExponentiatedPauliTerm],
        evolution_ordering: EvolutionOrdering,
        step_reps: int,
        num_qubits: int,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The list of exponentiated Pauli terms in a single step.
            evolution_ordering: A list of indices representing the order of evolution terms.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.

        """
        self.step_terms = step_terms
        evolution_ordering.validate_ordering(len(step_terms))
        self.evolution_ordering = evolution_ordering
        self.step_reps = step_reps
        self._num_qubits = num_qubits
        super().__init__()

    @property
    def type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """
        return "pauli_product_formula"

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on.

        Returns:
            The number of qubits.

        """
        return self._num_qubits

    def set_ordering(self, evolution_ordering: EvolutionOrdering) -> None:
        """Set the evolution ordering.

        Args:
            evolution_ordering: A list of indices representing the order of evolution terms.

        """
        evolution_ordering.validate_ordering(len(self.step_terms))
        self.evolution_ordering = evolution_ordering

    def to_json(self) -> dict[str, Any]:
        """Convert the TimeEvolutionUnitary to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the TimeEvolutionUnitary

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "step_terms": [{"pauli_term": term.pauli_term, "angle": term.angle} for term in self.step_terms],
            "evolution_ordering": self.evolution_ordering.indices,
            "step_reps": self.step_reps,
            "num_qubits": self.num_qubits,
        }
        return data

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the TimeEvolutionUnitary to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """
        group.attrs["container_type"] = self.type
        group.attrs["step_reps"] = self.step_reps
        group.attrs["num_qubits"] = self.num_qubits

        step_terms_group = group.create_group("step_terms")
        for i, term in enumerate(self.step_terms):
            term_group = step_terms_group.create_group(f"term_{i}")
            term_group.attrs["angle"] = term.angle
            pauli_term_group = term_group.create_group("pauli_term")
            for qubit_index, pauli_operator in term.pauli_term.items():
                pauli_term_group.attrs[str(qubit_index)] = pauli_operator

        group.attrs["evolution_ordering"] = self.evolution_ordering.indices

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "TimeEvolutionUnitaryContainer":
        """Create TimeEvolutionUnitary from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            TimeEvolutionUnitaryContainer

        """
        step_terms = [
            ExponentiatedPauliTerm(
                pauli_term=term_data["pauli_term"],
                angle=term_data["angle"],
            )
            for term_data in json_data["step_terms"]
        ]
        evolution_ordering = EvolutionOrdering(indices=json_data["evolution_ordering"])
        step_reps = json_data["step_reps"]
        num_qubits = json_data["num_qubits"]
        return cls(
            step_terms=step_terms,
            evolution_ordering=evolution_ordering,
            step_reps=step_reps,
            num_qubits=num_qubits,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "TimeEvolutionUnitaryContainer":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            TimeEvolutionUnitaryContainer

        """
        step_reps = group.attrs["step_reps"]
        num_qubits = group.attrs["num_qubits"]

        step_terms: list[ExponentiatedPauliTerm] = []
        step_terms_group = group["step_terms"]
        for term_name in step_terms_group:
            term_group = step_terms_group[term_name]
            angle = term_group.attrs["angle"]
            pauli_term: dict[int, str] = {}
            pauli_term_group = term_group["pauli_term"]
            for qubit_index_str in pauli_term_group.attrs:
                qubit_index = int(qubit_index_str)
                pauli_operator = pauli_term_group.attrs[qubit_index_str]
                pauli_term[qubit_index] = pauli_operator
            step_terms.append(ExponentiatedPauliTerm(pauli_term=pauli_term, angle=angle))

        evolution_ordering_indices = list(group.attrs["evolution_ordering"])
        evolution_ordering = EvolutionOrdering(indices=evolution_ordering_indices)

        return cls(
            step_terms=step_terms,
            evolution_ordering=evolution_ordering,
            step_reps=step_reps,
            num_qubits=num_qubits,
        )

    def get_summary(self) -> str:
        """Get summary of time evolution unitary.

        Returns:
            str: Summary string describing the TimeEvolutionUnitary's contents and properties

        """
        lines = ["Pauli Product Formula Container"]
        lines.append(f"  Number of qubits: {self.num_qubits}")
        lines.append(f"  Number of step terms: {len(self.step_terms)}")
        lines.append(f"  Step repetitions: {self.step_reps}")
        return "\n".join(lines)
