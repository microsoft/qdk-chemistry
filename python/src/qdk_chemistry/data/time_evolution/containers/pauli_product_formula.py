"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import h5py

from .base import TimeEvolutionUnitaryContainer

__all__ = ["ExponentiatedPauliTerm", "PauliProductFormulaContainer"]


@dataclass(frozen=True)
class ExponentiatedPauliTerm:
    r"""Dataclass for an exponentiated Pauli term.

    A single exponential factor of the form :math:`e^{-i \theta P}`, where:
        * :math:`P` is a Pauli string (e.g., :math:`X_0 Z_2`)
        * :math:`\theta` is rotation angle
    """

    pauli_term: dict[int, str]
    """A dictionary mapping qubit indices to Pauli operators ('X', 'Y', 'Z')."""

    angle: float
    """The rotation angle for the exponentiation."""


class PauliProductFormulaContainer(TimeEvolutionUnitaryContainer):
    r"""Dataclass for a Pauli product formula container.

    A Pauli Product Formula decomposes a time-evolution operator :math:`U(t) = e^{-i H t}`,
    into a product of exponentials of Pauli strings. A single product-formula step is represented as
    :math:`U_{\mathrm{step}}(t) = \prod_{j \in \pi} e^{-i \theta_j P_j}`, where:

    * :math:`P_j` is a Pauli string
    * :math:`\theta_j` is the rotation angle for that term
    * :math:`\prod_{j \in \pi}` is a permutation defining the multiplication order

    The full time-evolution unitary is:
    :math:`U(t) \approx \left[ U_{\mathrm{step}}\!\left(\tfrac{t}{r}\right) \right]^{r}`,
    where ``step_reps = r`` is the number of repeated steps.
    """

    # Class attribute for filename validation
    _data_type_name = "pauli_product_formula_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        step_terms: list[ExponentiatedPauliTerm],
        step_reps: int,
        num_qubits: int,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The list of exponentiated Pauli terms in a single step.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.

        """
        self.step_terms = step_terms
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

    def reorder_terms(self, permutation: list[int]) -> "PauliProductFormulaContainer":
        """Reorder the Pauli terms according to a given permutation.

        Args:
            permutation: A list where ``permutation[i]`` gives the old index of the term
                that should be placed at position ``i`` in the reordered list.

        Returns:
            PauliProductFormulaContainer: A new container with the updated ordering.

        Note:
            ``permutation[i]`` is the old index for new position ``i``. For example,
            ``permutation = [2, 0, 1]`` yields ``new_terms = [old_terms[2], old_terms[0], old_terms[1]]``.

        """
        # Validate permutation
        if len(permutation) != len(self.step_terms):
            raise ValueError(
                f"Permutation length ({len(permutation)}) must match the number of terms ({len(self.step_terms)})."
            )
        if set(permutation) != set(range(len(self.step_terms))):
            raise ValueError(f"Invalid permutation: must be a permutation of [0, 1, ..., {len(self.step_terms) - 1}].")

        reordered_step_terms: list[ExponentiatedPauliTerm] = []
        for i in permutation:
            reordered_step_terms.append(self.step_terms[i])

        return PauliProductFormulaContainer(
            step_terms=reordered_step_terms,
            step_reps=self.step_reps,
            num_qubits=self._num_qubits,
        )

    def combine(self, other_container: "PauliProductFormulaContainer", atol=1e-12) -> "PauliProductFormulaContainer":
        """Combine two Trotter evolutions, merging adjacent identical Pauli terms.

        The terms from ``self`` (repeated ``step_reps`` times) are followed by the
        terms from ``other_container`` (also repeated according to its
        ``step_reps``). When two consecutive terms act with the same Pauli operator
        string (i.e., have identical ``pauli_term`` dictionaries), their rotation
        angles are summed into a single ``ExponentiatedPauliTerm``. If the summed
        angle has magnitude less than ``atol``, the resulting term is removed.

        Args:
            other_container: The second ``PauliProductFormulaContainer`` appended
                after this container.
            atol: Absolute tolerance used when deciding whether a merged term with
                a small rotation angle should be dropped.

        Returns:
            A single ``PauliProductFormulaContainer`` representing the combined
            evolution with adjacent identical terms fused.

        """
        num_qubits = max(self.num_qubits, other_container.num_qubits)

        merged: list[ExponentiatedPauliTerm] = []
        for step_terms, step_reps in (
            (self.step_terms, self.step_reps),
            (other_container.step_terms, other_container.step_reps),
        ):
            for _ in range(step_reps):
                for term in step_terms:
                    if merged and merged[-1].pauli_term == term.pauli_term:
                        new_angle = merged[-1].angle + term.angle
                        if abs(new_angle) > atol:
                            merged[-1] = ExponentiatedPauliTerm(pauli_term=term.pauli_term, angle=new_angle)
                        else:
                            merged.pop()
                    else:
                        merged.append(term)
        return PauliProductFormulaContainer(step_terms=merged, step_reps=1, num_qubits=num_qubits)

    def to_json(self) -> dict[str, Any]:
        """Convert the PauliProductFormulaContainer to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the PauliProductFormulaContainer

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "step_terms": [
                {"pauli_term": {str(k): v for k, v in term.pauli_term.items()}, "angle": term.angle}
                for term in self.step_terms
            ],
            "step_reps": self.step_reps,
            "num_qubits": self.num_qubits,
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the PauliProductFormulaContainer to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """
        self._add_hdf5_version(group)
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

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "PauliProductFormulaContainer":
        """Create PauliProductFormulaContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            PauliProductFormulaContainer

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        step_terms = []
        for i, term_data in enumerate(json_data["step_terms"]):
            pauli_term: dict[int, str] = {}
            for k, v in term_data["pauli_term"].items():
                if not isinstance(k, str):
                    raise TypeError(f"step_terms[{i}].pauli_term: expected str key, got {type(k).__name__} ({k!r})")
                try:
                    qubit_index = int(k)
                except ValueError as exc:
                    raise ValueError(
                        f"step_terms[{i}].pauli_term: key {k!r} is not a valid integer qubit index"
                    ) from exc
                if str(qubit_index) != k:
                    raise ValueError(
                        f"step_terms[{i}].pauli_term: key {k!r} is not a canonical integer "
                        f"(expected {str(qubit_index)!r})"
                    )
                pauli_term[qubit_index] = v
            step_terms.append(ExponentiatedPauliTerm(pauli_term=pauli_term, angle=term_data["angle"]))
        step_reps = json_data["step_reps"]
        num_qubits = json_data["num_qubits"]
        return cls(
            step_terms=step_terms,
            step_reps=step_reps,
            num_qubits=num_qubits,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "PauliProductFormulaContainer":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            PauliProductFormulaContainer

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
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

        return cls(
            step_terms=step_terms,
            step_reps=step_reps,
            num_qubits=num_qubits,
        )

    def get_summary(self) -> str:
        """Get summary of PauliProductFormulaContainer.

        Returns:
            str: Summary string describing the PauliProductFormulaContainer's contents and properties

        """
        lines = ["Pauli Product Formula Container"]
        lines.append(f"  Number of qubits: {self.num_qubits}")
        lines.append(f"  Number of step terms: {len(self.step_terms)}")
        lines.append(f"  Step repetitions: {self.step_reps}")
        return "\n".join(lines)
