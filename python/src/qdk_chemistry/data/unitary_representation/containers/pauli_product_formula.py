"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_float, _hash_int, _hash_str, _hash_uint

from .base import UnitaryContainer

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


class PauliProductFormulaContainer(UnitaryContainer):
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

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for product-formula containers.

        Returns:
            ``"pauli_product_formula_container"``.

        """
        return "pauli_product_formula_container"

    # Serialization version for this class
    _serialization_version = "0.2.0"

    def __init__(
        self,
        step_terms: Sequence[ExponentiatedPauliTerm],
        step_reps: int,
        num_qubits: int,
        scale: float = 1.0,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The sequence of exponentiated Pauli terms in a single step.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.
            scale: The evolution time used for eigenvalue-phase conversion.

        Raises:
            TypeError: If ``step_reps`` is not an integer.
            ValueError: If ``step_reps`` is not positive.

        """
        # bool is an int subclass, but True as a repetition count is always a mistake.
        if isinstance(step_reps, bool) or not isinstance(step_reps, int | np.integer):
            raise TypeError(f"step_reps must be an integer, got {type(step_reps).__name__}.")
        if step_reps <= 0:
            raise ValueError(f"step_reps must be a positive integer, got {step_reps}.")

        self.step_terms: Sequence[ExponentiatedPauliTerm] = step_terms
        self.step_reps = int(step_reps)
        self._num_qubits = num_qubits
        self.scale = scale
        super().__init__()

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        r"""Recover a Hamiltonian eigenvalue from a time-evolution phase.

        For :math:`U(t) = e^{-iHt}` an eigenstate with energy :math:`E` acquires
        phase :math:`e^{-iEt}`, so QPE measures :math:`\varphi = (-Et / 2\pi) \bmod 1`.
        Inverting gives ``E = -angle / t``.

        Args:
            phase_fraction: Measured phase fraction :math:`\varphi \in [0, 1)`.

        Returns:
            float: The corresponding Hamiltonian eigenvalue.

        """
        angle = (phase_fraction % 1.0) * (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return float(-angle / self.scale)

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "pauli_product_formula")
        _hash_uint(h, len(self.step_terms))
        for term in self.step_terms:
            _hash_uint(h, len(term.pauli_term))
            for qubit_idx in sorted(term.pauli_term.keys()):
                _hash_int(h, qubit_idx)
                _hash_str(h, term.pauli_term[qubit_idx])
            _hash_float(h, term.angle)
        _hash_int(h, self.step_reps)
        _hash_int(h, self._num_qubits)
        _hash_float(h, self.scale)

    @property
    def type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            The type of the unitary container.

        """
        return "pauli_product_formula"

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits the unitary acts on.

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
        if len(permutation) != len(self.step_terms):
            raise ValueError(
                f"Permutation length ({len(permutation)}) must match the number of terms ({len(self.step_terms)})."
            )
        if set(permutation) != set(range(len(self.step_terms))):
            raise ValueError(f"Invalid permutation: must be a permutation of [0, 1, ..., {len(self.step_terms) - 1}].")
        return PauliProductFormulaContainer(
            [self.step_terms[i] for i in permutation], self.step_reps, self.num_qubits, self.scale
        )

    def combine(self, other_container: "PauliProductFormulaContainer", atol=1e-12) -> "PauliProductFormulaContainer":
        """Compose two evolutions, fusing only adjacent equal Pauli factors.

        Each input's repetitions are consumed in order. Angles are added sequentially;
        removing a cancelled pair can expose another matching pair on the stack.

        Args:
            other_container: Evolution to append, with matching register width and scale.
            atol: Drop a merged rotation when its absolute angle is at most this tolerance.

        Returns:
            A formula with ``step_reps=1``.

        """
        if self.num_qubits != other_container.num_qubits:
            raise ValueError(
                f"Cannot combine PauliProductFormulaContainer instances with different "
                f"num_qubits (self.num_qubits={self.num_qubits}, "
                f"other_container.num_qubits={other_container.num_qubits})."
            )
        if not np.isclose(self.scale, other_container.scale):
            raise ValueError(
                f"Cannot combine PauliProductFormulaContainer instances with different "
                f"scale (self.scale={self.scale}, other_container.scale={other_container.scale})."
            )

        merged: list[ExponentiatedPauliTerm] = []
        for container in (self, other_container):
            for _ in range(container.step_reps):
                for term in container.step_terms:
                    if merged and merged[-1].pauli_term == term.pauli_term:
                        angle = merged[-1].angle + term.angle
                        if abs(angle) > atol:
                            merged[-1] = ExponentiatedPauliTerm(term.pauli_term, angle)
                        else:
                            merged.pop()
                    else:
                        merged.append(term)
        return PauliProductFormulaContainer(merged, 1, self.num_qubits, self.scale)

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
        }
        data.update(step_reps=self.step_reps, num_qubits=self.num_qubits, scale=self.scale)
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
        group.attrs["scale"] = self.scale

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
            scale=json_data.get("scale", 1.0),
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
        for i in range(len(step_terms_group)):
            term_group = step_terms_group[f"term_{i}"]
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
            scale=float(group.attrs.get("scale", 1.0)),
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
