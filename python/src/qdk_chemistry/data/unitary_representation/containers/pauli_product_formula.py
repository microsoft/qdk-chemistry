"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from itertools import pairwise
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
    Optional ``layer_offsets`` delimit disjoint-support layers within the stored step.
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
        step_terms: list[ExponentiatedPauliTerm],
        step_reps: int,
        num_qubits: int,
        scale: float = 1.0,
        *,
        layer_offsets: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The list of exponentiated Pauli terms in a single step.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.
            scale: The evolution time used for eigenvalue-phase conversion.
            layer_offsets: Disjoint-layer boundaries spanning the step, starting at zero.

        Raises:
            TypeError: If ``step_reps`` is not an integer.
            ValueError: If ``step_reps`` is not positive or the layer boundaries are invalid.

        """
        # bool is an int subclass, but True as a repetition count is always a mistake.
        if isinstance(step_reps, bool) or not isinstance(step_reps, int | np.integer):
            raise TypeError(f"step_reps must be an integer, got {type(step_reps).__name__}.")
        if step_reps <= 0:
            raise ValueError(f"step_reps must be a positive integer, got {step_reps}.")

        self.step_terms = step_terms
        self.layer_offsets = None if layer_offsets is None else tuple(layer_offsets)
        if self.layer_offsets is not None:
            offsets = self.layer_offsets
            if (
                not offsets
                or any(
                    isinstance(offset, bool | np.bool_) or not isinstance(offset, int | np.integer)
                    for offset in offsets
                )
                or offsets[0] != 0
                or offsets[-1] != len(step_terms)
                or any(stop <= start for start, stop in pairwise(offsets))
            ):
                raise ValueError("layer_offsets must strictly increase from zero to the number of step terms.")
            for start, stop in pairwise(offsets):
                occupied: set[int] = set()
                for term in step_terms[start:stop]:
                    support = {qubit for qubit, pauli in term.pauli_term.items() if pauli != "I"}
                    if not occupied.isdisjoint(support):
                        raise ValueError("Terms in each layer_offsets interval must have disjoint qubit supports.")
                    occupied.update(support)
            self.layer_offsets = tuple(int(offset) for offset in offsets)
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
        if self.layer_offsets is not None:
            _hash_str(h, "layer_offsets")
            _hash_uint(h, len(self.layer_offsets))
            for offset in self.layer_offsets:
                _hash_uint(h, offset)

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
            Declared layers become singleton layers after reordering.

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
            scale=self.scale,
            layer_offsets=None if self.layer_offsets is None else tuple(range(len(self.step_terms) + 1)),
        )

    def combine(self, other_container: "PauliProductFormulaContainer", atol=1e-12) -> "PauliProductFormulaContainer":
        """Combine two Trotter evolutions, merging adjacent identical Pauli terms.

        The terms from ``self`` (repeated ``step_reps`` times) are followed by the
        terms from ``other_container`` (also repeated according to its
        ``step_reps``). When two consecutive terms act with the same Pauli operator
        string (i.e., have identical ``pauli_term`` dictionaries), their rotation
        angles are summed into a single ``ExponentiatedPauliTerm``. If the summed
        angle has magnitude less than ``atol``, the resulting term is removed.
        Surviving factors retain their declared layer boundaries without regrouping.

        Args:
            other_container: The second ``PauliProductFormulaContainer`` appended
                after this container.
            atol: Absolute tolerance used when deciding whether a merged term with
                a small rotation angle should be dropped.

        Returns:
            A single ``PauliProductFormulaContainer`` representing the combined
            evolution with adjacent identical terms fused.

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
        layers = [0] if self.layer_offsets is not None or other_container.layer_offsets is not None else None
        for container in (self, other_container):
            offsets = (
                container.layer_offsets
                if container.layer_offsets is not None
                else tuple(range(len(container.step_terms) + 1))
                if layers is not None
                else (0, len(container.step_terms))
            )
            for _ in range(container.step_reps):
                for begin, end in pairwise(offsets):
                    layer_start = None
                    for term in container.step_terms[begin:end]:
                        if merged and merged[-1].pauli_term == term.pauli_term:
                            new_angle = merged[-1].angle + term.angle
                            if abs(new_angle) > atol:
                                merged[-1] = ExponentiatedPauliTerm(pauli_term=term.pauli_term, angle=new_angle)
                            else:
                                merged.pop()
                                if layers is not None:
                                    while len(layers) > 1 and layers[-1] >= len(merged):
                                        layers.pop()
                                if layer_start is not None and len(merged) <= layer_start:
                                    layer_start = None
                        else:
                            if layers is not None and layer_start is None:
                                layer_start = len(merged)
                                if layers[-1] != layer_start:
                                    layers.append(layer_start)
                            merged.append(term)
        if layers is not None and layers[-1] != len(merged):
            layers.append(len(merged))
        return PauliProductFormulaContainer(
            step_terms=merged,
            step_reps=1,
            num_qubits=self.num_qubits,
            scale=self.scale,
            layer_offsets=None if layers is None else tuple(layers),
        )

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
            "scale": self.scale,
        }
        if self.layer_offsets is not None:
            data["layer_offsets"] = list(self.layer_offsets)
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
        if self.layer_offsets is not None:
            group.create_dataset("layer_offsets", data=self.layer_offsets, dtype="int64")

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
            layer_offsets=json_data.get("layer_offsets"),
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
        for index in range(len(step_terms_group)):
            term_group = step_terms_group[f"term_{index}"]
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
            layer_offsets=tuple(group["layer_offsets"][()]) if "layer_offsets" in group else None,
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
