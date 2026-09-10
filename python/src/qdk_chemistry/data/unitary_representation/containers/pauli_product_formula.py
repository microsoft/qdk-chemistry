"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_float, _hash_int, _hash_str, _hash_uint

from .base import UnitaryContainer

__all__ = [
    "BatchedExponentiatedPauliTerm",
    "ConjugatedExponentiatedPauliTerm",
    "ExponentiatedPauliTerm",
    "PauliProductFormulaContainer",
]


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


@dataclass(frozen=True)
class BatchedExponentiatedPauliTerm:
    r"""Equal-angle Pauli exponentials evaluated by Hamming-weight phasing.

    The operation is :math:`\prod_j e^{-i\theta P_j}`. The Pauli strings must
    have disjoint support so their parity representatives can be accumulated in
    one Hamming-weight register.
    """

    pauli_terms: Sequence[dict[int, str]]
    """Pairwise-disjoint Pauli strings sharing one rotation angle."""

    angle: float
    """The common rotation angle."""

    def __post_init__(self) -> None:
        """Validate the Hamming-weight phasing invariants."""
        if len(self.pauli_terms) < 2:
            raise ValueError("A batched Pauli exponential requires at least two Pauli strings.")
        support: set[int] = set()
        for pauli_term in self.pauli_terms:
            if not pauli_term:
                raise ValueError("A batched Pauli exponential cannot contain the identity term.")
            overlap = support.intersection(pauli_term)
            if overlap:
                raise ValueError(f"Batched Pauli exponentials must have disjoint support; overlap: {sorted(overlap)}.")
            support.update(pauli_term)


@dataclass(frozen=True)
class ConjugatedExponentiatedPauliTerm:
    r"""A structured conjugation :math:`V D V^\dagger`.

    Circuit mappers lower this to Q# ``within``/``apply`` syntax. Consequently,
    Q# controls only the ``apply`` block when the complete operation is controlled.
    """

    within_terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm]
    """The factors forming :math:`V` in execution order."""

    apply_terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm]
    """The factors forming :math:`D` in execution order."""

    def __post_init__(self) -> None:
        """Require both sides of the conjugation to be explicit."""
        if not self.within_terms:
            raise ValueError("A conjugated Pauli exponential requires at least one within term.")
        if not self.apply_terms:
            raise ValueError("A conjugated Pauli exponential requires at least one apply term.")


class PauliProductFormulaContainer(UnitaryContainer):
    r"""Dataclass for a Pauli product formula container.

    A Pauli Product Formula decomposes a time-evolution operator :math:`U(t) = e^{-i H t}`,
    into a product of exponentials of Pauli strings. A single product-formula step is represented as
    :math:`U_{\mathrm{step}}(t) = \prod_{j \in \pi} e^{-i \theta_j P_j}`, where:

    * :math:`P_j` is a Pauli string
    * :math:`\theta_j` is the rotation angle for that term
    * :math:`\prod_{j \in \pi}` is a permutation defining the multiplication order

    The optional ``conjugating_terms`` represent :math:`V` in
    :math:`V U_{\mathrm{step}}^r V^\dagger`. Circuit mappers preserve that
    conjugation structurally so Q# controls only the repeated body.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for product-formula containers.

        Returns:
            ``"pauli_product_formula_container"``.

        """
        return "pauli_product_formula_container"

    # Serialization version for this class
    _serialization_version = "0.2.1"

    def __init__(
        self,
        step_terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm],
        step_reps: int,
        num_qubits: int,
        scale: float = 1.0,
        conjugating_terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm] | None = None,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The list of exponentiated Pauli terms in a single step.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.
            scale: The evolution time used for eigenvalue-phase conversion.
            conjugating_terms: Terms forming the one-time conjugation around the repeated step.

        Raises:
            TypeError: If ``step_reps`` is not an integer.
            ValueError: If ``step_reps`` is not positive.

        """
        # bool is an int subclass, but True as a repetition count is always a mistake.
        if isinstance(step_reps, bool) or not isinstance(step_reps, int | np.integer):
            raise TypeError(f"step_reps must be an integer, got {type(step_reps).__name__}.")
        if step_reps <= 0:
            raise ValueError(f"step_reps must be a positive integer, got {step_reps}.")

        self.conjugating_terms = [] if conjugating_terms is None else list(conjugating_terms)
        self.step_terms = list(step_terms)
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

        Raises:
            ValueError: If the evolution time (``scale``) is zero, so the phase carries no
                energy information and the inversion ``E = -angle / t`` is undefined.

        """
        if self.scale == 0:
            raise ValueError(
                "Cannot recover an eigenvalue: the evolution time (scale) is zero, so the "
                "measured phase carries no energy information and E = -angle / t is undefined. "
                "Build the unitary with a non-zero evolution time."
            )
        angle = (phase_fraction % 1.0) * (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return float(-angle / self.scale)

    @staticmethod
    def _hash_pauli_term(h: Any, pauli_term: dict[int, str]) -> None:
        """Feed one sparse Pauli string into *h*."""
        _hash_uint(h, len(pauli_term))
        for qubit_idx in sorted(pauli_term):
            _hash_int(h, qubit_idx)
            _hash_str(h, pauli_term[qubit_idx])

    @classmethod
    def _hash_groups(
        cls,
        h: Any,
        terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm],
    ) -> None:
        """Feed a sequence of plain or batched terms into *h*."""
        _hash_uint(h, len(terms))
        for term in terms:
            if isinstance(term, ExponentiatedPauliTerm):
                _hash_str(h, "term")
                cls._hash_pauli_term(h, term.pauli_term)
                _hash_float(h, term.angle)
            else:
                _hash_str(h, "batch")
                _hash_uint(h, len(term.pauli_terms))
                for pauli_term in term.pauli_terms:
                    cls._hash_pauli_term(h, pauli_term)
                _hash_float(h, term.angle)

    def _hash_update(self, h: Any) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "pauli_product_formula")
        _hash_str(h, "conjugating")
        self._hash_groups(h, self.conjugating_terms)
        _hash_str(h, "step")
        _hash_uint(h, len(self.step_terms))
        for term in self.step_terms:
            if isinstance(term, ConjugatedExponentiatedPauliTerm):
                _hash_str(h, "conjugated")
                self._hash_groups(h, term.within_terms)
                self._hash_groups(h, term.apply_terms)
            else:
                self._hash_groups(h, [term])
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
        # Validate permutation
        if len(permutation) != len(self.step_terms):
            raise ValueError(
                f"Permutation length ({len(permutation)}) must match the number of terms ({len(self.step_terms)})."
            )
        if set(permutation) != set(range(len(self.step_terms))):
            raise ValueError(f"Invalid permutation: must be a permutation of [0, 1, ..., {len(self.step_terms) - 1}].")

        reordered_step_terms: list[
            ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm
        ] = []
        for i in permutation:
            reordered_step_terms.append(self.step_terms[i])

        return PauliProductFormulaContainer(
            step_terms=reordered_step_terms,
            step_reps=self.step_reps,
            num_qubits=self._num_qubits,
            scale=self.scale,
            conjugating_terms=self.conjugating_terms,
        )

    def combine(self, other_container: "PauliProductFormulaContainer", atol=1e-12) -> "PauliProductFormulaContainer":
        r"""Combine two Trotter evolutions, merging adjacent identical Pauli terms.

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

        Raises:
            ValueError: If the two containers act on a different number of qubits or
                carry a different ``scale``; or if either container contains explicit
                batching or conjugation structure, which ``combine`` cannot flatten safely.

        Note:
            ``atol`` is a *screening* threshold, not an error bound. Dropping a fused
            term whose summed angle falls below ``atol`` discards up to ``atol`` of
            rotation on that Pauli string, so merging a chain that drops :math:`k`
            terms permits up to :math:`k \cdot \mathrm{atol}` of accumulated operator
            error -- the error is additive in the number of terms removed, and is not
            re-scaled by how many were kept. The default (:math:`10^{-12}`) sits far
            below typical Trotter error, but a caller fusing many containers, or one
            needing a guaranteed operator-norm bound, should size ``atol`` against the
            total number of expected cancellations rather than a single one.

        """
        for label, container in (("self", self), ("other_container", other_container)):
            if container.conjugating_terms or any(
                not isinstance(term, ExponentiatedPauliTerm) for term in container.step_terms
            ):
                raise ValueError(
                    f"Cannot combine: {label} contains batched or conjugated terms. "
                    "Map structured product formulas to circuits before composing them."
                )
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
            terms = cast("list[ExponentiatedPauliTerm]", container.step_terms)
            for term in terms * container.step_reps:
                if merged and merged[-1].pauli_term == term.pauli_term:
                    new_angle = merged[-1].angle + term.angle
                    if abs(new_angle) > atol:
                        merged[-1] = ExponentiatedPauliTerm(pauli_term=term.pauli_term, angle=new_angle)
                    else:
                        merged.pop()
                else:
                    merged.append(term)
        return PauliProductFormulaContainer(
            step_terms=merged,
            step_reps=1,
            num_qubits=self.num_qubits,
            scale=self.scale,
        )

    def to_json(self) -> dict[str, Any]:
        """Convert the PauliProductFormulaContainer to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the PauliProductFormulaContainer

        """
        if self.conjugating_terms or any(not isinstance(term, ExponentiatedPauliTerm) for term in self.step_terms):
            raise ValueError("Structured Pauli product formulas cannot be serialized.")

        terms = cast("list[ExponentiatedPauliTerm]", self.step_terms)
        data: dict[str, Any] = {
            "container_type": self.type,
            "step_terms": [
                {"pauli_term": {str(k): v for k, v in term.pauli_term.items()}, "angle": term.angle} for term in terms
            ],
            "step_reps": self.step_reps,
            "num_qubits": self.num_qubits,
            "scale": self.scale,
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the PauliProductFormulaContainer to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """
        if self.conjugating_terms or any(not isinstance(term, ExponentiatedPauliTerm) for term in self.step_terms):
            raise ValueError("Structured Pauli product formulas cannot be serialized.")

        terms = cast("list[ExponentiatedPauliTerm]", self.step_terms)
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["step_reps"] = self.step_reps
        group.attrs["num_qubits"] = self.num_qubits
        group.attrs["scale"] = self.scale

        step_terms_group = group.create_group("step_terms")
        for i, term in enumerate(terms):
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
        step_terms: list[ExponentiatedPauliTerm] = []
        for i, term_data in enumerate(json_data["step_terms"]):
            pauli_term: dict[int, str] = {}
            for key, value in term_data["pauli_term"].items():
                if not isinstance(key, str):
                    raise TypeError(f"step_terms[{i}].pauli_term: expected str key, got {type(key).__name__} ({key!r})")
                try:
                    qubit_index = int(key)
                except ValueError as exc:
                    raise ValueError(
                        f"step_terms[{i}].pauli_term: key {key!r} is not a valid integer qubit index"
                    ) from exc
                if str(qubit_index) != key:
                    raise ValueError(
                        f"step_terms[{i}].pauli_term: key {key!r} is not a canonical integer "
                        f"(expected {str(qubit_index)!r})"
                    )
                pauli_term[qubit_index] = value
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
            pauli_term_group = term_group["pauli_term"]
            pauli_term = {
                int(qubit_index): pauli_term_group.attrs[qubit_index] for qubit_index in pauli_term_group.attrs
            }
            step_terms.append(ExponentiatedPauliTerm(pauli_term=pauli_term, angle=term_group.attrs["angle"]))

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
        lines.append(f"  Number of conjugating terms: {len(self.conjugating_terms)}")
        lines.append(f"  Number of step terms: {len(self.step_terms)}")
        lines.append(f"  Step repetitions: {self.step_reps}")
        return "\n".join(lines)
