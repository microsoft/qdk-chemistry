"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, pairwise, repeat
from math import isfinite
from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import (
    _hash_float,
    _hash_int,
    _hash_str,
    _hash_uint,
)

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


def _commute(terms: Sequence[ExponentiatedPauliTerm]) -> bool:
    """Certify QWC groups in sparse linear time, falling back to general commutation."""
    maps = [{q: p for q, p in term.pauli_term.items() if p != "I"} for term in terms]
    axes: dict[int, str] = {}
    qwc = True
    for word in maps:
        for qubit, axis in word.items():
            if axis not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli axis {axis!r} in commuting group.")
            if axes.setdefault(qubit, axis) != axis:
                qwc = False
    if qwc:
        return True
    # The public utility imports data classes through utils.__init__.
    from qdk_chemistry.utils.pauli_commutation import do_pauli_maps_commute  # noqa: PLC0415

    return all(do_pauli_maps_commute(a, b) for i, a in enumerate(maps) for b in maps[i + 1 :])


def _validate_groups(terms: Sequence[ExponentiatedPauliTerm], offsets: tuple[int, ...]) -> None:
    """Validate index coverage and commutation of each stored step group."""
    if (
        not offsets
        or any(isinstance(i, bool) or not isinstance(i, int | np.integer) for i in offsets)
        or offsets[0] != 0
        or offsets[-1] != len(terms)
        or any(b <= a for a, b in pairwise(offsets))
    ):
        raise ValueError("group_offsets must strictly increase from zero to the number of step terms.")
    for begin, end in pairwise(offsets):
        if not _commute(terms[begin:end]):
            raise ValueError("Terms in each group_offsets interval must commute.")


def _finite(angle: float) -> float:
    if not isfinite(angle):
        raise ValueError("Product-formula fusion requires finite angles and finite merged results.")
    return angle


def _merge_groups(
    left: Sequence[ExponentiatedPauliTerm], right: Sequence[ExponentiatedPauliTerm], atol: float
) -> list[ExponentiatedPauliTerm] | None:
    """Merge equal canonical word sets only when all words commute; None means no match."""
    words = [tuple(sorted((q, p) for q, p in t.pauli_term.items() if p != "I")) for t in chain(left, right)]
    if set(words[: len(left)]) != set(words[len(left) :]) or not _commute(left):
        return None
    angles: dict[tuple[tuple[int, str], ...], float] = {}
    for word, term in zip(words, chain(left, right), strict=True):
        angles[word] = _finite(angles.get(word, 0.0) + _finite(term.angle))
    return [ExponentiatedPauliTerm(dict(word), angle) for word, angle in angles.items() if abs(angle) > atol]


def _read_legacy_step_terms(
    num_qubits: int,
    term_offsets: np.ndarray,
    qubit_indices: np.ndarray,
    pauli_codes: np.ndarray,
    angles: np.ndarray,
) -> list[ExponentiatedPauliTerm]:
    """Convert the old packed 0.3.0 payload at the JSON/HDF5 loader boundary."""
    if (
        num_qubits <= 0
        or any(a.ndim != 1 or (a.size and a.dtype.kind not in "iu") for a in (term_offsets, qubit_indices, pauli_codes))
        or angles.ndim != 1
        or len(term_offsets) != len(angles) + 1
        or len(qubit_indices) != len(pauli_codes)
        or term_offsets[0] != 0
        or term_offsets[-1] != len(qubit_indices)
        or np.any(term_offsets[1:] < term_offsets[:-1])
        or np.any((qubit_indices < 0) | (qubit_indices >= num_qubits))
        or np.any((pauli_codes < 1) | (pauli_codes > 3))
    ):
        raise ValueError("Invalid packed product-formula arrays.")
    # Legacy terms used a dict: repeated indices retain their first position and last axis.
    return [
        ExponentiatedPauliTerm(
            {int(qubit_indices[i]): "IXYZ"[int(pauli_codes[i])] for i in range(int(begin), int(end))},
            float(angle),
        )
        for begin, end, angle in zip(term_offsets[:-1], term_offsets[1:], angles, strict=True)
    ]


class PauliProductFormulaContainer(UnitaryContainer):
    r"""Dataclass for a Pauli product formula container.

    A Pauli Product Formula decomposes a time-evolution operator :math:`U(t) = e^{-i H t}`,
    into a product of exponentials of Pauli strings. A single product-formula step is represented as
    :math:`U_{\mathrm{step}}(t) = \prod_{j \in \pi} e^{-i \theta_j P_j}`, where:

    * :math:`P_j` is a Pauli string
    * :math:`\theta_j` is the rotation angle for that term
    * :math:`\prod_{j \in \pi}` is a permutation defining the multiplication order

    Without endpoints, the full time-evolution unitary is:
    :math:`U(t) \approx \left[ U_{\mathrm{step}}\!\left(\tfrac{t}{r}\right) \right]^{r}`,
    where ``step_reps = r`` is the number of repeated steps.

    Optional ``beginning`` and ``end`` terms execute once before and after the repeated steps.
    All three term lists are stored explicitly without expanding repetitions;
    ``group_offsets`` certify commuting intervals of ``step_terms`` only.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for product-formula containers.

        Returns:
            ``"pauli_product_formula_container"``.

        """
        return "pauli_product_formula_container"

    # Serialization version for this class
    _serialization_version = "0.4.0"

    def __init__(
        self,
        step_terms: Sequence[ExponentiatedPauliTerm],
        step_reps: int,
        num_qubits: int,
        scale: float = 1.0,
        *,
        beginning: Sequence[ExponentiatedPauliTerm] = (),
        end: Sequence[ExponentiatedPauliTerm] = (),
        group_offsets: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The sequence of exponentiated Pauli terms in a single step.
            step_reps: The number of repetitions of the single step.
            num_qubits: The number of qubits the unitary acts on.
            scale: The evolution time used for eigenvalue-phase conversion.
            beginning: Terms executed once before the repeated steps.
            end: Terms executed once after the repeated steps.
            group_offsets: Strictly increasing commuting-group boundaries spanning step_terms, starting at zero.

        Raises:
            TypeError: If ``step_reps`` is not an integer.
            ValueError: If ``step_reps`` is not positive or ``group_offsets`` is invalid.

        """
        # bool is an int subclass, but True as a repetition count is always a mistake.
        if isinstance(step_reps, bool) or not isinstance(step_reps, int | np.integer):
            raise TypeError(f"step_reps must be an integer, got {type(step_reps).__name__}.")
        if step_reps <= 0:
            raise ValueError(f"step_reps must be a positive integer, got {step_reps}.")

        self.step_terms = [ExponentiatedPauliTerm(dict(t.pauli_term), t.angle) for t in step_terms]
        self.beginning = [ExponentiatedPauliTerm(dict(t.pauli_term), t.angle) for t in beginning]
        self.end = [ExponentiatedPauliTerm(dict(t.pauli_term), t.angle) for t in end]
        self.group_offsets = None if group_offsets is None else tuple(group_offsets)
        if self.group_offsets is not None:
            _validate_groups(self.step_terms, self.group_offsets)
            self.group_offsets = tuple(int(i) for i in self.group_offsets)
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
        for name, terms in (("beginning", self.beginning), ("end", self.end)):
            if not terms:
                continue
            _hash_str(h, name)
            _hash_uint(h, len(terms))
            for term in terms:
                _hash_uint(h, len(term.pauli_term))
                for qubit_idx in sorted(term.pauli_term.keys()):
                    _hash_int(h, qubit_idx)
                    _hash_str(h, term.pauli_term[qubit_idx])
                _hash_float(h, term.angle)
        _hash_int(h, self.step_reps)
        _hash_int(h, self._num_qubits)
        _hash_float(h, self.scale)
        if self.group_offsets is not None:
            _hash_str(h, "group_offsets")
            _hash_uint(h, len(self.group_offsets))
            for offset in self.group_offsets:
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

    @property
    def num_pauli_exponentials(self) -> int:
        """Count executed exponentials without expanding the repeated body."""
        return len(self.beginning) + self.step_reps * len(self.step_terms) + len(self.end)

    @property
    def num_stored_terms(self) -> int:
        """Count stored terms, independently of the repetition count."""
        return len(self.beginning) + len(self.step_terms) + len(self.end)

    def reorder_terms(self, permutation: list[int]) -> "PauliProductFormulaContainer":
        """Reorder the Pauli terms according to a given permutation.

        Only ``step_terms`` are permuted; endpoints are preserved and group certificates discarded.

        Args:
            permutation: A list where ``permutation[i]`` gives the old index of the term for new position ``i``.

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
            [self.step_terms[i] for i in permutation],
            self.step_reps,
            self.num_qubits,
            self.scale,
            beginning=self.beginning,
            end=self.end,
        )

    def combine(  # noqa: PLR0911 - Keep compact cases and the original fallback together.
        self, other_container: "PauliProductFormulaContainer | None" = None, atol: float = 1e-12
    ) -> "PauliProductFormulaContainer":
        """Fuse repetition boundaries, or append another formula.

        With no other formula, rewrite (L C R)^r as L (C merge(R,L))^(r-1) C R
        when L and R have equal commuting word sets. Missing metadata means
        singleton groups. A single commuting group instead scales its angles.
        Existing endpoints are left unchanged to avoid repeated endpoint growth;
        no within-step normalization or recursive optimization is performed.

        Two identical stored bodies have compact fast paths. Otherwise the original
        flatten-and-adjacent-merge algorithm returns step_reps=1, using memory
        proportional to the expanded output. Endpoints participate in that fallback.

        Args:
            other_container: Optional evolution to append, with matching width and compatible finite scale.
            atol: Finite nonnegative tolerance for dropping only merged near-zero angles.

        Returns:
            An equivalent formula retaining this container's scale.

        """
        if not isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative.")
        if other_container is None:
            if self.step_reps == 1 or not self.step_terms or self.beginning or self.end:
                return self
            offsets = self.group_offsets if self.group_offsets is not None else tuple(range(len(self.step_terms) + 1))
            _validate_groups(self.step_terms, offsets)
            if len(offsets) == 2:
                terms = [
                    ExponentiatedPauliTerm(t.pauli_term, _finite(t.angle * self.step_reps)) for t in self.step_terms
                ]
                return PauliProductFormulaContainer(terms, 1, self.num_qubits, self.scale, group_offsets=offsets)
            left, right = self.step_terms[: offsets[1]], self.step_terms[offsets[-2] :]
            boundary_terms = _merge_groups(right, left, atol)
            if boundary_terms is None:
                return self
            center = self.step_terms[offsets[1] : offsets[-2]]
            body_offsets = tuple(i - offsets[1] for i in offsets[1:-1])
            if boundary_terms:
                body_offsets += (len(center) + len(boundary_terms),)
            return PauliProductFormulaContainer(
                center + boundary_terms,
                self.step_reps - 1,
                self.num_qubits,
                self.scale,
                beginning=left,
                end=center + right,
                group_offsets=body_offsets,
            )

        if self.num_qubits != other_container.num_qubits:
            raise ValueError(
                f"Cannot combine PauliProductFormulaContainer instances with different "
                f"num_qubits (self.num_qubits={self.num_qubits}, "
                f"other_container.num_qubits={other_container.num_qubits})."
            )
        if (
            not isfinite(self.scale)
            or not isfinite(other_container.scale)
            or not np.isclose(self.scale, other_container.scale)
        ):
            raise ValueError(
                f"Cannot combine PauliProductFormulaContainer instances with different or nonfinite "
                f"scale (self.scale={self.scale}, other_container.scale={other_container.scale})."
            )
        if self.step_terms == other_container.step_terms:
            if not (self.beginning or self.end or other_container.beginning or other_container.end):
                return PauliProductFormulaContainer(
                    self.step_terms,
                    self.step_reps + other_container.step_reps,
                    self.num_qubits,
                    self.scale,
                    group_offsets=self.group_offsets or other_container.group_offsets,
                ).combine(atol=atol)
            if self.beginning == other_container.beginning and self.end == other_container.end:
                n = len(self.beginning)
                join = self.end if n == 0 else None
                if n and len(self.end) >= n:
                    boundary_terms = _merge_groups(self.end[-n:], self.beginning, atol)
                    if boundary_terms is not None:
                        join = self.end[:-n] + boundary_terms
                if join in ([], self.step_terms):
                    return PauliProductFormulaContainer(
                        self.step_terms,
                        self.step_reps + other_container.step_reps + int(bool(join)),
                        self.num_qubits,
                        self.scale,
                        beginning=self.beginning,
                        end=self.end,
                        group_offsets=self.group_offsets,
                    )

        merged: list[ExponentiatedPauliTerm] = []
        for container in (self, other_container):
            for term in chain(
                container.beginning,
                chain.from_iterable(repeat(container.step_terms, container.step_reps)),
                container.end,
            ):
                if merged and merged[-1].pauli_term == term.pauli_term:
                    angle = _finite(merged[-1].angle + term.angle)
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
        for name in ("beginning", "end"):
            data[name] = [
                {"pauli_term": {str(k): v for k, v in term.pauli_term.items()}, "angle": term.angle}
                for term in getattr(self, name)
            ]
        if self.group_offsets is not None:
            data["group_offsets"] = list(self.group_offsets)
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

        for name in ("step_terms", "beginning", "end"):
            terms_group = group.create_group(name)
            for i, term in enumerate(getattr(self, name)):
                term_group = terms_group.create_group(f"term_{i}")
                term_group.attrs["angle"] = term.angle
                pauli_term_group = term_group.create_group("pauli_term")
                for qubit_index, pauli_operator in term.pauli_term.items():
                    pauli_term_group.attrs[str(qubit_index)] = pauli_operator
        if self.group_offsets is not None:
            group.create_dataset("group_offsets", data=self.group_offsets, dtype="int64")

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "PauliProductFormulaContainer":
        """Create PauliProductFormulaContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            PauliProductFormulaContainer

        """
        version = json_data.get("version", "")
        expected_version = cls._serialization_version
        if version.startswith(("0.2.", "0.3.")):
            expected_version = ".".join(version.split(".")[:2]) + ".0"
        cls._validate_json_version(expected_version, json_data)
        if "segments" in json_data:
            raise ValueError("Recursive segments are not a supported product-formula format.")
        if expected_version == "0.3.0":
            return cls(
                _read_legacy_step_terms(
                    json_data["num_qubits"],
                    np.asarray(json_data["term_offsets"]),
                    np.asarray(json_data["qubit_indices"]),
                    np.asarray(json_data["pauli_codes"]),
                    np.asarray(json_data["angles"], dtype=float),
                ),
                step_reps=json_data["step_reps"],
                num_qubits=json_data["num_qubits"],
                scale=json_data.get("scale", 1.0),
            )
        lists: dict[str, list[ExponentiatedPauliTerm]] = {}
        for name in ("step_terms", "beginning", "end"):
            step_terms = []
            for i, term_data in enumerate(json_data[name] if name == "step_terms" else json_data.get(name, [])):
                pauli_term: dict[int, str] = {}
                for k, v in term_data["pauli_term"].items():
                    if not isinstance(k, str):
                        raise TypeError(f"{name}[{i}].pauli_term: expected str key, got {type(k).__name__} ({k!r})")
                    try:
                        qubit_index = int(k)
                    except ValueError as exc:
                        raise ValueError(
                            f"{name}[{i}].pauli_term: key {k!r} is not a valid integer qubit index"
                        ) from exc
                    if str(qubit_index) != k:
                        raise ValueError(
                            f"{name}[{i}].pauli_term: key {k!r} is not a canonical integer "
                            f"(expected {str(qubit_index)!r})"
                        )
                    pauli_term[qubit_index] = v
                step_terms.append(ExponentiatedPauliTerm(pauli_term=pauli_term, angle=term_data["angle"]))
            lists[name] = step_terms
        step_reps = json_data["step_reps"]
        num_qubits = json_data["num_qubits"]
        return cls(
            step_terms=lists["step_terms"],
            step_reps=step_reps,
            num_qubits=num_qubits,
            scale=json_data.get("scale", 1.0),
            beginning=lists["beginning"],
            end=lists["end"],
            group_offsets=json_data.get("group_offsets"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "PauliProductFormulaContainer":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            PauliProductFormulaContainer

        """
        version = group.attrs.get("version", "")
        expected_version = cls._serialization_version
        if version.startswith(("0.2.", "0.3.")):
            expected_version = ".".join(version.split(".")[:2]) + ".0"
        cls._validate_hdf5_version(expected_version, group)
        if "segments" in group:
            raise ValueError("Recursive segments are not a supported product-formula format.")
        step_reps = group.attrs["step_reps"]
        num_qubits = group.attrs["num_qubits"]

        if expected_version == "0.3.0":
            return cls(
                _read_legacy_step_terms(
                    num_qubits,
                    np.array(group["term_offsets"]),
                    np.array(group["qubit_indices"]),
                    np.array(group["pauli_codes"]),
                    np.array(group["angles"], dtype=float),
                ),
                step_reps=step_reps,
                num_qubits=num_qubits,
                scale=float(group.attrs.get("scale", 1.0)),
            )

        lists: dict[str, list[ExponentiatedPauliTerm]] = {}
        for name in ("step_terms", "beginning", "end"):
            step_terms: list[ExponentiatedPauliTerm] = []
            step_terms_group = group[name] if name == "step_terms" else group.get(name, {})
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
            lists[name] = step_terms

        return cls(
            step_terms=lists["step_terms"],
            step_reps=step_reps,
            num_qubits=num_qubits,
            scale=float(group.attrs.get("scale", 1.0)),
            beginning=lists["beginning"],
            end=lists["end"],
            group_offsets=tuple(group["group_offsets"][()]) if "group_offsets" in group else None,
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
        lines.append(f"  Stored terms: {self.num_stored_terms}")
        lines.append(f"  Pauli exponentials: {self.num_pauli_exponentials}")
        return "\n".join(lines)
