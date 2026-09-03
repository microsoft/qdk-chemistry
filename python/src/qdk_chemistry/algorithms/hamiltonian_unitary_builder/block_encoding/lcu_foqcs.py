r"""QDK/Chemistry FOQCS-LCU block encoding builder for spin-model Hamiltonians.

References:
    F. Della Chiara, M. Nibbi, Y. Shen, D. Camps, R. Van Beeumen, `Efficient
    LCU block encodings through Dicke states preparation
    <https://arxiv.org/abs/2507.20887>`_, 2025, arXiv:2507.20887.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import QubitOperator, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.foqcs import FoqcsContainer, FoqcsFamily
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer

__all__: list[str] = ["LCUFoqcsBuilder", "LCUFoqcsSettings"]


class LCUFoqcsSettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the FOQCS-LCU block encoding builder."""

    def __init__(self):
        """Initialize LCUFoqcsSettings with default values.

        Attributes:
            power: The power to which the block encoding is raised.
            quantum_walk: If True, wrap the block encoding with a quantum walk operator (use with QPE).
            tolerance: Coefficient-comparison tolerance for grouping homogeneous families.

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power to which the block encoding is raised.")
        self._set_default(
            "quantum_walk",
            "bool",
            False,
            "If True, wrap the block encoding with a quantum walk operator (use with QPE). "
            "If False, use the plain block encoding (use with Hadamard test).",
        )
        self._set_default(
            "tolerance",
            "float",
            1e-9,
            "Coefficient-comparison tolerance for grouping homogeneous families.",
        )


class LCUFoqcsBuilder(HamiltonianUnitaryBuilder):
    r"""Fast One-Qubit-Controlled Select LCU block encoding builder for translationally-structured spin models.

    Block-encodes a spin Hamiltonian by grouping its terms into homogeneous Pauli-term *families*.
    Each family is a single Pauli letter (``X``/``Y``/``Z``) acting either on every site (1-body field)
    or on every nearest-neighbour pair at a fixed offset ``k`` (2-body coupling).

    The block encoding is

    .. math::

        B[H] = \mathrm{PREP}(c^*)^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREP}(c)

    which encodes :math:`H / \lambda` in the ancilla-``|0>`` subspace, using balanced Dicke states for
    the family preparations and a transversal CX/CZ  SELECT.

    Only Hamiltonians whose terms decompose into homogeneous, translationally-invariant 1-body (full-chain)
    and 2-body (fixed-offset, nearest-neighbour) families are supported.  This covers the transverse-field
    Ising and anisotropic Heisenberg chains exactly.  An identity (constant-shift) term is also supported and
    is carried as a degenerate family that applies no Pauli.  Hamiltonians with inhomogeneous families,
    terms of weight > 2, mixed 2-body Pauli strings (e.g. ``XZ``), or families that do not span the
    expected geometry are rejected with a :class:`ValueError`.
    """

    def __init__(
        self,
        power: int = 1,
        quantum_walk: bool = False,
    ):
        r"""Initialize the FOQCS builder.

        Args:
            power: The power to raise the block encoding to. Defaults to 1.
            quantum_walk: If True, the circuit mapper wraps the block encoding with a quantum walk operator
                (use with QPE). If False, use the plain block encoding (use with Hadamard test). Defaults to False.

        """
        super().__init__()
        self._settings = LCUFoqcsSettings()
        self._settings.set("power", power)
        self._settings.set("quantum_walk", quantum_walk)

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"lcu_foqcs"``.

        """
        return "lcu_foqcs"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"hamiltonian_unitary_builder"``.

        """
        return "hamiltonian_unitary_builder"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        r"""Construct the FOQCS-LCU unitary representation.

        Groups the Hamiltonian terms into homogeneous families, computes the normalized sub-PREP amplitudes
        and phase corrections, and packages the result into a :class:`FoqcsContainer`.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to block-encode.

        Returns:
            UnitaryRepresentation: The unitary representation wrapping the built container.

        """
        power: int = self._settings.get("power")
        quantum_walk: bool = self._settings.get("quantum_walk")
        tolerance: float = self._settings.get("tolerance")

        if not qubit_hamiltonian.is_hermitian():
            raise ValueError("FOQCS block encoding requires a Hermitian Hamiltonian.")

        num_sites = qubit_hamiltonian.num_qubits
        if num_sites == 0:
            raise ValueError("FOQCS block encoding requires a non-empty Hamiltonian.")

        grouped = self._group_families(qubit_hamiltonian, num_sites, tolerance)

        # Compute normalization: lambda = sum_f |coeff_f| * count_f.
        mags = [math.sqrt(abs(coeff) * count) for (_letter, _weight, _offset, coeff, count) in grouped]
        norm_sq = float(sum(m * m for m in mags))
        if norm_sq <= tolerance:
            raise ValueError("FOQCS block encoding requires a Hamiltonian with a positive 1-norm.")
        norm = math.sqrt(norm_sq)
        lambda_ = norm_sq

        families: list[FoqcsFamily] = []
        for (letter, weight, offset, coeff, _count), mag in zip(grouped, mags, strict=True):
            num_y = weight if letter == "Y" else 0
            phase = (math.pi / 2.0 if coeff < 0 else 0.0) - (math.pi / 4.0) * num_y
            families.append(
                FoqcsFamily(
                    paulis=tuple([letter] * weight),
                    offset=offset,
                    abs_coeff=mag / norm,
                    phase=phase,
                )
            )

        foqcs_container = FoqcsContainer(
            num_sites=num_sites,
            families=families,
            lambda_=lambda_,
            power=power,
        )

        container = (
            LCUWalkContainer(block_encoding=foqcs_container, power=power, scale=lambda_)
            if quantum_walk
            else foqcs_container
        )

        return UnitaryRepresentation(container=container)

    def _group_families(
        self,
        qubit_hamiltonian: QubitOperator,
        num_sites: int,
        tolerance: float,
    ) -> list[tuple[str, int, int, float, int]]:
        """Group Hamiltonian terms into homogeneous FOQCS families.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.
            num_sites: Number of spin sites ``L``.
            tolerance: Coefficient-comparison tolerance.

        Returns:
            A list of ``(letter, weight, offset, coeff, count)`` tuples, with the identity family (if any) first,
            then fields (X, Y, Z), then couplings by ``(letter, offset)``.

        Raises:
            ValueError: If any term violates the FOQCS v1 scope.

        """
        # Accumulate terms into families keyed by (letter, weight, offset).
        acc: dict[tuple[str, int, int], list[tuple[float, tuple[int, ...]]]] = {}
        constant = 0.0
        for label, coeff in qubit_hamiltonian.get_real_coefficients():
            support = self._pauli_label_to_map(label)
            weight = len(support)
            if weight == 0:
                constant += float(coeff)
                continue
            if weight > 2:
                raise ValueError(
                    f"FOQCS block encoding supports only 1-body and 2-body terms; got a weight-{weight} term '{label}'."
                )

            sites = sorted(support)
            letters = {support[s] for s in sites}
            if len(letters) != 1:
                raise ValueError(
                    f"FOQCS block encoding requires homogeneous families; term '{label}' mixes Pauli letters."
                )
            letter = next(iter(letters))
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"FOQCS block encoding supports only X, Y, Z Paulis; got '{letter}'.")

            offset = 0 if weight == 1 else sites[1] - sites[0]
            key = (letter, weight, offset)
            acc.setdefault(key, []).append((float(coeff), tuple(sites)))

        # Deterministic ordering: fields first (X, Y, Z), then couplings by (letter, offset).
        letter_rank = {"X": 0, "Y": 1, "Z": 2}
        ordered_keys = sorted(acc.keys(), key=lambda k: (k[1], letter_rank[k[0]], k[2]))

        grouped: list[tuple[str, int, int, float, int]] = []
        for key in ordered_keys:
            letter, weight, offset = key
            terms = acc[key]
            coeffs = np.array([c for c, _ in terms])
            positions = frozenset(p for _, p in terms)
            count = len(terms)

            # Homogeneity: all coefficients in a family must be equal.
            if not np.allclose(coeffs, coeffs[0], atol=tolerance, rtol=0.0):
                raise ValueError(
                    f"FOQCS block encoding requires translationally-invariant families; the "
                    f"{'field' if weight == 1 else 'coupling'} family '{letter * weight}' (offset {offset}) "
                    "has non-uniform coefficients."
                )

            self._validate_geometry(letter, weight, offset, positions, count, num_sites)
            grouped.append((letter, weight, offset, float(coeffs[0]), count))

        # The identity family applies no Pauli, so SELECT is already the identity on its branch.
        if abs(constant) > tolerance:
            grouped.insert(0, ("I", 0, 0, constant, 1))

        return grouped

    @staticmethod
    def _validate_geometry(
        letter: str,
        weight: int,
        offset: int,
        positions: frozenset[tuple[int, ...]],
        count: int,
        num_sites: int,
    ) -> None:
        """Validate that a family spans the geometry expected by its Dicke preparation.

        Args:
            letter: The Pauli letter of the family.
            weight: 1 for a field family, 2 for a coupling family.
            offset: The nearest-neighbour separation ``k`` (0 for fields).
            positions: The set of site tuples covered by the family.
            count: Number of terms in the family.
            num_sites: Number of spin sites ``L``.

        Raises:
            ValueError: If the family does not span the expected chain geometry.

        """
        expected: set[tuple[int, ...]]
        if weight == 1:
            expected = {(s,) for s in range(num_sites)}
            if positions != expected:
                raise ValueError(
                    f"FOQCS block encoding requires a 1-body '{letter}' field on every site "
                    f"(expected {num_sites} sites, got {count})."
                )
        else:
            if offset < 1 or offset > num_sites - 1:
                raise ValueError(
                    f"FOQCS block encoding: 2-body '{letter * 2}' offset {offset} "
                    f"is out of range for {num_sites} sites."
                )
            expected = {(i, i + offset) for i in range(num_sites - offset)}
            if positions != expected:
                raise ValueError(
                    f"FOQCS block encoding requires a 2-body '{letter * 2}' coupling on every "
                    f"offset-{offset} nearest-neighbour pair (expected {num_sites - offset} pairs, got {count})."
                )
