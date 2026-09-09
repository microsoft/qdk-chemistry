"""Tests for the Hamming weight phasing Q# primitives and the batch-segment helper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import ClassVar

import numpy as np
import pytest
from scipy.linalg import expm

from .test_helpers import dense_matrix

#: dump_operation rounds to about six decimals, so exact agreement lands near 1e-6.
_TOL = 1e-5


def _pauli_string_matrix(qubits: list[int], axes: str, num_qubits: int) -> np.ndarray:
    """Dense matrix of a Pauli string, with qubit 0 as the leading tensor factor."""
    axis_of = dict(zip(qubits, axes, strict=True))
    out = np.array([[1.0 + 0j]])
    for q in range(num_qubits):
        out = np.kron(out, _PAULI[axis_of.get(q, "I")])
    return out


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class TestHammingWeightSchedule:
    """The adder tree's shape, which decides whether batching is worth anything."""

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 8, 16, 17, 32, 64, 1024])
    def test_uses_the_optimal_number_of_adders(self, count, qs_context):
        """Gidney's bound: reducing m bits to a binary weight takes m - w(m) full adders.

        This is the whole economic case for batching. A naive chain of controlled
        increments also computes the weight, but costs O(m log m) Toffolis, which at
        m = 16 is more expensive than the rotations it removes.
        """
        adders = qs_context.run(
            f"{{ import QDKChemistry.Utils.HammingWeightPhasing.*; HammingWeightScratchCount({count}) }}", 1
        )[0]
        assert adders == count - bin(count).count("1")

    @pytest.mark.parametrize("count", [4, 8, 16, 32])
    def test_arithmetic_costs_one_toffoli_per_adder(self, count, qs_context):
        """The adders use ``AND``, so compute and uncompute together cost one each.

        With ``CCNOT`` this would be twice as many, since the ``within`` block pays for
        the uncompute as well.
        """
        estimate = qs_context.logical_counts(
            "{ import QDKChemistry.Utils.HammingWeightPhasing.*; "
            f"use q = Qubit[{count}]; HammingWeightPhase(0.37, q) }}"
        ).estimate()
        assert estimate["logicalCounts"]["cczCount"] == count - bin(count).count("1")

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8, 9, 16])
    def test_weight_register_is_wide_enough(self, count, qs_context):
        """The register must hold ``count`` itself, the largest weight possible."""
        bits = qs_context.run(f"{{ import QDKChemistry.Utils.HammingWeightPhasing.*; HammingWeightBits({count}) }}", 1)[
            0
        ]
        assert 2**bits > count


class TestComputeHammingWeight:
    """The arithmetic itself, checked against a classical popcount."""

    @pytest.mark.parametrize("count", [2, 3, 4, 5])
    def test_counts_every_basis_state(self, count, qs_context):
        """Exhaustive over all 2^m inputs: the register must hold the true weight."""
        sizes = qs_context.run(
            f"{{ import QDKChemistry.Utils.HammingWeightPhasing.*; "
            f"(HammingWeightScratchCount({count}), HammingWeightBits({count})) }}",
            1,
        )[0]
        num_scratch, bits = sizes
        for value in range(2**count):
            pattern = format(value, f"0{count}b")
            prepare = "".join(f"if {bit} == 1 {{ X(q[{i}]); }} " for i, bit in enumerate(pattern))
            measure = ", ".join(f"MResetZ(w[{k}])" for k in range(bits))
            source = (
                "{ import QDKChemistry.Utils.HammingWeightPhasing.*; "
                f"use q = Qubit[{count}]; use s = Qubit[{num_scratch}]; use w = Qubit[{bits}]; "
                f"{prepare} ComputeHammingWeight(q, s, w); "
                f"let r = [{measure}]; ResetAll(q + s); r }}"
            )
            results = qs_context.run(source, 1)[0]
            measured = sum(1 << k for k, bit in enumerate(results) if str(bit) == "One")
            assert measured == pattern.count("1"), f"weight of {pattern}"


class TestHammingWeightPhase:
    """The phase the batched form imparts must equal the individual rotations."""

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    @pytest.mark.parametrize("angle", [0.31, -1.2])
    def test_matches_the_product_of_single_rotations(self, count, angle, qs_context):
        """``HammingWeightPhase`` stands in for one ``exp(-i angle Z)`` per qubit."""
        got = dense_matrix(
            f"qs => QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase({angle!r}, qs)",
            count,
            qs_context,
        )
        single = expm(-1j * angle * _PAULI["Z"])
        want = np.array([[1.0 + 0j]])
        for _ in range(count):
            want = np.kron(want, single)
        assert np.max(np.abs(got - want)) < _TOL

    @pytest.mark.parametrize("count", [2, 3])
    def test_keeps_the_identity_phase_under_a_control(self, count, qs_context):
        """The ``exp(-i angle m)`` piece is not a global phase once controlled.

        Phase estimation only ever applies this controlled, so dropping the identity
        part would leave the evolution right and the measured phase wrong.
        """
        angle = 0.31
        got = dense_matrix(
            f"qs => Controlled QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase("
            f"[qs[0]], ({angle!r}, qs[1...]))",
            count + 1,
            qs_context,
        )
        single = expm(-1j * angle * _PAULI["Z"])
        target = np.array([[1.0 + 0j]])
        for _ in range(count):
            target = np.kron(target, single)
        want = np.eye(2 ** (count + 1), dtype=complex)
        want[2**count :, 2**count :] = target
        assert np.max(np.abs(got - want)) < _TOL

    def test_control_falls_only_on_the_rotations(self, qs_context):
        """``C(V D V^dagger) = V C(D) V^dagger``, so the adder tree stays uncontrolled.

        Without this the Toffolis would double up under control and swamp the saving.
        """
        counts = {}
        for source in ("HammingWeightPhase(0.37, q)", "Controlled HammingWeightPhase([c], (0.37, q))"):
            estimate = qs_context.logical_counts(
                f"{{ import QDKChemistry.Utils.HammingWeightPhasing.*; use c = Qubit(); use q = Qubit[16]; {source} }}"
            ).estimate()
            counts[source] = estimate["logicalCounts"]
        plain, controlled = counts.values()
        assert plain["cczCount"] == controlled["cczCount"]

    def test_rotation_count_is_logarithmic(self, qs_context):
        """The saving itself: rotations grow with log(m), not m."""
        for count, expected in ((4, 3), (8, 4), (16, 5), (32, 6)):
            estimate = qs_context.logical_counts(
                "{ import QDKChemistry.Utils.HammingWeightPhasing.*; "
                f"use q = Qubit[{count}]; HammingWeightPhase(0.37, q) }}"
            ).estimate()
            assert estimate["logicalCounts"]["rotationCount"] == expected


class TestHammingWeightPhaseTerms:
    """Arbitrary commuting Pauli strings, rotated onto single-Z representatives."""

    CASES: ClassVar[dict] = {
        "single-qubit Z": (4, [([0], "Z"), ([1], "Z"), ([2], "Z"), ([3], "Z")]),
        "adjacent ZZ pairs": (4, [([0, 1], "ZZ"), ([2, 3], "ZZ")]),
        "non-adjacent ZZ pairs": (4, [([0, 2], "ZZ"), ([1, 3], "ZZ")]),
        "mixed axes": (4, [([0, 1], "XY"), ([2], "Z")]),
        "weight-three string": (4, [([0, 1, 2], "XYZ")]),
    }

    @pytest.mark.parametrize("name", list(CASES))
    @pytest.mark.parametrize("angle", [0.31, -1.2])
    def test_matches_the_individual_exponentials(self, name, angle, qs_context):
        """Each string must still get ``exp(-i angle P)``, whatever its axes."""
        num_qubits, terms = self.CASES[name]
        ops = ", ".join("[" + ", ".join(f"Pauli{a}" for a in axes) + "]" for _, axes in terms)
        targets = ", ".join("[" + ", ".join(f"qs[{q}]" for q in qubits) + "]" for qubits, _ in terms)
        got = dense_matrix(
            f"qs => QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhaseTerms({angle!r}, [{ops}], [{targets}])",
            num_qubits,
            qs_context,
        )
        want = np.eye(2**num_qubits, dtype=complex)
        for qubits, axes in terms:
            want = expm(-1j * angle * _pauli_string_matrix(qubits, axes, num_qubits)) @ want
        assert np.max(np.abs(got - want)) < _TOL


class TestBatchSegments:
    """Only consecutive equal identifiers may be applied as one block."""

    @pytest.mark.parametrize(
        ("ids", "expected"),
        [
            ([], []),
            ([0, 0, 0], [(0, 1), (1, 1), (2, 1)]),
            ([1, 1, 1], [(0, 3)]),
            ([1, 1, 0, 2, 2], [(0, 2), (2, 1), (3, 2)]),
            ([0, 1, 1, 0], [(0, 1), (1, 2), (3, 1)]),
            ([1, 2], [(0, 1), (1, 1)]),
        ],
    )
    def test_segments(self, ids, expected, qs_context):
        """Distinct identifiers must not merge, and zeros must stay singletons."""
        got = qs_context.run(f"{{ import QDKChemistry.Utils.HammingWeightPhasing.*; BatchSegments({ids!s}) }}", 1)[0]
        assert [tuple(pair) for pair in got] == expected
