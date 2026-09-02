"""Pinned baseline of SOSSA logical resource counts, for measuring the effect of cost reworks.

These tests do not assert that the current costs are *correct*. They record what the
implementation costs today so that a later rework can be shown to improve on it, and so
that an unrelated change cannot move the numbers silently.

The subject is the rotation QROM in ``QDKChemistry.Utils.SOSSAWalk.WithGivensRotationsQROM``,
which reads two angle tables. :cite:`Low2025` pays :math:`N + RB` to output the rotations
(Appendix B, step 5) and only :math:`R + B` to erase them by phase fixup (step 7);
``_paper_rotation_table_toffolis`` encodes that target and ``TestSelectBaseline`` measures
how far the implementation sits from it.

Both tables were originally read with ``Controlled Select``, whose adjoint does not use the
measurement-based unlookup that ``Adjoint Select`` does and so costs the same as the load.
The SF table -- the larger of the two -- is now read uncontrolled, which makes its uncompute
that unlookup and so realizes step 7. The DQ table stays controlled: giving it its own
``isSF`` address bit doubles it, and at every size measured the doubling costs more than the
cheaper unlookup saves. ``_baseline_rotation_table_toffolis`` records both halves.

When a further rework lands these tests are expected to fail. Re-derive the baselines from
the new implementation and record the delta; do not relax the assertions to make them pass.

The block-encoding numbers cover the PREPAREs as well, and those are now the executable
circuit: the ``Legacy*ResourceEstimate`` branches that ``IsResourceEstimating()`` used to
substitute for them have been deleted. They understated the two alias-sampling PREPAREs by
roughly 1,700 Toffolis per block encoding at Fe2S2-20.

The inner PREPARE's alias lookup is now erased by measurement rather than run backwards,
which moved every row: 141 -> 135, 263 -> 237, 342 -> 308, 554 -> 478 and 1,946 -> 1,582
Toffolis, the last an 18.7% cut. Qubits, rotations and ``tCount`` are unchanged and the
measurement column moves the other way, which is the trade the erasure makes. Traced at
production shapes the same change takes one FeMoCo-54 block encoding from 9,843 to 8,071
Toffolis, and the Fe2S2-20 end-to-end estimate from 37,873,827 to 31,837,599.

The end-to-end pin lives in
``test_phase_estimation_sossa.py::test_fe2s2_logical_resource_estimate``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from typing import Any

import pytest
import qdk

from qdk_chemistry.algorithms.circuit_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.utils.qsharp import create_qsharp_context

from .test_helpers import create_random_factorized_hamiltonian, to_sossa_operator

# (N, R, B, C, b_rot). Shapes are chosen so that B + 1 is a power of two in some rows and
# not in others, because the SF angle table is addressed by the full b register and so is
# padded up to 2^ceil(log2(B+1)) entries -- a padding that only shows up in the ragged rows.
_SELECT_SHAPES = [
    (2, 1, 1, 1, 8),
    (3, 2, 2, 1, 8),
    (4, 2, 3, 1, 10),
    (5, 3, 4, 1, 9),
    (6, 3, 3, 2, 10),
    (7, 2, 6, 3, 11),
    (8, 4, 5, 2, 12),
]

# (N, R, B, C, b_coeff, b_rot) -> (toffolis, measurements, qubits, rotations, tCount).
_BLOCK_ENCODING_BASELINE: dict[tuple[int, ...], tuple[int, ...]] = {
    (2, 1, 1, 1, 8, 8): (135, 155, 77, 63, 21),
    (3, 2, 2, 1, 8, 8): (237, 268, 92, 75, 21),
    (4, 2, 3, 1, 9, 10): (308, 366, 116, 108, 27),
    (6, 3, 3, 2, 10, 10): (478, 572, 147, 108, 27),
    # The only shape here whose inner PREPARE takes a non-trivial select-swap split rather
    # than a bare lookup, so the only one that exercises the swap network. Its qubit count is
    # above the others because the swap width the Toffoli-optimal lambda picks also sets the
    # swap register size.
    (12, 8, 9, 3, 10, 12): (1582, 1839, 287, 177, 33),
}


@pytest.fixture(scope="module")
def trace_ctx() -> qdk.Context:
    """A Q# context shared across the module for resource tracing.

    ``logical_counts`` traces each call independently, so unlike the simulation tests --
    which need a fresh context because ``TestSelectDQ`` allocates without releasing -- one
    context serves every shape here.
    """
    return create_qsharp_context()


def _select_params(num_orbitals: int, num_ranks: int, num_bases: int, num_copies: int, rot_bits: int) -> dict:
    """Build ``SelectParams`` for a given problem shape.

    The angle values are arbitrary. A table lookup's Toffoli cost is set by the number of
    entries and their width, not by their contents, so these counts depend only on the
    shape -- verified against randomly drawn Givens angles when the baseline was taken.
    """
    rank_bits = math.ceil(math.log2(num_ranks)) if num_ranks > 1 else 0
    angles = [[0.1 + 0.37 * j + 0.11 * entry for j in range(num_orbitals - 1)] for entry in range(num_orbitals)]
    sf_angles = [
        [0.1 + 0.37 * j + 0.11 * entry for j in range(num_orbitals - 1)] for entry in range(num_ranks * (num_bases + 1))
    ]
    return {
        "numOrbitals": num_orbitals,
        "numRanks": num_ranks,
        "numBases": num_bases,
        "numCopies": num_copies,
        "numPositiveOneBody": num_orbitals,
        "OneBodyRotationAngles": angles,
        "TwoBodyRotationAngles": sf_angles,
        "rotationBitPrecision": rot_bits,
        "numFreeRiderBits": 2 + rank_bits,
    }


def _select_toffolis(ctx: qdk.Context, params: dict) -> int:
    """Trace one SELECT and return its Toffoli count."""
    counts = ctx.logical_counts(ctx.code.QDKChemistry.Utils.SOSSAWalk.TestSelectDQ, params, 0, 0, True)
    return counts["cczCount"] + counts["ccixCount"]


def _select_unlookup_toffolis(num_entries: int) -> int:
    """Toffolis of ``Adjoint Select`` over ``num_entries``, the measurement-based unlookup.

    Matches ``Std.TableLookup.Select`` on QDK 1.31 for every table size measured, powers of
    two and not: it is the ``2**ceil(n/2) + 2**floor(n/2) - n - 2`` phase fixup on ``n``
    address qubits, which is what makes the erasure sublinear in the table size.
    """
    address_bits = max(1, math.ceil(math.log2(num_entries)))
    return 2 ** math.ceil(address_bits / 2) + 2 ** (address_bits // 2) - address_bits - 2


def _sf_table_entries(num_ranks: int, num_bases: int) -> int:
    """Entries in the SF angle table under the cheaper of the two address orderings.

    ``Select`` pads whichever register supplies the low address bits out to a power of two,
    so ``bReg ++ rBits`` costs ``R * 2**ceil(log2(B+1))`` and ``rBits ++ bReg`` costs
    ``(B+1) * 2**ceil(log2 R)``. ``SFTableRankAddressedFirst`` picks the smaller; neither
    reaches the paper's ``R*B``, which needs a non-power-of-two address stride.
    """
    b_bits = math.ceil(math.log2(num_bases + 1)) if num_bases > 0 else 1
    rank_bits = math.ceil(math.log2(num_ranks)) if num_ranks > 1 else 0
    return min(num_ranks * (1 << b_bits), (num_bases + 1) * (1 << rank_bits))


def _baseline_rotation_table_toffolis(num_orbitals: int, num_ranks: int, num_bases: int) -> int:
    """Toffolis the two rotation-table reads cost, load plus uncompute.

    The SF table is read uncontrolled, so it costs one Toffoli per entry less two and its
    uncompute is the measurement-based unlookup. The DQ table is still read through a
    ``Controlled Select``, one Toffoli per entry less one, whose adjoint costs the same
    again because a controlled unlookup is not measurement-based.
    """
    sf_entries = _sf_table_entries(num_ranks, num_bases)
    sf = (sf_entries - 2) + _select_unlookup_toffolis(sf_entries)
    dq = 2 * (num_orbitals - 1)
    return sf + dq


def _paper_rotation_table_toffolis(num_orbitals: int, num_ranks: int, num_bases: int) -> int:
    """Toffolis :cite:`Low2025` allots to the same work: ``N + RB`` to load, ``R + B`` to erase."""
    return (num_orbitals + num_ranks * num_bases) + (num_ranks + num_bases)


def _measured_rotation_table_toffolis(ctx: qdk.Context, shape: tuple[int, int, int, int, int]) -> tuple[int, int]:
    """Isolate the rotation-table Toffolis of one SELECT by measurement.

    Tracing at two rotation precisions cancels everything except the phase-gradient adders,
    whose cost is linear in ``b_rot``. Subtracting them and the shape-only ``SelectSpins``
    and ``MajoranaOp`` constants leaves the two angle-table reads. Deriving the term this
    way rather than from ``_baseline_select_toffolis`` keeps it an independent measurement.

    Returns:
        The table Toffolis, and the Toffolis per rotation bit used to remove the adders.

    """
    num_orbitals, num_ranks, num_bases, num_copies, rot_bits = shape
    wide = _select_toffolis(ctx, _select_params(num_orbitals, num_ranks, num_bases, num_copies, rot_bits))
    narrow = _select_toffolis(ctx, _select_params(num_orbitals, num_ranks, num_bases, num_copies, rot_bits - 1))

    per_bit = wide - narrow
    spin_and_majorana = 2 * (2 + num_orbitals) + 3
    return wide - per_bit * (rot_bits - 1) - spin_and_majorana, per_bit


def _baseline_select_toffolis(num_orbitals: int, num_ranks: int, num_bases: int, rot_bits: int) -> dict[str, int]:
    """Term-by-term Toffoli cost of one SELECT as currently implemented.

    ``givens`` is the phase-gradient adder, one Toffoli per angle bit less one, run forward
    and back around the Majorana step. ``spin`` is ``SelectSpins``: two Toffolis to compute
    the spin qubit plus ``N`` controlled swaps, uncomputed at full price rather than by the
    ``X``-measurement erasure of :cite:`Low2025`. ``majorana`` is the three doubly
    controlled ``Z`` gates of ``MajoranaOp``.
    """
    return {
        "givens": 2 * (num_orbitals - 1) * (rot_bits - 1),
        "spin": 2 * (2 + num_orbitals),
        "rotation_tables": _baseline_rotation_table_toffolis(num_orbitals, num_ranks, num_bases),
        "majorana": 3,
    }


class TestSelectBaseline:
    """Baseline for one SELECT, which carries the rotation QROM."""

    @pytest.mark.parametrize(("N", "R", "B", "C", "b_rot"), _SELECT_SHAPES)
    def test_select_toffolis_match_the_baseline_model(self, trace_ctx, N, R, B, C, b_rot):  # noqa: N803
        """The traced SELECT cost is fully accounted for by the four terms above.

        An exact match is what makes any later change attributable: if the total moves and
        only ``rotation_tables`` was meant to change, this pins that nothing else did.
        """
        measured = _select_toffolis(trace_ctx, _select_params(N, R, B, C, b_rot))
        terms = _baseline_select_toffolis(N, R, B, b_rot)

        assert measured == sum(terms.values()), (
            f"N={N},R={R},B={B},C={C},b_rot={b_rot}: SELECT costs {measured} Toffolis, "
            f"but the baseline model predicts {sum(terms.values())} from {terms}."
        )

    @pytest.mark.parametrize(("N", "R", "B", "C", "b_rot"), _SELECT_SHAPES)
    def test_sf_table_erasure_is_sublinear_in_the_table_size(self, trace_ctx, N, R, B, C, b_rot):  # noqa: N803
        """The SF table costs its size once, not twice: its erasure is the phase fixup.

        This is what the rework bought. Reading the table uncontrolled makes the uncompute
        ``Adjoint Select``, whose cost grows as the square root of the table rather than
        linearly, so the round trip is ``(L - 2) + O(sqrt(L))`` instead of ``2L``. A
        regression to a controlled read would push the measured term back to ``2L`` and fail
        here rather than only moving the pinned total.
        """
        measured_tables, per_bit = _measured_rotation_table_toffolis(trace_ctx, (N, R, B, C, b_rot))
        sf_entries = _sf_table_entries(R, B)
        dq = 2 * (N - 1)

        assert per_bit == 2 * (N - 1), (
            f"N={N},b_rot={b_rot}: expected 2(N-1)={2 * (N - 1)} Toffolis per rotation bit, got {per_bit}. "
            "The adders no longer scale as assumed, so the table term was not isolated."
        )
        sf_erasure = measured_tables - dq - (sf_entries - 2)
        assert sf_erasure == _select_unlookup_toffolis(sf_entries), (
            f"N={N},R={R},B={B}: the {sf_entries}-entry SF table loads for {sf_entries - 2} Toffolis and "
            f"erases for {sf_erasure}, not the {_select_unlookup_toffolis(sf_entries)} of a measurement-based "
            "unlookup. A controlled read would cost the full load again."
        )

    @pytest.mark.parametrize(("N", "R", "B", "C", "b_rot"), _SELECT_SHAPES)
    def test_rotation_table_toffolis_match_the_pinned_baseline(self, trace_ctx, N, R, B, C, b_rot):  # noqa: N803
        """Pin the measured table term per shape, against the paper's target for the same work.

        This is the headline number: the failure message reports the measured cost next to
        the target, so the effect of any further rework is read off directly. What is left of
        the gap is mostly the padding -- the SF table is padded from ``R(B+1)`` to
        ``R * 2**ceil(log2(B+1))`` entries because it is addressed by the full ``b``
        register -- plus the DQ table, which is still erased at full price.
        """
        measured_tables, _ = _measured_rotation_table_toffolis(trace_ctx, (N, R, B, C, b_rot))
        expected = _baseline_rotation_table_toffolis(N, R, B)
        target = _paper_rotation_table_toffolis(N, R, B)

        assert measured_tables == expected, (
            f"N={N},R={R},B={B}: rotation tables cost {measured_tables} Toffolis, baseline {expected}, "
            f"paper target {target} ({measured_tables / target:.2f}x the target, was "
            f"{expected / target:.2f}x). SF table holds {_sf_table_entries(R, B)} entries for "
            f"{R * (B + 1)} used."
        )


class TestBlockEncodingBaseline:
    """Baseline for the composed block encoding, to catch knock-on effects of a SELECT rework."""

    @pytest.mark.parametrize(("N", "R", "B", "C", "b_coeff", "b_rot"), list(_BLOCK_ENCODING_BASELINE))
    def test_block_encoding_counts_match_the_pinned_baseline(self, N, R, B, C, b_coeff, b_rot):  # noqa: N803
        """Pin the traced counts of ``B`` in the production mapper configuration.

        Rotations and ``tCount`` are pinned alongside the Toffolis because the outer and
        inner PREPARE reach ``PrepareUniformSuperposition``, which emits arbitrary-angle
        rotations rather than the phase-gradient rotation :cite:`Low2025` costs. A rework
        that moved work between those columns would otherwise look free.
        """
        expected = _BLOCK_ENCODING_BASELINE[N, R, B, C, b_coeff, b_rot]

        factorized = create_random_factorized_hamiltonian(
            num_orbitals=N, num_ranks=R, num_bases=B, num_copies=C, seed=42
        )
        unitary = SOSSABuilder().run(to_sossa_operator(factorized))

        mapper = SOSSAMapper()
        mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", "alias_sampling"))
        mapper.settings().set("inner_prepare_algorithm", "controlled_alias_sampling")
        mapper.settings().set("select_algorithm", "qrom_phase_gradient")
        mapper.settings().set("coefficient_bit_precision", b_coeff)
        mapper.settings().set("rotation_bit_precision", b_rot)

        factory: Any = mapper.run(unitary)._qsharp_factory
        counts = factory.program._qdk_context.logical_counts(factory.program, *factory.parameter.values())
        measured = (
            counts["cczCount"] + counts["ccixCount"],
            counts["measurementCount"],
            counts["numQubits"],
            counts["rotationCount"],
            counts["tCount"],
        )

        assert measured == expected, (
            f"N={N},R={R},B={B},C={C},b_coeff={b_coeff},b_rot={b_rot}: block encoding traced "
            f"(toffoli, measurements, qubits, rotations, tCount)={measured}, baseline={expected}."
        )
