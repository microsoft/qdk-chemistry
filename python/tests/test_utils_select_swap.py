"""Tests for the Q# SELECT-SWAP data-loading network."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import pytest

from qdk_chemistry.utils.qsharp import create_qsharp_context, get_qsharp_context

_DATA_1D = [
    [True, False, True],
    [False, True, True],
    [True, True, False],
    [False, False, True],
    [True, False, False],
    [True, True, True],
    [False, True, False],
    [False, False, False],
]

# Six rows, so two of the eight addressable states are unused. A table whose length is not a
# power of two is the case where the select path and the swap path can disagree on where an
# out-of-range address lands, which is invisible to a value test.
_DATA_1D_RAGGED = _DATA_1D[:6]

_DATA_2D = [
    [[True, False, True], [False, True, True], [True, True, False]],
    [[False, True, False], [True, True, True], [False, False, False]],
    [[True, False, False], [False, False, True], [True, True, True]],
]

# Six outer rows exercise the aliasing of unused *outer* addresses two levels down the unary
# iteration tree, which the 3-row table above does not reach: folding the outer index into a
# single flat lookup passes at nOuter = 3, 4, 5 and 8 but fails at 6.
_DATA_2D_RAGGED_OUTER = [
    [[True, False, True], [False, True, True], [True, True, False], [False, False, True]],
    [[False, True, False], [True, True, True], [False, False, False], [True, False, True]],
    [[True, False, False], [False, False, True], [True, True, True], [False, True, False]],
    [[False, False, False], [True, False, True], [False, True, True], [True, True, True]],
    [[True, True, True], [False, True, False], [True, False, False], [False, False, True]],
    [[False, True, True], [True, True, False], [False, False, True], [True, False, False]],
]

# A disagreeing phase oracle leaves the address register in a superposition that still
# collapses to |0...0> roughly half the time, so one shot of the agreement operation is a
# coin flip, not a verdict. Repeating drives the miss probability below 1e-3.
_PHASE_TRIALS = 12


def _assert_phase_agreement(operation_name, data, num_swap_bits, outer_always_valid=False):
    """Fail if any trial reports disagreement; a single passing trial proves nothing."""
    operation = getattr(get_qsharp_context().code.QDKChemistry.Utils.SelectSwap, operation_name)
    args = (data, num_swap_bits) if "1D" in operation_name else (data, num_swap_bits, outer_always_valid)
    failures = sum(1 for _ in range(_PHASE_TRIALS) if not operation(*args))
    assert failures == 0, (
        f"{operation_name} disagreed with the plain-select path in {failures}/{_PHASE_TRIALS} "
        f"trials at num_swap_bits={num_swap_bits}: the swap path corrupts the address phase."
    )


class TestSelectSwapLoadsCorrectValues:
    """The loaded word is the addressed word, for every select/swap split."""

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    def test_1d_loads_every_address(self, num_swap_bits):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap1DCorrectness(
            _DATA_1D, num_swap_bits
        )

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_1d_loads_every_address_when_length_is_not_a_power_of_two(self, num_swap_bits):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap1DCorrectness(
            _DATA_1D_RAGGED, num_swap_bits
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address(self, num_swap_bits, outer_always_valid):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap2DCorrectness(
            _DATA_2D, num_swap_bits, outer_always_valid
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_outer_length_is_not_a_power_of_two(
        self, num_swap_bits, outer_always_valid
    ):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap2DCorrectness(
            _DATA_2D_RAGGED_OUTER, num_swap_bits, outer_always_valid
        )


class TestSelectSwapPreservesAddressPhases:
    """The swap path must agree with the plain-select path as a *phase* oracle.

    A lookup that loads the right bits but routes unused addresses differently from the
    plain-select path, or that uncomputes incorrectly, still passes every value test above:
    the damage lands on the phase of the address register, where only these tests see it.

    The 2D sweeps include ``num_swap_bits == 0`` because the combined flattened lookup with
    an empty butterfly must preserve the same phases as the unary-iteration reference.
    """

    @pytest.mark.parametrize("num_swap_bits", [1, 2, 3])
    def test_1d_swap_path_matches_plain_select(self, num_swap_bits):
        _assert_phase_agreement("TestSelectSwap1DPhaseAgreement", _DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_1d_swap_path_matches_plain_select_when_length_is_not_a_power_of_two(self, num_swap_bits):
        _assert_phase_agreement("TestSelectSwap1DPhaseAgreement", _DATA_1D_RAGGED, num_swap_bits)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_load_matches_plain_select(self, num_swap_bits, outer_always_valid):
        """``SelectSwap2D`` erases against the flat table whatever swap width it loaded at.

        Its adjoint is written by hand rather than derived from its body, so this is the only
        test that can see an erasure that is wrong in phase but right in value.
        """
        _assert_phase_agreement("TestSelectSwap2DPhaseAgreement", _DATA_2D, num_swap_bits, outer_always_valid)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_load_matches_plain_select_when_outer_length_is_not_a_power_of_two(
        self, num_swap_bits, outer_always_valid
    ):
        _assert_phase_agreement(
            "TestSelectSwap2DPhaseAgreement", _DATA_2D_RAGGED_OUTER, num_swap_bits, outer_always_valid
        )


def _unlookup_toffolis(num_entries: int) -> int:
    """Toffolis of ``Adjoint Select`` over *num_entries*, the measurement-based unlookup.

    ``Std.TableLookup.Select`` erases by measuring the target and applying a phase fixup over
    the address, which costs ``2**ceil(n/2) + 2**floor(n/2) - n - 2`` on ``n`` address qubits
    rather than repeating the load. That is what makes the erasure sublinear in the table.
    """
    address_bits = max(1, math.ceil(math.log2(num_entries)))
    return 2 ** math.ceil(address_bits / 2) + 2 ** (address_bits // 2) - address_bits - 2


def _probe_toffolis(ctx, data, num_swap_bits, *, forward, adjoint):
    """Trace ``TestSelectSwap2DResourceProbe`` and return its Toffoli count."""
    counts = ctx.logical_counts(
        ctx.code.QDKChemistry.Utils.SelectSwap.TestSelectSwap2DResourceProbe,
        data,
        num_swap_bits,
        True,
        forward,
        adjoint,
    )
    return counts["cczCount"] + counts["ccixCount"]


class TestSelectSwap2DErasesByMeasurement:
    """The 2D word load's adjoint is a phase fixup over the flat address, not a second lookup."""

    @pytest.mark.parametrize(
        ("num_outer", "num_inner", "width"),
        [(4, 4, 5), (8, 4, 6), (6, 8, 7), (16, 8, 4)],
    )
    def test_adjoint_costs_the_unlookup_whatever_the_swap_width(self, num_outer, num_inner, width):
        """Erasure cost follows the table size alone, and undercuts the load it undoes.

        The forward pass varies with the swap width, but the adjoint resolves to a single
        ``Adjoint Select`` over ``outer x inner`` entries however the word was loaded, so its
        cost is the closed form above for every width. Running the load backwards instead
        would track the forward cost.
        """
        ctx = create_qsharp_context()
        data = [
            [[(o * 31 + i * 7 + b) % 2 == 0 for b in range(width)] for i in range(num_inner)] for o in range(num_outer)
        ]
        expected = _unlookup_toffolis(num_outer * num_inner)

        for num_swap_bits in (0, 1, 2):
            forward = _probe_toffolis(ctx, data, num_swap_bits, forward=True, adjoint=False)
            round_trip = _probe_toffolis(ctx, data, num_swap_bits, forward=True, adjoint=True)

            assert round_trip - forward == expected
            assert expected < forward
