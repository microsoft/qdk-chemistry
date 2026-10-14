"""Tests for the Q# SELECT-SWAP data-loading network."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import pytest
from qdk.test_utils import ArithmeticOpTester

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

_DATA_1D_RAGGED = _DATA_1D[:6]

_DATA_2D = [
    [[True, False, True], [False, True, True], [True, True, False]],
    [[False, True, False], [True, True, True], [False, False, False]],
    [[True, False, False], [False, False, True], [True, True, True]],
]

_DATA_2D_RAGGED_OUTER = [
    [[True, False, True], [False, True, True], [True, True, False], [False, False, True]],
    [[False, True, False], [True, True, True], [False, False, False], [True, False, True]],
    [[True, False, False], [False, False, True], [True, True, True], [False, True, False]],
    [[False, False, False], [True, False, True], [False, True, True], [True, True, True]],
    [[True, True, True], [False, True, False], [True, False, False], [False, False, True]],
    [[False, True, True], [True, True, False], [False, False, True], [True, False, False]],
]

_DATA_2D_POWER_OF_TWO = _DATA_2D_RAGGED_OUTER[:4]

_PHASE_TRIALS = 12


def _address_bits(length):
    return math.ceil(math.log2(length))


def _word(bits):
    """The little-endian integer ``ArithmeticOpTester`` reads back for *bits*."""
    return sum(1 << index for index, bit in enumerate(bits) if bit)


def _assert_1d_loads_every_address(context, data, num_swap_bits):
    """Every address loads its row, and the address and output registers come back clean."""
    op = context.code.QDKChemistry.TestUtils.SelectSwapTests.TestMakeSelectSwap1DCopyOp(data, num_swap_bits)
    width = len(data[0])
    tester = ArithmeticOpTester(op, [_address_bits(len(data)), width, width], context=context)
    failures = [
        (address, actual)
        for address, row in enumerate(data)
        if (actual := tester.run([address, 0, 0])) != [address, 0, _word(row)]
    ]
    assert not failures, f"(address, [address, output, copy]) mismatches at num_swap_bits={num_swap_bits}: {failures}"


def _assert_2d_loads_every_address(context, data, num_swap_bits, outer_always_valid):
    """Every (outer, inner) pair loads its word, and every other register comes back clean."""
    op = context.code.QDKChemistry.TestUtils.SelectSwapTests.TestMakeSelectSwap2DCopyOp(
        data, num_swap_bits, outer_always_valid
    )
    width = len(data[0][0])
    tester = ArithmeticOpTester(
        op, [_address_bits(len(data)), _address_bits(len(data[0])), width, width], context=context
    )
    failures = [
        (outer, inner, actual)
        for outer, rows in enumerate(data)
        for inner, row in enumerate(rows)
        if (actual := tester.run([outer, inner, 0, 0])) != [outer, inner, 0, _word(row)]
    ]
    assert not failures, (
        f"(outer, inner, [outer, inner, target, copy]) mismatches at num_swap_bits={num_swap_bits}, "
        f"outer_always_valid={outer_always_valid}: {failures}"
    )


def _assert_phase_agreement(test_utils, operation_name, data, num_swap_bits, outer_always_valid=False):
    """Fail if any trial reports disagreement; a single passing trial proves nothing."""
    operation = getattr(test_utils.SelectSwapTests, operation_name)
    args = (data, num_swap_bits) if "1D" in operation_name else (data, num_swap_bits, outer_always_valid)
    failures = sum(1 for _ in range(_PHASE_TRIALS) if not operation(*args))
    assert failures == 0, (
        f"{operation_name} disagreed with the plain-select path in {failures}/{_PHASE_TRIALS} "
        f"trials at num_swap_bits={num_swap_bits}: the swap path corrupts the address phase."
    )


class TestSelectSwapLoadsCorrectValues:
    """The loaded word is the addressed word, for every select/swap split."""

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    def test_1d_loads_every_address(self, qsharp_test_context, num_swap_bits):
        _assert_1d_loads_every_address(qsharp_test_context, _DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_1d_loads_every_address_when_length_is_not_a_power_of_two(self, qsharp_test_context, num_swap_bits):
        _assert_1d_loads_every_address(qsharp_test_context, _DATA_1D_RAGGED, num_swap_bits)

    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED])
    def test_1d_auto_swap_width_loads_every_address(self, qsharp_test_context, data):
        """``numSwapBits = -1`` hands the select/swap split to ``ComputeOptimalLambda1D``."""
        _assert_1d_loads_every_address(qsharp_test_context, data, -1)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address(self, qsharp_test_context, num_swap_bits, outer_always_valid):
        _assert_2d_loads_every_address(qsharp_test_context, _DATA_2D, num_swap_bits, outer_always_valid)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_both_lengths_are_powers_of_two(
        self, qsharp_test_context, num_swap_bits, outer_always_valid
    ):
        """No address is aliased onto a real row, so the in-range fixups must be no-ops."""
        _assert_2d_loads_every_address(qsharp_test_context, _DATA_2D_POWER_OF_TWO, num_swap_bits, outer_always_valid)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_outer_length_is_not_a_power_of_two(
        self, qsharp_test_context, num_swap_bits, outer_always_valid
    ):
        _assert_2d_loads_every_address(qsharp_test_context, _DATA_2D_RAGGED_OUTER, num_swap_bits, outer_always_valid)


class TestSelectSwapPreservesAddressPhases:
    """The swap path must agree with the plain-select path as a *phase* oracle."""

    @pytest.mark.parametrize("num_swap_bits", [1, 2, 3])
    def test_1d_swap_path_matches_plain_select(self, qsharp_test_utils, num_swap_bits):
        _assert_phase_agreement(qsharp_test_utils, "TestSelectSwap1DPhaseAgreement", _DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_1d_swap_path_matches_plain_select_when_length_is_not_a_power_of_two(
        self, qsharp_test_utils, num_swap_bits
    ):
        _assert_phase_agreement(qsharp_test_utils, "TestSelectSwap1DPhaseAgreement", _DATA_1D_RAGGED, num_swap_bits)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_load_matches_plain_select(self, qsharp_test_utils, num_swap_bits, outer_always_valid):
        """``SelectSwap2D`` erases against the flat table whatever swap width it loaded at.

        Its adjoint is written by hand rather than derived from its body, so this is the only
        test that can see an erasure that is wrong in phase but right in value.
        """
        _assert_phase_agreement(
            qsharp_test_utils, "TestSelectSwap2DPhaseAgreement", _DATA_2D, num_swap_bits, outer_always_valid
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_load_matches_plain_select_when_outer_length_is_not_a_power_of_two(
        self, qsharp_test_utils, num_swap_bits, outer_always_valid
    ):
        _assert_phase_agreement(
            qsharp_test_utils,
            "TestSelectSwap2DPhaseAgreement",
            _DATA_2D_RAGGED_OUTER,
            num_swap_bits,
            outer_always_valid,
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
        ctx.code.QDKChemistry.TestUtils.SelectSwapTests.TestSelectSwap2DResourceProbe,
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
    def test_adjoint_costs_the_unlookup_whatever_the_swap_width(self, qsharp_test_context, num_outer, num_inner, width):
        """Erasure cost follows the table size alone, and undercuts the load it undoes."""
        ctx = qsharp_test_context
        data = [
            [[(o * 31 + i * 7 + b) % 2 == 0 for b in range(width)] for i in range(num_inner)] for o in range(num_outer)
        ]
        expected = _unlookup_toffolis(num_outer * num_inner)

        for num_swap_bits in (0, 1, 2):
            forward = _probe_toffolis(ctx, data, num_swap_bits, forward=True, adjoint=False)
            round_trip = _probe_toffolis(ctx, data, num_swap_bits, forward=True, adjoint=True)

            assert round_trip - forward == expected
            assert expected < forward
