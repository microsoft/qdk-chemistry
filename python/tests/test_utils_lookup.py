"""Tests for the Q# SELECT-SWAP and box-restricted skew-tree table lookups."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import random

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
    """Convert little-endian bits to an integer."""
    return sum(1 << index for index, bit in enumerate(bits) if bit)


def _assert_1d_loads_every_address(context, test_utils, data, num_swap_bits):
    """Check 1D lookup values and register restoration."""
    op = test_utils.SelectSwapTests.TestMakeSelectSwap1DCopyOp(data, num_swap_bits)
    width = len(data[0])
    tester = ArithmeticOpTester(op, [_address_bits(len(data)), width, width], context=context)
    failures = [
        (address, actual)
        for address, row in enumerate(data)
        if (actual := tester.run([address, 0, 0])) != [address, 0, _word(row)]
    ]
    assert not failures, f"(address, [address, output, copy]) mismatches at num_swap_bits={num_swap_bits}: {failures}"


def _assert_2d_loads_every_address(context, test_utils, data, num_swap_bits, outer_always_valid):
    """Check 2D lookup values and register restoration."""
    op = test_utils.SelectSwapTests.TestMakeSelectSwap2DCopyOp(data, num_swap_bits, outer_always_valid)
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
    """Require phase agreement in every trial."""
    operation = getattr(test_utils.SelectSwapTests, operation_name)
    args = (data, num_swap_bits) if "1D" in operation_name else (data, num_swap_bits, outer_always_valid)
    failures = sum(1 for _ in range(_PHASE_TRIALS) if not operation(*args))
    assert failures == 0, (
        f"{operation_name} disagreed with the plain-select path in {failures}/{_PHASE_TRIALS} "
        f"trials at num_swap_bits={num_swap_bits}: the swap path corrupts the address phase."
    )


class TestSelectSwapLoadsCorrectValues:
    """Check lookup values across select/swap splits."""

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    def test_1d_loads_every_address(self, qsharp_test_context, qsharp_test_utils, num_swap_bits):
        _assert_1d_loads_every_address(qsharp_test_context, qsharp_test_utils, _DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_1d_loads_every_address_when_length_is_not_a_power_of_two(
        self, qsharp_test_context, qsharp_test_utils, num_swap_bits
    ):
        _assert_1d_loads_every_address(qsharp_test_context, qsharp_test_utils, _DATA_1D_RAGGED, num_swap_bits)

    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED])
    def test_1d_auto_swap_width_loads_every_address(self, qsharp_test_context, qsharp_test_utils, data):
        """Check lookup values with automatic swap width."""
        _assert_1d_loads_every_address(qsharp_test_context, qsharp_test_utils, data, -1)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address(
        self, qsharp_test_context, qsharp_test_utils, num_swap_bits, outer_always_valid
    ):
        _assert_2d_loads_every_address(
            qsharp_test_context, qsharp_test_utils, _DATA_2D, num_swap_bits, outer_always_valid
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_both_lengths_are_powers_of_two(
        self, qsharp_test_context, qsharp_test_utils, num_swap_bits, outer_always_valid
    ):
        """Check 2D lookup values for power-of-two dimensions."""
        _assert_2d_loads_every_address(
            qsharp_test_context, qsharp_test_utils, _DATA_2D_POWER_OF_TWO, num_swap_bits, outer_always_valid
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_outer_length_is_not_a_power_of_two(
        self, qsharp_test_context, qsharp_test_utils, num_swap_bits, outer_always_valid
    ):
        _assert_2d_loads_every_address(
            qsharp_test_context, qsharp_test_utils, _DATA_2D_RAGGED_OUTER, num_swap_bits, outer_always_valid
        )


class TestSelectSwapPreservesAddressPhases:
    """Check phase equivalence with plain SELECT."""

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
        """Check 2D erasure phases across swap widths."""
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
    """Return the measurement-based SELECT unlookup's Toffoli count."""
    address_bits = max(1, math.ceil(math.log2(num_entries)))
    return 2 ** math.ceil(address_bits / 2) + 2 ** (address_bits // 2) - address_bits - 2


def _probe_toffolis(ctx, test_utils, data, num_swap_bits, *, forward, adjoint):
    """Return the traced 2D lookup's Toffoli count."""
    counts = ctx.logical_counts(
        test_utils.SelectSwapTests.TestSelectSwap2DResourceProbe,
        data,
        num_swap_bits,
        True,
        forward,
        adjoint,
    )
    return counts["cczCount"] + counts["ccixCount"]


class TestSelectSwap2DErasesByMeasurement:
    """Check measurement-based 2D erasure costs."""

    @pytest.mark.parametrize(
        ("num_outer", "num_inner", "width"),
        [(4, 4, 5), (8, 4, 6), (6, 8, 7), (16, 8, 4)],
    )
    def test_adjoint_costs_the_unlookup_whatever_the_swap_width(
        self, qsharp_test_context, qsharp_test_utils, num_outer, num_inner, width
    ):
        """Check erasure costs depend only on table size and undercut loading."""
        ctx = qsharp_test_context
        data = [
            [[(o * 31 + i * 7 + b) % 2 == 0 for b in range(width)] for i in range(num_inner)] for o in range(num_outer)
        ]
        expected = _unlookup_toffolis(num_outer * num_inner)

        for num_swap_bits in (0, 1, 2):
            forward = _probe_toffolis(ctx, qsharp_test_utils, data, num_swap_bits, forward=True, adjoint=False)
            round_trip = _probe_toffolis(ctx, qsharp_test_utils, data, num_swap_bits, forward=True, adjoint=True)

            assert round_trip - forward == expected
            assert expected < forward


class TestSelectSwapCleanErasesByMeasurement:
    """Check clean 1D erasure costs and coherence."""

    @pytest.mark.parametrize("use_control", [False, True])
    @pytest.mark.parametrize(
        ("data", "num_swap_bits"),
        [(_DATA_1D, k) for k in (-1, 0, 1, 2, 3)] + [(_DATA_1D_RAGGED, k) for k in (-1, 0, 1, 2)],
    )
    def test_matches_plain_select_as_a_phase_oracle(self, qsharp_test_utils, data, num_swap_bits, use_control):
        """Check clean erasure phases against plain SELECT."""
        operation = qsharp_test_utils.SelectSwapTests.TestSelectSwapCleanPhaseAgreement
        failures = sum(1 for _ in range(_PHASE_TRIALS) if not operation(data, num_swap_bits, use_control))

        assert failures == 0, f"disagreed with plain select in {failures}/{_PHASE_TRIALS} trials"

    @pytest.mark.parametrize(("num_rows", "width"), [(8, 3), (64, 20), (250, 30)])
    def test_adjoint_costs_the_unlookup_rather_than_a_second_load(
        self, qsharp_test_context, qsharp_test_utils, num_rows, width
    ):
        """Check erasure costs depend only on row count and undercut loading."""
        ctx = qsharp_test_context
        probe = qsharp_test_utils.SelectSwapTests.TestSelectSwapCleanResourceProbe
        data = _make_table(num_rows, width)

        def toffolis(forward, adjoint):
            counts = ctx.logical_counts(probe, data, -1, forward, adjoint)
            return counts["cczCount"] + counts["ccixCount"]

        forward = toffolis(True, False)

        assert toffolis(False, True) == _unlookup_toffolis(num_rows) < forward


_DIRTY_TRIALS = 6


def _select_swap_ns(ctx):
    """Return the clean lookup namespace."""
    return ctx.code.QDKChemistry.Utils.SelectSwap


def _select_swap_dirty_ns(ctx):
    """Return the borrowed-qubit lookup namespace."""
    return ctx.code.QDKChemistry.Utils.SelectSwapDirty


def _make_table(num_rows: int, width: int) -> list[list[bool]]:
    """Build deterministic rows, distinct when the word width permits."""
    return [[bool(((r * 91 + 53) % (1 << width)) >> b & 1) for b in range(width)] for r in range(num_rows)]


def _routing_tables(num_rows: int, width: int) -> list[list[list[bool]]]:
    """Build tables that collectively distinguish every address pair."""
    if num_rows <= 1 << width:
        return [_make_table(num_rows, width)]
    num_address_bits = math.ceil(math.log2(num_rows))
    return [_make_table(num_rows, width)] + [
        [[bool(row >> ((first + b) % num_address_bits) & 1) for b in range(width)] for row in range(num_rows)]
        for first in range(0, num_address_bits, width)
    ]


def _make_table_2d(num_outer: int, num_inner: int, width: int) -> list[list[list[bool]]]:
    """Split deterministic rows into a 2D table."""
    rows = _make_table(num_outer * num_inner, width)
    return [rows[block * num_inner : (block + 1) * num_inner] for block in range(num_outer)]


class TestSelectSwapDirtyLoadsCorrectValues:
    """Check dirty lookup values and borrowed-qubit restoration."""

    @pytest.mark.parametrize("dirty_seed", [0, 1, 5])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    @pytest.mark.parametrize(("num_rows", "width"), [(8, 1), (16, 1)])
    def test_loads_every_address(self, qsharp_test_utils, num_rows, width, num_swap_bits, dirty_seed):
        """Check dirty lookup values with nonzero borrowed inputs."""
        for data in _routing_tables(num_rows, width):
            assert qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwapDirtyCorrectness(
                data, num_swap_bits, dirty_seed
            ), data

    @pytest.mark.parametrize("dirty_seed", [0, 3])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    @pytest.mark.parametrize(("num_rows", "width"), [(5, 2), (6, 2), (7, 1), (11, 2)])
    def test_loads_every_address_when_length_is_not_a_power_of_two(
        self, qsharp_test_utils, num_rows, width, num_swap_bits, dirty_seed
    ):
        """Check ragged-table aliasing against plain SELECT."""
        for data in _routing_tables(num_rows, width):
            assert qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwapDirtyCorrectness(
                data, num_swap_bits, dirty_seed
            ), data


class TestSelectSwapDirtyReturnsTheBorrowedQubits:
    """Check borrowed-qubit coherence."""

    @pytest.mark.parametrize(
        ("num_rows", "width", "num_swap_bits"),
        [(8, 1, 1), (8, 1, 2), (8, 1, 3), (6, 2, 1), (6, 2, 2), (5, 1, 2)],
    )
    def test_borrowed_register_is_restored_exactly(self, qsharp_test_utils, num_rows, width, num_swap_bits):
        """Check one dirty load against independent erasure on Hadamard-basis inputs."""
        for data in _routing_tables(num_rows, width):
            failures = sum(
                1
                for _ in range(_DIRTY_TRIALS)
                if not qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwapDirtyPhaseAgreement(data, num_swap_bits)
            )
            assert failures == 0, (
                f"the borrowed register was not restored in {failures}/{_DIRTY_TRIALS} trials at "
                f"num_swap_bits={num_swap_bits} on {data}: the lender is left entangled with the address."
            )


class TestSelectSwap2DBorrowedMatchesClean:
    """Check borrowed and clean 2D lookup equivalence."""

    @pytest.mark.parametrize("dirty_seed", [0, 5])
    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    @pytest.mark.parametrize("data", [_DATA_2D, _DATA_2D_RAGGED_OUTER], ids=["square", "ragged_outer"])
    def test_every_address_loads_the_clean_word_and_returns_the_lender(
        self, qsharp_test_utils, data, num_swap_bits, outer_always_valid, dirty_seed
    ):
        """Check borrowed 2D lookup values and lender restoration."""
        assert qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwap2DDirtyMatchesClean(
            data, num_swap_bits, outer_always_valid, dirty_seed
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    @pytest.mark.parametrize("data", [_DATA_2D, _DATA_2D_RAGGED_OUTER], ids=["square", "ragged_outer"])
    def test_one_load_preserves_address_and_lender_phases(
        self, qsharp_test_utils, data, num_swap_bits, outer_always_valid
    ):
        """Check address and lender phase preservation."""
        failures = sum(
            1
            for _ in range(_DIRTY_TRIALS)
            if not qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwap2DDirtyPhaseAgreement(
                data, num_swap_bits, outer_always_valid
            )
        )
        assert failures == 0, f"lender or address phases disturbed in {failures}/{_DIRTY_TRIALS} trials"


class TestSelectSwapDirtyCostModel:
    """Check dirty lookup cost models."""

    def test_select_swap_dirty_cost_1d(self, qsharp_test_context):
        """Check dirty 1D costs against the reference formula."""
        cost = _select_swap_dirty_ns(qsharp_test_context).SelectSwapDirtyCost1D
        for num_data in (8, 15, 224, 864):
            assert cost(0, num_data, 10) == num_data - 1

        # (90, 15) is Fe2S2's inner PREPARE, just above 2^6 where padded-height chunking overcharges.
        for num_data, num_bits in [(90, 15), (224, 15), (100, 8), (1000, 20), (4095, 8)]:
            for num_swap_bits in range(1, math.ceil(math.log2(num_data)) + 1):
                block = 1 << num_swap_bits
                reference = 2 * math.ceil(num_data / block) + 4 * num_bits * (block - 1)
                assert cost(num_swap_bits, num_data, num_bits) == reference - 2, (
                    f"width {num_swap_bits} on a {num_data}x{num_bits} table should cost "
                    f"{reference - 2}, got {cost(num_swap_bits, num_data, num_bits)}"
                )

    def test_compute_optimal_dirty_swap_bits(self, qsharp_test_context):
        """Check dirty swap-width selection, qubit budgets, and caps."""
        select_swap = _select_swap_dirty_ns(qsharp_test_context)
        # Fe2S2's 224-row, 15-bit angle table sits below the ``numData > 32 * numBits`` crossover.
        for num_data, num_bits in [(20, 15), (224, 15), (64, 10)]:
            assert select_swap.ComputeOptimalDirtySwapBits(num_data, num_bits, 4096, -1) == 0

        for num_data, num_bits in [(864, 10), (2048, 8)]:
            width = select_swap.ComputeOptimalDirtySwapBits(num_data, num_bits, 4096, -1)
            assert width > 0
            assert select_swap.SelectSwapDirtyCost1D(width, num_data, num_bits) < num_data - 1

        unconstrained = select_swap.ComputeOptimalDirtySwapBits(864, 10, 4096, -1)
        assert select_swap.SelectSwapDirtyBorrowedQubits(unconstrained, 10) > 10
        assert select_swap.ComputeOptimalDirtySwapBits(864, 10, 10, -1) == 0
        for available in (0, 10, 40, 80, 160, 640):
            width = select_swap.ComputeOptimalDirtySwapBits(864, 10, available, -1)
            assert width == 0 or select_swap.SelectSwapDirtyBorrowedQubits(width, 10) <= available

        # Width 1 costs 902 against the plain lookup's 863, so a cap of 1 keeps ``Select`` rather than clamping to it.
        assert select_swap.ComputeOptimalDirtySwapBits(864, 10, 4096, 1) == 0
        costs = [select_swap.SelectSwapDirtyCost1D(width, 864, 10) for width in range(11)]
        for cap in range(12):
            best = min(range(min(cap, 10) + 1), key=costs.__getitem__)
            assert select_swap.ComputeOptimalDirtySwapBits(864, 10, 4096, cap) == best, f"cap {cap}"

    def test_select_swap_dirty_cost_2d(self, qsharp_test_context, qsharp_test_utils):
        """Check dirty 2D costs against traced load and erasure."""
        ctx = qsharp_test_context
        select_swap = _select_swap_dirty_ns(ctx)
        probe = qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwap2DDirtyResourceProbe
        # (90, 16, 23) is Fe2S2's inner PREPARE; (1, 16, 2) at full width leaves both select passes free.
        for num_outer, num_inner, num_bits in [(1, 1, 2), (3, 5, 1), (4, 8, 3), (1, 16, 2), (90, 16, 23)]:
            data = _make_table_2d(num_outer, num_inner, num_bits)
            for width in range(math.ceil(math.log2(num_inner)) + 1):
                lent = select_swap.SelectSwapDirtyBorrowedQubits(width, num_bits)
                counts = ctx.logical_counts(probe, data, width, True, lent)
                assert select_swap.SelectSwapDirtyCost2D(width, num_outer, num_inner, num_bits, True) == (
                    counts["cczCount"] + counts["ccixCount"]
                ), f"width {width} on a {num_outer}x{num_inner}x{num_bits} table"

    def test_compute_optimal_dirty_swap_bits_2d(self, qsharp_test_context):
        """Check dirty 2D swap-width selection within budgets and caps."""
        select_swap = _select_swap_dirty_ns(qsharp_test_context)
        # Width 2 measures 25 Toffolis against the plain lookup's 27, but needs all four lent qubits.
        assert select_swap.ComputeOptimalDirtySwapBits2D(3, 5, 1, True, 4, -1) == 2
        assert select_swap.ComputeOptimalDirtySwapBits2D(3, 5, 1, True, 3, -1) == 0
        for available in (0, 1, 2, 4, 8):
            width = select_swap.ComputeOptimalDirtySwapBits2D(3, 5, 1, True, available, -1)
            assert width == 0 or select_swap.SelectSwapDirtyBorrowedQubits(width, 1) <= available
        # Width 1 measures 29, so a cap of 1 stays on the plain lookup instead of clamping width 2 down to it.
        assert [select_swap.ComputeOptimalDirtySwapBits2D(3, 5, 1, True, 4, cap) for cap in range(4)] == [0, 0, 2, 2]

    def test_select_swap_dirty_borrowed_qubits(self, qsharp_test_context, qsharp_test_utils):
        """Check the minimum borrowed-register size."""
        ctx = qsharp_test_context
        select_swap = _select_swap_dirty_ns(ctx)
        probe = qsharp_test_utils.SelectSwapDirtyTests.TestSelectSwap2DDirtyResourceProbe
        num_bits = 3
        data = _make_table_2d(4, 8, num_bits)

        assert select_swap.SelectSwapDirtyBorrowedQubits(0, num_bits) == 0
        for width in (1, 2, 3):
            lent = select_swap.SelectSwapDirtyBorrowedQubits(width, num_bits)
            ctx.logical_counts(probe, data, width, True, lent)
            with pytest.raises(Exception, match="dirty register needs"):
                ctx.logical_counts(probe, data, width, True, lent - 1)


class TestCleanSelectSwapForwardCostModel:
    """Check clean forward lookup cost models."""

    def test_select_swap_forward_cost(self, qsharp_test_context, qsharp_test_utils):
        """Check clean forward costs against traced Toffoli counts."""
        ctx = qsharp_test_context
        probe = qsharp_test_utils.SOSSAWalkTests.TestLoadRotationWordResourceProbe
        for num_data, num_bits in [(224, 15), (64, 10)]:
            data = _make_table(num_data, num_bits)
            for width in range(math.ceil(math.log2(num_data)) + 1):
                counts = ctx.logical_counts(probe, data, width)
                assert _select_swap_ns(ctx).SelectSwapForwardCost(width, num_data, num_bits) == (
                    counts["cczCount"] + counts["ccixCount"]
                ), f"width {width} on a {num_data}x{num_bits} table"

    def test_compute_optimal_swap_bits(self, qsharp_test_context):
        """Check minimum-cost swap-width selection within caps."""
        select_swap = _select_swap_ns(qsharp_test_context)
        for num_data, num_bits in [(224, 15), (864, 10), (32, 4)]:
            address_bits = math.ceil(math.log2(num_data))
            costs = [select_swap.SelectSwapForwardCost(k, num_data, num_bits) for k in range(address_bits + 1)]
            assert select_swap.SelectSwapForwardCost(
                select_swap.ComputeOptimalSwapBits(num_data, num_bits, -1), num_data, num_bits
            ) == min(costs)
            assert min(costs) < num_data - 1
            for cap in range(address_bits + 2):
                chosen = select_swap.ComputeOptimalSwapBits(num_data, num_bits, cap)
                assert chosen <= cap
                assert costs[chosen] == min(costs[: cap + 1]), f"cap {cap} on a {num_data}x{num_bits} table"
        # Every width pays the swaps twice plus a controlled copy, which these tables never win back.
        for num_data, num_bits in [(64, 10), (4, 64)]:
            assert select_swap.ComputeOptimalSwapBits(num_data, num_bits, -1) == 0

    def test_select_swap_scratch_qubits(self, qsharp_test_context, qsharp_test_utils):
        """Check scratch-qubit counts against traced peaks."""
        ctx = qsharp_test_context
        select_swap = _select_swap_ns(ctx)
        probe = qsharp_test_utils.SOSSAWalkTests.TestLoadRotationWordResourceProbe

        assert select_swap.SelectSwapScratchQubits(0, 15) == 0
        for num_data, num_bits in [(8, 3), (64, 10), (224, 15)]:
            width = math.ceil(math.log2(num_data))
            # Full width leaves no unary-iteration ancillas, so only the control, address and target remain.
            counts = ctx.logical_counts(probe, _make_table(num_data, num_bits), width)
            assert counts["numQubits"] - (1 + width + num_bits) == select_swap.SelectSwapScratchQubits(width, num_bits)


class TestCleanSelectSwap2DCostModel:
    """Check clean 2D lookup cost models."""

    def test_select_swap_cost_2d(self, qsharp_test_context, qsharp_test_utils):
        """Check clean 2D costs against traced load and erasure."""
        ctx = qsharp_test_context
        select_swap = _select_swap_ns(ctx)
        for num_outer, num_inner, num_bits in [(1, 1, 2), (3, 5, 1), (1, 8, 2), (1, 16, 3), (90, 16, 1), (90, 16, 23)]:
            data = _make_table_2d(num_outer, num_inner, num_bits)
            for width in range(math.ceil(math.log2(num_inner)) + 1):
                assert select_swap.SelectSwapCost2D(width, num_outer, num_inner, num_bits, True) == _probe_toffolis(
                    ctx, qsharp_test_utils, data, width, forward=True, adjoint=True
                ), f"width {width} on a {num_outer}x{num_inner}x{num_bits} table"

    def test_compute_optimal_lambda_2d(self, qsharp_test_context):
        """Check the narrowest allowed width within 20% of minimum cost."""
        select_swap = _select_swap_ns(qsharp_test_context)
        # One-bit words make the swaps cheap enough that only swapping the whole inner address is near the minimum.
        assert select_swap.ComputeOptimalLambda2D(90, 16, 1, True, -1) == 4
        for num_outer, num_inner, num_bits in [(90, 16, 1), (90, 16, 23), (1, 16, 3), (1, 8, 2), (3, 5, 1)]:
            address_bits = math.ceil(math.log2(num_inner))
            costs = [
                select_swap.SelectSwapCost2D(width, num_outer, num_inner, num_bits, True)
                for width in range(address_bits + 1)
            ]
            for cap in [-1, *range(address_bits + 2)]:
                allowed = costs if cap < 0 else costs[: cap + 1]
                narrowest = next(width for width, cost in enumerate(allowed) if cost <= min(allowed) * 1.2)
                assert select_swap.ComputeOptimalLambda2D(num_outer, num_inner, num_bits, True, cap) == narrowest, (
                    f"{num_outer}x{num_inner}x{num_bits} with cap {cap} and costs {costs}"
                )


class TestSelectSwapAliasedMatchesPlainSelect:
    """Check lookup address routing against plain SELECT."""

    @pytest.mark.parametrize("num_swap_bits", [-1, 0, 1, 2])
    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED], ids=["power_of_two", "ragged"])
    def test_aliased_load_matches_plain_select(self, qsharp_test_utils, num_swap_bits, data):
        assert qsharp_test_utils.SelectSwapTests.TestSelectSwapAliasedMatchesSelect1D(data, num_swap_bits)


# (fields, rows, word width): two padded fields, a padded high field, a ragged 1D table, an unpadded low field.
_SKEW_TREE_SHAPES = [
    ([(2, 3), (2, 3)], 11, 4),
    ([(3, 5), (2, 3)], 21, 3),
    ([], 6, 5),
    ([(1, 2), (3, 6)], 16, 2),
]
_SKEW_TREE_SHAPE_IDS = ["box3x3", "box5x3", "ragged1d", "box2x6"]


def _xor_rows(rows, width):
    """Return the bitwise XOR of rows."""
    out = [False] * width
    for row in rows:
        out = [a != b for a, b in zip(out, row, strict=True)]
    return out


def _skew_tree_shape(fields, num_rows, width, seed):
    """Return a random table, address width, and in-box addresses."""
    rng = random.Random(seed)
    num_bits = sum(bits for bits, _ in fields) or math.ceil(math.log2(num_rows))
    data = [[rng.random() < 0.5 for _ in range(width)] for _ in range(num_rows)]
    offsets = [sum(bits for bits, _ in fields[:j]) for j in range(len(fields))]
    box = [
        x
        for x in range(num_rows)
        if all(
            (x >> offset) & ((1 << bits) - 1) < extent for offset, (bits, extent) in zip(offsets, fields, strict=True)
        )
    ]
    return data, num_bits, box


@pytest.mark.parametrize("num_controls", [0, 1, 2])
@pytest.mark.parametrize(("fields", "num_rows", "width"), _SKEW_TREE_SHAPES, ids=_SKEW_TREE_SHAPE_IDS)
def test_skew_tree_loads_every_in_box_address(
    qsharp_test_context, qsharp_test_utils, fields, num_rows, width, num_controls
):
    """Check controlled skew-tree values and register restoration."""
    data, num_bits, box = _skew_tree_shape(fields, num_rows, width, seed=num_rows)
    op = qsharp_test_utils.SkewTreeTests.TestMakeSkewTreeCopyOp(data, fields, num_controls)
    tester = ArithmeticOpTester(op, [max(num_controls, 1), num_bits, width, width], context=qsharp_test_context)
    on = (1 << num_controls) - 1
    # ArithmeticOpTester reads registers back as little-endian integers.
    failures = [
        x for x in box if tester.run([on, x, 0, 0]) != [on, x, 0, sum(1 << i for i, bit in enumerate(data[x]) if bit)]
    ]
    if num_controls:
        failures += [("off", x) for x in box if tester.run([0, x, 0, 0]) != [0, x, 0, 0]]
    assert not failures


@pytest.mark.parametrize("num_controls", [0, 1, 2])
@pytest.mark.parametrize(("fields", "num_rows", "width"), _SKEW_TREE_SHAPES, ids=_SKEW_TREE_SHAPE_IDS)
def test_skew_tree_phase_matches_an_independent_reference(qsharp_test_utils, fields, num_rows, width, num_controls):
    """Check skew-tree phase equivalence and mutation detection."""
    data, num_bits, box = _skew_tree_shape(fields, num_rows, width, seed=num_rows + 1)
    # Outside the box the lookup writes the XOR of the in-box subset-XOR coefficients below the address.
    coefficients = {p: _xor_rows((data[q] for q in range(p + 1) if q & p == q), width) for p in box}
    reference = [_xor_rows((c for p, c in coefficients.items() if p & x == p), width) for x in range(1 << num_bits)]
    assert all(reference[x] == data[x] for x in box)
    tests = qsharp_test_utils.SkewTreeTests

    assert tests.TestSkewTreePhaseAgreement(data, reference, fields, num_bits, num_controls)
    mutated = [list(row) for row in data]
    mutated[box[-1]][0] = not mutated[box[-1]][0]
    assert not tests.TestSkewTreePhaseAgreement(mutated, reference, fields, num_bits, num_controls)


def test_skew_tree_costs_one_toffoli_per_box_node_and_undercuts_a_padded_select(qsharp_test_context, qsharp_test_utils):
    """Check skew-tree Toffoli counts against box size and padded SELECT."""
    fields, num_rows, width = _SKEW_TREE_SHAPES[0]
    data, num_bits, box = _skew_tree_shape(fields, num_rows, width, seed=0)

    def toffolis(*args):
        """Return the traced Toffoli count."""
        counts = qsharp_test_context.logical_counts(*args)
        return counts["cczCount"] + counts["ccixCount"]

    skew = toffolis(qsharp_test_utils.SkewTreeTests.TestSkewTreeResourceProbe, data, fields, num_bits, 1)
    padded = toffolis(qsharp_test_utils.SOSSAWalkTests.TestLoadRotationWordResourceProbe, data, 0)
    assert skew == len(box) - 1
    assert padded == num_rows - 1
    assert skew < padded
