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

    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED])
    def test_1d_auto_swap_width_loads_every_address(self, data):
        """``numSwapBits = -1`` hands the select/swap split to ``ComputeOptimalLambda1D``."""
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap1DCorrectness(data, -1)

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address(self, num_swap_bits, outer_always_valid):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap2DCorrectness(
            _DATA_2D, num_swap_bits, outer_always_valid
        )

    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_word_loads_every_address_when_both_lengths_are_powers_of_two(self, num_swap_bits, outer_always_valid):
        """No address is aliased onto a real row, so the in-range fixups must be no-ops."""
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwap2DCorrectness(
            _DATA_2D_POWER_OF_TWO, num_swap_bits, outer_always_valid
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
    """The swap path must agree with the plain-select path as a *phase* oracle."""

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
        """Erasure cost follows the table size alone, and undercuts the load it undoes."""
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


_DIRTY_TRIALS = 6


def _select_swap_ns(ctx=None):
    """The clean-strategy namespace: loaders that allocate their own swap scratch."""
    return (ctx or get_qsharp_context()).code.QDKChemistry.Utils.SelectSwap


def _select_swap_dirty_ns(ctx=None):
    """Return the dirty-strategy namespace for loaders that borrow live caller qubits.

    Keeping it separate catches calls routed through the clean namespace.
    """
    return (ctx or get_qsharp_context()).code.QDKChemistry.Utils.SelectSwapDirty


def _make_table(num_rows: int, width: int) -> list[list[bool]]:
    """Builds a deterministic bit table whose rows differ, so a misrouted address is visible.

    A constant or repeating pattern would let a lookup that returns the wrong row still pass.
    """
    return [[(r * 37 + b * 11 + r * b) % 3 == 0 for b in range(width)] for r in range(num_rows)]


class TestSelectSwapDirtyLoadsCorrectValues:
    """Dirty loads must match clean lookup words and return borrowed qubits.

    Sweep aliases too, because ragged tables route surplus addresses through ``Select``.
    """

    @pytest.mark.parametrize("dirty_seed", [0, 1, 5])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    @pytest.mark.parametrize(("num_rows", "width"), [(8, 1), (16, 1)])
    def test_loads_every_address(self, num_rows, width, num_swap_bits, dirty_seed):
        """Power-of-two tables are the baseline, with no surplus addresses to alias.

        ``dirty_seed`` catches loaders that only work from a zeroed borrow.
        """
        assert _select_swap_dirty_ns().TestSelectSwapDirtyCorrectness(
            _make_table(num_rows, width), num_swap_bits, dirty_seed
        )

    @pytest.mark.parametrize("dirty_seed", [0, 3])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    @pytest.mark.parametrize(("num_rows", "width"), [(5, 2), (6, 2), (7, 1), (11, 2)])
    def test_loads_every_address_when_length_is_not_a_power_of_two(self, num_rows, width, num_swap_bits, dirty_seed):
        """Surplus addresses have to alias exactly as the plain lookup aliases them."""
        assert _select_swap_dirty_ns().TestSelectSwapDirtyCorrectness(
            _make_table(num_rows, width), num_swap_bits, dirty_seed
        )


class TestSelectSwapDirtyReturnsTheBorrowedQubits:
    """The lender comes back unentangled, or the borrowing silently corrupts the caller."""

    @pytest.mark.parametrize(
        ("num_rows", "width", "num_swap_bits"),
        [(8, 1, 1), (8, 1, 2), (8, 1, 3), (6, 2, 1), (6, 2, 2), (5, 1, 2)],
    )
    def test_borrowed_register_is_restored_exactly(self, num_rows, width, num_swap_bits):
        """A phase oracle catches any entanglement left on the lender.

        Compare against width-0 select-swap so the same measurement-erasure path is used.
        """
        data = _make_table(num_rows, width)
        failures = sum(
            1
            for _ in range(_DIRTY_TRIALS)
            if not _select_swap_dirty_ns().TestSelectSwapDirtyPhaseAgreement(data, num_swap_bits)
        )
        assert failures == 0, (
            f"the borrowed register was not restored in {failures}/{_DIRTY_TRIALS} trials at "
            f"num_swap_bits={num_swap_bits}: the lender is left entangled with the address."
        )


class TestSelectSwap2DBorrowedMatchesClean:
    """Lending ``SelectSwap2D`` a register changes where the swap block lives, not what loads."""

    @pytest.mark.parametrize("dirty_seed", [0, 5])
    @pytest.mark.parametrize("outer_always_valid", [False, True])
    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    @pytest.mark.parametrize("data", [_DATA_2D, _DATA_2D_RAGGED_OUTER], ids=["square", "ragged_outer"])
    def test_every_address_loads_the_clean_word_and_returns_the_lender(
        self, data, num_swap_bits, outer_always_valid, dirty_seed
    ):
        """Nothing else reaches the borrowed 2D branch: at the Fe2S2 shapes its cost model declines."""
        assert _select_swap_dirty_ns().TestSelectSwap2DDirtyMatchesClean(
            data, num_swap_bits, outer_always_valid, dirty_seed
        )


class TestSelectSwapDirtyCostModel:
    """The width is chosen by cost, so the cost model is what decides if borrowing happens."""

    def test_select_swap_dirty_cost_1d(self):
        """Width 0 is the plain lookup; every wider width is the reference ``2*ceil(d/K) + 4b(K-1)``, less 2."""
        cost = _select_swap_dirty_ns().SelectSwapDirtyCost1D
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

    def test_compute_optimal_dirty_swap_bits(self):
        """Borrows only past the crossover, then beats the plain lookup without overdrawing the budget."""
        select_swap = _select_swap_dirty_ns()
        # Fe2S2's 224-row, 15-bit angle table sits below the ``numData > 32 * numBits`` crossover.
        for num_data, num_bits in [(20, 15), (224, 15), (64, 10)]:
            assert select_swap.ComputeOptimalDirtySwapBits(num_data, num_bits, 4096) == 0

        for num_data, num_bits in [(864, 10), (2048, 8)]:
            width = select_swap.ComputeOptimalDirtySwapBits(num_data, num_bits, 4096)
            assert width > 0
            assert select_swap.SelectSwapDirtyCost1D(width, num_data, num_bits) < num_data - 1

        unconstrained = select_swap.ComputeOptimalDirtySwapBits(864, 10, 4096)
        assert select_swap.SelectSwapDirtyBorrowedQubits(unconstrained, 10) > 10
        assert select_swap.ComputeOptimalDirtySwapBits(864, 10, 10) == 0
        for available in (0, 10, 40, 80, 160, 640):
            width = select_swap.ComputeOptimalDirtySwapBits(864, 10, available)
            assert width == 0 or select_swap.SelectSwapDirtyBorrowedQubits(width, 10) <= available


class TestCleanSelectSwapForwardCostModel:
    """The clean network is chosen for a load that is erased by measurement, not by its adjoint."""

    def test_select_swap_forward_cost(self):
        """Matches the traced Toffolis of the clean rotation-word load at every width."""
        ctx = get_qsharp_context()
        probe = ctx.code.QDKChemistry.Utils.SOSSAWalk.TestLoadRotationWordResourceProbe
        for num_data, num_bits in [(224, 15), (64, 10)]:
            data = _make_table(num_data, num_bits)
            for width in range(math.ceil(math.log2(num_data)) + 1):
                counts = ctx.logical_counts(probe, data, width)
                assert _select_swap_ns(ctx).SelectSwapForwardCost(width, num_data, num_bits) == (
                    counts["cczCount"] + counts["ccixCount"]
                ), f"width {width} on a {num_data}x{num_bits} table"

    def test_compute_optimal_swap_bits(self):
        """Picks the true argmin over every width, and stays on the plain lookup when no width beats it."""
        select_swap = _select_swap_ns()
        for num_data, num_bits in [(224, 15), (864, 10), (32, 4)]:
            chosen = select_swap.ComputeOptimalSwapBits(num_data, num_bits)
            address_bits = math.ceil(math.log2(num_data))
            best = min(select_swap.SelectSwapForwardCost(k, num_data, num_bits) for k in range(address_bits + 1))
            assert select_swap.SelectSwapForwardCost(chosen, num_data, num_bits) == best
            assert best < num_data - 1
        # Every width pays the swaps twice plus a controlled copy, which these tables never win back.
        for num_data, num_bits in [(64, 10), (4, 64)]:
            assert select_swap.ComputeOptimalSwapBits(num_data, num_bits) == 0

    def test_select_swap_scratch_qubits(self):
        """The Toffoli saving is only half the trade; callers need the width it is bought with."""
        select_swap = _select_swap_ns()
        width = select_swap.ComputeOptimalSwapBits(224, 15)

        assert select_swap.SelectSwapScratchQubits(0, 15) == 0
        assert select_swap.SelectSwapScratchQubits(width, 15) == 15 * (2**width - 1)


class TestSelectSwapAliasedMatchesPlainSelect:
    """A loader sharing ``Select`` erasure must share its address routing.

    The check is by value because phase-oracle harnesses disagree with ``Select`` on ragged tables.
    """

    @pytest.mark.parametrize("num_swap_bits", [-1, 0, 1, 2])
    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED], ids=["power_of_two", "ragged"])
    def test_aliased_load_matches_plain_select(self, num_swap_bits, data):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwapAliasedMatchesSelect1D(
            data, num_swap_bits
        )
