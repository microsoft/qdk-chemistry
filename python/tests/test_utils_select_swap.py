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
    """The dirty-strategy namespace: loaders that borrow live caller qubits instead.

    Split from ``_select_swap_ns`` because the two strategies now live in separate Q# files,
    so a test that reaches for a dirty symbol through the clean namespace fails to resolve
    rather than silently exercising the wrong loader.
    """
    return (ctx or get_qsharp_context()).code.QDKChemistry.Utils.SelectSwapDirty


def _make_table(num_rows: int, width: int) -> list[list[bool]]:
    """Builds a deterministic bit table whose rows differ, so a misrouted address is visible.

    A constant or repeating pattern would let a lookup that returns the wrong row still pass.
    """
    return [[(r * 37 + b * 11 + r * b) % 3 == 0 for b in range(width)] for r in range(num_rows)]


class TestDirtyQROAMLoadsCorrectValues:
    """Borrowing live qubits must load the same word a clean lookup would.

    The borrowed register is mid-computation and entangled with the rest of the machine, so
    the network has to put it back bit-for-bit. Sweeping the whole address space matters
    because surplus addresses -- those past the end of a non-power-of-two table -- are
    aliased onto real rows by ``Std.TableLookup.Select`` rather than reading as zero, and the
    swap path has to alias them the same way.

    Shapes are kept narrow on purpose: a swap width of ``k`` borrows ``width * 2**k`` qubits,
    which lands in the simulated statevector, so a wide word at a wide swap is unsimulable.
    """

    @pytest.mark.parametrize("dirty_seed", [0, 1, 5])
    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    @pytest.mark.parametrize(("num_rows", "width"), [(8, 1), (16, 1)])
    def test_loads_every_address(self, num_rows, width, num_swap_bits, dirty_seed):
        """Power-of-two tables are the baseline: every address is real, so nothing may alias.

        ``dirty_seed`` varies the junk the borrowed register starts in, since a network that
        only works from a zeroed borrow is not borrowing at all.
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


class TestDirtyQROAMReturnsTheBorrowedQubits:
    """The lender comes back unentangled, or the borrowing silently corrupts the caller."""

    @pytest.mark.parametrize(
        ("num_rows", "width", "num_swap_bits"),
        [(8, 1, 1), (8, 1, 2), (8, 1, 3), (6, 2, 1), (6, 2, 2), (5, 1, 2)],
    )
    def test_borrowed_register_is_restored_exactly(self, num_rows, width, num_swap_bits):
        """A phase oracle sees any residue the load leaves behind on the lender.

        Restoring the borrowed qubits' *values* is not enough: if the load leaves them
        correlated with the address, conjugating a phase kick will not close. Comparing
        against the width-0 path rather than a bare ``Select`` keeps the measurement on this
        network, since the library's measurement-based ``Adjoint Select`` is itself not
        self-cancelling on non-power-of-two tables.

        The load erases by measurement, so a single passing trial proves nothing.
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


class TestDirtyQROAMCostModel:
    """The width is chosen by cost, so the cost model is what decides if borrowing happens."""

    def test_width_zero_costs_the_plain_lookup(self):
        """Zero swap bits has to mean *no swap network*, not a one-block swap network.

        Modelling it as the ``K = 1`` limit of the two-pass formula would report roughly
        twice the true cost and make every swap width look like an improvement over a
        baseline that was never real.
        """
        cost = _select_swap_dirty_ns().DirtyQROAMCost
        for num_data in (8, 15, 224, 864):
            assert cost(0, num_data, 10) == num_data - 1

    @pytest.mark.parametrize(("num_data", "num_bits"), [(90, 15), (224, 15), (100, 8), (1000, 20), (4095, 8)])
    def test_the_cost_tracks_the_reference_formula_at_every_width(self, num_data, num_bits):
        """Each pass must address ``ceil(d/K)`` rows, not the padded height ``2^ceil(lg d)``.

        A strided chunking pins the table to its padded height, which nothing relational can
        see: the cost stays monotone in the width, still declines on short tables, still beats
        the plain lookup wherever it claims to. It is wrong only against the *reference*, and
        only far from a power of two -- at ``d = 90, K = 4`` it charged 62 Toffolis of
        ``Select`` against :cite:`Berry2019`'s 46, on the term the width search minimises.

        The ``- 2`` is not slack: ``ceil(d/K) - 1`` is the exact unary-iteration cost per pass
        where the reference quotes the bound.
        """
        cost = _select_swap_dirty_ns().DirtyQROAMCost
        for num_swap_bits in range(1, math.ceil(math.log2(num_data)) + 1):
            block = 1 << num_swap_bits
            reference = 2 * math.ceil(num_data / block) + 4 * num_bits * (block - 1)
            assert cost(num_swap_bits, num_data, num_bits) == reference - 2, (
                f"width {num_swap_bits} on a {num_data}x{num_bits} table should cost "
                f"{reference - 2}, got {cost(num_swap_bits, num_data, num_bits)}"
            )

    def test_the_fe2s2_inner_shape_costs_what_the_reference_charges(self):
        """One magnitude pin on the shape the overcharge was found at, so a regression names itself.

        ``d = 90`` is the worst case for padding: just above ``2^6``, so a strided table rounds
        all the way to ``2^7`` rows and charges nearly 40% more ``Select`` than it addresses.
        """
        select_cost = 2 * (math.ceil(90 / 4) - 1)
        butterfly_cost = 4 * 15 * (4 - 1)

        assert (select_cost, butterfly_cost) == (44, 180)
        assert _select_swap_dirty_ns().DirtyQROAMCost(2, 90, 15) == select_cost + butterfly_cost

    @pytest.mark.parametrize(("num_data", "num_bits"), [(20, 15), (224, 15), (64, 10)])
    def test_short_tables_decline_to_borrow(self, num_data, num_bits):
        """Below roughly ``numData > 32 * numBits`` a swap network cannot win, so it is refused.

        The two-pass structure costs ``2*ceil(d/K) + 4*b*(K-1)``, whose optimum ``4*sqrt(2*b*d)``
        only undercuts the plain ``d - 1`` once the table is tall relative to the word. Fe2S2's
        224-row, 15-bit angle table sits well under that line.
        """
        assert _select_swap_dirty_ns().ComputeOptimalDirtySwapBits(num_data, num_bits, 4096) == 0

    @pytest.mark.parametrize(("num_data", "num_bits"), [(864, 10), (2048, 8)])
    def test_tall_tables_borrow_and_come_out_ahead(self, num_data, num_bits):
        """Where the crossover is cleared the chosen width must actually beat the plain lookup."""
        select_swap = _select_swap_dirty_ns()
        width = select_swap.ComputeOptimalDirtySwapBits(num_data, num_bits, 4096)

        assert width > 0
        assert select_swap.DirtyQROAMCost(width, num_data, num_bits) < num_data - 1

    def test_a_tight_dirty_budget_forces_the_plain_lookup(self):
        """Borrowing is only legal for qubits that exist; a short budget must fall back, not overdraw."""
        select_swap = _select_swap_dirty_ns()
        unconstrained = select_swap.ComputeOptimalDirtySwapBits(864, 10, 4096)
        assert unconstrained > 0
        assert select_swap.DirtyQROAMBorrowedQubits(unconstrained, 10) > 10

        assert select_swap.ComputeOptimalDirtySwapBits(864, 10, 10) == 0

    def test_the_chosen_width_fits_the_budget_it_was_given(self):
        """Every budget must yield a width that borrows within it, not merely the loose ones.

        Sweeping from zero upward catches a rule that clamps only at one end, which would
        overdraw on the tight budgets while still passing a single generous-budget check.
        """
        select_swap = _select_swap_dirty_ns()
        for available in (0, 10, 40, 80, 160, 640):
            width = select_swap.ComputeOptimalDirtySwapBits(864, 10, available)
            assert select_swap.DirtyQROAMBorrowedQubits(width, 10) <= available or width == 0


class TestCleanSelectSwapForwardCostModel:
    """The clean network is chosen for a load that is erased by measurement, not by its adjoint."""

    def test_width_zero_costs_the_plain_lookup(self):
        """Zero swap bits means no network at all, so the baseline is the plain unary iteration."""
        cost = _select_swap_ns().SelectSwapForwardCost
        for num_data in (8, 15, 224, 864):
            assert cost(0, num_data, 10) == num_data - 1

    def test_the_forward_cost_is_cheaper_than_the_compute_uncompute_model(self):
        """Pricing only the forward pass is the whole reason this model exists.

        ``SelectSwapCost1D`` includes the swap network's own uncompute, which is right when the
        adjoint erases the load. A streamed rotation batch is erased by a shared measurement
        instead, so charging for that uncompute would pick a width tuned to a cost we never pay.
        """
        select_swap = _select_swap_ns()
        for width in (1, 2, 3):
            assert select_swap.SelectSwapForwardCost(width, 224, 15) < select_swap.SelectSwapCost1D(width, 224, 15)

    @pytest.mark.parametrize(("num_data", "num_bits"), [(224, 15), (64, 10), (864, 10), (32, 4)])
    def test_the_chosen_width_beats_the_plain_lookup(self, num_data, num_bits):
        """A clean network has no borrowing threshold: allocated scratch always buys Toffolis."""
        select_swap = _select_swap_ns()
        width = select_swap.ComputeOptimalSwapBits(num_data, num_bits)

        assert width > 0
        assert select_swap.SelectSwapForwardCost(width, num_data, num_bits) < num_data - 1

    @pytest.mark.parametrize(("num_data", "num_bits"), [(224, 15), (64, 10), (864, 10)])
    def test_the_chosen_width_is_the_optimum_over_every_width(self, num_data, num_bits):
        """The scan must be a true argmin, not merely an improvement over the baseline."""
        select_swap = _select_swap_ns()
        address_bits = math.ceil(math.log2(num_data))
        chosen = select_swap.ComputeOptimalSwapBits(num_data, num_bits)
        best = min(select_swap.SelectSwapForwardCost(k, num_data, num_bits) for k in range(address_bits + 1))

        assert select_swap.SelectSwapForwardCost(chosen, num_data, num_bits) == best

    def test_a_wide_word_against_a_short_table_declines_the_network(self):
        """Scratch costs ``numBits * (2^k - 1)``, so a wide word can make every width a loss."""
        select_swap = _select_swap_ns()

        assert select_swap.ComputeOptimalSwapBits(4, 64) == 0

    def test_the_scratch_cost_is_reported_for_the_width_that_was_chosen(self):
        """The Toffoli saving is only half the trade; callers need the width it is bought with."""
        select_swap = _select_swap_ns()
        width = select_swap.ComputeOptimalSwapBits(224, 15)

        assert select_swap.SelectSwapScratchQubits(0, 15) == 0
        assert select_swap.SelectSwapScratchQubits(width, 15) == 15 * (2**width - 1)


class TestSelectSwapAliasedMatchesPlainSelect:
    """A loader sharing ``Select``'s measurement-based erasure must share its address routing.

    ``SelectSwap`` zero-fills the addresses past the end of the table, which is correct when its
    own adjoint erases the load. ``Select`` instead aliases them onto real rows, and the phase
    fixup that erases a streamed rotation batch is written against that aliasing. A zero-filled
    forward load would therefore be phased against a word it never wrote.

    Checked by value rather than by phase: ``Select`` erases a ragged table by measurement, so
    the phase-oracle harness used elsewhere in this file disagrees with ``Select`` even when
    ``Select`` is compared against itself. The forward load is what the streamed rotation path
    takes from ``Select``, and the forward load is what these compare.
    """

    @pytest.mark.parametrize("num_swap_bits", [-1, 0, 1, 2])
    @pytest.mark.parametrize("data", [_DATA_1D, _DATA_1D_RAGGED], ids=["power_of_two", "ragged"])
    def test_aliased_load_matches_plain_select(self, num_swap_bits, data):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelectSwapAliasedMatchesSelect1D(
            data, num_swap_bits
        )
