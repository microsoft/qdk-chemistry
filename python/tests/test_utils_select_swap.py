"""Tests for the Q# SELECT-SWAP data-loading network."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.utils.qsharp import get_qsharp_context

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


def _assert_phase_agreement(operation_name, data, num_swap_bits):
    """Fail if any trial reports disagreement; a single passing trial proves nothing."""
    operation = getattr(get_qsharp_context().code.QDKChemistry.Utils.SelectSwap, operation_name)
    failures = sum(1 for _ in range(_PHASE_TRIALS) if not operation(data, num_swap_bits))
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

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_loads_every_address(self, num_swap_bits):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelect2DLoadCorrectness(
            _DATA_2D, num_swap_bits
        )

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_loads_every_address_when_outer_length_is_not_a_power_of_two(self, num_swap_bits):
        assert get_qsharp_context().code.QDKChemistry.Utils.SelectSwap.TestSelect2DLoadCorrectness(
            _DATA_2D_RAGGED_OUTER, num_swap_bits
        )


class TestSelectSwapPreservesAddressPhases:
    """The swap path must agree with the plain-select path as a *phase* oracle.

    A lookup that loads the right bits but routes unused addresses differently from the
    plain-select path, or that uncomputes incorrectly, still passes every value test above:
    the damage lands on the phase of the address register, where only these tests see it.
    """

    @pytest.mark.parametrize("num_swap_bits", [1, 2, 3])
    def test_1d_swap_path_matches_plain_select(self, num_swap_bits):
        _assert_phase_agreement("TestSelectSwap1DPhaseAgreement", _DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_1d_swap_path_matches_plain_select_when_length_is_not_a_power_of_two(self, num_swap_bits):
        _assert_phase_agreement("TestSelectSwap1DPhaseAgreement", _DATA_1D_RAGGED, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_2d_swap_path_matches_plain_select(self, num_swap_bits):
        _assert_phase_agreement("TestSelect2DLoadPhaseAgreement", _DATA_2D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_2d_swap_path_matches_plain_select_when_outer_length_is_not_a_power_of_two(self, num_swap_bits):
        _assert_phase_agreement("TestSelect2DLoadPhaseAgreement", _DATA_2D_RAGGED_OUTER, num_swap_bits)
