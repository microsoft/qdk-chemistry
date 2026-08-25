"""Tests for the Q# SELECT-SWAP data-loading network."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest
import qdk

from qdk_chemistry.utils.qsharp import create_qsharp_context


@pytest.fixture
def qdk_ctx() -> qdk.Context:
    """Fresh Q# context for a single test."""
    return create_qsharp_context()


#: Eight three-bit words, so the address register is three qubits and ``num_swap_bits``
#: can range over the whole select/swap split.
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

_DATA_2D = [
    [[True, False, True], [False, True, True], [True, True, False], [False, False, True]],
    [[False, True, False], [True, True, True], [False, False, False], [True, False, False]],
]


class TestSelectSwapLoadsCorrectValues:
    """The loaded word is the addressed word, for every select/swap split."""

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2, 3])
    def test_1d_loads_every_address(self, qdk_ctx, num_swap_bits):
        assert qdk_ctx.code.QDKChemistry.Utils.SelectSwap.TestSelectSwap1DCorrectness(_DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [0, 1, 2])
    def test_2d_loads_every_address(self, qdk_ctx, num_swap_bits):
        assert qdk_ctx.code.QDKChemistry.Utils.SelectSwap.TestSelect2DLoadCorrectness(_DATA_2D, num_swap_bits)


class TestSelectSwapPreservesAddressPhases:
    """The swap path must agree with the plain-select path as a *phase* oracle."""

    @pytest.mark.parametrize("num_swap_bits", [1, 2, 3])
    def test_1d_swap_path_matches_plain_select(self, qdk_ctx, num_swap_bits):
        assert qdk_ctx.code.QDKChemistry.Utils.SelectSwap.TestSelectSwap1DPhaseAgreement(_DATA_1D, num_swap_bits)

    @pytest.mark.parametrize("num_swap_bits", [1, 2])
    def test_2d_swap_path_matches_plain_select(self, qdk_ctx, num_swap_bits):
        assert qdk_ctx.code.QDKChemistry.Utils.SelectSwap.TestSelect2DLoadPhaseAgreement(_DATA_2D, num_swap_bits)
