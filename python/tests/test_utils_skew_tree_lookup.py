"""Tests for the clean-ancilla skew-tree QROM."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.utils.qsharp import get_qsharp_context

_DATA = [
    [True, False, True],
    [False, True, True],
    [True, True, False],
    [False, False, True],
    [True, False, False],
    [True, True, True],
    [False, True, False],
    [False, False, False],
]


@pytest.mark.parametrize("data", [[[True]], [[True], [False]], _DATA])
def test_skew_tree_lookup_loads_every_address(data: list[list[bool]]):
    assert get_qsharp_context().code.QDKChemistry.Utils.SkewTreeLookup.TestSkewTreeLookupCorrectness(data)


def test_skew_tree_lookup_matches_select_as_a_phase_oracle():
    operation = get_qsharp_context().code.QDKChemistry.Utils.SkewTreeLookup.TestSkewTreeLookupPhaseAgreement
    assert all(operation(_DATA) for _ in range(12))


def test_ccz_correction_order_is_reverse_dfs():
    function = get_qsharp_context().code.QDKChemistry.Utils.SkewTreeLookup.CczCorrectionOrder

    assert function(4) == [12, 14, 15, 13, 10, 11, 9, 6, 7, 5, 3]


def test_classical_ccz_absorption_handles_every_three_bit_outcome_pattern():
    function = get_qsharp_context().code.QDKChemistry.Utils.SkewTreeLookup.TestClassicalCczAbsorption
    correction_nodes = [6, 7, 5, 3]

    for packed_outcomes in range(1 << (2 * len(correction_nodes))):
        first_outcomes = [False] * len(_DATA)
        second_outcomes = [False] * len(_DATA)
        for offset, node in enumerate(correction_nodes):
            first_outcomes[node] = bool(packed_outcomes & (1 << (2 * offset)))
            second_outcomes[node] = bool(packed_outcomes & (1 << (2 * offset + 1)))

        for basis_entry in range(len(_DATA)):
            basis_data = [[entry == basis_entry] for entry in range(len(_DATA))]
            assert function(basis_data, first_outcomes, second_outcomes), (
                f"failed table basis {basis_entry}, outcome pattern {packed_outcomes:#x}"
            )


def test_skew_tree_architecture_model_matches_theorem_and_paper_layout():
    namespace = get_qsharp_context().code.QDKChemistry.Utils.SkewTreeLookup

    assert namespace.SkewTreeCczCount(8) == 4
    assert namespace.SkewTreeCleanAncillaCount(8) == 1
    assert namespace.CczCorrectionWaits(8, False, 20, 10) == 4
    assert namespace.CczCorrectionWaits(8, True, 20, 10) == 0
    assert namespace.ClassicalAbsorptionBitXorBound(8, 3) == 45
    assert namespace.PriorArtFastLookupComputePatches() / namespace.FastSkewTreeComputePatches() == 9


@pytest.mark.parametrize("num_address_qubits", [0, 1, 2, 3, 4])
def test_skew_tree_lookup_logical_resources(num_address_qubits: int):
    num_entries = 1 << num_address_qubits
    # This table has an all-true Mobius transform, so every possible skew-tree
    # data load is present and the count reaches the construction's worst case.
    data = [[entry == 0] for entry in range(num_entries)]
    context = get_qsharp_context()
    operation = context.code.QDKChemistry.Utils.SkewTreeLookup.SkewTreeLookupResourceEstimate

    counts = context.logical_counts(operation, data)

    assert counts["cczCount"] == num_entries - num_address_qubits - 1
    assert counts["numQubits"] == num_address_qubits + 1 + max(0, num_address_qubits - 2)
