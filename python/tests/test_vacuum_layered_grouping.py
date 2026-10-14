"""Vacuum-compatible grouping retains usable structure through circuit construction."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from itertools import combinations

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.term_grouper.vacuum_annihilating import _anticommuting_masks, _index_word
from qdk_chemistry.data import FlatPartition, LatticeGraph, LayeredPartition, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.sparse_pauli_decomposition import _sparse_terms
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian
from qdk_chemistry.utils.pauli_commutation import do_pauli_maps_commute
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .helpers import (
    apply_operation,
    assert_states_match_up_to_global_phase,
    controlled_product_formula_state,
    create_nontrivial_test_hamiltonian,
    random_state,
)


def _assert_grouping(operator: QubitOperator) -> None:
    """Check term coverage, disjoint layers, commuting matrices and the group's vacuum action."""
    partition = operator.term_partition
    assert isinstance(partition, LayeredPartition)
    assert sorted(partition.all_indices()) == list(range(len(operator.coefficients)))
    words = [dict(word) for word, _ in _sparse_terms(operator.get_container())]
    for layers in partition.groups:
        group = [index for layer in layers for index in layer]
        for layer in layers:
            occupied: set[int] = set()
            for index in layer:
                assert occupied.isdisjoint(words[index])
                occupied.update(words[index])
        matrices = [
            QubitOperator.from_sparse_terms(operator.num_qubits, [words[i]], np.ones(1)).to_matrix() for i in group
        ]
        for first, second in combinations(matrices, 2):
            np.testing.assert_allclose(first @ second, second @ first, atol=1e-14)
        action = sum(
            (operator.coefficients[i] * matrix[:, 0] for i, matrix in zip(group, matrices, strict=True)),
            np.zeros(2**operator.num_qubits, dtype=complex),
        )
        np.testing.assert_allclose(action[1:], 0.0, atol=1e-9)


def test_indexed_commutation_matches_pairwise_parity_across_chunks() -> None:
    """Odd overlaps remain per-term conflicts across chunk boundaries and huge sparse qubit indices."""
    rng = np.random.default_rng(771)
    qubits = (0, 1, 2, 2**32 - 1)
    words = [
        {q: axis for q, axis in zip(qubits, rng.choice(list("IXYZ"), size=4), strict=True) if axis != "I"}
        for _ in range(520)
    ]
    by_qubit: dict[int, dict[str, dict[int, int]]] = {}
    for index, word in enumerate(words):
        masks = _anticommuting_masks(word, by_qubit)
        actual = {
            chunk * 256 + bit for chunk, mask in masks.items() for bit in range(mask.bit_length()) if mask >> bit & 1
        }
        expected = {i for i in range(index) if not do_pauli_maps_commute(word, words[i])}
        assert actual == expected
        _index_word(word, index, by_qubit)


def test_packing_does_not_cancel_conflicts_between_members_of_one_group() -> None:
    """XX and YY both conflict with Z0; projecting to group IDs before parity would hide that."""
    source = QubitOperator(["XX", "YY", "IZ"], np.array([1.0, 1.0, 0.5]))
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.num_groups == 2
    _assert_grouping(output)


def test_packing_resolves_reordered_term_ordinals_across_chunks() -> None:
    """Conflicts in different chunks must resolve to the original term owners after producer reordering."""
    last = 2**32 - 1
    words = [{0: "X", last: "X"}, {0: "Y", last: "Y"}, *({q: "Z"} for q in (*range(259), last))]
    diagonal = tuple(range(2, len(words)))
    source = QubitOperator.from_sparse_terms(
        2**32,
        words,
        np.ones(len(words)),
        term_partition=FlatPartition(strategy="reordered", groups=((*diagonal, 1, 0),)),
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.groups == ((diagonal,), ((1,), (0,)))


@pytest.mark.parametrize(
    "words",
    [
        [{0: "Z", i: "Z"} for i in range(1, 33)],
        [{i: "Z"} for i in range(32)],
        [{0: "Z"}, {1: "Z"}, {0: "Z", 1: "Z"}, {}, {2: "Z"}, {1: "Z", 2: "Z"}, {0: "Z", 2: "Z"}, {2: "Z"}],
    ],
    ids=["star", "disjoint", "interleaved-and-identity"],
)
def test_derived_layers_keep_the_lowest_available_assignment(words: list[dict[int, str]]) -> None:
    """Sparse layer cursors must preserve the original first-fit schedule, including cursor gaps."""
    expected: list[list[int]] = []
    occupied: list[set[int]] = []
    for index, word in enumerate(words):
        layer = next((i for i, support in enumerate(occupied) if support.isdisjoint(word)), len(occupied))
        if layer == len(occupied):
            expected.append([])
            occupied.append(set())
        expected[layer].append(index)
        occupied[layer].update(word)
    source = QubitOperator.from_sparse_terms(2**32, words, np.ones(len(words)))
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.groups == (tuple(tuple(layer) for layer in expected),)


@pytest.mark.parametrize("sparse", [False, True])
def test_loaded_flat_partition_regroups_then_composes(sparse: bool) -> None:
    """A loaded flat partition can be regrouped for addition to layered output without changing serialization."""
    flat_partition = FlatPartition(strategy="vacuum_annihilating", groups=((0, 1),))
    source = QubitOperator(["XX", "YY"], np.ones(2), term_partition=flat_partition)
    if sparse:
        source = QubitOperator.from_sparse_terms(
            2, [{0: "X", 1: "X"}, {0: "Y", 1: "Y"}], np.ones(2), term_partition=flat_partition
        )
    loaded = QubitOperator.from_json(source.to_json())
    before = loaded.to_json()
    regrouped = create("term_grouper", "vacuum_annihilating").run(loaded)
    fresh = create("term_grouper", "vacuum_annihilating").run(QubitOperator(["XX", "YY"], np.ones(2)))
    combined = regrouped + fresh

    assert isinstance(loaded.term_partition, FlatPartition)
    assert isinstance(regrouped.term_partition, LayeredPartition)
    assert type(regrouped.get_container()) is type(loaded.get_container())
    assert loaded.to_json() == before
    assert regrouped.equiv(loaded)
    assert combined.equiv(2 * source)
    assert sorted(combined.term_partition.all_indices()) == [0, 1, 2, 3]


@pytest.mark.parametrize("sparse", [False, True])
def test_preserves_compatible_groups_layers_and_strategy(sparse: bool) -> None:
    """XX and YY cancel across layers; diagonal groups and their order remain untouched."""
    partition = LayeredPartition(strategy="producer", groups=(((4, 5),), ((1, 0), (3, 2))))
    words = [{0: "X", 1: "X"}, {2: "X", 3: "X"}, {0: "Y", 1: "Y"}, {2: "Y", 3: "Y"}, {0: "Z"}, {2: "Z"}]
    source = QubitOperator.from_sparse_terms(
        4, words, np.array([0.5, 0.5, 0.5, 0.5, 0.2, -0.2]), term_partition=partition
    )
    if not sparse:
        source = QubitOperator(list(source.pauli_strings), source.coefficients, term_partition=partition)
    before = source.to_json()
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output is not source
    assert output.term_partition == partition
    assert source.to_json() == before
    assert output.to_json() == before
    assert type(output.get_container()) is type(source.get_container())
    _assert_grouping(output)
    assert create("term_grouper", "vacuum_annihilating").run(output).term_partition == partition


def test_repairs_family_groups_but_keeps_an_independent_valid_group() -> None:
    """Do not merge whole XX/YY families or discard an unrelated certified group."""
    partition = LayeredPartition(strategy="families", groups=(((4,),), ((0,), (1,)), ((2,), (3,))))
    source = QubitOperator.from_sparse_terms(
        4,
        [{0: "X", 1: "X"}, {1: "X", 2: "X"}, {0: "Y", 1: "Y"}, {1: "Y", 2: "Y"}, {3: "Z"}],
        np.array([1.0, 1.0, 1.0, 1.0, 0.3]),
        term_partition=partition,
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.groups[0] == partition.groups[0]
    owners = {
        i: group_index
        for group_index, layers in enumerate(output.term_partition.groups)
        for layer in layers
        for i in layer
    }
    assert owners[0] == owners[2]
    assert owners[1] == owners[3]
    assert owners[0] != owners[1]
    _assert_grouping(output)


def test_layer_fragments_respect_full_support_including_parity_strings() -> None:
    """A valid old layer may need splitting, and Z tails count as occupied qubits."""
    partition = LayeredPartition(strategy="input", groups=(((0, 1),), ((2,), (3,))))
    source = QubitOperator.from_sparse_terms(
        4,
        [{0: "X", 1: "X"}, {2: "X", 3: "X"}, {0: "Y", 1: "Y", 2: "Z"}, {0: "Z", 2: "Y", 3: "Y"}],
        np.ones(4),
        term_partition=partition,
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.num_groups == 2
    assert all(len(layer) == 1 for group in output.term_partition.groups for layer in group)
    _assert_grouping(output)


@pytest.mark.parametrize("partition", [None, FlatPartition(strategy="split", groups=((0, 1), (2, 3)))])
def test_derives_wide_layers_when_no_layers_were_supplied(partition) -> None:
    """Cancellation blocks share an outer group but their overlapping XX/YY words do not share a layer."""
    source = QubitOperator.from_sparse_terms(
        4,
        [{0: "X", 1: "X"}, {2: "X", 3: "X"}, {0: "Y", 1: "Y"}, {2: "Y", 3: "Y"}],
        np.ones(4),
        term_partition=partition,
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.num_groups == 1
    assert sorted(len(layer) for layer in output.term_partition.groups[0]) == [2, 2]
    _assert_grouping(output)


def test_preserves_flat_group_membership_while_adding_layers() -> None:
    """A compatible producer's outer groups should not be merged merely because they commute."""
    partition = FlatPartition(strategy="producer", groups=((2, 3), (0, 1)))
    source = QubitOperator.from_sparse_terms(
        4,
        [{0: "X", 1: "X"}, {0: "Y", 1: "Y"}, {2: "X", 3: "X"}, {2: "Y", 3: "Y"}],
        np.ones(4),
        term_partition=partition,
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert (
        tuple(tuple(i for layer in group for i in layer) for group in output.term_partition.groups) == partition.groups
    )
    _assert_grouping(output)


def test_approximate_groups_do_not_spend_the_same_tolerance_repeatedly() -> None:
    """Two tolerated prefixes must rejoin their cancelling remainder instead of stranding it."""
    tolerance = 2.0**-30
    partition = LayeredPartition(strategy="prefixes", groups=(((0,),), ((1,),), ((2,),)))
    source = QubitOperator(["X", "X", "X"], np.array([0.75, 0.75, -1.5]) * tolerance, term_partition=partition)
    output = create("term_grouper", "vacuum_annihilating", tolerance=tolerance).run(source)

    assert output.term_partition.num_groups == 1
    _assert_grouping(output)


def test_noncommuting_input_group_is_repaired_not_mistaken_for_a_certificate() -> None:
    """Vanishing vacuum amplitudes alone do not certify a commuting product."""
    source = QubitOperator(
        ["X", "Y", "X", "Y"],
        np.array([1.0, 1.0, -1.0, -1.0]),
        term_partition=FlatPartition(strategy="claimed_commuting", groups=((0, 1, 2, 3),)),
    )
    output = create("term_grouper", "vacuum_annihilating").run(source)

    assert output.term_partition.num_groups == 2
    _assert_grouping(output)


def test_invalid_layer_is_rejected() -> None:
    """The producer may not claim overlapping Pauli terms form one disjoint layer."""
    source = QubitOperator(
        ["XX", "YY"],
        np.ones(2),
        term_partition=LayeredPartition(strategy="invalid", groups=(((0, 1),),)),
    )
    with pytest.raises(ValueError, match="disjoint"):
        create("term_grouper", "vacuum_annihilating").run(source)


@pytest.mark.parametrize("format_name", ["json", "hdf5"])
def test_layered_result_round_trips(tmp_path, format_name: str) -> None:
    source = QubitOperator(["XXII", "YYII", "IIXX", "IIYY"], np.ones(4))
    output = create("term_grouper", "vacuum_annihilating").run(source)
    filename = tmp_path / f"grouped.qubit_hamiltonian.{format_name}"
    output.to_file(str(filename), format_name)
    restored = QubitOperator.from_file(str(filename), format_name)

    assert restored.term_partition == output.term_partition
    np.testing.assert_array_equal(restored.coefficients, output.coefficients)
    _assert_grouping(restored)


@pytest.mark.parametrize("mapping_name", ["jordan_wigner", "bravyi_kitaev", "bravyi_kitaev_tree", "parity"])
@pytest.mark.parametrize("pre_group", [False, True])
def test_fermionic_pipeline_accepts_layered_output(mapping_name: str, pre_group: bool) -> None:
    """Real mapped Hamiltonians retain vacuum cancellation through fused fourth-order HWP evolution."""
    source = create("qubit_mapper", "qdk").run(
        create_nontrivial_test_hamiltonian(),
        getattr(MajoranaMapping, mapping_name)(4),
    )
    if pre_group:
        source = create("term_grouper", "commuting").run(source)
    grouped = create("term_grouper", "vacuum_annihilating").run(source)
    _assert_grouping(grouped)
    unitary = create(
        "hamiltonian_unitary_builder", "trotter", order=4, time=0.23, num_divisions=2, fuse_group_boundaries=True
    ).run(grouped)
    circuit = create(
        "controlled_circuit_mapper",
        "cswap_pauli_sequence",
        use_hamming_weight_phasing=True,
        max_hamming_weight_phasing_batch_size=3,
    ).run(unitary)
    system = random_state(unitary.get_num_qubits(), seed=769)
    state = np.kron(np.ones(2) / np.sqrt(2), system)
    actual = apply_operation(circuit._qsharp_op, state, split_control=True)
    expected = controlled_product_formula_state(unitary.get_container(), state)
    assert_states_match_up_to_global_phase(actual, expected, 1e-10)


@pytest.mark.parametrize("variant", ["pauli_sequence", "cswap_pauli_sequence"])
def test_repaired_model_partition_keeps_hwp_opportunities(variant: str) -> None:
    source = create_heisenberg_hamiltonian(LatticeGraph.chain(16, periodic=True), jx=1.0, jy=1.0, jz=0.0)
    grouped = create("term_grouper", "vacuum_annihilating").run(source)
    assert isinstance(grouped.term_partition, LayeredPartition)
    old_layers = [layer for group in source.term_partition.groups for layer in group]
    assert {layer for group in grouped.term_partition.groups for layer in group} == set(old_layers)
    unitary = create("hamiltonian_unitary_builder", "trotter", order=1, time=0.23).run(grouped)
    counts = []
    for enabled in (False, True):
        circuit = create("controlled_circuit_mapper", variant, use_hamming_weight_phasing=enabled).run(unitary)
        application = circuit.get_qre_application()
        counts.append(get_qsharp_context().logical_counts(application.entry_expr, *application.args))
    assert counts[1]["rotationCount"] < counts[0]["rotationCount"]


def test_final_mapper_still_detects_cancellation_lost_by_filtering() -> None:
    source = QubitOperator(["XX", "YY", "YY"], np.array([1.0, 1.0 - 2e-8, 2e-8]))
    grouped = create("term_grouper", "vacuum_annihilating").run(source)
    unitary = create("hamiltonian_unitary_builder", "trotter", order=1, time=1.0, weight_threshold=1e-7).run(grouped)

    with pytest.raises(ValueError, match="vacuum-preserving product formula"):
        create("controlled_circuit_mapper", "cswap_pauli_sequence", use_hamming_weight_phasing=True).run(unitary)


@pytest.mark.parametrize("variant", ["pauli_sequence", "cswap_pauli_sequence"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("with_layers", [False, True])
def test_explicit_hwp_requires_declared_layers(variant: str, enabled: bool, with_layers: bool) -> None:
    """Missing metadata is an error only when phasing was explicitly requested."""
    unitary = UnitaryRepresentation(
        PauliProductFormulaContainer(
            [ExponentiatedPauliTerm({0: "Z"}, 0.2), ExponentiatedPauliTerm({1: "Z"}, 0.2)],
            1,
            2,
            layer_offsets=(0, 2) if with_layers else None,
        )
    )
    mapper = create("controlled_circuit_mapper", variant, use_hamming_weight_phasing=enabled)
    if enabled and not with_layers:
        with pytest.raises(ValueError, match="layer_offsets"):
            mapper.run(unitary)
    else:
        mapper.run(unitary)
