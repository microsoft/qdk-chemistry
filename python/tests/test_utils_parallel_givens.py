"""Tests for tree-structured parallel Givens rotations."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest

from qdk_chemistry.utils.qsharp import create_qsharp_context, get_qsharp_context


def _tree_edges(num_orbitals: int) -> list[tuple[int, int]]:
    parents = [0]
    edges: list[tuple[int, int]] = []
    for depth in range(math.ceil(math.log2(num_orbitals)) if num_orbitals > 1 else 0):
        stride = 1 << depth
        next_parents: list[int] = []
        for parent in parents:
            next_parents.append(parent)
            child = parent + stride
            if child < num_orbitals:
                edges.append((parent, child))
                next_parents.append(child)
        parents = next_parents
    return edges


def _tree_angles(vector: np.ndarray) -> list[float]:
    work = vector.astype(float).copy()
    edges = _tree_edges(len(work))
    angles = [0.0] * len(edges)
    for index in range(len(edges) - 1, -1, -1):
        parent, child = edges[index]
        angles[index] = -math.atan2(work[child], work[parent])
        work[parent] = math.hypot(work[parent], work[child])
        work[child] = 0.0
    return angles


def test_givens_tree_uses_the_paper_edge_order():
    function = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens.GivensTreeEdges

    assert function(8) == [(0, 1), (0, 2), (1, 3), (0, 4), (2, 6), (1, 5), (3, 7)]


@pytest.mark.parametrize("num_orbitals", range(1, 17))
def test_givens_tree_has_logarithmic_disjoint_layers(num_orbitals: int):
    namespace = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens

    assert namespace.GivensTreeDepth(num_orbitals) == math.ceil(math.log2(num_orbitals))
    assert namespace.GivensTreeLayersAreDisjoint(num_orbitals)
    assert len(namespace.GivensTreeEdges(num_orbitals)) == num_orbitals - 1


def test_controlled_adder_batch_model_matches_paper_examples():
    namespace = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens
    distance = 22
    factory_cycles = 117

    assert namespace.ControlledAdderBatchCycles(5, factory_cycles, distance, 10) == 234
    assert namespace.ControlledAdderCczPerDistance(5, factory_cycles, distance, 10) == pytest.approx(0.94, abs=0.01)
    assert namespace.ControlledAdderCczPerDistance(10, factory_cycles, distance, 10) == pytest.approx(1.56, abs=0.01)


@pytest.mark.parametrize("angle", [-1.1, -0.25, 0.0, 0.4, 2.2])
def test_parallel_half_rotations_equal_controlled_rotation(angle: float):
    operation = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens.TestParallelGivensDecomposition

    assert all(operation(angle, input_kind) for input_kind in range(6))


@pytest.mark.parametrize(("angle_value", "num_bits"), [(1, 3), (3, 4), (13, 5)])
def test_phase_gradient_givens_matches_direct_decomposition(angle_value: int, num_bits: int):
    operation = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens.TestPhaseGradientGivensDecomposition

    assert all(operation(angle_value, num_bits, input_kind) for input_kind in range(6))


def test_full_phase_gradient_givens_tree_matches_direct_tree():
    operation = get_qsharp_context().code.QDKChemistry.Utils.ParallelGivens.TestPhaseGradientGivensTree

    assert operation([1, 3], 2)


@pytest.mark.parametrize("num_orbitals", [2, 3, 5, 8])
def test_givens_tree_prepares_requested_real_vector(num_orbitals: int):
    rng = np.random.default_rng(1701 + num_orbitals)
    expected_modes = rng.normal(size=num_orbitals)
    expected_modes /= np.linalg.norm(expected_modes)
    angles = _tree_angles(expected_modes)

    context = create_qsharp_context()
    context.eval(f"use target = Qubit[{num_orbitals}];")
    context.eval("X(target[0]);")
    context.eval(f"QDKChemistry.Utils.ParallelGivens.ApplyGivensTree({angles}, target);")
    actual = np.asarray(context.dump_machine().as_dense_state())

    expected = np.zeros(1 << num_orbitals, dtype=complex)
    for orbital, amplitude in enumerate(expected_modes):
        expected[1 << (num_orbitals - orbital - 1)] = amplitude

    assert np.allclose(actual, expected, atol=1e-12)


def test_parallel_givens_resource_counts_trade_depth_for_rotations():
    context = get_qsharp_context()
    operation = context.code.QDKChemistry.Utils.ParallelGivens.GivensTreeResourceEstimate
    angles = [0.125] * 7

    native = context.logical_counts(operation, angles, False)
    parallel = context.logical_counts(operation, angles, True)

    assert native["numQubits"] == parallel["numQubits"] == 8
    assert native["rotationCount"] == parallel["rotationCount"] == 14
    assert native["rotationDepth"] == 6
    assert parallel["rotationDepth"] == 3 == math.ceil(math.log2(8))
