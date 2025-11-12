"""Test for sparse isometry state preparation algorithms in QDK/Chemistry.

This module provides comprehensive tests for sparse isometry methods for
preparing quantum states from electronic structure wavefunctions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import builtins
import json
import re

import numpy as np
import pytest

from qdk.chemistry.data import CasWavefunctionContainer, Configuration, Wavefunction
from qdk.chemistry.state_preparation.sparse_isometry import (
    GF2XEliminationResult,
    SparseIsometryGF2XStatePrep,
    _eliminate_column,
    _find_pivot_row,
    _is_diagonal_matrix,
    _perform_gaussian_elimination,
    _reduce_diagonal_matrix,
    _remove_all_ones_rows_with_x,
    _remove_duplicate_rows_with_cnot,
    _remove_zero_rows,
    gf2x_with_tracking,
)

from .test_helpers import create_test_orbitals


def test_sparse_isometry_gf2x_basic(wavefunction_4e4o):
    """Test the SparseIsometryGF2XStatePrep class basic functionality."""
    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=False,
    )
    # Test circuit creation
    circuit_qasm = prep.create_circuit_qasm()
    assert isinstance(circuit_qasm, str)
    qubit_pattern = re.search(r"qubit\[(\d+)\] q;", circuit_qasm)
    assert qubit_pattern is not None
    assert int(qubit_pattern.group(1)) == 2 * 4
    # Test composite StatePreparation gate has been correctly decomposed in the qasm str
    assert "U(" in circuit_qasm  # decomposed into U gate
    expected_theta = 2 * np.arctan(9.8379475848252518e-01 / 1.7929827992011016e-01)
    assert f"{expected_theta:.6f}" in circuit_qasm  # expected angle


def test_sparse_isometry_gf2x_circuit_log_and_structure(wavefunction_4e4o, wavefunction_10e6o, tmp_path):
    """Test SparseIsometryGF2XStatePrep output logging and circuit structure from the logging."""
    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=True,
        output_dir=tmp_path,
    )
    # Test loading and validating the saved summary JSON file
    prep.create_circuit_qasm()
    with open(tmp_path / "sparse_isometry_gf2x_summary.state_preparation.json") as f:
        gf2x_summary = json.load(f)

    # Verify the expected structure of the summary file
    assert gf2x_summary.keys() == {"metadata", "matrix_operations", "untranspiled_circuit_summary"}
    assert gf2x_summary["metadata"].keys() == {"num_orbitals", "num_determinants", "amplitude_threshold", "max_dets"}
    assert gf2x_summary["matrix_operations"].keys() == {
        "original_shape",
        "original_rank",
        "reduced_shape",
        "reduced_rank",
        "row_map",
        "operation_order",
        "total_operations",
        "cnot_operations",
        "x_operations",
    }
    assert gf2x_summary["untranspiled_circuit_summary"].keys() == {
        "num_qubits",
        "depth",
        "total_gates",
        "single_qubit_clifford",
        "two_qubit_clifford",
        "non_clifford",
    }

    # Verify circuit structure from the logged operations
    # Dense prep on 1 qubit and qubit index is 1
    assert gf2x_summary["matrix_operations"]["reduced_rank"] == 1
    assert gf2x_summary["matrix_operations"]["row_map"][0] == 1

    # Check X and CNOT operations on target qubits from matrix mapping
    # Extract X and CNOT operations from the matrix operations
    x_ops = [op[1] for op in gf2x_summary["matrix_operations"]["operation_order"] if op[0] == "x"]
    cnot_ops = [op[1] for op in gf2x_summary["matrix_operations"]["operation_order"] if op[0] == "cnot"]
    assert len(x_ops) == 2  # Two X operations expected
    assert set(x_ops) == {0, 2}  # X on qubits 0 and 2
    assert len(cnot_ops) == 4  # Four CNOT operations expected
    assert cnot_ops == [[4, 0], [5, 1], [6, 2], [2, 1]]  # CNOTs from target, control pairs

    # Test with 10e6o wavefunction
    prep_10e6o = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_10e6o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=True,
        output_dir=tmp_path / "10e6o_test_dir",
    )
    # Test circuit creation
    circuit_10e6o = prep_10e6o.create_circuit_qasm()
    assert isinstance(circuit_10e6o, str)
    qubit_pattern_10e6o = re.search(r"qubit\[(\d+)\] q;", circuit_10e6o)
    assert qubit_pattern_10e6o is not None
    assert int(qubit_pattern_10e6o.group(1)) == 2 * 6  # 6 orbitals

    with open(tmp_path / "10e6o_test_dir/sparse_isometry_gf2x_summary.state_preparation.json") as f:
        gf2x_summary_10e6o = json.load(f)

    # Dense prep on 2 qubits and qubit indices are qubit 2 and qubit 4
    assert gf2x_summary_10e6o["matrix_operations"]["reduced_rank"] == 2
    assert gf2x_summary_10e6o["matrix_operations"]["row_map"] == [2, 4]

    # Check X and CNOT operations on target qubits from matrix mapping
    # Extract X and CNOT operations from the 10e6o test
    x_ops_10e6o = [op[1] for op in gf2x_summary_10e6o["matrix_operations"]["operation_order"] if op[0] == "x"]
    cnot_ops_10e6o = [op[1] for op in gf2x_summary_10e6o["matrix_operations"]["operation_order"] if op[0] == "cnot"]
    assert len(x_ops_10e6o) == 1  # One X operation expected
    assert x_ops_10e6o[0] == 0  # X on qubit 0
    assert len(cnot_ops_10e6o) == 10  # Ten CNOT operations expected
    # Check the ordering and (target, control) pairs of CNOT operations
    assert cnot_ops_10e6o == [[1, 0], [3, 0], [6, 0], [7, 0], [9, 0], [8, 2], [10, 4], [11, 5], [4, 2], [5, 4]]


def test_sparse_isometry_gf2x_parameter_validation(wavefunction_4e4o):
    """Test parameter validation and initialization for SparseIsometryGF2XStatePrep.

    Validates that the class accepts various parameter combinations including
    None max_dets (use all determinants), different amplitude thresholds,
    and successfully creates quantum circuits with valid configurations.
    Tests edge cases and error conditions for robust parameter validation.
    """
    # Test valid parameters with specific max_dets
    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=2,
        save_outputs=False,
    )
    assert prep is not None
    assert prep.max_dets == 2
    assert prep.amplitude_threshold == 0.001
    assert prep.save_outputs is False

    # Test with None max_dets (should use all determinants)
    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=False,
    )
    assert prep is not None
    assert prep.max_dets == 2

    # Test with zero amplitude threshold (should include all determinants)
    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.0,
        max_dets=None,
        save_outputs=False,
    )
    assert prep is not None
    assert prep.amplitude_threshold == 0.0

    # Test that circuits can be created successfully with valid parameters
    circuit = prep.create_circuit_qasm()
    assert isinstance(circuit, str)
    qubit_pattern = re.search(r"qubit\[(\d+)\] q;", circuit)
    assert qubit_pattern is not None
    assert int(qubit_pattern.group(1)) == 2 * 4

    # Test with invalid max_dets (negative)
    prep_invalid = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=-1,
        save_outputs=False,
    )
    with pytest.raises(ValueError, match="max_dets must be a positive integer"):
        prep_invalid.create_circuit_qasm()

    # Test with very high amplitude threshold (should filter out all determinants)
    prep_high_threshold = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=10.0,
        max_dets=None,
        save_outputs=False,
    )
    with pytest.raises(ValueError, match="No determinants remain after filtering"):
        prep_high_threshold.create_circuit_qasm()

    # Test with save_outputs=True
    prep_save = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=True,
    )
    assert prep_save.save_outputs is True
    assert prep_save.output_dir == "sparse_isometry_gf2x_output"  # default value

    # Test with custom output directory
    prep_custom_dir = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        amplitude_threshold=0.001,
        max_dets=None,
        save_outputs=True,
        output_dir="custom_test_dir",
    )
    assert prep_custom_dir.output_dir == "custom_test_dir"


def test_sparse_isometry_gf2x_single_reference_state():
    """Test SparseIsometryGF2XStatePrep with single reference state after filtering."""
    # Create a wavefunction with coefficients that will be filtered out
    test_orbitals = create_test_orbitals(2)

    det1 = Configuration("0200")
    det2 = Configuration("2000")
    det3 = Configuration("du00")
    dets = [det1, det2, det3]
    coeffs = np.array([0.0, 0.6, 0.8])

    container = CasWavefunctionContainer(coeffs, dets, test_orbitals)
    wavefunction = Wavefunction(container)

    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction, amplitude_threshold=0.75
    )  # Filter out one determinant

    single_ref_circuit = prep.create_circuit_qasm()
    assert isinstance(single_ref_circuit, str)
    # Count number of qubits in qasm
    qubit_pattern = re.search(r"qubit\[(\d+)\] q;", single_ref_circuit)
    assert qubit_pattern is not None
    assert int(qubit_pattern.group(1)) == 4
    # Count x operation on qubit "x q[*]"
    assert single_ref_circuit.count("x q[") == 2


def test_sparse_isometry_gf2x_save_outputs_file_error(wavefunction_4e4o, monkeypatch, tmp_path):
    """Test SparseIsometryGF2XStatePrep save_outputs with file write error."""

    def mock_open(*_, **__):
        raise PermissionError("Permission denied")

    # Monkeypatch open to simulate file write error
    monkeypatch.setattr(builtins, "open", mock_open)
    output_dir = tmp_path / "test_dir"

    prep = SparseIsometryGF2XStatePrep(
        wavefunction=wavefunction_4e4o,
        save_outputs=True,
        output_dir=output_dir,
    )

    # Should not raise error but log warning (test passes if no exception)
    circuit = prep.create_circuit_qasm()
    assert circuit is not None


def test_find_pivot_row():
    """Test the _find_pivot_row helper function."""
    # Test matrix with clear pivot patterns
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int8)

    # Test finding pivot in column 0 starting from row 0
    pivot = _find_pivot_row(m, 0, 4, 0)
    assert pivot == 1  # Row 1 has a 1 in column 0

    # Test finding pivot in column 1 starting from row 0
    pivot = _find_pivot_row(m, 0, 4, 1)
    assert pivot == 0  # Row 0 has a 1 in column 1

    # Test finding pivot in column 2 starting from row 1
    pivot = _find_pivot_row(m, 1, 4, 2)
    assert pivot == 1  # Row 1 has a 1 in column 2

    # Test no pivot found
    pivot = _find_pivot_row(m, 3, 4, 0)
    assert pivot is None  # No 1 found in column 0 starting from row 3


def test_eliminate_column():
    """Test the _eliminate_column helper function."""
    # Setup test matrix
    matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8)

    row_map = [0, 1, 2, 3]
    cnot_ops: list[tuple[int, int]] = []

    # Eliminate column 0 with pivot row 0
    m_result, cnot_ops_result = _eliminate_column(matrix, 4, 0, 0, row_map, cnot_ops)

    # Verify matrix is modified correctly (rows 1 and 3 should be XORed with row 0)
    expected = np.array(
        [
            [1, 1, 0],  # pivot row unchanged
            [0, 1, 1],  # row 1 XOR row 0: [1,0,1] XOR [1,1,0] = [0,1,1]
            [0, 1, 1],  # row 2 unchanged (no 1 in column 0)
            [0, 0, 0],  # row 3 XOR row 0: [1,1,0] XOR [1,1,0] = [0,0,0]
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(m_result, expected)

    # Verify CNOT operations are recorded correctly
    expected_cnots = [(1, 0), (3, 0)]  # (target, control) pairs
    assert cnot_ops_result == expected_cnots

    # Verify original inputs are not modified
    assert cnot_ops == []  # Original list should be empty
    original_expected = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(matrix, original_expected)


def test_perform_gaussian_elimination():
    """Test the _perform_gaussian_elimination helper function."""
    # Test matrix for Gaussian elimination
    matrix = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8)

    m, n = matrix.shape
    row_map = [0, 1, 2]
    cnot_ops: list[tuple[int, int]] = []

    # Perform Gaussian elimination
    m_result, row_map_result, cnot_ops_result = _perform_gaussian_elimination(matrix, m, n, row_map, cnot_ops)

    # Verify the result is in row echelon form
    # After elimination, we should have:
    # - Leading 1s in different columns
    # - Below each leading 1, all entries should be 0

    # Check that we get a valid result
    assert m_result.shape == (3, 3)
    assert len(row_map_result) == 3
    assert isinstance(cnot_ops_result, list)

    # Assert the specific CNOT sequence: CNOT(0,2), CNOT(0,1), CNOT(2,1)
    # For matrix [[1,1,0], [0,1,1], [1,0,1]], Gaussian elimination should produce:
    # Column 0: CNOT(2,0) to eliminate position [2,0]
    # Column 1: CNOT(0,1) to eliminate position [0,1]
    # Column 1: CNOT(2,1) to eliminate position [2,1] (redundant but part of algorithm)
    assert len(cnot_ops_result) == 3, f"Expected 3 CNOT operations, got {len(cnot_ops_result)}"
    assert cnot_ops_result[0] == (2, 0), f"First CNOT should be CNOT(0,2), got {cnot_ops_result[0]}"
    assert cnot_ops_result[1] == (0, 1), f"Second CNOT should be CNOT(0,1), got {cnot_ops_result[1]}"
    assert cnot_ops_result[2] == (2, 1), f"Third CNOT should be CNOT(2,1), got {cnot_ops_result[2]}"

    # Verify original inputs are not modified
    original_expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(matrix, original_expected)
    assert row_map == [0, 1, 2]
    assert cnot_ops == []


def test_remove_zero_rows():
    """Test the _remove_zero_rows helper function."""
    # Test matrix with some zero rows
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],  # zero row
            [0, 1, 0],
            [0, 0, 0],  # zero row
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    row_map = [10, 11, 12, 13, 14]  # Original indices

    # Remove zero rows
    m_result, row_map_result, rank = _remove_zero_rows(matrix, row_map)

    # Verify zero rows are removed
    expected_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int8)

    np.testing.assert_array_equal(m_result, expected_matrix)
    assert row_map_result == [10, 12, 14]  # Indices of non-zero rows
    assert rank == 3

    # Test with no zero rows
    m_no_zeros = np.array([[1, 0], [0, 1]], dtype=np.int8)

    row_map_no_zeros = [5, 6]
    m_result2, row_map_result2, rank2 = _remove_zero_rows(m_no_zeros, row_map_no_zeros)

    np.testing.assert_array_equal(m_result2, m_no_zeros)
    assert row_map_result2 == [5, 6]
    assert rank2 == 2

    # Test with all zero rows
    m_all_zeros = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int8)

    row_map_all_zeros = [0, 1]
    m_result3, row_map_result3, rank3 = _remove_zero_rows(m_all_zeros, row_map_all_zeros)

    assert m_result3.shape == (0, 3)  # Empty matrix with 0 rows, 3 columns
    assert row_map_result3 == []
    assert rank3 == 0


# ======================================================================================
# GF2X Enhanced Elimination Tests
# ======================================================================================


def test_gf2x_with_tracking_basic():
    """Test basic functionality of gf2x_with_tracking."""
    # Simple 3x3 matrix
    matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.int8)

    elimination_results = gf2x_with_tracking(matrix)

    # Verify the result structure
    assert isinstance(elimination_results, GF2XEliminationResult)
    assert isinstance(elimination_results.reduced_matrix, np.ndarray)
    assert isinstance(elimination_results.row_map, list)
    assert isinstance(elimination_results.col_map, list)
    assert isinstance(elimination_results.operations, list)
    assert isinstance(elimination_results.rank, int)

    # Verify column mapping is preserved (no column operations in GF2X)
    assert elimination_results.col_map == [0, 1, 2]

    # Verify rank doesn't exceed original
    original_rank = np.linalg.matrix_rank(matrix)
    assert 0 <= elimination_results.rank <= original_rank

    # Verify operations format
    for op in elimination_results.operations:
        assert isinstance(op, tuple)
        assert len(op) == 2
        assert op[0] in ["cnot", "x"]
        if op[0] == "cnot":
            assert isinstance(op[1], tuple)
            assert len(op[1]) == 2
        elif op[0] == "x":
            assert isinstance(op[1], int)


def test_gf2x_with_tracking_duplicate_rows():
    """Test gf2x_with_tracking with duplicate rows."""
    # Matrix with duplicate rows
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],  # Duplicate of row 0
            [0, 0, 1],
        ],
        dtype=np.int8,
    )

    elimination_results = gf2x_with_tracking(matrix)

    # Should have CNOT operations to eliminate duplicates
    cnot_ops = [op for op in elimination_results.operations if op[0] == "cnot"]
    assert len(cnot_ops) > 0, "Expected CNOT operations for duplicate elimination"

    # Verify rank reduction due to duplicate elimination
    original_rank = np.linalg.matrix_rank(matrix)
    assert elimination_results.rank <= original_rank

    # Verify operations use original matrix indices
    for op in elimination_results.operations:
        if op[0] == "cnot":
            target, control = op[1]
            assert 0 <= target < matrix.shape[0]
            assert 0 <= control < matrix.shape[0]
        elif op[0] == "x":
            assert 0 <= op[1] < matrix.shape[0]


def test_gf2x_with_tracking_all_ones_rows():
    """Test gf2x_with_tracking with all-ones rows."""
    # Matrix with all-ones rows
    matrix = np.array(
        [
            [1, 1, 1],  # All ones
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],  # All ones (duplicate)
        ],
        dtype=np.int8,
    )

    elimination_results = gf2x_with_tracking(matrix)

    # Should have X operations to eliminate all-ones rows
    x_ops = [op for op in elimination_results.operations if op[0] == "x"]
    assert len(x_ops) > 0, "Expected X operations for all-ones row elimination"

    # Verify rank reduction
    original_rank = np.linalg.matrix_rank(matrix)
    assert elimination_results.rank <= original_rank


def test_gf2x_with_tracking_diagonal_matrix():
    """Test gf2x_with_tracking with diagonal matrix."""
    # 3x3 identity matrix (diagonal)
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)

    elimination_results = gf2x_with_tracking(matrix)

    # For diagonal matrix, should apply diagonal reduction
    # This should reduce rank from 3 to 2
    assert elimination_results.rank == 2, f"Expected rank 2 for diagonal reduction, got {elimination_results.rank}"

    # Should have CNOT and X operations from diagonal reduction
    cnot_ops = [op for op in elimination_results.operations if op[0] == "cnot"]
    x_ops = [op for op in elimination_results.operations if op[0] == "x"]

    # Diagonal reduction uses CNOT(i, i+1) for i=0 to rank-2, then X on last row
    # For 3x3: CNOT(0,1), CNOT(1,2), X(2)
    assert len(cnot_ops) >= 2, f"Expected at least 2 CNOT operations, got {len(cnot_ops)}"
    assert len(x_ops) >= 1, f"Expected at least 1 X operation, got {len(x_ops)}"


def test_gf2x_with_tracking_empty_matrix():
    """Test gf2x_with_tracking with empty matrix."""
    # Empty matrix (no rows)
    m_no_rows = np.empty((0, 3), dtype=np.int8)

    with pytest.raises(ValueError, match="Input matrix has no rows"):
        gf2x_with_tracking(m_no_rows)

    # Empty matrix (no columns)
    m_no_cols = np.empty((3, 0), dtype=np.int8)

    with pytest.raises(ValueError, match="Input matrix has no columns"):
        gf2x_with_tracking(m_no_cols)

    # All-zero matrix (rank 0)
    m_zeros = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="Input matrix has rank 0"):
        gf2x_with_tracking(m_zeros)


def test_gf2x_final_rank_zero_return():
    # Create a matrix where preprocessing cancels everything out
    matrix = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # duplicate row → removed
        ],
        dtype=int,
    )

    # Should return empty matrix, not raise
    elimination_results = gf2x_with_tracking(matrix)

    assert elimination_results.reduced_matrix.shape == (0, matrix.shape[1])
    assert elimination_results.row_map == []
    assert elimination_results.rank == 0


def test_gf2x_with_tracking_reconstruction():
    """Test that gf2x_with_tracking operations can reconstruct the original matrix."""
    test_matrices = [
        # Basic matrix
        np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.int8),
        # Matrix with duplicates and all-ones
        np.array(
            [
                [1, 1, 1],  # All ones
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],  # Duplicate
                [1, 1, 1],  # All ones duplicate
            ],
            dtype=np.int8,
        ),
        # Larger matrix
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],  # Duplicate
                [1, 1, 1, 1],  # All ones
                [0, 0, 1, 1],
            ],
            dtype=np.int8,
        ),
    ]

    for i, original_matrix in enumerate(test_matrices):
        elimination_results = gf2x_with_tracking(original_matrix)

        # Verify operations use valid indices
        for op in elimination_results.operations:
            if op[0] == "cnot":
                target, control = op[1]
                assert 0 <= target < original_matrix.shape[0]
                assert 0 <= control < original_matrix.shape[0]
            elif op[0] == "x":
                assert 0 <= op[1] < original_matrix.shape[0]

        # Verify rank consistency
        computed_rank = (
            np.linalg.matrix_rank(elimination_results.reduced_matrix)
            if elimination_results.reduced_matrix.size > 0
            else 0
        )
        assert elimination_results.rank == computed_rank

        # Reconstruct original matrix from reduced matrix
        if elimination_results.rank > 0:
            # Create full-size matrix with reduced matrix data
            reconstructed = np.zeros_like(original_matrix)
            for reduced_row_idx, original_row_idx in enumerate(elimination_results.row_map):
                reconstructed[original_row_idx, :] = elimination_results.reduced_matrix[reduced_row_idx, :]

            # Apply operations in reverse order to reconstruct
            for op in reversed(elimination_results.operations):
                if op[0] == "cnot":
                    target, control = op[1]
                    reconstructed[target] = reconstructed[target] ^ reconstructed[control]
                elif op[0] == "x":
                    row_idx = op[1]
                    # X operation flips all bits in the row
                    reconstructed[row_idx] = 1 - reconstructed[row_idx]

            # Verify reconstruction
            np.testing.assert_array_equal(original_matrix, reconstructed, f"Matrix {i + 1} reconstruction failed")


def test_remove_duplicate_rows_with_cnot():
    """Test the _remove_duplicate_rows_with_cnot helper function."""
    # Matrix with duplicate rows
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],  # Duplicate of row 0
            [0, 0, 1],
        ],
        dtype=np.int8,
    )

    row_map = [0, 1, 2, 3]
    operations: list[tuple[str, int | tuple[int, int]]] = []

    m_result, row_map_result, operations_result = _remove_duplicate_rows_with_cnot(matrix, row_map, operations)

    # Should have eliminated duplicate rows
    assert m_result.shape[0] < matrix.shape[0], "Expected rows to be eliminated"

    # Should have CNOT operations
    cnot_ops = [op for op in operations_result if op[0] == "cnot"]
    assert len(cnot_ops) > 0, "Expected CNOT operations for duplicate elimination"

    # Verify row mapping consistency
    assert len(row_map_result) == m_result.shape[0]


def test_remove_all_ones_rows_with_x():
    """Test the _remove_all_ones_rows_with_x helper function."""
    # Matrix with all-ones rows
    matrix = np.array(
        [
            [1, 1, 1],  # All ones
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],  # All ones
        ],
        dtype=np.int8,
    )

    row_map = [0, 1, 2, 3]
    operations: list[tuple[str, int | tuple[int, int]]] = []

    m_result, row_map_result, operations_result = _remove_all_ones_rows_with_x(matrix, row_map, operations)

    # Should have eliminated all-ones rows
    assert m_result.shape[0] < matrix.shape[0], "Expected rows to be eliminated"

    # Should have X operations
    x_ops = [op for op in operations_result if op[0] == "x"]
    assert len(x_ops) > 0, "Expected X operations for all-ones row elimination"

    # Verify row mapping consistency
    assert len(row_map_result) == m_result.shape[0]

    # Verify no all-ones rows remain
    for i in range(m_result.shape[0]):
        assert not np.all(m_result[i] == 1), "All-ones row should be eliminated"


def test_is_diagonal_matrix():
    """Test the _is_diagonal_matrix helper function."""
    # True diagonal matrices (Scenario 1: Square matrices with diagonal structure and rank > 1)
    diagonal_matrices = [
        np.array([[1, 0], [0, 1]], dtype=np.int8),  # 2x2 identity (rank 2)
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8),  # 3x3 identity (rank 3)
    ]

    for matrix in diagonal_matrices:
        assert _is_diagonal_matrix(matrix), f"Matrix should be diagonal (square case):\n{matrix}"

    # Pseudo-diagonal matrices (Scenario 2: Rectangular with diagonal square part,
    # ALL remaining columns must be all 1s, AND must have odd number of rows)
    pseudo_diagonal_matrices = [
        # Example: 3x4 matrix (3 rows = odd, 3x3 square part is diagonal, remaining column all 1s)
        np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.int8),  # VALID: odd rows + all-1s remaining
        # 3x5 matrix with multiple remaining columns (all must be all 1s)
        np.array([[1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=np.int8),
    ]

    for matrix in pseudo_diagonal_matrices:
        assert _is_diagonal_matrix(matrix), f"Matrix should be valid pseudo-diagonal:\n{matrix}"

    # False diagonal matrices
    non_diagonal_matrices = [
        # Rank 1 matrices (should be rejected regardless of structure)
        np.array([[1, 1]], dtype=np.int8),  # 1x2 all ones (rank 1)
        np.array([[1], [1]], dtype=np.int8),  # 2x1 all ones (rank 1)
        # Non-binary matrices (should be rejected due to invalid values)
        np.array([[2, 0], [0, 1]], dtype=np.int8),  # Contains value 2
        np.array([[1, -1], [0, 1]], dtype=np.int8),  # Contains negative value
        # Square matrices that are not true diagonal
        np.array([[1, 1], [0, 1]], dtype=np.int8),  # Upper triangular
        np.array([[1, 0], [1, 1]], dtype=np.int8),  # Lower triangular
        np.array([[0, 0], [0, 0]], dtype=np.int8),  # All zeros
        np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.int8),  # Off-diagonal elements
        # Rectangular matrices that don't meet criteria - now ALL these should be rejected
        # Even rows (regardless of remaining column content)
        np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8),  # 2x3 (2 rows = even, remaining all 1s)
        np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int8),  # 2x4 even rows with all-1s columns
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8),  # 2x3 even rows with all-0s remaining
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int8),  # 2x4 even rows, all-0s remaining
        # Odd rows but remaining columns contain all-0s (not valid for pseudo-diagonal)
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8),  # INVALID: odd rows but all-0s remaining
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.int8),  # 3x5 odd rows, remaining all 0s
        # Odd rows but remaining columns have mixed 0s and 1s
        np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1]], dtype=np.int8),  # Mixed 0s and 1s
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8),  # Mixed 0s and 1s in remaining column
        # Rectangular with odd rows but square part not diagonal
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8),  # 3x3 but not diagonal
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8),  # 3x3 with off-diagonal element
        # Other invalid cases
        np.array([[1, 1, 0]], dtype=np.int8),  # 1x3 with mixed remaining columns
    ]

    for matrix in non_diagonal_matrices:
        assert not _is_diagonal_matrix(matrix), f"Matrix should not be diagonal:\n{matrix}"


def test_reduce_diagonal_matrix():
    """Test the _reduce_diagonal_matrix helper function."""
    # 3x3 diagonal matrix
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)

    row_map = [0, 1, 2]
    col_map = [0, 1, 2]
    operations: list[tuple[str, int | tuple[int, int]]] = []

    elimination_results = _reduce_diagonal_matrix(matrix, row_map, col_map, operations)

    # Should reduce rank by 1
    assert elimination_results.rank == 2, f"Expected rank 2, got {elimination_results.rank}"
    assert elimination_results.reduced_matrix.shape == (2, 3), (
        f"Expected shape (2, 3), got {elimination_results.reduced_matrix.shape}"
    )

    # Should have CNOT and X operations
    cnot_ops = [op for op in elimination_results.operations if op[0] == "cnot"]
    x_ops = [op for op in elimination_results.operations if op[0] == "x"]

    # For 3x3 diagonal matrix: CNOT(0,2), CNOT(1,2), X(2)
    assert len(cnot_ops) == 2, f"Expected 2 CNOT operations, got {len(cnot_ops)}"
    assert len(x_ops) == 1, f"Expected 1 X operation, got {len(x_ops)}"

    # Assert exact CNOT sequence: CNOT(1,0), CNOT(2,1)
    assert cnot_ops[0] == ("cnot", (1, 0)), f"First CNOT should be CNOT(0,1), got {cnot_ops[0]}"
    assert cnot_ops[1] == ("cnot", (2, 1)), f"Second CNOT should be CNOT(1,2), got {cnot_ops[1]}"

    # Verify specific X operation: X(2) - the last row gets X operation
    expected_x_qubits = {2}
    actual_x_qubits = {op[1] for op in x_ops}  # Extract qubit indices
    assert actual_x_qubits == expected_x_qubits, (
        f"Expected X operations on qubits {expected_x_qubits}, got {actual_x_qubits}"
    )

    # Verify row mapping updated correctly (last row removed)
    assert len(elimination_results.row_map) == 2
    assert elimination_results.row_map == [0, 1]

    # Column mapping should be unchanged
    assert elimination_results.col_map == col_map

    non_diagonal_m = np.array([[1, 0], [1, 0]], dtype=np.int8)
    elimination_results = _reduce_diagonal_matrix(non_diagonal_m, row_map, col_map, operations)
    assert all(non_diagonal_m[i, j] == elimination_results.reduced_matrix[i, j] for i in range(2) for j in range(2))


def test_gf2x_edge_cases():
    """Test gf2x_with_tracking edge cases."""
    # All-zero matrix (should raise ValueError)
    m_zeros = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="Input matrix has rank 0"):
        gf2x_with_tracking(m_zeros)

    # Single row matrix
    m_single = np.array([[1, 0, 1]], dtype=np.int8)

    elimination_results = gf2x_with_tracking(m_single)
    assert elimination_results.rank <= 1
    assert len(elimination_results.row_map) == elimination_results.reduced_matrix.shape[0]

    # Single column matrix
    m_col = np.array([[1], [0], [1]], dtype=np.int8)

    elimination_results = gf2x_with_tracking(m_col)
    assert elimination_results.rank <= 1
    assert len(elimination_results.row_map) == elimination_results.reduced_matrix.shape[0]


def test_gf2x_with_tracking_edge_case_pseudo_diagonal():
    """Test gf2x_with_tracking with a specific pseudo-diagonal edge case.

    This test verifies that GF2X correctly applies diagonal reduction optimization
    when a matrix becomes pseudo-diagonal after initial GF2 processing, achieving
    better rank reduction than standard GF2 elimination alone.
    """
    # Starting matrix that should lead to pseudo-diagonal intermediate form
    # This matrix has rank 3 but after GF2 processing should become pseudo-diagonal
    # and allow further diagonal reduction to rank 2
    matrix = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.int8)

    # Test GF2X - should detect pseudo-diagonal and reduce to rank 2
    elimination_results = gf2x_with_tracking(matrix)
    assert elimination_results.rank == 2, f"Expected GF2X rank 2, got {elimination_results.rank}"

    # Verify the reduced matrix dimensions
    assert elimination_results.reduced_matrix.shape[0] == elimination_results.rank, (
        f"Reduced matrix should have {elimination_results.rank} rows, got {elimination_results.reduced_matrix.shape[0]}"
    )
    assert elimination_results.reduced_matrix.shape[1] == matrix.shape[1], (
        f"Reduced matrix should have {matrix.shape[1]} columns, got {elimination_results.reduced_matrix.shape[1]}"
    )

    # Verify row mapping consistency
    assert len(elimination_results.row_map) == elimination_results.rank, (
        f"Row map length should be {elimination_results.rank}, got {len(elimination_results.row_map)}"
    )

    # Verify that operations list contains both CNOT and X operations
    cnot_ops = [op for op in elimination_results.operations if op[0] == "cnot"]
    x_ops = [op for op in elimination_results.operations if op[0] == "x"]
    assert len(cnot_ops) > 0, "Should have recorded some CNOT operations"
    assert len(x_ops) > 0, "Should have recorded some X operations"

    # Should have some operations recorded
    assert len(elimination_results.operations) > 0, "Should have recorded some operations"
