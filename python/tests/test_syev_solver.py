"""Test for LAPACK SYEV solver utility."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry._core._algorithms import syev_solver

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_syev_solver_small_dense_matrix():
    """Test syev_solver with a small dense symmetric matrix."""
    matrix = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )

    eigenvalues, _ = syev_solver(matrix)

    k = np.array([1, 2, 3], dtype=float)
    expected_eigenvalues = 2.0 - 2.0 * np.cos(k * np.pi / 4)

    # Check eigenvalues are sorted in ascending order
    assert np.allclose(
        eigenvalues,
        expected_eigenvalues,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )


def test_syev_solver_heisenberg_model():
    """Test syev_solver with the Heisenberg model matrix."""
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 2, 0],
            [0, 2, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    expected_eigenvalues = [-3.0, 1.0, 1.0, 1.0]
    expected_eigenvectors = [
        np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
        np.array([0, 0, 0, 1]),
    ]

    eigenvalues, eigenvectors = syev_solver(matrix)

    # Check eigenvalues are correct
    assert np.allclose(
        eigenvalues,
        expected_eigenvalues,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )

    # Verify eigenvectors by reconstructing the matrix
    for i in range(len(expected_eigenvectors)):
        dot = float(np.dot(eigenvectors[:, i], expected_eigenvectors[i]))
        assert np.isclose(
            abs(dot), 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )


def test_syev_solver_invalid_input():
    """Test syev_solver with invalid input."""
    with pytest.raises(ValueError, match="Input matrix must be square"):
        syev_solver(np.array([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match="Input matrix must be 2-dimensional"):
        syev_solver(np.array([1, 2, 3]))
