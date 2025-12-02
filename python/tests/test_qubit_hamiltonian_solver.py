"""Test qubit hamiltonian solver algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.qubit_hamiltonian_solver import DenseMatrixSolver, SparseMatrixSolver
from qdk_chemistry.data import QubitHamiltonian

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_dense_matrix_solver_for_small_system():
    """Test dense matrix solver for small qubit systems."""
    qubit_hamiltonian = QubitHamiltonian(["ZZ", "XX"], [1.0, 0.5])
    q_solver = create("qubit_hamiltonian_solver", "dense_matrix_solver")
    assert isinstance(q_solver, DenseMatrixSolver)

    energy, state = q_solver.run(qubit_hamiltonian)
    assert np.isclose(energy, -1.5, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)
    assert state.shape == (4,)
    expected_state = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])
    dot = float(np.dot(state, expected_state))
    assert np.isclose(abs(dot), 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)


def test_sparse_matrix_solver():
    """Test sparse matrix solver."""
    # Create a 4-qubit Hamiltonian
    qubit_hamiltonian = QubitHamiltonian(
        ["ZZII", "YYII", "XXII", "IIZZ", "IIYY", "IIXX"], [1.0, 0.5, 0.5, 1.0, 0.5, 0.5]
    )
    q_solver = create("qubit_hamiltonian_solver", "sparse_matrix_solver")
    assert isinstance(q_solver, SparseMatrixSolver)
    assert np.isclose(
        q_solver.settings().tol,
        1e-8,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )
    assert q_solver.settings().max_m == 20

    energy, state = q_solver.run(qubit_hamiltonian)
    assert np.isclose(energy, -4, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)
    assert state.shape == (16,)
    expected_state = np.zeros(16, dtype=float)
    expected_state[10] = 0.5
    expected_state[6] = -0.5
    expected_state[9] = -0.5
    expected_state[5] = 0.5
    expected_state /= np.linalg.norm(expected_state)

    dot = float(np.dot(state, expected_state))
    assert np.isclose(abs(dot), 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)
