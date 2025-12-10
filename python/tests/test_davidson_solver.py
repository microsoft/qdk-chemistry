"""Test for Davidson solver utility."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sp

from qdk_chemistry._core._algorithms import davidson_solver
from qdk_chemistry.data import QubitHamiltonian

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_davidson_solver_matrix():
    """Davidson solver test using a 6x6 matrix with known analytical eigenvalues/eigenvectors."""
    test_matrix = np.array(
        [
            [2, -1, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 2, -1, 0],
            [0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, -1, 2],
        ],
        dtype=float,
    )

    test_matrix_csr = sp.csr_matrix(test_matrix)

    # Run Davidson
    eigval, eigvec = davidson_solver(test_matrix_csr)

    # Expected analytical ground state:
    expected_energy = 2 - 2 * np.cos(np.pi / 7)

    # eigenvector components: v_i = sin(i*pi/7), i=1..6
    indices = np.arange(1, 7)
    expected_vec = np.sin(indices * np.pi / 7)

    # Normalize expected vec
    expected_vec /= np.linalg.norm(expected_vec)
    dot = float(np.dot(eigvec, expected_vec))

    assert np.isclose(
        eigval, expected_energy, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )
    assert eigvec.shape == (6,)
    assert np.isclose(abs(dot), 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)


def test_davidson_solver_model_hamiltonian():
    """Test Davidson solver on a model Hamiltonian."""
    heisenberg_model = QubitHamiltonian(pauli_strings=["XX", "YY", "ZZ"], coefficients=[1.0, 1.0, 1.0])
    csr_matrix = heisenberg_model.pauli_ops.to_matrix(sparse=True).real.copy()

    eigval, eigvec = davidson_solver(csr_matrix)

    expected_energy = -3.0
    expected_eigvec = np.array([0.0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0.0])

    # Check energy
    assert np.isclose(
        eigval, expected_energy, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )
    # Check eigenvector
    assert eigvec.shape == (4,)
    dot = float(np.dot(eigvec, expected_eigvec))
    assert np.isclose(abs(dot), 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance)


def test_davidson_solver_hamiltonian(hamiltonian_10e6o):
    """Test Davidson solver on a larger Hamiltonian (10 electrons in 6 orbitals)."""
    hamiltonian_csr = hamiltonian_10e6o.pauli_ops.to_matrix(sparse=True).real.copy()
    eigval, eigvec = davidson_solver(hamiltonian_csr)

    # Reference values obtained from SCI calculation (test_data/make_f2.py)
    expected_energy = -33.34889127359048
    assert np.isclose(
        eigval, expected_energy, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )
    assert eigvec.shape == (2**12,)
