"""Tests for pybind11 Pauli matrix utility functions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import random as stdlib_random

import numpy as np
import pytest

from qdk_chemistry._core.utils import (
    pauli_expectation,
    pauli_string_to_masks,
    pauli_to_dense_matrix,
    pauli_to_sparse_matrix,
)

from .reference_tolerances import float_comparison_absolute_tolerance

_SEED = 2026

_COMPREHENSIVE_LABELS = [
    "XI",
    "IX",
    "ZI",
    "IZ",
    "YI",
    "IY",
    "XY",
    "YX",
    "ZZ",
    "XX",
    "YY",
    "XYZ",
    "ZIX",
    "YZX",
    "XXYZ",
    "IIZZXY",
    "XYZIXYZZ",
]


def _pauli_mat(label):
    """Return Pauli matrix from a Pauli label."""
    mat = np.eye(1, dtype=complex)
    for i in label:
        if i == "I":
            mat = np.kron(mat, np.eye(2, dtype=complex))
        elif i == "X":
            mat = np.kron(mat, np.array([[0, 1], [1, 0]], dtype=complex))
        elif i == "Y":
            mat = np.kron(mat, np.array([[0, -1j], [1j, 0]], dtype=complex))
        elif i == "Z":
            mat = np.kron(mat, np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            raise ValueError(f"Invalid Pauli character '{i}'")
    return mat


class TestPauliStringToMasks:
    """Tests for pauli_string_to_masks.

    Encoding convention: label position i maps to bit (n-1-i),
    so the leftmost character (position 0) is the most-significant bit.
    X sets x_mask, Z sets z_mask, Y sets both. Phase = i^(number of Y's).
    """

    @staticmethod
    def _bit(n_qubits, position):
        """Return the bitmask for a given qubit position in an n-qubit label."""
        return 1 << (n_qubits - 1 - position)

    def test_identity(self):
        """Identity has no X or Z bits set, phase = 1."""
        x_mask, z_mask, phase = pauli_string_to_masks("II")
        assert x_mask == 0
        assert z_mask == 0
        assert np.isclose(phase, 1.0, atol=float_comparison_absolute_tolerance)

    def test_single_x(self):
        """X on qubit 0 (label position 0 = MSB) sets x_mask high bit."""
        x_mask, z_mask, phase = pauli_string_to_masks("XI")
        assert x_mask == self._bit(2, 0)
        assert z_mask == 0
        assert np.isclose(phase, 1.0, atol=float_comparison_absolute_tolerance)

    def test_single_z(self):
        """Z on qubit 0 sets z_mask high bit."""
        x_mask, z_mask, phase = pauli_string_to_masks("ZI")
        assert x_mask == 0
        assert z_mask == self._bit(2, 0)
        assert np.isclose(phase, 1.0, atol=float_comparison_absolute_tolerance)

    def test_single_y(self):
        """Y sets both x and z masks; phase = i."""
        x_mask, z_mask, phase = pauli_string_to_masks("Y")
        assert x_mask == self._bit(1, 0)
        assert z_mask == self._bit(1, 0)
        assert np.isclose(phase, 1j, atol=float_comparison_absolute_tolerance)

    def test_two_y_phase(self):
        """Two Y operators: phase = i^2 = -1."""
        x_mask, z_mask, phase = pauli_string_to_masks("YY")
        assert x_mask == self._bit(2, 0) | self._bit(2, 1)
        assert z_mask == self._bit(2, 0) | self._bit(2, 1)
        assert np.isclose(phase, -1.0, atol=float_comparison_absolute_tolerance)

    def test_three_y_phase(self):
        """Three Y operators: phase = i^3 = -i."""
        _, _, phase = pauli_string_to_masks("YYY")
        assert np.isclose(phase, -1j, atol=float_comparison_absolute_tolerance)

    def test_four_y_phase(self):
        """Four Y operators: phase = i^4 = 1."""
        _, _, phase = pauli_string_to_masks("YYYY")
        assert np.isclose(phase, 1.0, atol=float_comparison_absolute_tolerance)

    def test_mixed_xzy(self):
        """Mixed label: XZY on 3 qubits."""
        x_mask, z_mask, phase = pauli_string_to_masks("XZY")
        # X at position 0, Z at position 1, Y at position 2
        assert x_mask == self._bit(3, 0) | self._bit(3, 2)  # X and Y set x
        assert z_mask == self._bit(3, 1) | self._bit(3, 2)  # Z and Y set z
        assert np.isclose(phase, 1j, atol=float_comparison_absolute_tolerance)


class TestPauliExpectation:
    """Tests for pauli_expectation."""

    def test_on_bell_state(self):
        """<Bell|ZZ|Bell> = 1 for Bell state (|00>+|11>)/sqrt(2)."""
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.isclose(pauli_expectation("ZZ", psi), 1.0, atol=float_comparison_absolute_tolerance)
        assert np.isclose(pauli_expectation("XX", psi), 1.0, atol=float_comparison_absolute_tolerance)
        assert np.isclose(pauli_expectation("YY", psi), -1.0, atol=float_comparison_absolute_tolerance)

    @pytest.mark.parametrize("label", _COMPREHENSIVE_LABELS)
    def test_agrees_with_matrix_comprehensive(self, label):
        """pauli_expectation matches <psi|P|psi> for all labels in the comprehensive set."""
        n = len(label)
        rng = np.random.default_rng(_SEED + n)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mat = _pauli_mat(label)
        expected = np.real(psi.conj() @ mat @ psi)
        got = pauli_expectation(label, psi)
        assert np.isclose(got, expected, atol=float_comparison_absolute_tolerance), (
            f"Mismatch for {label}: expected {expected}, got {got}"
        )


class TestPauliToDenseMatrix:
    """Tests for pauli_to_dense_matrix."""

    @pytest.mark.parametrize("label", _COMPREHENSIVE_LABELS)
    def test_single_term_with_reference(self, label):
        """Each single-term Pauli matrix matches reference."""
        mat = np.asarray(pauli_to_dense_matrix([label], np.array([1.0 + 0j])))
        expected = _pauli_mat(label)
        assert np.allclose(mat, expected, atol=float_comparison_absolute_tolerance), f"Reference mismatch for {label}"

    def test_multi_term_3terms(self):
        """H = 0.5*ZI + 0.3*IX + 0.2*XY matches reference sum."""
        labels = ["ZI", "IX", "XY"]
        coeffs = np.array([0.5, 0.3, 0.2], dtype=complex)
        mat = np.asarray(pauli_to_dense_matrix(labels, coeffs))
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        assert np.allclose(mat, expected, atol=float_comparison_absolute_tolerance)

    def test_multi_term_8terms_2qubit(self):
        """8-term 2-qubit Hamiltonian matches reference sum."""
        labels = ["ZI", "IX", "XY", "YZ", "XX", "YY", "ZZ", "IZ"]
        coeffs = np.array([0.5, 0.3, 0.2, -0.1, 0.4, -0.25, 0.15, -0.35], dtype=complex)
        mat = np.asarray(pauli_to_dense_matrix(labels, coeffs))
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        assert np.allclose(mat, expected, atol=float_comparison_absolute_tolerance)

    def test_multi_term_6terms_3qubit(self):
        """6-term 3-qubit Hamiltonian matches reference sum."""
        labels = ["XYZ", "ZIX", "YZX", "III", "ZZZ", "XXX"]
        coeffs = np.array([0.3, -0.2, 0.15, 1.0, -0.5, 0.25], dtype=complex)
        mat = np.asarray(pauli_to_dense_matrix(labels, coeffs))
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        assert np.allclose(mat, expected, atol=float_comparison_absolute_tolerance)


class TestPauliToSparseMatrix:
    """Tests for pauli_to_sparse_matrix."""

    def test_identity_sparse(self):
        """Sparse identity matches reference."""
        sp = pauli_to_sparse_matrix(["II"], np.array([1.0], dtype=complex))
        assert np.allclose(sp.toarray(), _pauli_mat("II"), atol=float_comparison_absolute_tolerance)

    def test_matches_dense(self):
        """Sparse and dense give identical results for a multi-term Hamiltonian."""
        labels = ["ZI", "IZ", "XX", "YY"]
        coeffs = np.array([0.5, -0.3, 0.2, 0.1], dtype=complex)
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        dense = np.asarray(pauli_to_dense_matrix(labels, coeffs))
        sparse = pauli_to_sparse_matrix(labels, coeffs)
        assert np.allclose(dense, expected, atol=float_comparison_absolute_tolerance)
        assert np.allclose(sparse.toarray(), expected, atol=float_comparison_absolute_tolerance)

    def test_sparse_complex_coeffs(self):
        """Sparse matrix with complex coefficients matches pauli_mat reference."""
        labels = ["XY", "ZI", "IZ", "ZZ"]
        coeffs = np.array([0.4 + 0.2j, -0.2 - 0.5j, 0.7j, -0.1 + 0.3j], dtype=complex)
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        sp = pauli_to_sparse_matrix(labels, coeffs)
        assert np.allclose(sp.toarray(), expected, atol=float_comparison_absolute_tolerance)

    def test_sparse_3qubit(self):
        """3-qubit sparse matrix matches reference."""
        labels = ["ZZI", "IXX", "YYZ"]
        coeffs = np.array([1.0, -0.5, 0.3], dtype=complex)
        sp = pauli_to_sparse_matrix(labels, coeffs)
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        assert np.allclose(sp.toarray(), expected, atol=float_comparison_absolute_tolerance)

    def test_sparse_nnz_efficiency(self):
        """Sparse matrix has fewer stored elements than dense for diagonal Hamiltonian."""
        labels = ["ZI", "IZ", "ZZ"]
        coeffs = np.array([1.0, -0.5, 0.3], dtype=complex)
        sp = pauli_to_sparse_matrix(labels, coeffs)
        assert sp.nnz <= sp.shape[0]

    def test_6qubit_100terms_sparse_matches_pauli_mat(self):
        """6-qubit, 100-term sparse matrix matches pauli_mat reference with complex coefficients."""
        stdlib_random.seed(_SEED)
        rng = np.random.default_rng(_SEED)
        pauli_chars = "IXYZ"
        n_qubits = 6
        n_terms = 100
        labels = ["".join(stdlib_random.choice(pauli_chars) for _ in range(n_qubits)) for _ in range(n_terms)]
        coeffs = rng.standard_normal(n_terms) + 1j * rng.standard_normal(n_terms)
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        sparse = pauli_to_sparse_matrix(labels, coeffs)
        assert np.allclose(sparse.toarray(), expected, atol=float_comparison_absolute_tolerance)

    def test_12qubit_5terms_sparse_matches_pauli_mat(self):
        """12-qubit, 5-term sparse matrix matches pauli_mat reference."""
        stdlib_random.seed(_SEED)
        rng = np.random.default_rng(_SEED)
        pauli_chars = "IXYZ"
        n_qubits = 12
        n_terms = 5
        labels = ["".join(stdlib_random.choice(pauli_chars) for _ in range(n_qubits)) for _ in range(n_terms)]
        coeffs = rng.standard_normal(n_terms) + 1j * rng.standard_normal(n_terms)
        expected = sum(c * _pauli_mat(label) for c, label in zip(coeffs, labels, strict=True))
        sparse = pauli_to_sparse_matrix(labels, coeffs)
        assert np.allclose(sparse.toarray(), expected, atol=float_comparison_absolute_tolerance)

    @pytest.mark.parametrize("n_qubits", [16, 20, 24])
    def test_large_qubit_construction(self, n_qubits):
        """Test that a large-qubit sparse matrix with 2 random terms can be constructed."""
        stdlib_random.seed(_SEED)
        rng = np.random.default_rng(_SEED)
        pauli_chars = "IXYZ"
        n_terms = 2
        dim = 2**n_qubits
        labels = ["".join(stdlib_random.choice(pauli_chars) for _ in range(n_qubits)) for _ in range(n_terms)]
        coeffs = rng.standard_normal(n_terms) + 1j * rng.standard_normal(n_terms)
        sparse = pauli_to_sparse_matrix(labels, coeffs)
        assert sparse.shape == (dim, dim)


class TestPauliExpectationVsDenseMatrix:
    """Cross-validation: pauli_expectation should agree with matrix-based <psi|H|psi>."""

    def test_2qubit_hamiltonian_energy(self):
        """For a 2-qubit random state, expectation of Pauli sum = sum of individual expectations."""
        rng = np.random.default_rng(_SEED)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)
        labels = ["ZI", "IZ", "XX", "YY"]
        coeffs = np.array([0.5, -0.3, 0.2, 0.1])
        mat = np.asarray(pauli_to_dense_matrix(labels, coeffs.astype(complex)))
        energy_matrix = np.real(psi.conj() @ mat @ psi)
        energy_sum = sum(c * pauli_expectation(label, psi) for c, label in zip(coeffs, labels, strict=True))
        assert np.isclose(energy_matrix, energy_sum, atol=float_comparison_absolute_tolerance)
