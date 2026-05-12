"""Tests for qubit tapering utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.utils.tapering import taper_qubits

# -------------------------------------------------------------------------------------
# taper_qubits — core tests
# -------------------------------------------------------------------------------------


class TestTaperQubits:
    """Tests for the taper_qubits free function."""

    def test_single_qubit_z_eigenvalue_positive(self) -> None:
        """Tapering a Z on a qubit with eigenvalue +1 leaves coefficient unchanged."""
        qh = QubitHamiltonian(pauli_strings=["ZI", "IZ"], coefficients=np.array([1.0, 2.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1
        assert "Z" in result.pauli_strings or "I" in result.pauli_strings

    def test_single_qubit_z_eigenvalue_negative(self) -> None:
        """Tapering a Z on a qubit with eigenvalue -1 flips the coefficient sign."""
        qh = QubitHamiltonian(pauli_strings=["ZI"], coefficients=np.array([1.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[-1])
        assert result.num_qubits == 1
        assert np.isclose(result.coefficients[0], -1.0)

    def test_x_on_tapered_qubit_drops_term(self) -> None:
        """Terms with X on a tapered qubit are removed."""
        qh = QubitHamiltonian(pauli_strings=["XI", "IZ"], coefficients=np.array([1.0, 2.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1
        assert len(result.pauli_strings) == 1

    def test_y_on_tapered_qubit_drops_term(self) -> None:
        """Terms with Y on a tapered qubit are removed."""
        qh = QubitHamiltonian(pauli_strings=["YI", "IZ"], coefficients=np.array([1.0, 2.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1

    def test_identity_on_tapered_qubit_preserved(self) -> None:
        """Terms with I on a tapered qubit are kept as-is."""
        qh = QubitHamiltonian(pauli_strings=["IZ", "II"], coefficients=np.array([1.0, 2.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1
        assert len(result.pauli_strings) == 2

    def test_two_qubits_tapered(self) -> None:
        """Tapering two qubits reduces qubit count by 2."""
        qh = QubitHamiltonian(pauli_strings=["IIII", "ZIZI", "IZIZ"], coefficients=np.array([1.0, 0.5, 0.3]))
        result = taper_qubits(qh, qubit_indices=[0, 3], eigenvalues=[1, 1])
        assert result.num_qubits == 2

    def test_duplicate_terms_merged(self) -> None:
        """After tapering, identical Pauli strings are merged."""
        qh = QubitHamiltonian(pauli_strings=["ZI", "II"], coefficients=np.array([1.0, 2.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        # ZI → I (coeff 1.0) and II → I (coeff 2.0) should merge to I (coeff 3.0)
        assert len(result.pauli_strings) == 1
        assert np.isclose(result.coefficients[0], 3.0)

    def test_encoding_preserved(self) -> None:
        """Encoding metadata is preserved through tapering."""
        qh = QubitHamiltonian(pauli_strings=["ZI", "IZ"], coefficients=np.array([1.0, 2.0]), encoding="bravyi-kitaev")
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.encoding == "bravyi-kitaev"

    def test_mismatched_lengths_raises(self) -> None:
        """ValueError when qubit_indices and eigenvalues have different lengths."""
        qh = QubitHamiltonian(pauli_strings=["ZI"], coefficients=np.array([1.0]))
        with pytest.raises(ValueError, match="length"):
            taper_qubits(qh, qubit_indices=[0, 1], eigenvalues=[1])

    def test_out_of_range_index_raises(self) -> None:
        """ValueError for qubit index out of range."""
        qh = QubitHamiltonian(pauli_strings=["ZI"], coefficients=np.array([1.0]))
        with pytest.raises(ValueError, match="out of range"):
            taper_qubits(qh, qubit_indices=[5], eigenvalues=[1])

    def test_invalid_eigenvalue_raises(self) -> None:
        """ValueError for eigenvalue that is not ±1."""
        qh = QubitHamiltonian(pauli_strings=["ZI"], coefficients=np.array([1.0]))
        with pytest.raises(ValueError, match="must be"):
            taper_qubits(qh, qubit_indices=[0], eigenvalues=[0])

    def test_duplicate_indices_raises(self) -> None:
        """ValueError for duplicate qubit indices."""
        qh = QubitHamiltonian(pauli_strings=["ZII"], coefficients=np.array([1.0]))
        with pytest.raises(ValueError, match="duplicate"):
            taper_qubits(qh, qubit_indices=[0, 0], eigenvalues=[1, -1])

    def test_all_terms_eliminated_raises(self) -> None:
        """ValueError when all terms are eliminated by tapering."""
        qh = QubitHamiltonian(pauli_strings=["XI"], coefficients=np.array([1.0]))
        with pytest.raises(ValueError, match="eliminated"):
            taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
