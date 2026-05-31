"""Tests for qubit tapering utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.data import QubitHamiltonian, Symmetries
from qdk_chemistry.data.tapering import TaperingSpecification, taper_qubits, taper_to_scbk

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    orthonormality_error_tolerance,
)

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

    def test_all_terms_eliminated_returns_zero(self) -> None:
        """When all terms are eliminated, return a zero-coefficient identity operator."""
        qh = QubitHamiltonian(pauli_strings=["XI"], coefficients=np.array([1.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1
        assert len(result.pauli_strings) == 1
        assert result.pauli_strings[0] == "I"
        assert np.isclose(result.coefficients[0], 0.0)

    def test_full_cancellation_returns_zero(self) -> None:
        """When terms cancel to zero after merging, return a zero-coefficient identity."""
        # Z with +1 eigenvalue → +1.0·I, and I → -1.0·I → merge to 0
        qh = QubitHamiltonian(pauli_strings=["ZI", "II"], coefficients=np.array([1.0, -1.0]))
        result = taper_qubits(qh, qubit_indices=[1], eigenvalues=[1])
        assert result.num_qubits == 1
        assert len(result.pauli_strings) == 1
        assert result.pauli_strings[0] == "I"
        assert np.isclose(result.coefficients[0], 0.0)


# -------------------------------------------------------------------------------------
# taper_to_scbk — two-step SCBK tests
# -------------------------------------------------------------------------------------


class TestTaperToScbk:
    """Tests for the taper_to_scbk two-step function."""

    def test_output_has_tapering_attribute(self) -> None:
        """taper_to_scbk sets tapering metadata on the output QubitHamiltonian."""
        qh = QubitHamiltonian(
            pauli_strings=["IIII", "ZIZI", "IZIZ", "ZZII"],
            coefficients=np.array([1.0, 0.5, 0.3, 0.2]),
            encoding="bravyi-kitaev-tree",
        )
        symmetries = Symmetries(n_alpha=1, n_beta=1)
        result = taper_to_scbk(qh, symmetries)

        assert result.tapering is not None
        assert isinstance(result.tapering, TaperingSpecification)
        assert result.tapering.qubit_indices == (1, 3)
        assert result.tapering.source_num_qubits == 4
        assert result.tapering.source_encoding == "bravyi-kitaev-tree"
        assert result.encoding == "symmetry-conserving-bravyi-kitaev"

    def test_output_tapering_eigenvalues(self) -> None:
        """Tapering eigenvalues match the symmetry sector."""
        qh = QubitHamiltonian(
            pauli_strings=["IIII", "ZIII"],
            coefficients=np.array([1.0, 0.5]),
            encoding="bravyi-kitaev",
        )
        symmetries = Symmetries(n_alpha=1, n_beta=0)
        result = taper_to_scbk(qh, symmetries)

        # n_alpha=1 (odd) → ev_alpha=-1, n_total=1 (odd) → ev_total=-1
        assert result.tapering is not None
        assert result.tapering.eigenvalues == (-1, -1)

    def test_tapered_eigenvalues_subset_of_full(self) -> None:
        """Tapered Hamiltonian eigenvalues are a subset of the full BK eigenvalues."""
        base_strings = ["IIII", "IIIZ", "IIZZ", "IZII", "ZZZI"]
        base_coeffs = [2.0, -0.5, -0.5, -0.5, -0.5]

        qh_clean = QubitHamiltonian(
            pauli_strings=base_strings,
            coefficients=np.array(base_coeffs),
            encoding="bravyi-kitaev",
        )

        qh_with_xy = QubitHamiltonian(
            pauli_strings=[*base_strings, "IIXI", "IIYI", "XIIX"],
            coefficients=np.array([*base_coeffs, 0.3, 0.3, 0.2]),
            encoding="bravyi-kitaev",
        )

        symmetries = Symmetries(n_alpha=1, n_beta=1)
        result_clean = taper_to_scbk(qh_clean, symmetries)
        result_with_xy = taper_to_scbk(qh_with_xy, symmetries)

        # X/Y terms on tapered qubits must be dropped — results should be identical
        assert result_with_xy.num_qubits == 2
        assert result_clean.pauli_strings == result_with_xy.pauli_strings
        np.testing.assert_allclose(
            result_clean.coefficients, result_with_xy.coefficients, atol=float_comparison_absolute_tolerance
        )

        # Also verify eigenvalue subset for the symmetry-preserving Hamiltonian
        full_eigs = np.linalg.eigvalsh(qh_clean.to_matrix())
        tapered_eigs = np.linalg.eigvalsh(result_clean.to_matrix())
        for e in tapered_eigs:
            assert np.any(np.isclose(full_eigs, e, atol=orthonormality_error_tolerance))
