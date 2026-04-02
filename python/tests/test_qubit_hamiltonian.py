"""Test Hamiltonian loading and grouping functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import random as stdlib_random
import re

import numpy as np
import pytest
import scipy.sparse
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _pauli_matrix(label):
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


class TestQubitHamiltonian:
    """Test suite for QubitHamiltonian data class."""

    def test_initialization(self):
        """Test that QubitHamiltonian initializes correctly."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        assert qubit_hamiltonian.pauli_strings == pauli_strings
        assert np.array_equal(qubit_hamiltonian.coefficients, coefficients)
        assert qubit_hamiltonian.num_qubits == 2

    def test_initialization_mismatch(self):
        """Test that initialization raises ValueError on mismatched lengths."""
        with pytest.raises(ValueError, match=r"Mismatch between number of Pauli strings and coefficients\."):
            QubitHamiltonian(pauli_strings=["X", "Y"], coefficients=np.array([1.0]))

    def test_initialization_invalid_pauli(self):
        """Test that initialization raises ValueError on invalid Pauli strings."""
        with pytest.raises(ValueError, match="invalid characters"):
            QubitHamiltonian(pauli_strings=["X", "A"], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="has length"):
            QubitHamiltonian(pauli_strings=["X", "ZY"], coefficients=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="invalid characters"):
            QubitHamiltonian(pauli_strings=["XZ", "A1"], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="empty"):
            QubitHamiltonian(pauli_strings=["X", ""], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="empty"):
            QubitHamiltonian(pauli_strings=[], coefficients=[])

    def test_group_commuting(self):
        """Test group_commuting."""
        qubit_hamiltonian = QubitHamiltonian(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = qubit_hamiltonian.group_commuting(qubit_wise=False)
        assert len(grouped) == 2

        # Verify coefficients are preserved
        coeff_map = dict(zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True))
        for group in grouped:
            for pauli_str, coeff in zip(group.pauli_strings, group.coefficients, strict=True):
                assert np.isclose(
                    coeff,
                    coeff_map[pauli_str],
                    atol=float_comparison_absolute_tolerance,
                    rtol=float_comparison_relative_tolerance,
                )

    def test_group_commuting_qubitwise(self):
        """Test group_commuting without qubit-wise commuting."""
        qubit_hamiltonian = QubitHamiltonian(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = qubit_hamiltonian.group_commuting(qubit_wise=True)
        assert len(grouped) == 4  # Qubit-wise commuting returns four groups

        # Check that all original Pauli strings are present across all groups
        all_grouped_strings = []
        for group in grouped:
            assert len(group.pauli_strings) == 1  # Each group should contain only one Pauli string
            all_grouped_strings.extend(group.pauli_strings)
        assert set(all_grouped_strings) == {"XX", "YY", "ZZ", "XY"}

    def test_group_commuting_all_commute(self):
        """Test that fully commuting operators go into one group."""
        # ZI, IZ, ZZ all commute with each other
        qh = QubitHamiltonian(["ZI", "IZ", "ZZ"], np.array([1.0, -0.5, 0.3]))
        grouped = qh.group_commuting(qubit_wise=False)
        assert len(grouped) == 1
        assert len(grouped[0].pauli_strings) == 3

    def test_group_commuting_none_commute(self):
        """Test that non-commuting operators each get their own group."""
        # X and Z anticommute; Y and X anticommute; Y and Z anticommute
        qh = QubitHamiltonian(["X", "Z", "Y"], np.array([1.0, -0.5, 0.3]))
        grouped = qh.group_commuting(qubit_wise=False)
        assert len(grouped) == 3

    def test_group_commuting_single_term(self):
        """Test group_commuting with a single term."""
        qh = QubitHamiltonian(["ZZ"], np.array([1.0]))
        grouped = qh.group_commuting(qubit_wise=False)
        assert len(grouped) == 1
        assert grouped[0].pauli_strings == ["ZZ"]

    def test_group_commuting_reconstruct_matrix(self):
        """Test group_commuting with matrix verification."""
        qh = QubitHamiltonian(
            ["II", "IZ", "ZI", "ZZ", "XX", "YY"],
            np.array([-0.8, 0.17, -0.17, 0.12, 0.04, 0.04]),
        )
        # General commuting: all diagonal terms commute, XX and YY commute with each other and with diag terms
        grouped = qh.group_commuting(qubit_wise=False)
        total_terms = sum(len(g.pauli_strings) for g in grouped)
        assert total_terms == 6
        # Verify ground state energy via eigenvalues
        mat = qh.to_matrix()
        gs_energy = np.min(np.linalg.eigvalsh(mat))
        # Reconstruct from groups and check same ground state energy
        full_mat = np.zeros_like(mat)
        for g in grouped:
            full_mat += g.to_matrix()
        gs_energy_grouped = np.min(np.linalg.eigvalsh(full_mat))
        assert np.isclose(gs_energy, gs_energy_grouped, atol=float_comparison_absolute_tolerance)

    def test_group_commuting_qw_reconstruct_matrix(self):
        """Test that QW-grouped Hamiltonian reconstructs the original matrix exactly."""
        labels = ["ZI", "IZ", "ZZ", "XI", "IX", "XX", "YY"]
        coeffs = np.array([0.5, 0.3, 0.2, -0.1, 0.4, -0.25, 0.15])
        qh = QubitHamiltonian(labels, coeffs)
        original_mat = qh.to_matrix()
        groups = qh.group_commuting(qubit_wise=True)
        reconstructed = np.zeros_like(original_mat)
        for g in groups:
            reconstructed += g.to_matrix()
        assert np.allclose(
            reconstructed,
            original_mat,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_schatten_norm_basic(self):
        """Test Schatten norm with basic Hamiltonian."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        # Schatten norm = |1.0| + |-0.5| + |0.75| = 2.25
        expected_norm = 2.25
        assert np.isclose(
            qubit_hamiltonian.schatten_norm,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_schatten_norm_with_negative_coefficients(self):
        """Test Schatten norm handles negative coefficients correctly."""
        pauli_strings = ["X", "Y", "Z"]
        coefficients = np.array([-2.0, -1.5, -0.5])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        # Schatten norm = |-2.0| + |-1.5| + |-0.5| = 4.0
        expected_norm = 4.0
        assert np.isclose(
            qubit_hamiltonian.schatten_norm,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_schatten_norm_with_complex_coefficients(self):
        """Test Schatten norm with complex coefficients."""
        pauli_strings = ["XX", "YY"]
        coefficients = np.array([3.0 + 4.0j, -1.0 + 0.0j])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        # Schatten norm = |3.0+4.0j| + |-1.0| = 5.0 + 1.0 = 6.0
        expected_norm = 6.0
        assert np.isclose(
            qubit_hamiltonian.schatten_norm,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_schatten_norm_single_term(self):
        """Test Schatten norm with single term Hamiltonian."""
        pauli_strings = ["Z"]
        coefficients = np.array([3.5])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        expected_norm = 3.5
        assert np.isclose(
            qubit_hamiltonian.schatten_norm,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_schatten_norm_zero_coefficients(self):
        """Test Schatten norm with zero coefficients."""
        pauli_strings = ["X", "Y", "Z"]
        coefficients = np.array([0.0, 0.0, 0.0])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        expected_norm = 0.0
        assert np.isclose(
            qubit_hamiltonian.schatten_norm,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_to_interleaved_4_qubits(self):
        """Test blocked to interleaved conversion for 4 qubits."""
        # Blocked: [α₀, α₁, β₀, β₁] -> Interleaved: [α₀, β₀, α₁, β₁]
        qh = QubitHamiltonian(["XYZZ"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert interleaved.pauli_strings == ["XZYZ"]

    def test_to_interleaved_preserves_coefficients(self):
        """Test that interleaving preserves coefficient values."""
        qh = QubitHamiltonian(["XIZI", "IYII"], np.array([0.5 + 0.1j, 0.3], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert np.allclose(interleaved.coefficients, qh.coefficients)

    def test_to_interleaved_invalid_n_spatial(self):
        """Test that invalid n_spatial raises error."""
        qh = QubitHamiltonian(["XIZI"], np.array([1.0], dtype=complex))
        with pytest.raises(ValueError, match=re.escape("must be 2 * n_spatial")):
            qh.to_interleaved(n_spatial=3)

    def test_to_interleaved_single_orbital(self):
        """Test that single spatial orbital (2 qubits) is unchanged."""
        qh = QubitHamiltonian(["XY"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=1)
        assert interleaved.pauli_strings == ["XY"]

    def test_to_matrix_hermitian(self):
        """Test that to_matrix produces a Hermitian matrix for real coefficients."""
        qh = QubitHamiltonian(["IX", "ZI", "ZZ", "YY"], np.array([0.5, -0.3, 0.8, -0.2]))
        mat = qh.to_matrix()
        assert np.allclose(
            mat, mat.conj().T, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_to_matrix(self):
        """Test to_matrix returns a matrix matching reference."""
        labels = ["IX", "ZZ", "YY"]
        coeffs = np.array([0.5, -0.3, 0.1])
        qh = QubitHamiltonian(labels, coeffs)
        expected = sum(c * _pauli_matrix(label) for c, label in zip(coeffs, labels, strict=True))
        dense = qh.to_matrix(sparse=False)
        sparse = qh.to_matrix(sparse=True)
        assert scipy.sparse.issparse(sparse)
        assert np.allclose(
            dense, expected, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )
        assert np.allclose(
            sparse.toarray(),
            expected,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_to_matrix_eigenvalues(self):
        """Test that sparse and dense matrices give the same eigenvalues."""
        qh = QubitHamiltonian(["ZI", "IZ", "XX"], np.array([0.7, -0.4, 0.3]))
        dense = qh.to_matrix(sparse=False)
        sparse = qh.to_matrix(sparse=True)
        eigvals_dense = np.sort(np.linalg.eigvalsh(dense))
        eigvals_sparse = np.sort(np.linalg.eigvalsh(sparse.toarray()))
        assert np.allclose(
            eigvals_dense,
            eigvals_sparse,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_to_matrix_complex_coefficients(self):
        """Test to_matrix with complex coefficients."""
        labels = ["X", "Y"]
        coeffs = np.array([1.0 + 0.5j, 0.0 - 0.3j])
        qh = QubitHamiltonian(labels, coeffs)
        expected = sum(c * _pauli_matrix(label) for c, label in zip(coeffs, labels, strict=True))
        mat = qh.to_matrix()
        assert np.allclose(
            mat, expected, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_to_matrix_large_10qubit_random(self):
        """Test to_matrix on a 10-qubit, 20-term random Hamiltonian."""
        stdlib_random.seed(2026)
        rng = np.random.default_rng(2026)
        pauli_chars = "IXYZ"
        n_qubits = 10
        n_terms = 20
        labels = ["".join(stdlib_random.choice(pauli_chars) for _ in range(n_qubits)) for _ in range(n_terms)]
        coeffs = rng.standard_normal(n_terms) + 1j * rng.standard_normal(n_terms)
        qh = QubitHamiltonian(labels, coeffs)
        dim = 2**n_qubits
        expected = np.zeros((dim, dim), dtype=complex)
        for coeff, label in zip(coeffs, labels, strict=True):
            expected += coeff * _pauli_matrix(label)
        dense = qh.to_matrix(sparse=False)
        sparse = qh.to_matrix(sparse=True)
        assert np.allclose(
            dense, expected, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )
        assert np.allclose(
            sparse.toarray(),
            expected,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )


class TestQubitHamiltonianSerialization:
    """Test suite for QubitHamiltonian serialization (JSON and HDF5)."""

    def test_json_serialization_real_coefficients(self):
        """Test JSON serialization with real coefficients."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)

        # Test to_json() returns valid JSON
        json_data = qubit_hamiltonian.to_json()
        assert "pauli_strings" in json_data
        assert "coefficients" in json_data
        assert "version" in json_data

        # Verify the coefficients are serialized as dict with real and imag
        assert isinstance(json_data["coefficients"], dict)
        assert "real" in json_data["coefficients"]
        assert "imag" in json_data["coefficients"]

        # Serialize to string and back (validates JSON compatibility)
        json_string = json.dumps(json_data)
        parsed = json.loads(json_string)
        assert parsed == json_data

    def test_json_serialization_complex_coefficients(self):
        """Test JSON serialization with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)

        # Test to_json() returns valid JSON
        json_data = qubit_hamiltonian.to_json()

        # Serialize to string and back (validates JSON compatibility)
        json_string = json.dumps(json_data)
        parsed = json.loads(json_string)

        # Verify coefficients structure
        assert isinstance(parsed["coefficients"], dict)
        assert parsed["coefficients"]["real"] == [1.0, -0.5, 0.0, 2.0]
        assert parsed["coefficients"]["imag"] == [0.5, -0.25, 0.75, 0.0]

    def test_json_roundtrip_real_coefficients(self):
        """Test JSON roundtrip with real coefficients."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        original = QubitHamiltonian(pauli_strings, coefficients)

        # Roundtrip through JSON
        json_data = original.to_json()
        reconstructed = QubitHamiltonian.from_json(json_data)

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_json_roundtrip_complex_coefficients(self):
        """Test JSON roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitHamiltonian(pauli_strings, coefficients)

        # Roundtrip through JSON
        json_data = original.to_json()
        reconstructed = QubitHamiltonian.from_json(json_data)

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_json_file_roundtrip_complex_coefficients(self, tmp_path):
        """Test JSON file roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitHamiltonian(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.json"
        original.to_json_file(str(filename))

        # Load and verify
        reconstructed = QubitHamiltonian.from_json_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_hdf5_roundtrip_real_coefficients(self, tmp_path):
        """Test HDF5 roundtrip with real coefficients."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        original = QubitHamiltonian(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        # Load and verify
        reconstructed = QubitHamiltonian.from_hdf5_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_hdf5_roundtrip_complex_coefficients(self, tmp_path):
        """Test HDF5 roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitHamiltonian(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        # Load and verify
        reconstructed = QubitHamiltonian.from_hdf5_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_json_to_json_file_no_complex_error(self, tmp_path):
        """Regression test: to_json_file must not raise TypeError for complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.json"

        # This should not raise TypeError: Object of type complex is not JSON serializable
        qubit_hamiltonian.to_json_file(str(filename))

        # Verify the file can be read
        with open(filename) as f:
            data = json.load(f)

        assert "pauli_strings" in data
        assert "coefficients" in data


class TestFermionModeOrder:
    """Test suite for fermion_mode_order metadata on QubitHamiltonian."""

    def test_default_is_none(self):
        """fermion_mode_order defaults to None when not specified."""
        qh = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]))
        assert qh.fermion_mode_order is None

    def test_set_blocked(self):
        """fermion_mode_order can be set to BLOCKED."""
        qh = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order=FermionModeOrder.BLOCKED)
        assert qh.fermion_mode_order == FermionModeOrder.BLOCKED
        assert qh.fermion_mode_order == "blocked"

    def test_set_interleaved(self):
        """fermion_mode_order can be set to INTERLEAVED."""
        qh = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order=FermionModeOrder.INTERLEAVED)
        assert qh.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_set_from_string(self):
        """fermion_mode_order accepts a raw string and coerces to the enum."""
        qh = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order="blocked")
        assert qh.fermion_mode_order is FermionModeOrder.BLOCKED

    def test_json_roundtrip(self):
        """fermion_mode_order survives JSON serialization."""
        original = QubitHamiltonian(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            encoding="jordan-wigner",
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        json_data = original.to_json()
        assert json_data["fermion_mode_order"] == "blocked"

        restored = QubitHamiltonian.from_json(json_data)
        assert restored.fermion_mode_order == FermionModeOrder.BLOCKED

    def test_json_roundtrip_none(self):
        """fermion_mode_order=None is omitted from JSON and restored as None."""
        original = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]))
        json_data = original.to_json()
        assert "fermion_mode_order" not in json_data

        restored = QubitHamiltonian.from_json(json_data)
        assert restored.fermion_mode_order is None

    def test_hdf5_roundtrip(self, tmp_path):
        """fermion_mode_order survives HDF5 serialization."""
        original = QubitHamiltonian(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            encoding="jordan-wigner",
            fermion_mode_order=FermionModeOrder.INTERLEAVED,
        )
        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        restored = QubitHamiltonian.from_hdf5_file(str(filename))
        assert restored.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_hdf5_roundtrip_none(self, tmp_path):
        """fermion_mode_order=None is omitted from HDF5 and restored as None."""
        original = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]))
        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        restored = QubitHamiltonian.from_hdf5_file(str(filename))
        assert restored.fermion_mode_order is None

    def test_group_commuting_preserves(self):
        """group_commuting preserves fermion_mode_order."""
        qh = QubitHamiltonian(
            ["XX", "YY", "ZZ"],
            np.array([1.0, 0.5, -0.5]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        for group in qh.group_commuting(qubit_wise=True):
            assert group.fermion_mode_order == FermionModeOrder.BLOCKED

    def test_to_interleaved_sets_order(self):
        """to_interleaved sets fermion_mode_order to INTERLEAVED."""
        qh = QubitHamiltonian(
            ["IIIX", "ZZII"],
            np.array([0.5, 0.3]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        interleaved = qh.to_interleaved(n_spatial=2)
        assert interleaved.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_summary_includes_order(self):
        """get_summary includes fermion_mode_order when set."""
        qh = QubitHamiltonian(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        summary = qh.get_summary()
        assert "blocked" in summary
        assert "Fermion mode order" in summary

    def test_summary_omits_when_none(self):
        """get_summary omits fermion_mode_order when None."""
        qh = QubitHamiltonian(["IX", "ZZ"], np.array([0.5, 0.3]))
        summary = qh.get_summary()
        assert "Fermion mode order" not in summary
