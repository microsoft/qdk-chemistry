"""Test Hamiltonian loading and grouping functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import random as stdlib_random
import re
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import scipy.sparse

from qdk_chemistry.algorithms import registry
from qdk_chemistry.data import SparsePauliTerms, TaperingSpecification
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.data.term_partition import FlatPartition, LayeredPartition

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


def _sparse_copy(operator: QubitOperator) -> QubitOperator:
    """Copy an operator into sparse storage while retaining its metadata."""
    return QubitOperator.from_sparse_terms(
        operator.num_qubits,
        (term for term, _ in operator.iter_sparse_terms()),
        operator.coefficients,
        encoding=operator.encoding,
        fermion_mode_order=operator.fermion_mode_order,
        term_partition=operator.term_partition,
        tapering=operator.tapering,
    )


class TestQubitHamiltonian:
    """Test suite for QubitOperator data class."""

    def test_initialization(self):
        """Test that QubitOperator initializes correctly."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
        assert qubit_hamiltonian.pauli_strings == pauli_strings
        assert np.array_equal(qubit_hamiltonian.coefficients, coefficients)
        assert qubit_hamiltonian.num_qubits == 2

    def test_initialization_mismatch(self):
        """Test that initialization raises ValueError on mismatched lengths."""
        with pytest.raises(ValueError, match=r"Mismatch between number of Pauli strings and coefficients\."):
            QubitOperator(pauli_strings=["X", "Y"], coefficients=np.array([1.0]))

    def test_initialization_invalid_pauli(self):
        """Test that initialization raises ValueError on invalid Pauli strings."""
        with pytest.raises(ValueError, match="invalid characters"):
            QubitOperator(pauli_strings=["X", "A"], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="has length"):
            QubitOperator(pauli_strings=["X", "ZY"], coefficients=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="invalid characters"):
            QubitOperator(pauli_strings=["XZ", "A1"], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="empty"):
            QubitOperator(pauli_strings=["X", ""], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="empty"):
            QubitOperator(pauli_strings=[], coefficients=[])

    def test_sparse_terms_match_dense_operator(self):
        dense = QubitOperator(["IX", "YY", "ZI"], np.array([1.0, -0.5, 0.75]))
        sparse = QubitOperator.from_sparse_terms(
            2,
            [((0, "X"),), ((0, "Y"), (1, "Y")), ((1, "Z"),)],
            np.array([1.0, -0.5, 0.75]),
        )

        assert sparse.pauli_strings == dense.pauli_strings
        assert sparse.num_qubits == 2
        assert sparse.num_terms == 3
        assert sparse.equiv(dense)
        np.testing.assert_array_equal(sparse.to_matrix(), dense.to_matrix())
        assert (2.0 * sparse).pauli_strings == dense.pauli_strings

    def test_sparse_terms_validate_factors(self):
        with pytest.raises(ValueError, match="in-range integer indices"):
            QubitOperator.from_sparse_terms(2, [((2, "X"),)], np.array([1.0]))
        with pytest.raises(ValueError, match="X/Y/Z axes"):
            QubitOperator.from_sparse_terms(2, [((0, "A"),)], np.array([1.0]))

    def test_content_hash_includes_fermion_mode_order(self):
        """Content hash changes when fermion_mode_order changes."""
        blocked = QubitOperator(["IX", "ZI"], np.array([1.0, 0.5]), fermion_mode_order="blocked")
        interleaved = QubitOperator(["IX", "ZI"], np.array([1.0, 0.5]), fermion_mode_order="interleaved")
        assert blocked.content_hash() != interleaved.content_hash()

    def test_content_hash_includes_term_partition(self):
        """Content hash changes when term_partition changes."""
        grouped = QubitOperator(
            ["IX", "ZI"],
            np.array([1.0, 0.5]),
            term_partition=FlatPartition(strategy="commuting", groups=((0, 1),)),
        )
        ungrouped = QubitOperator(
            ["IX", "ZI"],
            np.array([1.0, 0.5]),
            term_partition=FlatPartition(strategy="commuting", groups=((0,), (1,))),
        )
        assert grouped.content_hash() != ungrouped.content_hash()

    def test_content_hash_includes_tapering(self):
        """Content hash changes when tapering metadata changes."""
        from qdk_chemistry.data import TaperingSpecification  # noqa: PLC0415

        h1 = QubitOperator(
            ["IX", "ZI"],
            np.array([1.0, 0.5]),
            tapering=TaperingSpecification(qubit_indices=(3, 1), eigenvalues=(1, -1)),
        )
        h2 = QubitOperator(
            ["IX", "ZI"],
            np.array([1.0, 0.5]),
            tapering=TaperingSpecification(qubit_indices=(3, 1), eigenvalues=(1, 1)),
        )
        assert h1.content_hash() != h2.content_hash()

    def test_group_commuting(self):
        """Test full-commuting term grouper produces correct groups."""
        qubit_hamiltonian = QubitOperator(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = registry.create("term_grouper", "commuting").run(qubit_hamiltonian)
        partition = grouped.term_partition
        assert isinstance(partition, FlatPartition)
        assert partition.num_groups == 2

        # Verify coefficients are preserved
        for group_indices in partition.groups:
            for idx in group_indices:
                assert np.isclose(
                    grouped.coefficients[idx],
                    qubit_hamiltonian.coefficients[idx],
                    atol=float_comparison_absolute_tolerance,
                    rtol=float_comparison_relative_tolerance,
                )

    def test_group_commuting_qubitwise(self):
        """Test qubit-wise commuting grouper."""
        qubit_hamiltonian = QubitOperator(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = registry.create("term_grouper", "qubit_wise_commuting").run(qubit_hamiltonian)
        partition = grouped.term_partition
        assert isinstance(partition, FlatPartition)
        assert partition.num_groups == 4

        # Each group should contain exactly one term
        for group_indices in partition.groups:
            assert len(group_indices) == 1
        # All original terms are present
        all_indices = sorted(partition.all_indices())
        assert all_indices == list(range(4))

    def test_group_commuting_all_commute(self):
        """Test that fully commuting operators go into one group."""
        qh = QubitOperator(["ZI", "IZ", "ZZ"], np.array([1.0, -0.5, 0.3]))
        grouped = registry.create("term_grouper", "commuting").run(qh)
        assert grouped.term_partition.num_groups == 1
        assert len(grouped.term_partition.groups[0]) == 3

    def test_group_commuting_none_commute(self):
        """Test that non-commuting operators each get their own group."""
        qh = QubitOperator(["X", "Z", "Y"], np.array([1.0, -0.5, 0.3]))
        grouped = registry.create("term_grouper", "commuting").run(qh)
        assert grouped.term_partition.num_groups == 3

    def test_group_commuting_single_term(self):
        """Test grouping with a single term."""
        qh = QubitOperator(["ZZ"], np.array([1.0]))
        grouped = registry.create("term_grouper", "commuting").run(qh)
        assert grouped.term_partition.num_groups == 1
        assert grouped.pauli_strings == ["ZZ"]

    def test_group_commuting_reconstruct_matrix(self):
        """Test that grouped terms reconstruct the same matrix."""
        qh = QubitOperator(
            ["II", "IZ", "ZI", "ZZ", "XX", "YY"],
            np.array([-0.8, 0.17, -0.17, 0.12, 0.04, 0.04]),
        )
        grouped = registry.create("term_grouper", "commuting").run(qh)
        partition = grouped.term_partition
        total_terms = sum(len(g) for g in partition.groups)
        assert total_terms == 6

        mat = qh.to_matrix()
        gs_energy = np.min(np.linalg.eigvalsh(mat))
        # Reconstruct from groups and check same ground state energy
        full_mat = np.zeros_like(mat)
        for group_indices in partition.groups:
            sub = QubitOperator(
                [grouped.pauli_strings[i] for i in group_indices],
                np.array([grouped.coefficients[i] for i in group_indices]),
            )
            full_mat += sub.to_matrix()
        gs_energy_grouped = np.min(np.linalg.eigvalsh(full_mat))
        assert np.isclose(gs_energy, gs_energy_grouped, atol=float_comparison_absolute_tolerance)

    def test_group_commuting_qw_reconstruct_matrix(self):
        """Test that QW-grouped Hamiltonian reconstructs the original matrix exactly."""
        labels = ["ZI", "IZ", "ZZ", "XI", "IX", "XX", "YY"]
        coeffs = np.array([0.5, 0.3, 0.2, -0.1, 0.4, -0.25, 0.15])
        qh = QubitOperator(labels, coeffs)
        original_mat = qh.to_matrix()

        grouped = registry.create("term_grouper", "qubit_wise_commuting").run(qh)
        partition = grouped.term_partition
        reconstructed = np.zeros_like(original_mat)
        for group_indices in partition.groups:
            sub = QubitOperator(
                [grouped.pauli_strings[i] for i in group_indices],
                np.array([grouped.coefficients[i] for i in group_indices]),
            )
            reconstructed += sub.to_matrix()
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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)
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
        qh = QubitOperator(["XYZZ"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert interleaved.pauli_strings == ["XZYZ"]

    def test_to_interleaved_preserves_coefficients(self):
        """Test that interleaving preserves coefficient values."""
        qh = QubitOperator(["XIZI", "IYII"], np.array([0.5 + 0.1j, 0.3], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert np.allclose(interleaved.coefficients, qh.coefficients)

    def test_to_interleaved_invalid_n_spatial(self):
        """Test that invalid n_spatial raises error."""
        qh = QubitOperator(["XIZI"], np.array([1.0], dtype=complex))
        with pytest.raises(ValueError, match=re.escape("must be 2 * n_spatial")):
            qh.to_interleaved(n_spatial=3)

    def test_to_interleaved_single_orbital(self):
        """Test that single spatial orbital (2 qubits) is unchanged."""
        qh = QubitOperator(["XY"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=1)
        assert interleaved.pauli_strings == ["XY"]

    def test_to_matrix_hermitian(self):
        """Test that to_matrix produces a Hermitian matrix for real coefficients."""
        qh = QubitOperator(["IX", "ZI", "ZZ", "YY"], np.array([0.5, -0.3, 0.8, -0.2]))
        mat = qh.to_matrix()
        assert np.allclose(
            mat, mat.conj().T, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_to_matrix(self):
        """Test to_matrix returns a matrix matching reference."""
        labels = ["IX", "ZZ", "YY"]
        coeffs = np.array([0.5, -0.3, 0.1])
        qh = QubitOperator(labels, coeffs)
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
        qh = QubitOperator(["ZI", "IZ", "XX"], np.array([0.7, -0.4, 0.3]))
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
        qh = QubitOperator(labels, coeffs)
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
        qh = QubitOperator(labels, coeffs)
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
    """Test suite for QubitOperator serialization (JSON and HDF5)."""

    def test_json_serialization_real_coefficients(self):
        """Test JSON serialization with real coefficients."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)

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
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)

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
        original = QubitOperator(pauli_strings, coefficients)

        # Roundtrip through JSON
        json_data = original.to_json()
        reconstructed = QubitOperator.from_json(json_data)

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_json_roundtrip_complex_coefficients(self):
        """Test JSON roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitOperator(pauli_strings, coefficients)

        # Roundtrip through JSON
        json_data = original.to_json()
        reconstructed = QubitOperator.from_json(json_data)

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_sparse_roundtrip(self, tmp_path, file_format):
        """PR 0.2.0 words, including identity, round-trip without packed keys."""
        original = QubitOperator.from_sparse_terms(
            4,
            [((0, "X"), (3, "Z")), ((1, "Y"),), ()],
            np.array([1.0, -0.5j, 0.25]),
        )
        filename = tmp_path / f"test.qubit_hamiltonian.{file_format}"
        original.to_file(str(filename), file_format)
        reconstructed = QubitOperator.from_file(str(filename), file_format)

        assert original.to_json()["version"] == "0.2.0"
        assert "pauli_terms" in original.to_json()
        assert not {"term_offsets", "qubit_indices", "pauli_codes", "pauli_strings"} & original.to_json().keys()
        assert reconstructed.has_sparse_terms
        assert reconstructed.content_hash(0) == original.content_hash(0)
        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_equal(reconstructed.coefficients, original.coefficients)

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    @pytest.mark.parametrize("identity_only", [False, True])
    def test_old_packed_load_writes_canonical(self, tmp_path, file_format, identity_only):
        """Old 0.2.0 arrays become canonical words, preserving width, coefficients and metadata."""
        partition = LayeredPartition(strategy="legacy", groups=(((2, 0),), ((1,),)))
        tapering = TaperingSpecification(qubit_indices=(3, 1), eigenvalues=(1, -1))
        coefficients = np.array([1.0, -0.5j, 0.25])
        payload = {
            "version": "0.2.0",
            "num_qubits": 4,
            "term_offsets": [0, 0, 0, 0] if identity_only else [0, 2, 2, 3],
            "qubit_indices": [] if identity_only else [0, 3, 1],
            "pauli_codes": [] if identity_only else [1, 3, 2],
            "coefficients": {"real": coefficients.real.tolist(), "imag": coefficients.imag.tolist()},
            "encoding": "jordan-wigner",
            "fermion_mode_order": "blocked",
            "term_partition": partition.to_json(),
            "tapering": tapering.to_json(),
        }
        if file_format == "json":
            restored = QubitOperator.from_json(json.loads(json.dumps(payload)))
        else:
            with h5py.File(tmp_path / "old.h5", "w") as group:
                for key in ("version", "num_qubits", "encoding", "fermion_mode_order"):
                    group.attrs[key] = payload[key]
                for key in ("term_partition", "tapering"):
                    group.attrs[key] = json.dumps(payload[key])
                for key in ("term_offsets", "qubit_indices", "pauli_codes"):
                    group.create_dataset(key, data=payload[key])
                group.create_dataset("coefficients", data=coefficients)
                restored = QubitOperator.from_hdf5(group)
        expected = QubitOperator.from_sparse_terms(
            4,
            [{}, {}, {}] if identity_only else [{0: "X", 3: "Z"}, {}, {1: "Y"}],
            coefficients,
            encoding="jordan-wigner",
            fermion_mode_order="blocked",
            term_partition=partition,
            tapering=tapering,
        )
        assert isinstance(restored.pauli_strings, SparsePauliTerms)
        assert restored.to_json() == expected.to_json()
        assert restored.content_hash(0) == expected.content_hash(0)
        filename = tmp_path / f"canonical.qubit_hamiltonian.{file_format}"
        restored.to_file(filename, file_format)
        if file_format == "json":
            written = json.loads(filename.read_text())
            assert written == expected.to_json()
            assert "term_offsets" not in written
        else:
            with h5py.File(filename, "r") as group:
                assert group.attrs["version"] == "0.2.0"
                assert "pauli_terms" in group
                assert not {"term_offsets", "qubit_indices", "pauli_codes", "pauli_strings"} & group.keys()
        assert QubitOperator.from_file(filename, file_format).content_hash(0) == expected.content_hash(0)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("term_offsets", [0, 2, 1]),
            ("qubit_indices", [[0]]),
            ("qubit_indices", [0.5]),
            ("pauli_codes", [4]),
        ],
    )
    def test_old_packed_rejects_invalid_arrays(self, key, value):
        """Malformed packed payloads must fail rather than truncate or reinterpret factors."""
        payload = {
            "version": "0.2.0",
            "num_qubits": 2,
            "term_offsets": [0, 1],
            "qubit_indices": [0],
            "pauli_codes": [1],
            "coefficients": [1.0],
        }
        payload[key] = value
        with pytest.raises(ValueError, match="packed sparse Pauli"):
            QubitOperator.from_json(payload)

    def test_json_file_roundtrip_complex_coefficients(self, tmp_path):
        """Test JSON file roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitOperator(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.json"
        original.to_json_file(str(filename))

        # Load and verify
        reconstructed = QubitOperator.from_json_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_hdf5_roundtrip_real_coefficients(self, tmp_path):
        """Test HDF5 roundtrip with real coefficients."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        original = QubitOperator(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        # Load and verify
        reconstructed = QubitOperator.from_hdf5_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_hdf5_roundtrip_complex_coefficients(self, tmp_path):
        """Test HDF5 roundtrip with complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        original = QubitOperator(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        # Load and verify
        reconstructed = QubitOperator.from_hdf5_file(str(filename))

        assert reconstructed.pauli_strings == original.pauli_strings
        np.testing.assert_array_almost_equal(reconstructed.coefficients, original.coefficients)

    def test_json_to_json_file_no_complex_error(self, tmp_path):
        """Regression test: to_json_file must not raise TypeError for complex coefficients."""
        pauli_strings = ["IX", "YY", "ZZ", "XY"]
        coefficients = np.array([1.0 + 0.5j, -0.5 - 0.25j, 0.75j, 2.0])
        qubit_hamiltonian = QubitOperator(pauli_strings, coefficients)

        filename = tmp_path / "test.qubit_hamiltonian.json"

        # This should not raise TypeError: Object of type complex is not JSON serializable
        qubit_hamiltonian.to_json_file(str(filename))

        # Verify the file can be read
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)

        assert "pauli_strings" in data
        assert "coefficients" in data


class TestFermionModeOrder:
    """Test suite for fermion_mode_order metadata on QubitOperator."""

    def test_default_is_none(self):
        """fermion_mode_order defaults to None when not specified."""
        qh = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]))
        assert qh.fermion_mode_order is None

    def test_set_blocked(self):
        """fermion_mode_order can be set to BLOCKED."""
        qh = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order=FermionModeOrder.BLOCKED)
        assert qh.fermion_mode_order == FermionModeOrder.BLOCKED
        assert qh.fermion_mode_order == "blocked"

    def test_set_interleaved(self):
        """fermion_mode_order can be set to INTERLEAVED."""
        qh = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order=FermionModeOrder.INTERLEAVED)
        assert qh.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_set_from_string(self):
        """fermion_mode_order accepts a raw string and coerces to the enum."""
        qh = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]), fermion_mode_order="blocked")
        assert qh.fermion_mode_order is FermionModeOrder.BLOCKED

    def test_json_roundtrip(self):
        """fermion_mode_order survives JSON serialization."""
        original = QubitOperator(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            encoding="jordan-wigner",
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        json_data = original.to_json()
        assert json_data["fermion_mode_order"] == "blocked"

        restored = QubitOperator.from_json(json_data)
        assert restored.fermion_mode_order == FermionModeOrder.BLOCKED

    def test_json_roundtrip_none(self):
        """fermion_mode_order=None is omitted from JSON and restored as None."""
        original = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]))
        json_data = original.to_json()
        assert "fermion_mode_order" not in json_data

        restored = QubitOperator.from_json(json_data)
        assert restored.fermion_mode_order is None

    def test_hdf5_roundtrip(self, tmp_path):
        """fermion_mode_order survives HDF5 serialization."""
        original = QubitOperator(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            encoding="jordan-wigner",
            fermion_mode_order=FermionModeOrder.INTERLEAVED,
        )
        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        restored = QubitOperator.from_hdf5_file(str(filename))
        assert restored.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_hdf5_roundtrip_none(self, tmp_path):
        """fermion_mode_order=None is omitted from HDF5 and restored as None."""
        original = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]))
        filename = tmp_path / "test.qubit_hamiltonian.h5"
        original.to_hdf5_file(str(filename))

        restored = QubitOperator.from_hdf5_file(str(filename))
        assert restored.fermion_mode_order is None

    def test_group_commuting_preserves(self):
        """term_grouper preserves fermion_mode_order."""
        qh = QubitOperator(
            ["XX", "YY", "ZZ"],
            np.array([1.0, 0.5, -0.5]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        grouped = registry.create("term_grouper", "qubit_wise_commuting").run(qh)
        assert grouped.fermion_mode_order == FermionModeOrder.BLOCKED

    def test_to_interleaved_sets_order(self):
        """to_interleaved sets fermion_mode_order to INTERLEAVED."""
        qh = QubitOperator(
            ["IIIX", "ZZII"],
            np.array([0.5, 0.3]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        interleaved = qh.to_interleaved(n_spatial=2)
        assert interleaved.fermion_mode_order == FermionModeOrder.INTERLEAVED

    def test_summary_includes_order(self):
        """get_summary includes fermion_mode_order when set."""
        qh = QubitOperator(
            ["IX", "ZZ"],
            np.array([0.5, 0.3]),
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        summary = qh.get_summary()
        assert "blocked" in summary
        assert "Fermion mode order" in summary

    def test_summary_omits_when_none(self):
        """get_summary omits fermion_mode_order when None."""
        qh = QubitOperator(["IX", "ZZ"], np.array([0.5, 0.3]))
        summary = qh.get_summary()
        assert "Fermion mode order" not in summary


class TestQubitHamiltonianArithmetic:
    """Tests for __add__, __mul__, __rmul__ and partition merging."""

    def test_add_concatenates_terms(self):
        """H1 + H2 should concatenate pauli_strings and coefficients."""
        h1 = QubitOperator(["XI"], np.array([1.0]))
        h2 = QubitOperator(["IZ"], np.array([2.0]))
        result = h1 + h2
        assert result.pauli_strings == ["XI", "IZ"]
        np.testing.assert_allclose(result.coefficients, [1.0, 2.0])

    @pytest.mark.parametrize("sparse", [False, True])
    def test_add_merges_flat_partitions(self, sparse):
        """__add__ should merge FlatPartitions with offset."""
        h1 = QubitOperator(
            ["XI", "IZ"], np.array([1.0, 1.0]), term_partition=FlatPartition(strategy="s", groups=((0, 1),))
        )
        h2 = QubitOperator(["XX"], np.array([0.5]), term_partition=FlatPartition(strategy="s", groups=((0,),)))
        if sparse:
            h1 = _sparse_copy(h1)
        result = h1 + h2
        assert result.term_partition is not None
        assert isinstance(result.term_partition, FlatPartition)
        assert result.term_partition.groups == ((0, 1), (2,))
        assert result.has_sparse_terms == sparse
        assert result.pauli_strings == ["XI", "IZ", "XX"]
        np.testing.assert_array_equal(result.coefficients, [1.0, 1.0, 0.5])

    @pytest.mark.parametrize("sparse", [False, True])
    def test_add_merges_layered_partitions(self, sparse):
        """__add__ should merge LayeredPartitions with offset."""
        h1 = QubitOperator(
            ["XI", "IZ"], np.array([1.0, 1.0]), term_partition=LayeredPartition(strategy="s", groups=(((0,), (1,)),))
        )
        h2 = QubitOperator(["XX"], np.array([0.5]), term_partition=LayeredPartition(strategy="s", groups=(((0,),),)))
        if sparse:
            h2 = _sparse_copy(h2)
        result = h1 + h2
        assert isinstance(result.term_partition, LayeredPartition)
        assert result.term_partition.groups == (((0,), (1,)), ((2,),))
        assert result.has_sparse_terms == sparse
        assert result.pauli_strings == ["XI", "IZ", "XX"]
        np.testing.assert_array_equal(result.coefficients, [1.0, 1.0, 0.5])

    def test_add_mismatched_partition_types_raises(self):
        """__add__ with FlatPartition + LayeredPartition should raise TypeError."""
        h1 = QubitOperator(["XI"], np.array([1.0]), term_partition=FlatPartition(strategy="s", groups=((0,),)))
        h2 = QubitOperator(["IZ"], np.array([1.0]), term_partition=LayeredPartition(strategy="s", groups=(((0,),),)))
        with pytest.raises(TypeError, match="Cannot merge partitions of different types"):
            h1 + h2

    def test_add_no_partition_when_either_missing(self):
        """__add__ should produce None partition when only one operand has one."""
        h1 = QubitOperator(["XI"], np.array([1.0]), term_partition=FlatPartition(strategy="s", groups=((0,),)))
        h2 = QubitOperator(["IZ"], np.array([1.0]))
        result = h1 + h2
        assert result.term_partition is None

    def test_add_preserves_encoding(self):
        """__add__ should propagate matching encoding."""
        h1 = QubitOperator(["XI"], np.array([1.0]), encoding="jordan-wigner")
        h2 = QubitOperator(["IZ"], np.array([2.0]), encoding="jordan-wigner")
        assert (h1 + h2).encoding == "jordan-wigner"

    def test_add_mismatched_encoding_raises(self):
        """__add__ with different encodings should raise ValueError."""
        h1 = QubitOperator(["XI"], np.array([1.0]), encoding="jordan-wigner")
        h2 = QubitOperator(["IZ"], np.array([2.0]), encoding="bravyi-kitaev")
        with pytest.raises(ValueError, match="different encodings"):
            h1 + h2

    def test_add_mismatched_qubits_raises(self):
        """__add__ with different qubit counts should raise ValueError."""
        h1 = QubitOperator(["XI"], np.array([1.0]))
        h2 = QubitOperator(["IIZ"], np.array([2.0]))
        with pytest.raises(ValueError, match="Cannot add"):
            h1 + h2

    def test_mul_scales_coefficients(self):
        """Scalar * H should scale coefficients."""
        h = QubitOperator(["XI", "IZ"], np.array([1.0, 2.0]))
        result = 3.0 * h
        np.testing.assert_allclose(result.coefficients, [3.0, 6.0])

    def test_mul_preserves_partition(self):
        """Scalar * H should keep the partition unchanged."""
        p = FlatPartition(strategy="s", groups=((0,), (1,)))
        h = QubitOperator(["XI", "IZ"], np.array([1.0, 2.0]), term_partition=p)
        result = 2.0 * h
        assert result.term_partition is not None
        assert result.term_partition.groups == p.groups

    def test_rmul_equals_mul(self):
        """H * scalar and scalar * H should give the same result."""
        h = QubitOperator(["XI"], np.array([3.0]))
        np.testing.assert_allclose((h * 2.0).coefficients, (2.0 * h).coefficients)


class TestTaperingPropagation:
    """Verify tapering metadata survives arithmetic and reordering."""

    @pytest.fixture
    def tapering(self):
        """Create a sample tapering specification."""
        from qdk_chemistry.data import TaperingSpecification  # noqa: PLC0415

        return TaperingSpecification(
            qubit_indices=(3, 1),
            eigenvalues=(1, -1),
        )

    @pytest.fixture(params=[False, True], ids=["dense", "sparse"])
    def tapered_h(self, tapering, request):
        """Create a QubitOperator with tapering metadata."""
        operator = QubitOperator(
            ["XI", "IZ"],
            np.array([1.0, 0.5]),
            encoding="symmetry-conserving-bravyi-kitaev",
            tapering=tapering,
        )
        return _sparse_copy(operator) if request.param else operator

    def test_add_preserves_tapering(self, tapered_h, tapering):
        """H1 + H2 with matching tapering should preserve it."""
        h2 = QubitOperator(
            ["XX"],
            np.array([0.3]),
            encoding="symmetry-conserving-bravyi-kitaev",
            tapering=tapering,
        )
        result = tapered_h + h2
        assert result.tapering == tapering

    def test_add_mismatched_tapering_raises(self, tapered_h):
        """H1 + H2 with different tapering should raise ValueError."""
        from qdk_chemistry.data import TaperingSpecification  # noqa: PLC0415

        other_tapering = TaperingSpecification(
            qubit_indices=(3, 1),
            eigenvalues=(1, 1),
        )
        h2 = QubitOperator(
            ["XX"],
            np.array([0.3]),
            encoding="symmetry-conserving-bravyi-kitaev",
            tapering=other_tapering,
        )
        with pytest.raises(ValueError, match="tapering"):
            tapered_h + h2

    def test_add_missing_tapering_raises_both_directions(self, tapered_h):
        """Missing tapering must raise ValueError rather than invoke its bound comparison."""
        untapered = QubitOperator(["XX"], np.array([0.3]), encoding=tapered_h.encoding)
        with pytest.raises(ValueError, match="tapering"):
            _ = tapered_h + untapered
        with pytest.raises(ValueError, match="tapering"):
            _ = untapered + tapered_h

    def test_mul_preserves_tapering(self, tapered_h, tapering):
        """Scalar * H should preserve tapering metadata."""
        result = 2.0 * tapered_h
        assert result.tapering == tapering
        result2 = tapered_h * 3.0
        assert result2.tapering == tapering

    def test_to_interleaved_preserves_tapering(self, tapering):
        """to_interleaved should preserve tapering metadata."""
        h = QubitOperator(
            ["XIZI", "IZIX"],
            np.array([1.0, 0.5]),
            encoding="symmetry-conserving-bravyi-kitaev",
            tapering=tapering,
        )
        result = h.to_interleaved(n_spatial=2)
        assert result.tapering == tapering


class TestSparseQubitOperator:
    """Sparse words preserve Pauli algebra without allocating dense labels."""

    def test_sparse_words_and_coefficients_are_isolated(self):
        """Own mutable inputs while sharing immutable factors during scalar multiplication."""
        coefficients = np.array([-1.25, 0.5])
        words = [{0: "Y"}, {}]
        sparse = QubitOperator.from_sparse_terms(3, words, coefficients)
        scaled = 2 * sparse
        before = sparse.content_hash()
        coefficients[:] = 0
        words[0][0] = "Z"
        with pytest.raises(AttributeError):
            sparse.pauli_strings.words = ()
        assert not np.shares_memory(scaled.coefficients, sparse.coefficients)
        assert not scaled.coefficients.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            sparse.coefficients[0] = 0
        assert sparse.content_hash() == before
        assert list(sparse.iter_sparse_terms()) == [(((0, "Y"),), -1.25 + 0j), ((), 0.5 + 0j)]

    @pytest.mark.parametrize("term", [[(True, "X")], [(0.5, "X")], [(2**32, "X")], [(0, "I")], [(0, "X"), (0, "Y")]])
    def test_sparse_factors_reject_invalid_values(self, term):
        """Reject invalid indices, axes and duplicate qubits."""
        with pytest.raises(ValueError, match="[Ss]parse Pauli"):
            QubitOperator.from_sparse_terms(2, [term], np.array([1.0]))

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_large_sparse_operations_never_access_full_labels(self, file_format, monkeypatch, tmp_path):
        """Arithmetic, reordering and persistence scale with support, not register width."""
        coefficients = np.array([1.25, -2j, 0.5])
        sparse = QubitOperator.from_sparse_terms(
            100_000,
            [{0: "X", 99_999: "Y"}, {65_536: "Z"}, {}],
            coefficients,
            term_partition=LayeredPartition(strategy="s", groups=(((2, 0),), ((1,),))),
        )
        monkeypatch.setattr(SparsePauliTerms, "__getitem__", Mock(side_effect=AssertionError("Dense labels")))
        assert (0.0 * sparse).get_real_coefficients() == []
        assert (sparse + (-0.5 * sparse)).equiv(0.5 * sparse)
        assert (2j * sparse).equiv(sparse * 2j)
        expected = QubitOperator.from_sparse_terms(100_000, [{0: "X", 99_999: "Y"}, {31_073: "Z"}, {}], coefficients)
        interleaved = sparse.to_interleaved(50_000)
        assert interleaved.equiv(expected)
        assert interleaved.fermion_mode_order is FermionModeOrder.INTERLEAVED
        assert interleaved.term_partition is None
        filename = tmp_path / f"large.qubit_hamiltonian.{file_format}"
        sparse.to_file(filename, file_format)
        restored = QubitOperator.from_file(filename, file_format)
        assert restored.to_json() == sparse.to_json()
        assert restored.content_hash(0) == sparse.content_hash(0)
        wider = QubitOperator.from_sparse_terms(100_001, sparse.pauli_strings.words, coefficients)
        assert not sparse.equiv(wider)
        assert sparse.content_hash() != wider.content_hash()

    def test_json_rejects_broadcasting_coefficient_components(self):
        """Do not silently broadcast a short imaginary component across sparse coefficients."""
        payload = QubitOperator.from_sparse_terms(2, [{0: "X"}, {}], np.array([1.0, 0.5])).to_json()
        payload["coefficients"]["imag"] = [0.0]
        with pytest.raises(ValueError, match="matching shapes"):
            QubitOperator.from_json(payload)
