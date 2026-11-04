"""Tests for the real orthogonal matrix and isometry unitary container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tracemalloc

import h5py
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import ortho_group

from qdk_chemistry.data import RealUnitaryContainer, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)


def _orthogonal(dimension: int, seed: int) -> np.ndarray:
    """Return a random real orthogonal matrix."""
    return ortho_group.rvs(dimension, random_state=seed)


def _block_sparse(seed: int) -> sparse.csc_array:
    """Return a row-permuted block-diagonal 8x8 orthogonal matrix with 2x2, 4x4, 1x1 and 1x1 blocks."""
    blocks = sparse.block_diag(
        [sparse.coo_array(_orthogonal(2, seed)), _orthogonal(4, seed + 1), [[1.0]], [[-1.0]]], format="csc"
    )
    return sparse.csc_array(blocks[np.random.default_rng(seed).permutation(8)])


def _signed_permutation(num_qubits: int, seed: int) -> sparse.csc_array:
    """Return a random signed permutation matrix on ``num_qubits`` qubits."""
    dimension = 1 << num_qubits
    permutation = np.random.default_rng(seed).permutation(dimension)
    return sparse.csc_array(
        (np.where(permutation % 2 == 0, 1.0, -1.0), permutation, np.arange(dimension + 1)),
        shape=(dimension, dimension),
    )


class TestValidation:
    """Construction accepts real isometries and rejects everything else with a clear error."""

    @pytest.mark.parametrize(("rows", "cols"), [(2, 1), (2, 2), (4, 3), (8, 8)])
    def test_dense_isometry(self, rows, cols):
        """Leading columns of an orthogonal matrix are accepted and stored read-only."""
        matrix = _orthogonal(rows, rows + cols)[:, :cols]
        container = RealUnitaryContainer(matrix)

        assert container.type == "real_unitary"
        assert container.num_qubits == rows.bit_length() - 1
        assert container.num_columns == cols
        assert container.is_square == (rows == cols)
        assert not container.is_sparse
        assert container.matrix.dtype == np.float64
        assert not container.matrix.flags.writeable
        np.testing.assert_array_equal(container.to_dense(), matrix)

    def test_input_is_copied(self):
        """Mutating the caller's array after construction does not change the container."""
        matrix = np.eye(4)
        container = RealUnitaryContainer(matrix)
        matrix[0, 0] = 5.0
        np.testing.assert_array_equal(container.to_dense(), np.eye(4))

    @pytest.mark.parametrize("stored_sparse", [False, True], ids=["dense", "sparse"])
    def test_matrix_is_returned_by_copy(self, stored_sparse):
        """Mutating the returned matrix leaves the container, its hash and its JSON unchanged."""
        matrix = _block_sparse(4)
        expected = matrix.toarray()
        container = RealUnitaryContainer(matrix if stored_sparse else expected)
        content_hash = container.content_hash()
        serialized = container.to_json()

        returned = container.matrix
        if stored_sparse:
            returned.data[:] = 0.0
            returned.indices[:] = 0
        else:
            assert not returned.flags.writeable
            returned.setflags(write=True)
            returned[:] = 0.0

        np.testing.assert_array_equal(container.to_dense(), expected)
        np.testing.assert_array_equal(
            container.matrix.toarray() if stored_sparse else container.matrix,
            expected,
        )
        assert container.content_hash() == content_hash
        assert container.to_json() == serialized

    def test_sparse_validation_stays_sparse(self):
        """Validating a wide sparse matrix never forms a dense Gram matrix or identity."""
        matrix = _signed_permutation(12, seed=0)
        tracemalloc.start()
        try:
            container = RealUnitaryContainer(matrix)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert container.num_qubits == 12
        # A dense 4096 x 4096 float64 array alone takes 128 MiB.
        assert peak < 8 * 1024 * 1024, f"validation peaked at {peak / 2**20:.1f} MiB"

    @pytest.mark.parametrize("kind", ["csc_array", "csr_matrix", "coo_array"])
    def test_sparse_is_canonical_csc(self, kind):
        """Any SciPy sparse input is stored as a canonical CSC array without explicit zeros."""
        dense = _block_sparse(3).toarray()
        matrix = getattr(sparse, kind)(dense)
        container = RealUnitaryContainer(matrix)

        assert container.is_sparse
        assert isinstance(container.matrix, sparse.csc_array)
        assert container.matrix.has_canonical_format
        assert container.matrix.nnz == np.count_nonzero(dense)
        np.testing.assert_array_equal(container.to_dense(), dense)

    def test_integer_permutation_is_accepted(self):
        """An integer permutation matrix is a valid orthogonal matrix."""
        container = RealUnitaryContainer(np.eye(4, dtype=int)[[2, 0, 3, 1]])
        assert container.matrix.dtype == np.float64

    @pytest.mark.parametrize(
        "matrix",
        [np.eye(2, dtype=complex), sparse.csc_array(np.eye(2, dtype=complex)), np.array([["a", "b"], ["c", "d"]])],
        ids=["complex", "sparse-complex", "string"],
    )
    def test_rejects_non_real(self, matrix):
        """Complex and non-numeric matrices raise TypeError."""
        with pytest.raises(TypeError, match="real"):
            RealUnitaryContainer(matrix)

    @pytest.mark.parametrize(
        ("matrix", "match"),
        [
            (np.ones(4) / 2, "two-dimensional"),
            (np.eye(3), "power-of-two row count"),
            (np.eye(1), "power-of-two row count"),
            (np.zeros((4, 0)), "between 1 and 4 columns"),
            (np.vstack([np.eye(2), np.zeros((2, 2))]).T, "between 1 and 2 columns"),
            (np.full((2, 2), np.nan), "finite"),
            (np.array([[1.0, 1.0], [0.0, 1.0]]), "orthonormal"),
            (2.0 * np.eye(2), "orthonormal"),
            (sparse.csc_array(np.array([[1.0, 0.0], [0.0, 0.5]])), "orthonormal"),
            (sparse.csc_array(np.array([[1.0, 1.0], [0.0, 1.0]]) / np.array([1.0, np.sqrt(2.0)])), "orthonormal"),
        ],
        ids=[
            "1d",
            "three-rows",
            "one-row",
            "no-columns",
            "wide",
            "nan",
            "non-orthogonal",
            "not-normalized",
            "sparse-not-normalized",
            "sparse-non-orthogonal",
        ],
    )
    def test_rejects_invalid(self, matrix, match):
        """Bad shapes, non-finite values and non-orthonormal columns raise ValueError."""
        with pytest.raises(ValueError, match=match):
            RealUnitaryContainer(matrix)

    def test_tolerates_rounding(self):
        """Columns orthonormal to within rounding are accepted."""
        matrix = _orthogonal(8, 1)[:, :5] + 1e-12
        RealUnitaryContainer(matrix)


class TestSerialization:
    """JSON and HDF5 round trips preserve the matrix exactly, dense or sparse."""

    @pytest.fixture(params=["dense-square", "dense-isometry", "sparse-square", "sparse-isometry"])
    def container(self, request):
        """A container of each storage format and shape."""
        storage, shape = request.param.split("-")
        matrix = _block_sparse(5)
        if shape == "isometry":
            matrix = sparse.csc_array(matrix[:, :3])
        return RealUnitaryContainer(matrix if storage == "sparse" else matrix.toarray())

    @staticmethod
    def _assert_same(restored, container):
        """Assert that two containers store the same matrix in the same format."""
        assert isinstance(restored, RealUnitaryContainer)
        assert restored.is_sparse == container.is_sparse
        assert restored.matrix.shape == container.matrix.shape
        np.testing.assert_array_equal(restored.to_dense(), container.to_dense())
        assert restored.content_hash() == container.content_hash()

    def test_json_roundtrip(self, container):
        """The container survives a JSON text round trip, directly and through UnitaryRepresentation."""
        data = json.loads(json.dumps(container.to_json()))
        assert data["container_type"] == "real_unitary"
        self._assert_same(RealUnitaryContainer.from_json(data), container)
        self._assert_same(UnitaryRepresentation.from_json(data).get_container(), container)

    def test_hdf5_roundtrip(self, container, tmp_path):
        """The container survives an HDF5 round trip, directly and through UnitaryRepresentation."""
        path = tmp_path / "real_unitary.h5"
        with h5py.File(path, "w") as file:
            UnitaryRepresentation(container).to_hdf5(file.create_group("unitary"))
        with h5py.File(path, "r") as file:
            self._assert_same(RealUnitaryContainer.from_hdf5(file["unitary"]), container)
            self._assert_same(UnitaryRepresentation.from_hdf5(file["unitary"]).get_container(), container)

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_file_roundtrip(self, container, tmp_path, file_format):
        """The container survives the DataClass file round trip."""
        path = tmp_path / f"matrix.real_unitary_container.{file_format}"
        container.to_file(path, file_format)
        self._assert_same(RealUnitaryContainer.from_file(path, file_format), container)

    def test_unknown_format_raises(self, container):
        """An unknown storage format in the serialized data is rejected."""
        data = container.to_json()
        data["format"] = "banded"
        with pytest.raises(ValueError, match="storage format"):
            RealUnitaryContainer.from_json(data)

    def test_hash_distinguishes_storage(self):
        """The same matrix stored dense and sparse hashes differently, and equal inputs hash equally."""
        matrix = _block_sparse(9)
        assert RealUnitaryContainer(matrix).content_hash() == RealUnitaryContainer(matrix.copy()).content_hash()
        assert RealUnitaryContainer(matrix).content_hash() != RealUnitaryContainer(matrix.toarray()).content_hash()


class TestOperations:
    """``combine``, the summary and the unsupported eigenvalue conversion."""

    def test_combine_appends_other(self):
        """``a.combine(b)`` applies ``a`` first, so its matrix is ``b @ a``."""
        first = _orthogonal(4, 1)[:, :2]
        second = _orthogonal(4, 2)
        combined = RealUnitaryContainer(first).combine(RealUnitaryContainer(second))
        assert not combined.is_sparse
        np.testing.assert_allclose(combined.to_dense(), second @ first, atol=1e-14)

    def test_combine_keeps_sparse(self):
        """Combining two sparse containers stays sparse."""
        first, second = _block_sparse(1), _block_sparse(2)
        combined = RealUnitaryContainer(first).combine(RealUnitaryContainer(second))
        assert combined.is_sparse
        np.testing.assert_allclose(combined.to_dense(), (second @ first).toarray(), atol=1e-14)

    @pytest.mark.parametrize("sparse_first", [True, False], ids=["sparse-then-dense", "dense-then-sparse"])
    def test_combine_mixed_storage(self, sparse_first):
        """Combining dense and sparse containers gives the dense product in either order."""
        sparse_matrix, dense_matrix = _block_sparse(3), _orthogonal(8, 4)
        first, second = (sparse_matrix, dense_matrix) if sparse_first else (dense_matrix, sparse_matrix)
        combined = RealUnitaryContainer(first).combine(RealUnitaryContainer(second))
        assert not combined.is_sparse
        expected = second @ first.toarray() if sparse_first else second.toarray() @ first
        np.testing.assert_allclose(combined.to_dense(), expected, atol=1e-14)

    def test_combine_mixed_storage_keeps_sparse_factor_sparse(self):
        """Appending a wide sparse unitary to a dense column never densifies the sparse factor."""
        permutation = _signed_permutation(12, seed=1)
        column = np.random.default_rng(1).normal(size=(permutation.shape[0], 1))
        column /= np.linalg.norm(column)
        state, other = RealUnitaryContainer(column), RealUnitaryContainer(permutation)
        tracemalloc.start()
        try:
            combined = state.combine(other)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        np.testing.assert_array_equal(combined.to_dense(), permutation @ column)
        # A dense 4096 x 4096 float64 array alone takes 128 MiB.
        assert peak < 8 * 1024 * 1024, f"combine peaked at {peak / 2**20:.1f} MiB"

    def test_combine_rejects_mismatch(self):
        """Different widths, a non-square second factor or another container type are rejected."""
        container = RealUnitaryContainer(np.eye(4))
        with pytest.raises(ValueError, match="2 and 1 qubits"):
            container.combine(RealUnitaryContainer(np.eye(2)))
        with pytest.raises(ValueError, match="square"):
            container.combine(RealUnitaryContainer(np.eye(4)[:, :2]))
        other = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, 0.1)], 1, 2)
        with pytest.raises(TypeError, match="PauliProductFormulaContainer"):
            container.combine(other)

    def test_summary(self):
        """The summary names the kind, width, shape and storage."""
        assert RealUnitaryContainer(np.eye(4)).get_summary() == "Real orthogonal matrix (2 qubits, 4x4, dense)"
        summary = UnitaryRepresentation(RealUnitaryContainer(_block_sparse(4)[:, :3])).get_summary()
        assert summary.startswith("Real isometry (3 qubits, 8x3, sparse, ")

    def test_eigenvalue_from_phase_is_unsupported(self):
        """There is no Hamiltonian to convert a phase into an eigenvalue for."""
        with pytest.raises(NotImplementedError):
            RealUnitaryContainer(np.eye(2)).eigenvalue_from_phase(0.25)

    def test_representation_reports_width(self):
        """``UnitaryRepresentation`` reports the register width as both total and system qubits."""
        representation = UnitaryRepresentation(RealUnitaryContainer(np.eye(8)[:, :3]))
        assert representation.get_container_type() == "real_unitary"
        assert representation.get_num_qubits() == 3
        assert representation.get_num_system_qubits() == 3
