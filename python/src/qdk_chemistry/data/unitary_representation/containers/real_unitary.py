"""Unitary container for a real orthogonal matrix or isometry given by its matrix."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from qdk_chemistry.data._hashing import _hash_array, _hash_bool, _hash_str
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer

if TYPE_CHECKING:
    import h5py

__all__ = ["RealUnitaryContainer"]

_ORTHOGONALITY_TOLERANCE = 1.0e-8  # Per-column Frobenius norm of M.T @ M - I, as in the C++ synthesis.


class RealUnitaryContainer(UnitaryContainer):
    r"""Container for a real orthogonal matrix, or the leading columns of one, stored dense or sparse.

    The matrix :math:`M` has :math:`2^n` rows for an :math:`n`-qubit register and
    :math:`1 \le m \le 2^n` orthonormal columns. A square :math:`M` is an orthogonal unitary.
    Otherwise :math:`M` is an isometry, and any circuit that synthesizes it realizes a unitary
    :math:`U` whose first :math:`m` columns equal :math:`M`; the other columns of :math:`U` are
    unspecified.

    Entry ``matrix[r, c]`` is :math:`\langle r | U | c \rangle` for little-endian basis
    states: bit :math:`k` of the index is qubit :math:`k` of the register. Column :math:`c < m`
    is therefore the image of the register value :math:`c`, so an isometry with
    :math:`m \le 2^{n-k}` columns maps the low :math:`n - k` qubits, with the high :math:`k`
    qubits in :math:`|0\rangle`, into the full register.

    Only real matrices are supported. Sparse input is stored as a canonical
    :class:`scipy.sparse.csc_array` without explicit zeros.

    Args:
        matrix: Real matrix with orthonormal columns, as a NumPy array or a SciPy sparse
            array or matrix.

    Raises:
        TypeError: If the matrix is complex or not numeric.
        ValueError: If the matrix is not two-dimensional or finite, its row count is not a
            power of two of at least two, it has no columns or more columns than rows, or its
            columns are not orthonormal.

    """

    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for real unitary containers.

        Returns:
            ``"real_unitary_container"``.

        """
        return "real_unitary_container"

    def __init__(self, matrix: np.ndarray | sparse.sparray | sparse.spmatrix) -> None:
        """Initialize the container."""
        stored = self._canonical_sparse(matrix) if sparse.issparse(matrix) else self._canonical_dense(matrix)
        self._validate_isometry(stored)
        self._matrix = stored
        super().__init__()

    @staticmethod
    def _check_dtype(dtype: np.dtype) -> None:
        """Reject complex and non-numeric dtypes."""
        if np.issubdtype(dtype, np.complexfloating):
            raise TypeError("RealUnitaryContainer requires a real matrix, got a complex one.")
        if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer) or dtype == np.bool_):
            raise TypeError(f"RealUnitaryContainer requires a real numeric matrix, got dtype {dtype}.")

    @classmethod
    def _canonical_dense(cls, matrix: Any) -> np.ndarray:
        """Return a read-only float64 copy of a dense matrix."""
        array = np.asarray(matrix)
        cls._check_dtype(array.dtype)
        array = np.array(array, dtype=np.float64, order="C")
        array.setflags(write=False)
        return array

    @classmethod
    def _canonical_sparse(cls, matrix: sparse.sparray | sparse.spmatrix) -> sparse.csc_array:
        """Return a canonical float64 CSC copy of a sparse matrix without explicit zeros."""
        cls._check_dtype(matrix.dtype)
        array = sparse.csc_array(matrix, dtype=np.float64, copy=True)
        array.sum_duplicates()
        array.eliminate_zeros()
        array.sort_indices()
        return array

    @staticmethod
    def _validate_isometry(matrix: np.ndarray | sparse.csc_array) -> None:
        """Check the shape, finiteness, and orthonormal columns of a canonical matrix."""
        if matrix.ndim != 2:
            raise ValueError(f"RealUnitaryContainer requires a two-dimensional matrix, got {matrix.ndim} dimensions.")
        rows, cols = matrix.shape
        if rows < 2 or rows & (rows - 1):
            raise ValueError(f"RealUnitaryContainer requires a power-of-two row count of at least two, got {rows}.")
        if not 1 <= cols <= rows:
            raise ValueError(f"RealUnitaryContainer requires between 1 and {rows} columns, got {cols}.")
        values = matrix.data if sparse.issparse(matrix) else matrix
        if not np.all(np.isfinite(values)):
            raise ValueError("RealUnitaryContainer requires a finite matrix.")
        if sparse.issparse(matrix):
            # Keep the residual sparse: a dense m-by-m Gram matrix would dominate memory for wide isometries.
            identity = sparse.csc_array(
                (np.ones(cols), np.arange(cols), np.arange(cols + 1)), shape=(cols, cols), dtype=np.float64
            )
            residual = float(np.linalg.norm(sparse.csc_array(matrix.T @ matrix - identity).data))
        else:
            residual = float(np.linalg.norm(matrix.T @ matrix - np.eye(cols)))
        if not residual <= _ORTHOGONALITY_TOLERANCE * cols:
            raise ValueError(
                "RealUnitaryContainer requires orthonormal columns; "
                f"the Frobenius norm of M.T @ M - I is {residual:.3e}."
            )

    @property
    def type(self) -> str:
        """Return the container type."""
        return "real_unitary"

    @property
    def matrix(self) -> np.ndarray | sparse.csc_array:
        """Return a copy of the stored matrix, as a read-only NumPy array or a SciPy CSC array.

        Both storages are copied to keep the validated container immutable: a read-only NumPy
        view can be made writable again with ``setflags(write=True)``, and SciPy sparse arrays
        cannot be made read-only.
        """
        if self.is_sparse:
            return self._matrix.copy()
        matrix = self._matrix.copy()
        matrix.setflags(write=False)
        return matrix

    @property
    def is_sparse(self) -> bool:
        """Return whether the matrix is stored sparse."""
        return sparse.issparse(self._matrix)

    @property
    def num_columns(self) -> int:
        """Return the number of specified columns :math:`m`."""
        return int(self._matrix.shape[1])

    @property
    def is_square(self) -> bool:
        """Return whether every column is specified, so the matrix is the full unitary."""
        return self._matrix.shape[0] == self._matrix.shape[1]

    @property
    def num_qubits(self) -> int:
        """Return the width :math:`n` of the register, the base-2 logarithm of the row count."""
        return int(self._matrix.shape[0]).bit_length() - 1

    def to_dense(self) -> np.ndarray:
        """Return the matrix as a NumPy array.

        Returns:
            numpy.ndarray: A float64 copy with :math:`2^n` rows and :math:`m` columns.

        """
        if self.is_sparse:
            return self._matrix.toarray()
        return np.array(self._matrix)

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        """Converting phases to eigenvalues is not supported.

        The container holds a unitary with no associated Hamiltonian.

        Args:
            phase_fraction: Measured phase fraction.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("RealUnitaryContainer has no Hamiltonian to recover eigenvalues for.")

    def combine(self, other: UnitaryContainer) -> RealUnitaryContainer:
        """Append a square real unitary after this one.

        Args:
            other: Square container on the same register, applied after this one.

        Returns:
            RealUnitaryContainer: The product ``other.matrix @ self.matrix``, stored sparse when
            both inputs are sparse.

        Raises:
            TypeError: If ``other`` is not a :class:`~qdk_chemistry.data.RealUnitaryContainer`.
            ValueError: If ``other`` is not square or acts on a different register width.

        """
        if not isinstance(other, RealUnitaryContainer):
            raise TypeError(f"Cannot combine RealUnitaryContainer with {type(other).__name__}.")
        if other.num_qubits != self.num_qubits:
            raise ValueError(
                f"Cannot combine RealUnitaryContainer instances on {self.num_qubits} and {other.num_qubits} qubits."
            )
        if not other.is_square:
            raise ValueError("Only a square RealUnitaryContainer can be appended after another.")
        if self.is_sparse and other.is_sparse:
            return RealUnitaryContainer(other.matrix @ self.matrix)
        return RealUnitaryContainer(other.to_dense() @ self.to_dense())

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_bool(h, self.is_sparse)
        if self.is_sparse:
            _hash_array(h, np.asarray(self._matrix.shape, dtype=np.int64))
            _hash_array(h, self._matrix.indptr.astype(np.int64))
            _hash_array(h, self._matrix.indices.astype(np.int64))
            _hash_array(h, self._matrix.data)
        else:
            _hash_array(h, self._matrix)

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary.

        Returns:
            dict: The container type, the storage format, the shape, and either the dense rows
            or the CSC arrays ``data``, ``indices``, and ``indptr``.

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "format": "sparse" if self.is_sparse else "dense",
            "shape": [int(dim) for dim in self._matrix.shape],
        }
        if self.is_sparse:
            data["data"] = self._matrix.data.tolist()
            data["indices"] = self._matrix.indices.tolist()
            data["indptr"] = self._matrix.indptr.tolist()
        else:
            data["matrix"] = self._matrix.tolist()
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the container to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to.

        """
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["format"] = "sparse" if self.is_sparse else "dense"
        group.attrs["shape"] = np.asarray(self._matrix.shape, dtype=np.int64)
        if self.is_sparse:
            group.create_dataset("data", data=self._matrix.data)
            group.create_dataset("indices", data=self._matrix.indices.astype(np.int64))
            group.create_dataset("indptr", data=self._matrix.indptr.astype(np.int64))
        else:
            group.create_dataset("matrix", data=self._matrix)

    @classmethod
    def _from_arrays(cls, storage_format: str, shape: Any, arrays: dict[str, Any]) -> RealUnitaryContainer:
        """Rebuild a container from its serialized storage format, shape, and arrays."""
        if storage_format == "dense":
            matrix = np.asarray(arrays["matrix"], dtype=np.float64).reshape(tuple(int(dim) for dim in shape))
            return cls(matrix)
        if storage_format == "sparse":
            csc = sparse.csc_array(
                (
                    np.asarray(arrays["data"], dtype=np.float64),
                    np.asarray(arrays["indices"], dtype=np.int64),
                    np.asarray(arrays["indptr"], dtype=np.int64),
                ),
                shape=tuple(int(dim) for dim in shape),
            )
            return cls(csc)
        raise ValueError(f"Unsupported RealUnitaryContainer storage format: {storage_format}")

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RealUnitaryContainer:
        """Create a container from JSON.

        Args:
            json_data: The serialized container.

        Returns:
            RealUnitaryContainer: The reconstructed container.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls._from_arrays(json_data["format"], json_data["shape"], json_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RealUnitaryContainer:
        """Create a container from an HDF5 group.

        Args:
            group: The HDF5 group holding the serialized container.

        Returns:
            RealUnitaryContainer: The reconstructed container.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        storage_format = group.attrs["format"]
        if isinstance(storage_format, bytes):
            storage_format = storage_format.decode("utf-8")
        names = ("data", "indices", "indptr") if storage_format == "sparse" else ("matrix",)
        arrays = {name: group[name][()] for name in names if name in group}
        return cls._from_arrays(str(storage_format), group.attrs["shape"], arrays)

    def get_summary(self) -> str:
        """Return a human-readable summary of the container."""
        rows, cols = self._matrix.shape
        kind = "orthogonal matrix" if self.is_square else "isometry"
        storage = f"sparse, {self._matrix.nnz} nonzeros" if self.is_sparse else "dense"
        return f"Real {kind} ({self.num_qubits} qubits, {rows}x{cols}, {storage})"
