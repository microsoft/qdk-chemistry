"""Unitary container for a unitary matrix or isometry given by its matrix."""

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
from qdk_chemistry.utils.serialization import complex_array_from_json, complex_array_to_json

if TYPE_CHECKING:
    import h5py

__all__ = ["UnitaryMatrixContainer"]

_ORTHONORMALITY_TOLERANCE = 1.0e-8  # Per-column Frobenius norm of M^H M - I, as in the C++ synthesis.


class UnitaryMatrixContainer(UnitaryContainer):
    """Unitary matrix, or its leading orthonormal columns, on a little-endian register, stored dense or sparse.

    Args:
        matrix: Matrix with :math:`2^n` rows and orthonormal columns, as a NumPy array or SciPy sparse matrix.

    Raises:
        TypeError: If the matrix is not numeric.
        ValueError: If the matrix has an invalid shape, non-finite entries or non-orthonormal columns.

    """

    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier ``"unitary_matrix_container"``."""
        return "unitary_matrix_container"

    def __init__(self, matrix: np.ndarray | sparse.sparray | sparse.spmatrix) -> None:
        """Initialize the container."""
        stored = self._canonical_sparse(matrix) if sparse.issparse(matrix) else self._canonical_dense(matrix)
        self._validate_isometry(stored)
        self._matrix = stored
        super().__init__()

    @staticmethod
    def _is_complex_dtype(dtype: np.dtype) -> bool:
        """Return whether ``dtype`` is complex, rejecting non-numeric dtypes."""
        if np.issubdtype(dtype, np.complexfloating):
            return True
        if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer) or dtype == np.bool_:
            return False
        raise TypeError(f"UnitaryMatrixContainer requires a numeric matrix, got dtype {dtype}.")

    @classmethod
    def _canonical_dense(cls, matrix: Any) -> np.ndarray:
        """Return a read-only float64 or complex128 copy of a dense matrix."""
        array = np.asarray(matrix)
        is_complex = cls._is_complex_dtype(array.dtype)
        array = np.array(array, dtype=np.complex128 if is_complex else np.float64, order="C")
        # Adding zero turns -0.0 into +0.0, which JSON serialization of complex values does not keep.
        array = np.ascontiguousarray(np.real_if_close(array) + 0.0)
        array.setflags(write=False)
        return array

    @classmethod
    def _canonical_sparse(cls, matrix: sparse.sparray | sparse.spmatrix) -> sparse.csc_array:
        """Return a canonical float64 or complex128 CSC copy of a sparse matrix without explicit zeros."""
        is_complex = cls._is_complex_dtype(matrix.dtype)
        array = sparse.csc_array(matrix, dtype=np.complex128 if is_complex else np.float64, copy=True)
        array.sum_duplicates()
        array = sparse.csc_array((np.real_if_close(array.data) + 0.0, array.indices, array.indptr), shape=array.shape)
        array.eliminate_zeros()
        array.sort_indices()
        return array

    @staticmethod
    def _validate_isometry(matrix: np.ndarray | sparse.csc_array) -> None:
        """Check the shape, finiteness, and orthonormal columns of a canonical matrix."""
        if matrix.ndim != 2:
            raise ValueError(f"UnitaryMatrixContainer requires a two-dimensional matrix, got {matrix.ndim} dimensions.")
        rows, cols = matrix.shape
        if rows < 2 or rows & (rows - 1):
            raise ValueError(f"UnitaryMatrixContainer requires a power-of-two row count of at least two, got {rows}.")
        if not 1 <= cols <= rows:
            raise ValueError(f"UnitaryMatrixContainer requires between 1 and {rows} columns, got {cols}.")
        values = matrix.data if sparse.issparse(matrix) else matrix
        if not np.all(np.isfinite(values)):
            raise ValueError("UnitaryMatrixContainer requires a finite matrix.")
        adjoint = matrix.conj().T if np.iscomplexobj(values) else matrix.T
        if sparse.issparse(matrix):
            # Keep the residual sparse: a dense m-by-m Gram matrix would dominate memory for wide isometries.
            identity = sparse.csc_array(
                (np.ones(cols), np.arange(cols), np.arange(cols + 1)), shape=(cols, cols), dtype=np.float64
            )
            residual = float(np.linalg.norm(sparse.csc_array(adjoint @ matrix - identity).data))
        else:
            residual = float(np.linalg.norm(adjoint @ matrix - np.eye(cols)))
        if not residual <= _ORTHONORMALITY_TOLERANCE * cols:
            raise ValueError(
                "UnitaryMatrixContainer requires orthonormal columns; "
                f"the Frobenius norm of M^H @ M - I is {residual:.3e}."
            )

    @property
    def type(self) -> str:
        """Return the container type."""
        return "unitary_matrix"

    @property
    def matrix(self) -> np.ndarray | sparse.csc_array:
        """Return a copy of the stored matrix, read-only when dense, so the container stays immutable."""
        if self.is_sparse:
            return self._matrix.copy()
        matrix = self._matrix.copy()
        matrix.setflags(write=False)
        return matrix

    @property
    def is_real(self) -> bool:
        """Return whether every entry is real, so the matrix is stored as float64."""
        return not np.issubdtype(self._matrix.dtype, np.complexfloating)

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
        """Return a dense float64 or complex128 copy of the matrix."""
        if self.is_sparse:
            return self._matrix.toarray()
        return np.array(self._matrix)

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        """Raise NotImplementedError, since the container has no Hamiltonian.

        Args:
            phase_fraction: Measured phase fraction.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("UnitaryMatrixContainer has no Hamiltonian to recover eigenvalues for.")

    def combine(self, other: UnitaryContainer) -> UnitaryMatrixContainer:
        """Return ``other.matrix @ self.matrix``, stored sparse only when both factors are sparse.

        Args:
            other: Square container on the same register, applied after this one.

        Returns:
            UnitaryMatrixContainer: The product.

        Raises:
            TypeError: If ``other`` is not a :class:`UnitaryMatrixContainer`.
            ValueError: If ``other`` is not square or has a different register width.

        """
        if not isinstance(other, UnitaryMatrixContainer):
            raise TypeError(f"Cannot combine UnitaryMatrixContainer with {type(other).__name__}.")
        if other.num_qubits != self.num_qubits:
            raise ValueError(
                f"Cannot combine UnitaryMatrixContainer instances on {self.num_qubits} and {other.num_qubits} qubits."
            )
        if not other.is_square:
            raise ValueError("Only a square UnitaryMatrixContainer can be appended after another.")
        return UnitaryMatrixContainer(other._matrix @ self._matrix)  # noqa: SLF001

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

    def _values_to_json(self, values: np.ndarray) -> list | dict[str, Any]:
        """Return real values as nested lists and complex values split into real and imaginary parts."""
        return values.tolist() if self.is_real else complex_array_to_json(values)

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary.

        Returns:
            dict: The storage format, the shape, and the dense rows or the CSC arrays.

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "format": "sparse" if self.is_sparse else "dense",
            "shape": [int(dim) for dim in self._matrix.shape],
        }
        if self.is_sparse:
            data["data"] = self._values_to_json(self._matrix.data)
            data["indices"] = self._matrix.indices.tolist()
            data["indptr"] = self._matrix.indptr.tolist()
        else:
            data["matrix"] = self._values_to_json(self._matrix)
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

    @staticmethod
    def _values_from_json(values: Any) -> np.ndarray:
        """Return the array written by ``_values_to_json``, or an HDF5 dataset's array unchanged."""
        return complex_array_from_json(values) if isinstance(values, dict) else np.asarray(values)

    @classmethod
    def _from_arrays(cls, storage_format: str, shape: Any, arrays: dict[str, Any]) -> UnitaryMatrixContainer:
        """Rebuild a container from its serialized storage format, shape, and arrays."""
        shape = tuple(int(dim) for dim in shape)
        if storage_format == "dense":
            return cls(cls._values_from_json(arrays["matrix"]).reshape(shape))
        if storage_format == "sparse":
            csc = sparse.csc_array(
                (
                    cls._values_from_json(arrays["data"]),
                    np.asarray(arrays["indices"], dtype=np.int64),
                    np.asarray(arrays["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
            return cls(csc)
        raise ValueError(f"Unsupported UnitaryMatrixContainer storage format: {storage_format}")

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> UnitaryMatrixContainer:
        """Create a container from JSON.

        Args:
            json_data: The serialized container.

        Returns:
            UnitaryMatrixContainer: The reconstructed container.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls._from_arrays(json_data["format"], json_data["shape"], json_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> UnitaryMatrixContainer:
        """Create a container from an HDF5 group.

        Args:
            group: The HDF5 group holding the serialized container.

        Returns:
            UnitaryMatrixContainer: The reconstructed container.

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
        kind = ("orthogonal matrix" if self.is_real else "unitary matrix") if self.is_square else "isometry"
        storage = f"sparse, {self._matrix.nnz} nonzeros" if self.is_sparse else "dense"
        return f"{'Real' if self.is_real else 'Complex'} {kind} ({self.num_qubits} qubits, {rows}x{cols}, {storage})"
