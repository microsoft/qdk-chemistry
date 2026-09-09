"""Validation contracts shared by packed Pauli data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.data._sparse_pauli import _validate_sparse_pauli_arrays
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer


@pytest.fixture
def packed_arrays():
    """Provide local factors with an identity term and a reset between term indices."""
    return {
        "num_qubits": 4,
        "term_offsets": np.array([0, 2, 2, 3]),
        "qubit_indices": np.array([1, 3, 0]),
        "pauli_codes": np.array([1, 2, 3]),
        "num_terms": 3,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_qubits", 0),
        ("num_qubits", -1),
        ("num_qubits", 4.0),
        ("num_qubits", True),
        ("num_qubits", 2**32 + 1),
        ("num_terms", -1),
        ("num_terms", 3.0),
        ("num_terms", True),
        ("num_terms", 2**64),
        ("num_terms", 2),
        ("term_offsets", []),
        ("term_offsets", [[0, 2, 2, 3]]),
        ("term_offsets", [1, 2, 2, 3]),
        ("term_offsets", [0, 2, 1, 3]),
        ("term_offsets", [0, 2, 2, 4]),
        ("term_offsets", [0, 2, 2, -1]),
        ("term_offsets", [0, 2, 2, 2**64]),
        ("term_offsets", [0, 2, True, 3]),
        ("term_offsets", np.array([0.0, 2.0, 2.0, 3.0])),
        ("qubit_indices", [1, 3]),
        ("qubit_indices", [1, 1, 0]),
        ("qubit_indices", [3, 1, 0]),
        ("qubit_indices", [1, 3, 4]),
        ("qubit_indices", [1, 3, -1]),
        ("qubit_indices", [1, 3, 0.5]),
        ("qubit_indices", [1, 3, False]),
        ("qubit_indices", np.array([1, 3, 2**32], dtype=np.uint64)),
        ("qubit_indices", np.array([1, 3, 2**100], dtype=object)),
        ("pauli_codes", [1, 2, 0]),
        ("pauli_codes", [1, 2, 4]),
        ("pauli_codes", [1, 2, -1]),
        ("pauli_codes", [1, 2, 257]),
        ("pauli_codes", [1, 2, 2.5]),
        ("pauli_codes", np.array([True, False, True])),
        ("pauli_codes", [1, 2, True]),
        ("pauli_codes", ["1", "2", "3"]),
    ],
)
def test_sparse_arrays_reject_invalid_values_before_casting(packed_arrays, field, value):
    """Reject malformed boundaries, factors, and counts before narrowing integer storage."""
    packed_arrays[field] = value
    with pytest.raises(ValueError, match=r"num_qubits|num_terms|term_offsets|qubit_indices|pauli_codes|Pauli"):
        _validate_sparse_pauli_arrays(**packed_arrays)


@pytest.mark.parametrize(
    ("num_qubits", "offsets", "indices", "codes"),
    [
        (1, [0], [], []),
        (np.int64(4), [0, 0, 2, 2, 3, 3], [1, 3, 0], [1, 2, 3]),
        (2**32, [0, 1], [2**32 - 1], [3]),
    ],
)
def test_sparse_arrays_allow_empty_identity_and_full_width_terms(num_qubits, offsets, indices, codes):
    """Empty supports and the full uint32 range remain valid without dense labels."""
    normalized = _validate_sparse_pauli_arrays(num_qubits, offsets, indices, codes, np.int64(len(offsets) - 1))
    for actual, expected, dtype in zip(
        normalized, (offsets, indices, codes), (np.uint64, np.uint32, np.uint8), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == dtype
        assert actual.flags.owndata
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable


@pytest.mark.parametrize("container_type", [QubitOperator, PauliProductFormulaContainer])
@pytest.mark.parametrize("source", ["arrays", "json", "hdf5"])
@pytest.mark.parametrize(("field", "value"), [("term_offsets", [0, True]), ("pauli_codes", [257])])
def test_public_sparse_loaders_do_not_coerce_raw_values(container_type, source, field, value, tmp_path):
    """Public entry points must preserve booleans and overflowing codes for the shared validator."""
    arrays = {"num_qubits": 2, "term_offsets": [0, 1], "qubit_indices": [0], "pauli_codes": [1]}
    if container_type is QubitOperator:
        arrays["coefficients"] = np.array([1.0])
    else:
        arrays.update(angles=np.array([0.5]), step_reps=1)
    if source == "arrays":
        arrays[field] = value
        with pytest.raises(ValueError, match="term_offsets|pauli_codes"):
            container_type.from_sparse_arrays(**arrays)
        return

    original = container_type.from_sparse_arrays(**arrays)
    if source == "json":
        data = original.to_json()
        data[field] = value
        with pytest.raises(ValueError, match="term_offsets|pauli_codes"):
            container_type.from_json(data)
    else:
        with h5py.File(tmp_path / "invalid.h5", "w") as group:
            original.to_hdf5(group)
            del group[field]
            group.create_dataset(field, data=value, dtype=bool if field == "term_offsets" else None)
            with pytest.raises(ValueError, match="term_offsets|pauli_codes"):
                container_type.from_hdf5(group)
