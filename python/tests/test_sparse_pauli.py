"""Validation contracts shared by packed Pauli data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import SparsePauliProductFormulaContainer, SparsePauliTerms
from qdk_chemistry.data.qubit_operator import QubitOperator


@pytest.mark.parametrize("field", ["num_qubits", "term_offsets", "qubit_indices", "pauli_codes"])
def test_sparse_arrays_reject_invalid_values_before_casting(field):
    """Reject raw booleans, fractions, strings, and overflow before narrowing integer storage."""
    arrays: dict[str, Any] = {"num_qubits": 4, "term_offsets": [0, 2], "qubit_indices": [1, 3], "pauli_codes": [1, 2]}
    for value in (True, -1, 0.5, "1", 2**100):
        invalid = value if field == "num_qubits" else [value, *arrays[field][1:]]
        with pytest.raises(ValueError, match="num_qubits|term_offsets|qubit_indices|pauli_codes"):
            SparsePauliTerms(**(arrays | {field: invalid}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_qubits", 0),
        ("num_qubits", 2**32 + 1),
        ("term_offsets", []),
        ("term_offsets", [[0, 2]]),
        ("term_offsets", [1, 2]),
        ("term_offsets", [0, 3, 2]),
        ("term_offsets", [0, 3]),
        ("qubit_indices", [1]),
        ("qubit_indices", [1, 1]),
        ("qubit_indices", [3, 1]),
        ("qubit_indices", [1, 4]),
        ("pauli_codes", [1, 0]),
        ("pauli_codes", [1, 4]),
    ],
)
def test_sparse_arrays_reject_invalid_structure(field, value):
    """Validate register width, row boundaries, matching factor lengths, and canonical Pauli support."""
    arrays = {"num_qubits": 4, "term_offsets": [0, 2], "qubit_indices": [1, 3], "pauli_codes": [1, 2]}
    with pytest.raises(ValueError, match="num_qubits|term_offsets|Pauli"):
        SparsePauliTerms(**(arrays | {field: value}))


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
    sources = tuple(
        np.asarray(values, dtype=dtype)
        for values, dtype in zip((offsets, indices, codes), (np.uint64, np.uint32, np.uint8), strict=True)
    )
    normalized = SparsePauliTerms(num_qubits, *sources).arrays()
    for actual, expected, dtype in zip(normalized, sources, (np.uint64, np.uint32, np.uint8), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == dtype
        assert not np.shares_memory(actual, expected)
        assert actual.flags.owndata
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable


@pytest.mark.parametrize("container_type", [QubitOperator, SparsePauliProductFormulaContainer])
@pytest.mark.parametrize("source", ["json", "hdf5"])
@pytest.mark.parametrize(("field", "value"), [("term_offsets", [0, True]), ("pauli_codes", [257])])
def test_public_sparse_loaders_do_not_coerce_raw_values(container_type, source, field, value, tmp_path):
    """Public entry points must preserve booleans and overflowing codes for the shared validator."""
    arrays = {"num_qubits": 2, "term_offsets": [0, 1], "qubit_indices": [0], "pauli_codes": [1]}
    parameters = (
        {"coefficients": np.ones(1)} if container_type is QubitOperator else {"angles": np.ones(1), "step_reps": 1}
    )
    arrays.update(parameters)
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
