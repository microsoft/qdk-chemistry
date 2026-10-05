"""Migrate Pauli product formulas written before prefix and suffix term segments existed."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py

from . import _io

OLD_VERSION = "0.2.0"
PRODUCT_FORMULA_VERSION = "0.3.0"

_PRODUCT_FORMULA = "pauli_product_formula"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed legacy product formula into an old-doc."""
    _require_product_formula(doc.get("container_type"))
    old = dict(doc)
    old["_source_version"] = str(old.pop("version", None))
    return old


def from_hdf5_file(path) -> dict:
    """Read a legacy product-formula HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        _require_product_formula(_io.read_attr(handle, "container_type"))
        terms = handle["step_terms"]
        return {
            "_source_version": str(_io.read_attr(handle, "version")),
            "container_type": _PRODUCT_FORMULA,
            # Read by index: sorted names would put term_10 before term_2.
            "step_terms": [_read_term(terms[f"term_{i}"]) for i in range(len(terms))],
            "step_reps": int(_io.read_attr(handle, "step_reps")),
            "num_qubits": int(_io.read_attr(handle, "num_qubits")),
            "scale": float(_io.read_attr(handle, "scale", 1.0)),
        }


def to_new_json(old: dict) -> dict:
    """Add empty prefix and suffix term segments; the repeated body is unchanged."""
    new = {key: value for key, value in old.items() if key != "_source_version"}
    new.update(prefix_terms=[], suffix_terms=[], version=PRODUCT_FORMULA_VERSION)
    return new


def _read_term(group: h5py.Group) -> dict:
    """Read one exponentiated Pauli term from the legacy HDF5 layout."""
    pauli_term = group["pauli_term"]
    return {
        "pauli_term": {key: _io.read_attr(pauli_term, key) for key in pauli_term.attrs},
        "angle": float(_io.read_attr(group, "angle")),
    }


def _require_product_formula(container_type) -> None:
    """Reject unitary containers whose serialization schema did not change."""
    if container_type != _PRODUCT_FORMULA:
        raise ValueError(
            f"Only {_PRODUCT_FORMULA} containers have a migration step; "
            f"{container_type!r} files load without conversion."
        )


STEPS = {OLD_VERSION: (PRODUCT_FORMULA_VERSION, to_new_json)}
