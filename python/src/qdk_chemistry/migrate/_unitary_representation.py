"""Migrate Pauli product formulas and Hubbard plaquette evolutions written against older schemas."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import warnings

import h5py

from . import _io

OLD_VERSION = "0.2.0"
PRODUCT_FORMULA_VERSION = "0.3.0"
PLAQUETTE_OLD_VERSION = "0.1.0"
PLAQUETTE_VERSION = "0.2.0"

_PRODUCT_FORMULA = "pauli_product_formula"
_PLAQUETTE = "hubbard_plaquette"


def plaquette_step_key(version: str) -> str:
    """Key of ``version`` in ``STEPS``; the plaquette container versions on its own."""
    return f"{_PLAQUETTE} {version}"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed legacy product formula or plaquette evolution into an old-doc."""
    container_type = doc.get("container_type")
    _require_migratable(container_type)
    old = dict(doc)
    old["_source_version"] = str(old.pop("version", None))
    if container_type == _PLAQUETTE:
        old["_step_key"] = plaquette_step_key(old["_source_version"])
    return old


def from_hdf5_file(path) -> dict:
    """Read a legacy product-formula or plaquette HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        container_type = _io.read_attr(handle, "container_type")
        _require_migratable(container_type)
        if container_type == _PLAQUETTE:
            # A plaquette evolution stores its JSON document as one payload attribute.
            return from_json_doc(json.loads(_io.read_attr(handle, "payload")))
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
    if old["step_reps"] > 1:
        warnings.warn(
            "Legacy product-formula scale is preserved unchanged, but may not represent the total evolution time. "
            "The saved repetition count does not distinguish divisions from power. Rebuild powered repeat-strategy "
            "or multi-step composed formulas before using phase-to-energy inversion.",
            UserWarning,
            stacklevel=2,
        )
    new = {key: value for key, value in old.items() if key != "_source_version"}
    new.update(prefix_terms=[], suffix_terms=[], version=PRODUCT_FORMULA_VERSION)
    return new


def plaquette_to_new_json(old: dict) -> dict:
    """Add ``order``, which was always 2 before it was stored."""
    new = {key: value for key, value in old.items() if key not in ("_source_version", "_step_key")}
    new.update(order=2, version=PLAQUETTE_VERSION)
    return new


def _read_term(group: h5py.Group) -> dict:
    """Read one exponentiated Pauli term from the legacy HDF5 layout."""
    pauli_term = group["pauli_term"]
    return {
        "pauli_term": {key: _io.read_attr(pauli_term, key) for key in pauli_term.attrs},
        "angle": float(_io.read_attr(group, "angle")),
    }


def _require_migratable(container_type) -> None:
    """Reject unitary containers whose serialization schema did not change."""
    if container_type not in (_PRODUCT_FORMULA, _PLAQUETTE):
        raise ValueError(
            f"Only {_PRODUCT_FORMULA} and {_PLAQUETTE} containers have migration steps; "
            f"{container_type!r} files load without conversion."
        )


STEPS = {
    OLD_VERSION: (PRODUCT_FORMULA_VERSION, to_new_json),
    plaquette_step_key(PLAQUETTE_OLD_VERSION): (plaquette_step_key(PLAQUETTE_VERSION), plaquette_to_new_json),
}
