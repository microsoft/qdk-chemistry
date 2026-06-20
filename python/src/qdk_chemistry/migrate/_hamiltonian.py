"""Convert v1 ``Hamiltonian`` serialization to the v2 schema.

The Hamiltonian envelope (``{version, container}``) is schema-stable; only the
container payload changed (dense integral arrays -> ``SymmetryBlockedTensor``).

The v1 ``cholesky`` container did not persist the MO three-center vectors, but
it *did* serialize the full four-center two-body tensor (it derived from the
four-center container). It is therefore migrated to a ``canonical_four_center``
container, dropping the now-unused AO Cholesky vectors.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py
import numpy as np

from . import _io, _orbitals, _sbt, _sparse

HAMILTONIAN_VERSION = "0.1.0"
CONTAINER_VERSION = "0.2.0"

_FOUR_CENTER = "canonical_four_center"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed v1 Hamiltonian JSON object into a container old-doc."""
    container = doc["container"]
    container_type = container["container_type"]
    if container_type == "sparse":
        return _sparse.from_json_doc(container)
    return _four_center_from_json(container)


def from_hdf5_file(path) -> dict:
    """Normalize a v1 Hamiltonian HDF5 file into a container old-doc."""
    with h5py.File(path, "r") as handle:
        container = handle["container"]
        container_type = _io.read_attr(container, "container_type")
        if container_type == "sparse":
            return _sparse.from_hdf5_group(container)
        return _four_center_from_hdf5(container)


def to_new_json(old: dict) -> dict:
    """Build the v2 Hamiltonian JSON object from a normalized container old-doc."""
    is_sparse = old["container_type"] == "sparse"
    container = _sparse.to_new_json(old) if is_sparse else _four_center_to_new_json(old)
    return {"version": HAMILTONIAN_VERSION, "container": container}


def _four_center_from_json(container: dict) -> dict:
    """Read four-center / Cholesky container fields from old JSON."""
    two_body = container.get("two_body_integrals") or {}
    return {
        "container_type": _FOUR_CENTER,
        "core_energy": container.get("core_energy", 0.0),
        "type": container.get("type", "Hermitian"),
        "is_restricted": bool(container.get("is_restricted", True)),
        "one_body_alpha": _opt_array(container.get("one_body_integrals_alpha")),
        "one_body_beta": _opt_array(container.get("one_body_integrals_beta")),
        "two_body_aaaa": _opt_array(two_body.get("aaaa")),
        "two_body_aabb": _opt_array(two_body.get("aabb")),
        "two_body_bbbb": _opt_array(two_body.get("bbbb")),
        "fock_alpha": _opt_array(container.get("inactive_fock_matrix_alpha")),
        "fock_beta": _opt_array(container.get("inactive_fock_matrix_beta")),
        "orbitals": _orbitals.from_json_doc(container["orbitals"]),
    }


def _four_center_from_hdf5(container: h5py.Group) -> dict:
    """Read four-center / Cholesky container fields from old HDF5."""
    metadata = container["metadata"]
    return {
        "container_type": _FOUR_CENTER,
        "core_energy": float(_io.read_attr(metadata, "core_energy", 0.0)),
        "type": _io.read_attr(metadata, "type", "Hermitian"),
        "is_restricted": bool(_io.read_attr(metadata, "is_restricted", True)),
        "one_body_alpha": _io.read_matrix(container, "one_body_integrals_alpha"),
        "one_body_beta": _io.read_matrix(container, "one_body_integrals_beta"),
        "two_body_aaaa": _io.read_vector(container, "two_body_integrals_aaaa"),
        "two_body_aabb": _io.read_vector(container, "two_body_integrals_aabb"),
        "two_body_bbbb": _io.read_vector(container, "two_body_integrals_bbbb"),
        "fock_alpha": _io.read_matrix(container, "inactive_fock_matrix_alpha"),
        "fock_beta": _io.read_matrix(container, "inactive_fock_matrix_beta"),
        "orbitals": _orbitals.from_hdf5_group(container["orbitals"]),
    }


def _four_center_to_new_json(old: dict) -> dict:
    """Build the v2 four-center container JSON from an old-doc."""
    restricted = old["is_restricted"]
    container: dict = {
        "version": CONTAINER_VERSION,
        "container_type": _FOUR_CENTER,
        "core_energy": float(old["core_energy"]),
        "type": old["type"],
        "is_restricted": restricted,
        "orbitals": _orbitals.to_new_json(old["orbitals"]),
    }

    if old.get("one_body_alpha") is not None:
        beta = None if restricted else old.get("one_body_beta")
        container["one_body_integrals"] = _sbt.rank2_dict(old["one_body_alpha"], beta)

    if old.get("two_body_aaaa") is not None:
        if restricted:
            container["two_body_integrals"] = _sbt.rank4_dict(old["two_body_aaaa"])
        else:
            container["two_body_integrals"] = _sbt.rank4_dict(
                old["two_body_aaaa"], old["two_body_aabb"], old["two_body_bbbb"]
            )

    if old.get("fock_alpha") is not None:
        beta = None if restricted else old.get("fock_beta")
        container["inactive_fock_matrix"] = _sbt.rank2_dict(old["fock_alpha"], beta)

    return container


def _opt_array(value):
    """Return ``value`` as a float64 array, or None when absent."""
    return None if value is None else np.asarray(value, dtype=np.float64)
