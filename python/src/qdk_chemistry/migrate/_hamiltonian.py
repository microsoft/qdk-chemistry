"""Migrate the ``Hamiltonian`` serialization schema to the current version.

The Hamiltonian envelope (``{version, container}``) is schema-stable; only the
container payload changed (dense integral arrays -> ``SymmetryBlockedTensor``).

Two distinct ``cholesky`` container layouts both serialize at container version
``0.1.0`` and are told apart by which integrals they carry:

- The released (1.1.0) container derived from the four-center container and stored
  the full dense four-center two-body tensor (keys ``aaaa``/``aabb``/``bbbb``),
  never the MO three-center vectors. It is migrated to a ``canonical_four_center``
  container, dropping the now-unused AO Cholesky vectors.
- The later container stored the MO three-center vectors directly (key ``aa``/
  ``bb``, or the ``three_center_integrals_aa`` HDF5 dataset). Those vectors are the
  current Cholesky data model, so the container is preserved as ``cholesky`` with
  the vectors re-expressed as a ``SymmetryBlockedTensor``.
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
OLD_CONTAINER_VERSION = "0.1.0"

_FOUR_CENTER = "canonical_four_center"
_CHOLESKY = "cholesky"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed legacy Hamiltonian JSON object into a container old-doc."""
    container = doc["container"]
    container_type = container["container_type"]
    if container_type == "sparse":
        return _sparse.from_json_doc(container)
    if container_type == _CHOLESKY and "three_center_integrals" in container:
        return _cholesky_from_json(container)
    return _four_center_from_json(container)


def from_hdf5_file(path) -> dict:
    """Normalize a legacy Hamiltonian HDF5 file into a container old-doc."""
    with h5py.File(path, "r") as handle:
        return from_hdf5_group(handle)


def from_hdf5_group(group) -> dict:
    """Normalize a legacy Hamiltonian HDF5 group (with a ``container`` subgroup)."""
    container = group["container"]
    container_type = _io.read_attr(container, "container_type")
    if container_type == "sparse":
        return _sparse.from_hdf5_group(container)
    if container_type == _CHOLESKY and "three_center_integrals_aa" in container:
        return _cholesky_from_hdf5(container)
    return _four_center_from_hdf5(container)


def to_new_json(old: dict) -> dict:
    """Build the migrated Hamiltonian JSON object from a normalized container old-doc."""
    container_type = old["container_type"]
    if container_type == "sparse":
        container = _sparse.to_new_json(old)
    elif container_type == _CHOLESKY:
        container = _cholesky_to_new_json(old)
    else:
        container = _four_center_to_new_json(old)
    return {"version": HAMILTONIAN_VERSION, "container": container}


def _four_center_from_json(container: dict) -> dict:
    """Read four-center / Cholesky container fields from old JSON."""
    two_body = container.get("two_body_integrals") or {}
    return {
        "_source_version": str(container.get("version")),
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
        "_source_version": _io.read_attr(container, "version"),
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
    """Build the migrated four-center container JSON from an old-doc."""
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


def _cholesky_from_json(container: dict) -> dict:
    """Read a genuine (three-center) Cholesky container from old JSON."""
    three_center = container.get("three_center_integrals") or {}
    return {
        "_source_version": str(container.get("version")),
        "container_type": _CHOLESKY,
        "core_energy": container.get("core_energy", 0.0),
        "type": container.get("type", "Hermitian"),
        "is_restricted": bool(container.get("is_restricted", True)),
        "one_body_alpha": _opt_array(container.get("one_body_integrals_alpha")),
        "one_body_beta": _opt_array(container.get("one_body_integrals_beta")),
        "three_center_aa": _opt_array(three_center.get("aa")),
        "three_center_bb": _opt_array(three_center.get("bb")),
        "fock_alpha": _opt_array(container.get("inactive_fock_matrix_alpha")),
        "fock_beta": _opt_array(container.get("inactive_fock_matrix_beta")),
        "ao_cholesky_vectors": _opt_array(container.get("ao_cholesky_vectors")),
        "orbitals": _orbitals.from_json_doc(container["orbitals"]),
    }


def _cholesky_from_hdf5(container: h5py.Group) -> dict:
    """Read a genuine (three-center) Cholesky container from old HDF5."""
    metadata = container["metadata"]
    return {
        "_source_version": _io.read_attr(container, "version"),
        "container_type": _CHOLESKY,
        "core_energy": float(_io.read_attr(metadata, "core_energy", 0.0)),
        "type": _io.read_attr(metadata, "type", "Hermitian"),
        "is_restricted": bool(_io.read_attr(metadata, "is_restricted", True)),
        "one_body_alpha": _io.read_matrix(container, "one_body_integrals_alpha"),
        "one_body_beta": _io.read_matrix(container, "one_body_integrals_beta"),
        "three_center_aa": _io.read_matrix(container, "three_center_integrals_aa"),
        "three_center_bb": _io.read_matrix(container, "three_center_integrals_bb"),
        "fock_alpha": _io.read_matrix(container, "inactive_fock_matrix_alpha"),
        "fock_beta": _io.read_matrix(container, "inactive_fock_matrix_beta"),
        "ao_cholesky_vectors": _io.read_matrix(container, "ao_cholesky_vectors"),
        "orbitals": _orbitals.from_hdf5_group(container["orbitals"]),
    }


def _cholesky_to_new_json(old: dict) -> dict:
    """Build the migrated Cholesky container JSON from an old-doc."""
    restricted = old["is_restricted"]
    container: dict = {
        "version": CONTAINER_VERSION,
        "container_type": _CHOLESKY,
        "core_energy": float(old["core_energy"]),
        "type": old["type"],
        "is_restricted": restricted,
        "orbitals": _orbitals.to_new_json(old["orbitals"]),
    }

    beta = None if restricted else old.get("one_body_beta")
    container["one_body_integrals"] = _sbt.rank2_dict(old["one_body_alpha"], beta)

    beta = None if restricted else old.get("three_center_bb")
    container["three_center_integrals"] = _sbt.rank3_three_center_dict(old["three_center_aa"], beta)

    if old.get("fock_alpha") is not None:
        beta = None if restricted else old.get("fock_beta")
        container["inactive_fock_matrix"] = _sbt.rank2_dict(old["fock_alpha"], beta)

    if old.get("ao_cholesky_vectors") is not None:
        container["ao_cholesky_vectors"] = np.asarray(old["ao_cholesky_vectors"], dtype=np.float64).tolist()

    return container


# The Hamiltonian envelope version is unchanged; the chain is keyed on the
# container's serialization version (the legacy cholesky/sparse/four-center
# containers all serialized version 0.1.0).
STEPS = {OLD_CONTAINER_VERSION: (CONTAINER_VERSION, to_new_json)}
