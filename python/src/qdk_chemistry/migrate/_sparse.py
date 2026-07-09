"""Migrate the ``sparse`` Hamiltonian container serialization to the current version.

Only the two-body payload changed: the legacy flat ``two_body_integrals_sparse``
list of ``[p, q, r, s, value]`` entries became a ``SymmetryBlockedSparseMap``.
The (restricted) one-body sparse triplet list and the scalar metadata are
schema-stable. Sparse model Hamiltonians carry no orbitals.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py
import numpy as np

from . import _io, _sbt

CONTAINER_VERSION = "0.2.0"


def from_json_doc(container: dict) -> dict:
    """Normalize a parsed legacy sparse container JSON object into an old-doc."""
    return {
        "_source_version": str(container.get("version")),
        "container_type": "sparse",
        "core_energy": container.get("core_energy", 0.0),
        "type": container.get("type", "Hermitian"),
        "num_orbitals": int(container["num_orbitals"]),
        "one_body_sparse": container.get("one_body_integrals_alpha_sparse"),
        "two_body_sparse": container.get("two_body_integrals_sparse"),
    }


def _unpack_indices(packed_column) -> tuple:
    """Decode a column of ``pack_indices`` doubles into two integer arrays.

    The legacy HDF5 sparse format packs an ``(a, b)`` uint32 index pair into the
    bytes of a ``double`` (``uint32[2] = {a, b}`` memcpy'd to a ``double``). On a
    little-endian host this is recovered by viewing the double as two uint32s.
    """
    pairs = np.ascontiguousarray(packed_column, dtype=np.float64).view(np.uint32).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def _read_scalar_int(node: h5py.Group | h5py.Dataset, name: str) -> int | None:
    """Read an integer stored as either an HDF5 attribute or a scalar dataset member."""
    if name in node.attrs:
        return int(np.asarray(node.attrs[name]).ravel()[0])
    if isinstance(node, h5py.Group) and name in node:
        return int(np.asarray(node[name]).ravel()[0])
    return None


def from_hdf5_group(container: h5py.Group) -> dict:
    """Normalize a legacy sparse container HDF5 group into an old-doc."""
    metadata = container["metadata"]
    one_body_ds = container.get("one_body_integrals_alpha_sparse")
    two_body_ds = container.get("two_body_integrals_sparse")

    # The legacy writer stores num_orbitals as an attribute on the one-body dataset;
    # fall back to the metadata group (attribute or scalar dataset) otherwise.
    num_orbitals = None
    if one_body_ds is not None:
        num_orbitals = _read_scalar_int(one_body_ds, "num_orbitals")
    if num_orbitals is None:
        num_orbitals = _read_scalar_int(metadata, "num_orbitals")
    if num_orbitals is None or num_orbitals <= 0:
        raise ValueError(
            "Sparse container is missing a positive num_orbitals value (looked for an "
            "attribute on one_body_integrals_alpha_sparse and in metadata)."
        )

    one_body = None
    if one_body_ds is not None:
        # Each row is [pack_indices(row, col), value].
        arr = np.asarray(one_body_ds, dtype=np.float64)
        rows, cols = _unpack_indices(arr[:, 0])
        one_body = [[int(r), int(c), float(v)] for r, c, v in zip(rows, cols, arr[:, 1], strict=True)]

    two_body = None
    if two_body_ds is not None:
        # Each row is [pack_indices(p, q), pack_indices(r, s), value].
        arr = np.asarray(two_body_ds, dtype=np.float64)
        p, q = _unpack_indices(arr[:, 0])
        r, s = _unpack_indices(arr[:, 1])
        two_body = [
            [int(pp), int(qq), int(rr), int(ss), float(v)]
            for pp, qq, rr, ss, v in zip(p, q, r, s, arr[:, 2], strict=True)
        ]

    return {
        "_source_version": _io.read_attr(container, "version"),
        "container_type": "sparse",
        "core_energy": float(_io.read_attr(metadata, "core_energy", 0.0)),
        "type": _io.read_attr(metadata, "type", "Hermitian"),
        "num_orbitals": num_orbitals,
        "one_body_sparse": one_body,
        "two_body_sparse": two_body,
    }


def to_new_json(old: dict) -> dict:
    """Build the migrated sparse container JSON object from a normalized old-doc."""
    norb = old["num_orbitals"]
    one_body = old.get("one_body_sparse") or []
    return {
        "version": CONTAINER_VERSION,
        "container_type": "sparse",
        "core_energy": float(old["core_energy"]),
        "type": old["type"],
        "is_restricted": True,
        "num_orbitals": norb,
        "has_one_body_integrals": len(one_body) > 0,
        "one_body_integrals_alpha_sparse": [[int(r), int(c), float(v)] for r, c, v in one_body],
        "two_body_integrals": _sbt.sparse_rank4_dict(old.get("two_body_sparse") or [], norb),
    }
