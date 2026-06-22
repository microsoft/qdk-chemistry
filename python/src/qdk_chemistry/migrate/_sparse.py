"""Convert v1 ``sparse`` Hamiltonian containers to the v2 schema.

Only the two-body payload changed: the v1 flat ``two_body_integrals_sparse``
list of ``[p, q, r, s, value]`` entries became a ``SymmetryBlockedSparseMap``.
The (restricted) one-body sparse triplet list and the scalar metadata are
schema-stable. Sparse model Hamiltonians carry no orbitals.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _io, _sbt

if TYPE_CHECKING:
    import h5py

CONTAINER_VERSION = "0.2.0"


def from_json_doc(container: dict) -> dict:
    """Normalize a parsed v1 sparse container JSON object into an old-doc."""
    return {
        "container_type": "sparse",
        "core_energy": container.get("core_energy", 0.0),
        "type": container.get("type", "Hermitian"),
        "num_orbitals": int(container["num_orbitals"]),
        "one_body_sparse": container.get("one_body_integrals_alpha_sparse"),
        "two_body_sparse": container.get("two_body_integrals_sparse"),
    }


def _unpack_indices(packed_column) -> tuple:
    """Decode a column of ``pack_indices`` doubles into two integer arrays.

    The v1 HDF5 sparse format packs an ``(a, b)`` uint32 index pair into the
    bytes of a ``double`` (``uint32[2] = {a, b}`` memcpy'd to a ``double``). On a
    little-endian host this is recovered by viewing the double as two uint32s.
    """
    pairs = np.ascontiguousarray(packed_column, dtype=np.float64).view(np.uint32).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def from_hdf5_group(container: h5py.Group) -> dict:
    """Normalize a v1 sparse container HDF5 group into an old-doc."""
    metadata = container["metadata"]
    one_body_ds = container.get("one_body_integrals_alpha_sparse")
    two_body_ds = container.get("two_body_integrals_sparse")

    num_orbitals = None
    if one_body_ds is not None and "num_orbitals" in one_body_ds.attrs:
        num_orbitals = int(np.asarray(one_body_ds.attrs["num_orbitals"]).ravel()[0])
    elif "num_orbitals" in metadata.attrs or "num_orbitals" in metadata:
        num_orbitals = int(_io.read_attr(metadata, "num_orbitals", 0))
    if num_orbitals is None:
        raise ValueError("Sparse container is missing num_orbitals metadata")

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
        "container_type": "sparse",
        "core_energy": float(_io.read_attr(metadata, "core_energy", 0.0)),
        "type": _io.read_attr(metadata, "type", "Hermitian"),
        "num_orbitals": num_orbitals,
        "one_body_sparse": one_body,
        "two_body_sparse": two_body,
    }


def to_new_json(old: dict) -> dict:
    """Build the v2 sparse container JSON object from a normalized old-doc."""
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
