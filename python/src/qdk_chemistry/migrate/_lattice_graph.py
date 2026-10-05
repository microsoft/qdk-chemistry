"""Migrate ``LatticeGraph`` files written before graph serialization was versioned."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py

from . import _io

# Unversioned files have no version field, which the readers record as str(None).
UNVERSIONED = str(None)
LATTICE_GRAPH_VERSION = "0.1.0"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed legacy lattice graph into an old-doc."""
    old = dict(doc)
    old["_source_version"] = str(old.pop("version", None))
    return old


def from_hdf5_file(path) -> dict:
    """Read a legacy lattice-graph HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        old: dict = {
            "_source_version": str(_io.read_attr(handle, "version")),
            "num_sites": int(_io.read_attr(handle, "num_sites")),
            "adjacency_sparse": [
                [int(row), int(col), float(value)] for row, col, value in handle["adjacency_sparse"][:]
            ],
        }
        if "edge_coloring" in handle:
            old["edge_coloring"] = [[int(i), int(j), int(color)] for i, j, color in handle["edge_coloring"][:]]
        return old


def to_new_json(old: dict) -> dict:
    """Add the first graph serialization version; the stored graph is unchanged."""
    new = {key: value for key, value in old.items() if key != "_source_version"}
    new["version"] = LATTICE_GRAPH_VERSION
    return new


STEPS = {UNVERSIONED: (LATTICE_GRAPH_VERSION, to_new_json)}
