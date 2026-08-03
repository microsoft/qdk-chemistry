"""Migrate the ``QpeResult`` serialization schema to the current version."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import h5py

from . import _io

OLD_VERSION = "0.1.0"
QPE_RESULT_VERSION = "0.2.0"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed legacy QPE result into an old-doc."""
    old = dict(doc)
    old["_source_version"] = str(old.pop("version", None))
    return old


def from_hdf5_file(path) -> dict:
    """Read a legacy QPE-result HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        old: dict = {
            "_source_version": str(_io.read_attr(handle, "version")),
            "method": _io.read_attr(handle, "method"),
            "phase_fraction": float(_io.read_attr(handle, "phase_fraction")),
            "phase_angle": float(_io.read_attr(handle, "phase_angle")),
            "canonical_phase_fraction": float(_io.read_attr(handle, "canonical_phase_fraction")),
            "canonical_phase_angle": float(_io.read_attr(handle, "canonical_phase_angle")),
            "raw_energy": float(_io.read_attr(handle, "raw_energy")),
            "branching": [float(value) for value in handle["branching"][:]],
        }
        if "resolved_energy" in handle.attrs:
            old["resolved_energy"] = float(_io.read_attr(handle, "resolved_energy"))
        if "bits_msb_first" in handle:
            old["bits_msb_first"] = [int(value) for value in handle["bits_msb_first"][:]]
        if "bitstring_msb_first" in handle.attrs:
            old["bitstring_msb_first"] = _io.read_attr(handle, "bitstring_msb_first")
        if "metadata" in handle.attrs:
            old["metadata"] = json.loads(_io.read_attr(handle, "metadata"))
        return old


def to_new_json(old: dict) -> dict:
    """Remove the obsolete evolution time and emit the current schema."""
    new = {key: value for key, value in old.items() if key not in {"_source_version", "evolution_time"}}
    new["version"] = QPE_RESULT_VERSION
    return new


STEPS = {OLD_VERSION: (QPE_RESULT_VERSION, to_new_json)}
