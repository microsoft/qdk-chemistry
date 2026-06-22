"""Convert v1 ``Ansatz`` serialization to the v2 schema.

An ``Ansatz`` is an envelope around a :class:`~qdk_chemistry.data.Hamiltonian`
and a :class:`~qdk_chemistry.data.Wavefunction`. The envelope itself
(``{type, version, hamiltonian, wavefunction}``) is schema-stable; only the two
embedded payloads changed, so this converter simply migrates each in place.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py

from . import _hamiltonian, _wavefunction

ANSATZ_VERSION = "0.1.0"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed v1 Ansatz JSON object into an old-doc."""
    hamiltonian = doc.get("hamiltonian")
    wavefunction = doc.get("wavefunction")
    return {
        "hamiltonian": None if not hamiltonian else _hamiltonian.from_json_doc(hamiltonian),
        "wavefunction": None if not wavefunction else _wavefunction.from_json_doc(wavefunction),
    }


def from_hdf5_file(path) -> dict:
    """Read a v1 Ansatz HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        group = handle.get("ansatz", handle)
        hamiltonian = _hamiltonian.from_hdf5_group(group["hamiltonian"]) if "hamiltonian" in group else None
        wavefunction = _wavefunction.from_hdf5_group(group["wavefunction"]) if "wavefunction" in group else None
    return {"hamiltonian": hamiltonian, "wavefunction": wavefunction}


def to_new_json(old: dict) -> dict:
    """Build the v2 Ansatz JSON object from a normalized old-doc."""
    new: dict = {"version": ANSATZ_VERSION, "type": "Ansatz"}
    new["hamiltonian"] = None if old["hamiltonian"] is None else _hamiltonian.to_new_json(old["hamiltonian"])
    new["wavefunction"] = None if old["wavefunction"] is None else _wavefunction.to_new_json(old["wavefunction"])
    return new
