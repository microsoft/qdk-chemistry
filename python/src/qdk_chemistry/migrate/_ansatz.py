"""Convert v1 ``Ansatz`` serialization to the v2 schema.

An ``Ansatz`` is an envelope around a :class:`~qdk_chemistry.data.Hamiltonian`
and a :class:`~qdk_chemistry.data.Wavefunction`. The envelope itself
(``{type, version, hamiltonian, wavefunction}``) is schema-stable, so this
converter has no version step of its own; it migrates each embedded payload
through that payload's own serialization-version chain.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py

from . import _hamiltonian, _io, _wavefunction

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
    """Build the v2 Ansatz JSON object by migrating each embedded payload's chain."""
    new: dict = {"version": ANSATZ_VERSION, "type": "Ansatz"}
    hamiltonian = old["hamiltonian"]
    wavefunction = old["wavefunction"]
    new["hamiltonian"] = (
        None if hamiltonian is None else _io.migrate_doc(_hamiltonian.STEPS, hamiltonian, "embedded Hamiltonian")
    )
    new["wavefunction"] = (
        None if wavefunction is None else _io.migrate_doc(_wavefunction.STEPS, wavefunction, "embedded Wavefunction")
    )
    return new
