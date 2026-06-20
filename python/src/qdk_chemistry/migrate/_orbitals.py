"""Convert v1 ``Orbitals`` serialization to the v2 schema.

The only fields whose representation changed are ``coefficients`` and
``energies`` (dense per-spin arrays -> ``SymmetryBlockedTensor``). Everything
else (active/inactive index sets, AO overlap, basis set, scalar metadata) is
schema-stable and carried through unchanged.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.data import BasisSet

from . import _io, _sbt

if TYPE_CHECKING:
    import h5py

NEW_VERSION = "0.2.0"
OLD_VERSION = "0.1.0"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed v1 orbitals JSON object into the internal old-doc."""
    coeff = doc["coefficients"]
    energies = doc.get("energies")
    return {
        "is_restricted": bool(doc.get("is_restricted", True)),
        "num_atomic_orbitals": doc.get("num_atomic_orbitals"),
        "num_molecular_orbitals": doc.get("num_molecular_orbitals"),
        "coefficients_alpha": np.asarray(coeff["alpha"], dtype=np.float64),
        "coefficients_beta": np.asarray(coeff["beta"], dtype=np.float64),
        "energies_alpha": None if energies is None else np.asarray(energies["alpha"], dtype=np.float64),
        "energies_beta": None if energies is None else np.asarray(energies["beta"], dtype=np.float64),
        "ao_overlap": None if "ao_overlap" not in doc else np.asarray(doc["ao_overlap"], dtype=np.float64),
        "active_space_indices": doc.get("active_space_indices"),
        "inactive_space_indices": doc.get("inactive_space_indices"),
        "basis_set": doc.get("basis_set"),
    }


def from_hdf5_group(group: h5py.Group) -> dict:
    """Normalize a v1 orbitals HDF5 group into the internal old-doc."""
    metadata = group["metadata"]
    is_restricted = bool(np.asarray(metadata["is_restricted"]).ravel()[0])

    active = _io.read_index_vector(group, "active_space_indices_alpha")
    inactive = _io.read_index_vector(group, "inactive_space_indices_alpha")
    energies_alpha = _io.read_vector(group, "energies_alpha")

    basis_set = None
    if "basis_set" in group:
        basis_set = _io.subgroup_to_json(group["basis_set"], BasisSet)

    return {
        "is_restricted": is_restricted,
        "num_atomic_orbitals": int(np.asarray(metadata["num_atomic_orbitals"]).ravel()[0]),
        "num_molecular_orbitals": int(np.asarray(metadata["num_molecular_orbitals"]).ravel()[0]),
        "coefficients_alpha": _io.read_matrix(group, "coefficients_alpha"),
        "coefficients_beta": _io.read_matrix(group, "coefficients_beta"),
        "energies_alpha": energies_alpha,
        "energies_beta": _io.read_vector(group, "energies_beta"),
        "ao_overlap": _io.read_matrix(group, "ao_overlap"),
        "active_space_indices": None
        if active is None
        else {"alpha": active, "beta": _io.read_index_vector(group, "active_space_indices_beta")},
        "inactive_space_indices": None
        if inactive is None
        else {"alpha": inactive, "beta": _io.read_index_vector(group, "inactive_space_indices_beta")},
        "basis_set": basis_set,
    }


def to_new_json(old: dict) -> dict:
    """Build the v2 orbitals JSON object from a normalized old-doc."""
    restricted = old["is_restricted"]
    coeff_beta = None if restricted else old["coefficients_beta"]
    new: dict = {
        "version": NEW_VERSION,
        "type": "Orbitals",
        "is_restricted": restricted,
        "coefficients": _sbt.rank2_dict(old["coefficients_alpha"], coeff_beta),
    }
    if old.get("num_atomic_orbitals") is not None:
        new["num_atomic_orbitals"] = int(old["num_atomic_orbitals"])
    if old.get("num_molecular_orbitals") is not None:
        new["num_molecular_orbitals"] = int(old["num_molecular_orbitals"])

    if old.get("energies_alpha") is not None:
        energy_beta = None if restricted else old["energies_beta"]
        new["energies"] = _sbt.rank1_dict(old["energies_alpha"], energy_beta)

    ao_overlap = old.get("ao_overlap")
    new["has_overlap_matrix"] = ao_overlap is not None
    if ao_overlap is not None:
        new["ao_overlap"] = np.asarray(ao_overlap, dtype=np.float64).tolist()

    if old.get("active_space_indices") is not None:
        new["active_space_indices"] = _index_pair(old["active_space_indices"])
    if old.get("inactive_space_indices") is not None:
        new["inactive_space_indices"] = _index_pair(old["inactive_space_indices"])
    if old.get("basis_set") is not None:
        new["basis_set"] = old["basis_set"]
    return new


def _index_pair(indices: dict) -> dict:
    """Return an alpha/beta index dict with integer entries."""
    return {
        "alpha": [int(i) for i in indices["alpha"]],
        "beta": [int(i) for i in indices["beta"]],
    }
