"""Convert v1 ``Wavefunction`` serialization to the v2 schema.

The v1 single-determinant (``sd``), complete-active-space (``cas``) and
selected-CI (``sci``) containers were flattened into ``state_vector``; the v1
``mp2``/``coupled_cluster`` containers were flattened into ``amplitude``.

The envelope (``{version, container_type, container}``) and the coefficient /
determinant / amplitude / spin-traced-RDM payloads are schema-stable. What
changed: the embedded orbitals (dense -> ``SymmetryBlockedTensor``) and the
spin-dependent active RDMs (dense ``one_rdm_aa/bb`` and ``two_rdm_aaaa/aabb/bbbb``
-> ``active_one_rdm`` / ``active_two_rdm`` symmetry-blocked tensors).

Spin-dependent RDM components are carried across by their serialized name; the
``(aaaa, aabb, bbbb)`` ordering fix (#499) is a separate data-correctness
concern and is intentionally not reinterpreted here.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py
import numpy as np

from . import _io, _orbitals, _sbt

WAVEFUNCTION_VERSION = "0.2.0"
CONTAINER_VERSION = "0.2.0"
AMPLITUDE_CONTAINER_VERSION = "0.1.0"

_STATE_VECTOR_TYPES = {"sd", "cas", "sci", "state_vector"}
_AMPLITUDE_TYPES = {"mp2", "coupled_cluster", "cc", "amplitude"}
_SPIN_HALF_CHARS = "0ud2"


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed v1 Wavefunction JSON object into an old-doc."""
    container = doc.get("container", doc)
    tag = container.get("container_type", container.get("type"))
    return {"tag": tag, "container": container}


def from_hdf5_file(path) -> dict:
    """Read a v1 Wavefunction HDF5 file into an old-doc."""
    with h5py.File(path, "r") as handle:
        return _read_wavefunction_group(handle["wavefunction"])


def from_hdf5_group(group) -> dict:
    """Read a v1 Wavefunction HDF5 group (the ``wavefunction`` group) into an old-doc."""
    return _read_wavefunction_group(group)


def to_new_json(old: dict) -> dict:
    """Build the v2 Wavefunction JSON object from a normalized old-doc."""
    container = _convert_container(old["container"], old["tag"])
    return {
        "version": WAVEFUNCTION_VERSION,
        "container_type": container["container_type"],
        "container": container,
    }


def _convert_container(container: dict, tag: str) -> dict:
    """Dispatch a legacy wavefunction container to its v2 converter."""
    if tag in _STATE_VECTOR_TYPES:
        return _to_state_vector(container, tag)
    if tag in _AMPLITUDE_TYPES:
        return _to_amplitude(container, tag)
    raise ValueError(f"Unknown legacy wavefunction container type: {tag!r}")


def _convert_orbitals(orbitals_json: dict) -> dict:
    """Convert an embedded v1 orbitals JSON object to the v2 schema (idempotent)."""
    coefficients = orbitals_json.get("coefficients")
    if isinstance(coefficients, dict) and coefficients.get("type") == "SymmetryBlockedTensor":
        return orbitals_json  # already migrated (HDF5 reader path)
    return _orbitals.to_new_json(_orbitals.from_json_doc(orbitals_json))


def _to_state_vector(container: dict, tag: str) -> dict:
    """Convert a legacy sd/cas/sci container to a state-vector container."""
    new: dict = {
        "version": CONTAINER_VERSION,
        "container_type": "state_vector",
        "wavefunction_type": container.get("wavefunction_type", "self_dual"),
    }

    if tag == "sd":
        new["is_complex"] = False
        new["coefficients"] = [1.0]
        new["configuration_set"] = {
            "orbitals": _convert_orbitals(container["orbitals"]),
            "configurations": [container["determinant"]],
        }
        return new

    new["is_complex"] = container.get("is_complex", False)
    new["coefficients"] = container["coefficients"]
    config_set = container["configuration_set"]
    new["configuration_set"] = {
        "orbitals": _convert_orbitals(config_set["orbitals"]),
        "configurations": config_set["configurations"],
    }
    if "sector" in config_set:
        new["configuration_set"]["sector"] = config_set["sector"]

    rdms = _convert_rdms(container.get("rdms"))
    if rdms is not None:
        new["rdms"] = rdms
    return new


def _convert_rdms(rdms):
    """Convert legacy RDM fields to the v2 RDM layout."""
    if not rdms:
        return None
    new: dict = {}
    # Spin-traced RDMs are dense and schema-stable: carry through verbatim.
    for key in (
        "is_one_rdm_spin_traced_complex",
        "one_rdm_spin_traced",
        "is_two_rdm_spin_traced_complex",
        "two_rdm_spin_traced",
    ):
        if key in rdms:
            new[key] = rdms[key]

    # Spin-dependent active RDMs: dense per-channel arrays -> symmetry-blocked.
    # Restricted (closed-shell) files omit the redundant beta channels; the
    # symmetry-blocked builders alias them from the alpha channels.
    if "one_rdm_aa" in rdms:
        if rdms.get("is_one_rdm_aa_complex") or rdms.get("is_one_rdm_bb_complex"):
            raise NotImplementedError("Complex active-RDM migration is not supported.")
        bb = np.asarray(rdms["one_rdm_bb"], dtype=np.float64) if "one_rdm_bb" in rdms else None
        new["active_one_rdm"] = _sbt.rank2_dict(np.asarray(rdms["one_rdm_aa"], dtype=np.float64), bb)
    if "two_rdm_aaaa" in rdms and "two_rdm_aabb" in rdms:
        if any(rdms.get(f"is_two_rdm_{c}_complex") for c in ("aaaa", "aabb", "bbbb")):
            raise NotImplementedError("Complex active-RDM migration is not supported.")
        bbbb = np.asarray(rdms["two_rdm_bbbb"], dtype=np.float64) if "two_rdm_bbbb" in rdms else None
        new["active_two_rdm"] = _sbt.rank4_rdm_dict(
            np.asarray(rdms["two_rdm_aaaa"], dtype=np.float64),
            np.asarray(rdms["two_rdm_aabb"], dtype=np.float64),
            bbbb,
        )
    return new or None


def _to_amplitude(container: dict, tag: str) -> dict:
    """Convert a legacy mp2/coupled-cluster container to an amplitude container."""
    amplitude_type = "moller_plesset" if tag == "mp2" else "coupled_cluster"
    new: dict = {
        "version": AMPLITUDE_CONTAINER_VERSION,
        "container_type": "amplitude",
        "amplitude_type": amplitude_type,
        "is_complex": container.get("is_complex", False),
    }
    if "orbitals" in container:
        new["orbitals"] = _convert_orbitals(container["orbitals"])
    nested = container.get("wavefunction")
    if nested is not None:
        new["wavefunction"] = (
            nested if nested.get("version") == WAVEFUNCTION_VERSION else to_new_json(from_json_doc(nested))
        )
    for key in (
        "t1_amplitudes_aa",
        "t1_amplitudes_bb",
        "t2_amplitudes_abab",
        "t2_amplitudes_aaaa",
        "t2_amplitudes_bbbb",
    ):
        if key in container:
            new[key] = container[key]
    return new


# --------------------------------------------------------------------------- #
# HDF5 readers: rebuild the old-JSON-equivalent container from the v1 layout.
# --------------------------------------------------------------------------- #
def _read_wavefunction_group(group: h5py.Group) -> dict:
    """Read a v1 wavefunction HDF5 group into an old-doc."""
    container_group = group["container"]
    tag = _io.read_attr(container_group, "container_type")
    if tag in _AMPLITUDE_TYPES:
        container = _read_amplitude_hdf5(container_group, tag)
    else:
        container = _read_state_vector_hdf5(container_group, tag)
    return {"tag": tag, "container": container}


def _orbitals_to_new(group: h5py.Group) -> dict:
    return _orbitals.to_new_json(_orbitals.from_hdf5_group(group))


def _read_real_vector(group: h5py.Group, name: str) -> list:
    return np.asarray(group[name], dtype=np.float64).ravel().tolist()


def _decode_configuration(packed, orbital_capacity: int, bits_per_mode: int) -> dict:
    row = np.asarray(packed).ravel()
    modes_per_byte = 8 // bits_per_mode
    mask = (1 << bits_per_mode) - 1
    chars = _SPIN_HALF_CHARS if bits_per_mode == 2 else "01"
    occ = ""
    for pos in range(orbital_capacity):
        byte = int(row[pos // modes_per_byte])
        value = (byte >> ((pos % modes_per_byte) * bits_per_mode)) & mask
        occ += chars[value]
    return {"bits_per_mode": bits_per_mode, "configuration": occ}


def _decode_configurations(dataset) -> list:
    capacity = int(np.asarray(dataset.attrs["orbital_capacity"]).ravel()[0])
    packed_size = int(np.asarray(dataset.attrs["packed_size"]).ravel()[0])
    bits_per_mode = (packed_size * 8) // capacity
    return [_decode_configuration(row, capacity, bits_per_mode) for row in np.asarray(dataset)]


def _read_state_vector_hdf5(container_group: h5py.Group, tag: str) -> dict:
    if _io.read_attr(container_group, "is_complex", 0):
        raise NotImplementedError("Complex wavefunction migration is not supported.")
    new: dict = {
        "container_type": tag,
        "wavefunction_type": _io.read_attr(container_group, "wavefunction_type", "self_dual"),
    }
    if tag == "sd":
        new["orbitals"] = _orbitals_to_new(container_group["orbitals"])
        det = container_group["determinant"]["configuration"]
        capacity = int(np.asarray(det).ravel().shape[0]) * 4
        new["determinant"] = _decode_configuration(det, capacity, 2)
        return new

    config_set = container_group["configuration_set"]
    new["is_complex"] = False
    new["coefficients"] = _read_real_vector(container_group, "coefficients")
    new["configuration_set"] = {
        "orbitals": _orbitals_to_new(config_set["orbitals"]),
        "configurations": _decode_configurations(config_set["configurations"]),
    }
    if "rdms" in container_group:
        new["rdms"] = _read_rdms_hdf5(container_group["rdms"])
    return new


def _read_rdms_hdf5(group: h5py.Group) -> dict:
    rdms: dict = {}
    for name in ("one_rdm_spin_traced", "one_rdm_aa", "one_rdm_bb"):
        if name in group:
            rdms[name] = _io.read_matrix(group, name).tolist()
    for name in ("two_rdm_spin_traced", "two_rdm_aaaa", "two_rdm_aabb", "two_rdm_bbbb"):
        if name in group:
            rdms[name] = _read_real_vector(group, name)
    return rdms


def _read_amplitude_hdf5(container_group: h5py.Group, tag: str) -> dict:
    if _io.read_attr(container_group, "is_complex", 0):
        raise NotImplementedError("Complex wavefunction migration is not supported.")
    new: dict = {"container_type": tag, "is_complex": False}
    if "orbitals" in container_group:
        new["orbitals"] = _orbitals_to_new(container_group["orbitals"])
    if "wavefunction" in container_group:
        nested = _read_wavefunction_group(container_group["wavefunction"])
        new["wavefunction"] = to_new_json(nested)
    for name in (
        "t1_amplitudes_aa",
        "t1_amplitudes_bb",
        "t2_amplitudes_abab",
        "t2_amplitudes_aaaa",
        "t2_amplitudes_bbbb",
    ):
        if name in container_group:
            new[name] = _read_real_vector(container_group, name)
    return new
