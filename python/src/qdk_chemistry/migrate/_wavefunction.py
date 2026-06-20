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

import numpy as np

from . import _orbitals, _sbt

WAVEFUNCTION_VERSION = "0.2.0"
CONTAINER_VERSION = "0.2.0"
AMPLITUDE_CONTAINER_VERSION = "0.1.0"

_STATE_VECTOR_TYPES = {"sd", "cas", "sci", "state_vector"}
_AMPLITUDE_TYPES = {"mp2", "coupled_cluster", "cc", "amplitude"}


def from_json_doc(doc: dict) -> dict:
    """Normalize a parsed v1 Wavefunction JSON object into an old-doc."""
    container = doc.get("container", doc)
    tag = container.get("container_type", container.get("type"))
    return {"tag": tag, "container": container}


def from_hdf5_file(path) -> dict:
    """Read a v1 Wavefunction HDF5 file into an old-doc (not yet supported)."""
    raise NotImplementedError(
        "HDF5 wavefunction migration is not yet supported; convert the file to "
        "JSON with an older qdk-chemistry release first, or migrate the JSON form."
    )


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
    """Convert an embedded v1 orbitals JSON object to the v2 schema."""
    return _orbitals.to_new_json(_orbitals.from_json_doc(orbitals_json))


def _to_state_vector(container: dict, tag: str) -> dict:
    """Convert a legacy sd/cas/sci container to a state-vector container."""
    new: dict = {
        "version": CONTAINER_VERSION,
        "container_type": "state_vector",
        "wavefunction_type": container.get("wavefunction_type", "self_dual"),
    }

    if tag == "sd":
        # Single determinant: unit coefficient on the one stored configuration.
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
    if "one_rdm_aa" in rdms and "one_rdm_bb" in rdms:
        new["active_one_rdm"] = _sbt.rank2_dict(
            np.asarray(rdms["one_rdm_aa"], dtype=np.float64),
            np.asarray(rdms["one_rdm_bb"], dtype=np.float64),
        )
    if "two_rdm_aaaa" in rdms and "two_rdm_aabb" in rdms and "two_rdm_bbbb" in rdms:
        new["active_two_rdm"] = _sbt.rank4_dict(
            np.asarray(rdms["two_rdm_aaaa"], dtype=np.float64),
            np.asarray(rdms["two_rdm_aabb"], dtype=np.float64),
            np.asarray(rdms["two_rdm_bbbb"], dtype=np.float64),
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
    if "wavefunction" in container and container["wavefunction"] is not None:
        nested = from_json_doc(container["wavefunction"])
        new["wavefunction"] = to_new_json(nested)
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
