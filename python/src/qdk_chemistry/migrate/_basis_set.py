"""Migrate the ``BasisSet`` serialization schema to the current version.

``BasisSet`` 0.1.0 stored each atom's local ECP term at the atom's highest ECP
angular momentum; 0.2.0 labels it ``OrbitalType.UL``. The layout is otherwise
unchanged, so HDF5 groups are relabeled in place, while JSON objects go through
``STEPS``. A ``BasisSet`` keeps its own serialization version wherever another
file embeds it, so :func:`upgrade_embedded` migrates those before the enclosing
type's own steps run.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import h5py
import numpy as np

from . import _io

if TYPE_CHECKING:
    from pathlib import Path

NEW_VERSION = "0.2.0"
OLD_VERSION = "0.1.0"

_KEY = "basis_set"
_LOCAL = "ul"
# Orbital-type labels indexed by angular momentum + 1.
_ORBITAL_TYPES = ("ul", "s", "p", "d", "f", "g", "h", "i")


def from_json_doc(doc: dict) -> dict:
    """Normalize a legacy basis-set JSON object into the internal old-doc."""
    old = dict(doc)
    old["_source_version"] = str(old.pop("version", None))
    return old


def from_hdf5_file(path) -> dict:
    """Read only the version of a basis-set HDF5 file; :func:`upgrade_embedded` migrates old ones in place."""
    with h5py.File(path, "r") as handle:
        return {"_source_version": _io.read_attr(handle[_KEY], "version")}


def to_new_json(old: dict) -> dict:
    """Build the migrated basis-set JSON object from a normalized old-doc."""
    new = {key: value for key, value in old.items() if key != "_source_version"}
    new["version"] = NEW_VERSION
    new["atoms"] = [_label_local_term(atom) for atom in old.get("atoms", [])]
    return new


def upgrade_embedded(src: Path, dst: Path, fmt: str) -> bool:
    """Copy ``src`` to ``dst`` with every embedded old ``BasisSet`` migrated.

    HDF5 stores a standalone basis set in a ``basis_set`` group as well, so that is migrated here too.

    Args:
        src: File to read.
        dst: File to write; left unwritten when nothing needs migrating.
        fmt: Serialization format of ``src`` (``"json"`` or ``"hdf5"``).

    Returns:
        Whether ``src`` embedded an old ``BasisSet`` and ``dst`` was written.

    """
    if fmt == "json":
        return _upgrade_embedded_json(src, dst)
    return _upgrade_embedded_hdf5(src, dst)


def _upgrade_embedded_json(src: Path, dst: Path) -> bool:
    """Migrate the old basis sets stored under ``basis_set`` keys of a JSON file."""
    upgraded = False

    def upgrade(obj: dict) -> dict:
        """Replace an old basis set stored under ``obj["basis_set"]``."""
        nonlocal upgraded
        payload = obj.get(_KEY)
        if isinstance(payload, dict) and payload.get("version") in STEPS:
            obj[_KEY] = _io.migrate_doc(STEPS, from_json_doc(payload), "embedded BasisSet")
            upgraded = True
        return obj

    doc = json.loads(src.read_text(encoding="utf-8"), object_hook=upgrade)
    if upgraded:
        dst.write_text(json.dumps(doc), encoding="utf-8")
    return upgraded


def _upgrade_embedded_hdf5(src: Path, dst: Path) -> bool:
    """Copy an HDF5 file to ``dst`` and migrate its old basis-set groups there in place."""
    with h5py.File(src, "r") as handle:
        names = _old_group_names(handle)
    if not names:
        return False
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as handle:
        for name in names:
            _upgrade_group(handle[name])
    return True


def _old_group_names(handle: h5py.File) -> list[str]:
    """Return the paths of the basis-set groups in ``handle`` that have a migration step."""
    names: list[str] = []

    def collect(name: str, obj) -> None:
        """Record ``name`` if it is a basis-set group with a migration step."""
        if isinstance(obj, h5py.Group) and name.rpartition("/")[2] == _KEY and _io.read_attr(obj, "version") in STEPS:
            names.append(name)

    handle.visititems(collect)
    return names


def _upgrade_group(group: h5py.Group) -> None:
    """In place, label each atom's highest ECP channel UL (-1) unless one already is, and bump the version."""
    if "ecp_shells" in group:
        atoms = group["ecp_shells/atom_indices"][()]
        types = group["ecp_shells/orbital_types"][()]
        for atom in np.unique(atoms):
            on_atom = atoms == atom
            if not (types[on_atom] == -1).any():
                types[on_atom & (types == types[on_atom].max())] = -1
        group["ecp_shells/orbital_types"][...] = types
    group.attrs.modify("version", NEW_VERSION)


def _label_local_term(atom: dict) -> dict:
    """Return ``atom`` with its highest ECP channel labelled local, unless one already is."""
    shells = atom.get("ecp_shells")
    if not shells or any(shell["orbital_type"] == _LOCAL for shell in shells):
        return atom
    highest = max((shell["orbital_type"] for shell in shells), key=_ORBITAL_TYPES.index)
    labelled = [{**shell, "orbital_type": _LOCAL} if shell["orbital_type"] == highest else shell for shell in shells]
    return {**atom, "ecp_shells": labelled}


STEPS = {OLD_VERSION: (NEW_VERSION, to_new_json)}
