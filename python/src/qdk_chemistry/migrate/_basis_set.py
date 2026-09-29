"""Migrate the ``BasisSet`` serialization schema to the current version.

``BasisSet`` 0.1.0 stored each atom's local ECP term at the atom's highest ECP
angular momentum; 0.2.0 labels it ``OrbitalType.UL``. The layout is otherwise
unchanged. A ``BasisSet`` keeps its own serialization version wherever another
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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from qdk_chemistry.data import BasisSet, Structure

from . import _io

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """Normalize a legacy basis-set HDF5 file into the internal old-doc."""
    with h5py.File(path, "r") as handle:
        return from_hdf5_group(handle[_KEY])


def from_hdf5_group(group: h5py.Group) -> dict:
    """Normalize a legacy basis-set HDF5 group into the internal old-doc."""
    metadata = group["metadata"]
    old: dict = {"_source_version": _io.read_attr(group, "version"), "name": _io.read_attr(metadata, "name")}
    if "atomic_orbital_type" in metadata.attrs:
        old["atomic_orbital_type"] = _io.read_attr(metadata, "atomic_orbital_type")
    atoms: dict[int, dict] = {}
    for key in ("shells", "ecp_shells"):
        if key in group:
            for atom_index, shell in _read_shells(group[key]):
                atoms.setdefault(atom_index, {"atom_index": atom_index}).setdefault(key, []).append(shell)
    old["atoms"] = [atoms[index] for index in sorted(atoms)]
    if "ecp_name" in group.attrs:
        old["ecp_name"] = _io.read_attr(group, "ecp_name")
        old["ecp_electrons"] = _io.read_index_vector(group, "ecp_electrons") or []
    if "structure" in group:
        old["structure"] = _io.subgroup_to_json(group["structure"], Structure, "structure")
    return old


def to_new_json(old: dict) -> dict:
    """Build the migrated basis-set JSON object from a normalized old-doc."""
    new = {key: value for key, value in old.items() if key != "_source_version"}
    new["version"] = NEW_VERSION
    new["atoms"] = [_label_local_term(atom) for atom in old.get("atoms", [])]
    return new


def upgrade_embedded(src: Path, dst: Path, fmt: str) -> bool:
    """Copy ``src`` to ``dst`` with every embedded old ``BasisSet`` migrated.

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
    """Replace the old basis-set groups of an HDF5 file with migrated ones."""
    with h5py.File(src, "r") as handle:
        names = _old_group_names(handle)
    if not names:
        return False
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as handle, tempfile.TemporaryDirectory() as tmp:
        for index, name in enumerate(names):
            new_json = _io.migrate_doc(STEPS, from_hdf5_group(handle[name]), "embedded BasisSet")
            migrated_path = Path(tmp) / f"{index}.basis_set.h5"
            BasisSet.from_json(json.dumps(new_json)).to_hdf5_file(str(migrated_path))
            parent, _, leaf = name.rpartition("/")
            del handle[name]
            with h5py.File(migrated_path, "r") as migrated:
                migrated.copy(migrated[_KEY], handle[parent or "/"], name=leaf)
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


def _read_shells(group: h5py.Group) -> Iterator[tuple[int, dict]]:
    """Yield ``(atom_index, shell)`` JSON pairs from a legacy flat shell group."""
    exponents = _io.read_vector(group, "exponents")
    coefficients = _io.read_vector(group, "coefficients")
    rpowers = _io.read_index_vector(group, "rpowers")
    offset = 0
    for atom_index, orbital_type, count in zip(
        group["atom_indices"][()].tolist(),
        group["orbital_types"][()].tolist(),
        group["num_primitives"][()].tolist(),
        strict=True,
    ):
        primitives = slice(offset, offset + count)
        shell = {
            "orbital_type": _ORBITAL_TYPES[orbital_type + 1],
            "exponents": [] if exponents is None else exponents[primitives].tolist(),
            "coefficients": [] if coefficients is None else coefficients[primitives].tolist(),
        }
        if rpowers is not None:
            shell["rpowers"] = rpowers[primitives]
        offset += count
        yield atom_index, shell


def _label_local_term(atom: dict) -> dict:
    """Return ``atom`` with its highest ECP channel labelled local, unless one already is."""
    shells = atom.get("ecp_shells")
    if not shells or any(shell["orbital_type"] == _LOCAL for shell in shells):
        return atom
    highest = max((shell["orbital_type"] for shell in shells), key=_ORBITAL_TYPES.index)
    labelled = [{**shell, "orbital_type": _LOCAL} if shell["orbital_type"] == highest else shell for shell in shells]
    return {**atom, "ecp_shells": labelled}


STEPS = {OLD_VERSION: (NEW_VERSION, to_new_json)}
