"""Migrate v1 (<= 1.1.0) serialized data files to the v2 (2.0) schema.

The v2 data-class deserializers intentionally accept only the current schema. Use
this module to upgrade files written by an older ``qdk-chemistry`` release.

Command line::

    python -m qdk_chemistry.migrate OLD_FILE NEW_FILE

Python::

    from qdk_chemistry import migrate
    migrate.convert_file("old.hamiltonian.json", "new.hamiltonian.h5")

The data type is taken from the ``name.type.ext`` filename convention
(``orbitals`` / ``hamiltonian`` / ``wavefunction``) and the serialization format
from the file extension (``.json`` or ``.h5`` / ``.hdf5``). Input and output
formats may differ.

This module is transitional and will be removed once v1 files are no longer
supported; it lives outside the data classes so that no legacy-schema knowledge
leaks into the core serialization.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import h5py

from qdk_chemistry.data import Hamiltonian, Orbitals, Wavefunction

from . import _hamiltonian, _io, _orbitals, _wavefunction

__all__ = ["MigrationError", "convert_file"]

_PathLike = str | Path


class MigrationError(RuntimeError):
    """Raised when a file cannot be migrated to the current schema."""


_MODULES = {
    "orbitals": _orbitals,
    "hamiltonian": _hamiltonian,
    "wavefunction": _wavefunction,
}

_CLASSES = {
    "orbitals": Orbitals,
    "hamiltonian": Hamiltonian,
    "wavefunction": Wavefunction,
}


def convert_file(src: _PathLike, dst: _PathLike) -> Path:
    """Migrate a single v1 data file to the v2 schema.

    Args:
        src: Path to the v1 file to read (format inferred from its extension).
        dst: Path to write the migrated v2 file (format inferred from its extension).

    Returns:
        The ``dst`` path as a :class:`pathlib.Path`.

    Raises:
        MigrationError: If the data type/format cannot be determined, the file is
            already in the v2 schema, or the migration fails.

    """
    src_path = Path(src)
    dst_path = Path(dst)
    try:
        data_type = _io.detect_type(src_path)
        src_format = _io.detect_format(src_path)
        dst_format = _io.detect_format(dst_path)
    except ValueError as error:
        raise MigrationError(str(error)) from error

    module = _MODULES[data_type]
    try:
        old_doc = _read_old(module, data_type, src_path, src_format)
        new_json = module.to_new_json(old_doc)
        obj = _CLASSES[data_type].from_json(json.dumps(new_json))
        _io.write_object(obj, dst_path, dst_format)
    except NotImplementedError as error:
        raise MigrationError(str(error)) from error
    except (KeyError, ValueError, RuntimeError, OSError) as error:
        raise MigrationError(f"Failed to migrate '{src_path}': {error}") from error
    return dst_path


def _read_old(module, data_type: str, src: Path, src_format: str):
    """Read a v1 file into a normalized old-doc for ``data_type``."""
    if src_format == "json":
        doc = json.loads(src.read_text(encoding="utf-8"))
        return module.from_json_doc(doc)

    if data_type == "orbitals":
        with h5py.File(src, "r") as handle:
            return module.from_hdf5_group(handle)
    return module.from_hdf5_file(src)
