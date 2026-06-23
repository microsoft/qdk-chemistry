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

Migration is keyed point-by-point on each object's serialization version, not on
a release: every migratable type exposes a ``STEPS`` table mapping a source
version to a ``(next_version, transform)`` pair, and the chain is followed until
it reaches the schema the installed library accepts. To support a future
serialization-version bump, register the next step in that type's ``STEPS`` table
(``_orbitals``/``_hamiltonian``/``_wavefunction``); the migrated document is
validated against the live deserializer, so a missing step fails loudly.

This module lives outside the data classes so that no legacy-schema knowledge
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

from qdk_chemistry.data import Ansatz, Hamiltonian, Orbitals, Wavefunction

from . import _ansatz, _hamiltonian, _io, _orbitals, _wavefunction

__all__ = ["MigrationError", "convert_file"]

_PathLike = str | Path


class MigrationError(RuntimeError):
    """Raised when a file cannot be migrated to the current schema."""


_MODULES = {
    "orbitals": _orbitals,
    "hamiltonian": _hamiltonian,
    "wavefunction": _wavefunction,
    "ansatz": _ansatz,
}

_CLASSES = {
    "orbitals": Orbitals,
    "hamiltonian": Hamiltonian,
    "wavefunction": Wavefunction,
    "ansatz": Ansatz,
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
        if data_type == "ansatz":
            # An Ansatz has no version step of its own; it delegates to each
            # embedded payload's serialization-version chain.
            new_json = module.to_new_json(old_doc)
        else:
            new_json = _io.migrate_doc(module.STEPS, old_doc, data_type)
    except NotImplementedError as error:
        raise MigrationError(str(error)) from error
    except (KeyError, ValueError, RuntimeError, OSError) as error:
        raise MigrationError(f"Failed to migrate '{src_path}': {error}") from error

    try:
        obj = _CLASSES[data_type].from_json(json.dumps(new_json))
    except RuntimeError as error:
        raise MigrationError(
            f"The installed qdk-chemistry rejected the migrated {data_type} ({error}). If the "
            f"{data_type} serialization schema changed, register the next step in {module.__name__}.STEPS."
        ) from error

    try:
        _io.write_object(obj, dst_path, dst_format)
    except (OSError, RuntimeError) as error:
        raise MigrationError(f"Failed to write '{dst_path}': {error}") from error
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
