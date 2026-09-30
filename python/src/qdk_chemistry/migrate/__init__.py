"""Migrate serialized data files across serialization-schema versions.

Each qdk-chemistry data class versions its serialization schema independently, and
its deserializer accepts only the version the installed library was built against.
Loading a file written against an older version of that class's schema raises an
error; use this module to upgrade such a file to the version the library accepts.

Command line::

    python -m qdk_chemistry.migrate OLD_FILE NEW_FILE

Python::

    from qdk_chemistry import migrate
    migrate.convert_file("old.hamiltonian.json", "new.hamiltonian.h5")

The data type is taken from the ``name.type.ext`` filename convention
(``basis_set`` / ``orbitals`` / ``hamiltonian`` / ``wavefunction`` / ``ansatz`` / ``qpe_result``) and the
serialization format from the file extension (``.json`` or ``.h5`` / ``.hdf5``).
Input and output formats may differ.

Migration is keyed point-by-point on each data class's serialization version, not
on a library release: every migratable type exposes a ``STEPS`` table mapping a
source version to a ``(next_version, transform)`` pair, and the chain is followed
until it reaches the version the installed library accepts. To support a future
serialization-version bump for a data class, register the next step in that type's
``STEPS`` table (``_basis_set``/``_orbitals``/``_hamiltonian``/``_wavefunction``/``_qpe_result``);
the migrated document is validated against the live deserializer, so a missing step
fails loudly.

A ``BasisSet`` keeps its own serialization version wherever another file embeds it,
so embedded basis sets are migrated first; a file whose own version is already
current is then complete.

This module lives outside the data classes so that no legacy-schema knowledge
leaks into the core serialization.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py

from qdk_chemistry.data import Ansatz, BasisSet, Hamiltonian, Orbitals, QpeResult, Wavefunction

from . import _ansatz, _basis_set, _hamiltonian, _io, _orbitals, _qpe_result, _wavefunction

__all__ = ["MigrationError", "convert_file"]

_PathLike = str | Path


class MigrationError(RuntimeError):
    """Raised when a file cannot be migrated to the current serialization version."""


def _resolves_to_same_file(src: Path, dst: Path) -> bool:
    """Return True if ``src`` and ``dst`` point at the same file on disk."""
    try:
        if src.exists() and dst.exists():
            return src.samefile(dst)
    except OSError:
        pass
    try:
        return src.resolve() == dst.resolve()
    except OSError:
        return src == dst


_MODULES = {
    "basis_set": _basis_set,
    "orbitals": _orbitals,
    "hamiltonian": _hamiltonian,
    "wavefunction": _wavefunction,
    "ansatz": _ansatz,
    "qpe_result": _qpe_result,
}

_CLASSES = {
    "basis_set": BasisSet,
    "orbitals": Orbitals,
    "hamiltonian": Hamiltonian,
    "wavefunction": Wavefunction,
    "ansatz": Ansatz,
    "qpe_result": QpeResult,
}


def convert_file(src: _PathLike, dst: _PathLike) -> Path:
    """Migrate a single data file to the serialization version the library accepts.

    Args:
        src: Path to the file to read (format inferred from its extension).
        dst: Path to write the migrated file (format inferred from its extension).

    Returns:
        The ``dst`` path as a :class:`pathlib.Path`.

    Raises:
        MigrationError: If the data type/format cannot be determined, the source and
            destination are the same file, the file is already at the current
            serialization version, or the migration fails.

    """
    src_path = Path(src)
    dst_path = Path(dst)
    if _resolves_to_same_file(src_path, dst_path):
        raise MigrationError(
            f"Refusing to migrate '{src_path}' in place: the source and destination resolve to "
            "the same file. Choose a different output path so the original is not overwritten."
        )
    try:
        data_type = _io.detect_type(src_path)
        src_format = _io.detect_format(src_path)
        dst_format = _io.detect_format(dst_path)
    except ValueError as error:
        raise MigrationError(str(error)) from error

    with tempfile.TemporaryDirectory() as tmp:
        # The installed deserializers expect the name.type.ext pattern.
        staged = Path(tmp) / f"staged.{data_type}.{'json' if src_format == 'json' else 'h5'}"
        try:
            upgraded = _basis_set.upgrade_embedded(src_path, staged, src_format)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            raise MigrationError(f"Failed to migrate '{src_path}': {error}") from error
        obj = _load_current(data_type, staged, src_format) if upgraded else None
        if obj is None:
            obj = _migrate(data_type, staged if upgraded else src_path, src_format, src_path)

    try:
        _io.write_object(obj, dst_path, dst_format)
    except (OSError, RuntimeError) as error:
        raise MigrationError(f"Failed to write '{dst_path}': {error}") from error
    return dst_path


def _load_current(data_type: str, path: Path, fmt: str):
    """Load ``path`` with the installed class, or return None if its own version is outdated."""
    try:
        return _CLASSES[data_type].from_file(str(path), fmt)
    except (RuntimeError, ValueError):
        return None


def _migrate(data_type: str, path: Path, fmt: str, src_path: Path):
    """Migrate ``path`` through its type's serialization-version steps and load the result."""
    module = _MODULES[data_type]
    try:
        old_doc = _read_old(module, data_type, path, fmt)
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
        json_input = new_json if data_type == "qpe_result" else json.dumps(new_json)
        return _CLASSES[data_type].from_json(json_input)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MigrationError(
            f"The installed qdk-chemistry rejected the migrated {data_type} ({error}). If the "
            f"{data_type} serialization schema changed, register the next step in {module.__name__}.STEPS."
        ) from error


def _read_old(module, data_type: str, src: Path, src_format: str):
    """Read a legacy file into a normalized old-doc for ``data_type``."""
    if src_format == "json":
        doc = json.loads(src.read_text(encoding="utf-8"))
        return module.from_json_doc(doc)

    if data_type == "orbitals":
        with h5py.File(src, "r") as handle:
            return module.from_hdf5_group(handle)
    return module.from_hdf5_file(src)
