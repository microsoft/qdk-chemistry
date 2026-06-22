"""Low-level I/O helpers for the v1 -> v2 schema migration.

Handles serialization-format and data-type detection, reading dense
arrays out of the v1 HDF5 layout (which stores Eigen column-major matrices
in ``[rows, cols]`` datasets), and writing a reconstructed v2 object back
out in the requested format.
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
import numpy as np

_JSON_SUFFIXES = {".json"}
_HDF5_SUFFIXES = {".h5", ".hdf5", ".he5"}

_TYPE_TOKENS = ("orbitals", "hamiltonian", "wavefunction", "ansatz")


def major_minor(version) -> tuple[int, int]:
    """Parse a ``"major.minor.patch"`` schema version string into ``(major, minor)``."""
    parts = str(version).split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Missing or malformed schema version: {version!r}") from error


def detect_format(path: Path) -> str:
    """Return ``"json"`` or ``"hdf5"`` for a path, based on its suffix."""
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        return "json"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    raise ValueError(f"Cannot determine serialization format from suffix '{path.suffix}' of '{path}'")


def detect_type(path: Path) -> str:
    """Return the data type ('orbitals'/'hamiltonian'/'wavefunction').

    Uses the ``name.type.ext`` filename convention (e.g. ``foo.hamiltonian.h5``).
    """
    tokens = {p.lower() for p in path.name.split(".")}
    matches = [t for t in _TYPE_TOKENS if t in tokens]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Cannot determine data type from filename '{path.name}'. Expected one of "
        f"{_TYPE_TOKENS} in the 'name.type.ext' filename pattern."
    )


def read_matrix(group: h5py.Group, name: str) -> np.ndarray | None:
    """Read a dense matrix written by the C++ ``save_matrix_to_group``.

    The C++ side writes Eigen column-major data into a ``[rows, cols]`` dataset,
    so the row-major bytes h5py returns are the transpose of the logical matrix.
    """
    if name not in group:
        return None
    raw = np.asarray(group[name], dtype=np.float64)
    rows, cols = raw.shape
    return np.ascontiguousarray(raw.reshape(rows * cols).reshape((cols, rows)).T)


def read_vector(group: h5py.Group, name: str) -> np.ndarray | None:
    """Read a 1-D dataset written by the C++ ``save_vector_to_group``."""
    if name not in group:
        return None
    return np.ascontiguousarray(np.asarray(group[name], dtype=np.float64).ravel())


def read_index_vector(group: h5py.Group, name: str) -> list | None:
    """Read a 1-D integer index dataset as a Python list."""
    if name not in group:
        return None
    return [int(v) for v in np.asarray(group[name]).ravel()]


def read_attr(obj, name: str, default=None):
    """Read an HDF5 attribute, decoding bytes to str."""
    if name not in obj.attrs:
        return default
    value = obj.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "O"):
        return value[0].decode("utf-8") if isinstance(value[0], bytes) else str(value[0])
    return value


def subgroup_to_json(group: h5py.Group, data_class, type_token: str) -> dict:
    """Serialize an embedded, schema-unchanged sub-object to JSON.

    Copies ``group`` into a standalone temporary HDF5 file and loads it with the
    given data class' (current, unchanged-schema) ``from_hdf5_file``, then emits
    JSON. Used for nested objects whose schema did not change between v1 and v2
    (e.g. ``BasisSet``), so the migration need not know their internal layout.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / f"sub.{type_token}.h5"
        with h5py.File(h5_path, "w") as dst:
            # The standalone file wraps the payload in a "/<type_token>" group,
            # whereas the embedded form stores it directly in ``group``.
            dst.copy(group, type_token)
        obj = data_class.from_hdf5_file(str(h5_path))
        json_path = Path(tmp) / f"sub.{type_token}.json"
        obj.to_json_file(str(json_path))
        return json.loads(json_path.read_text(encoding="utf-8"))


def write_object(obj, dst: Path, fmt: str) -> None:
    """Write a reconstructed v2 data object to ``dst`` in the given format."""
    if fmt == "json":
        obj.to_json_file(str(dst))
    elif fmt == "hdf5":
        obj.to_hdf5_file(str(dst))
    else:
        raise ValueError(f"Unsupported output format '{fmt}'")
