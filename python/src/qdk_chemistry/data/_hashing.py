"""Hashing utilities for deterministic content hashing of data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import hashlib
import struct
import sys
from typing import Any

import numpy as np

__all__: list[str] = []

_NATIVE_IS_BIG = sys.byteorder == "big"


def _hash_bytes(h: "hashlib._Hash", data: bytes) -> None:
    """Feed raw bytes into the hasher."""
    h.update(data)


def _hash_str(h: "hashlib._Hash", s: str) -> None:
    """Hash a string with length prefix to avoid collisions.

    The string is hashed byte-exact (no normalization).
    Use ``_hash_normalized_str`` for inputs where case/whitespace
    should be ignored (e.g. basis set names).
    """
    encoded = s.encode("utf-8")
    h.update(struct.pack("<Q", len(encoded)))
    h.update(encoded)


def _hash_normalized_str(h: "hashlib._Hash", s: str) -> None:
    """Hash a string after normalizing case and whitespace.

    Use this for identifiers where case differences should not
    produce distinct hashes (e.g. ``"STO-3G"`` vs ``"sto-3g"``).
    """
    encoded = s.strip().lower().encode("utf-8")
    h.update(struct.pack("<Q", len(encoded)))
    h.update(encoded)


def _hash_float(h: "hashlib._Hash", f: float) -> None:
    """Hash a float (8 bytes, little-endian)."""
    h.update(struct.pack("<d", float(f)))


def _hash_int(h: "hashlib._Hash", i: int) -> None:
    """Hash an int (8 bytes, little-endian signed)."""
    h.update(struct.pack("<q", int(i)))


def _hash_uint(h: "hashlib._Hash", u: int) -> None:
    """Hash an unsigned int (8 bytes, little-endian)."""
    h.update(struct.pack("<Q", int(u)))


def _hash_bool(h: "hashlib._Hash", b: bool) -> None:
    """Hash a boolean (1 byte)."""
    h.update(b"\x01" if b else b"\x00")


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    """Hash a numpy array deterministically."""
    # Hash shape first for disambiguation
    _hash_uint(h, len(arr.shape))
    for dim in arr.shape:
        _hash_int(h, dim)
    # Force little-endian byte order for cross-platform determinism
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder not in ("<", "=", "|") or (arr.dtype.byteorder == "=" and _NATIVE_IS_BIG):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    # Include dtype to avoid collisions between equal byte payloads of different dtypes
    _hash_str(h, arr.dtype.str)
    h.update(arr.tobytes())


def _hash_optional(h: "hashlib._Hash", val: Any, hash_fn) -> None:
    """Hash an optional value: 0x00 if None, 0x01 + data if present."""
    if val is None:
        h.update(b"\x00")
    else:
        h.update(b"\x01")
        hash_fn(h, val)


def _hash_arg(h: "hashlib._Hash", arg: Any) -> None:
    """Hash an arbitrary argument, dispatching by type."""
    if hasattr(arg, "content_hash"):
        _hash_str(h, arg.content_hash())
    elif isinstance(arg, bool):
        _hash_bool(h, arg)
    elif isinstance(arg, np.bool_):
        _hash_bool(h, bool(arg))
    elif isinstance(arg, int):
        _hash_int(h, arg)
    elif isinstance(arg, np.integer):
        _hash_int(h, int(arg))
    elif isinstance(arg, float):
        _hash_float(h, arg)
    elif isinstance(arg, np.floating):
        _hash_float(h, float(arg))
    elif isinstance(arg, str):
        _hash_str(h, arg)
    elif isinstance(arg, bytes):
        _hash_uint(h, len(arg))
        _hash_bytes(h, arg)
    elif isinstance(arg, np.ndarray):
        _hash_array(h, arg)
    elif isinstance(arg, list | tuple):
        _hash_uint(h, len(arg))
        for item in arg:
            _hash_arg(h, item)
    elif isinstance(arg, dict):
        _hash_uint(h, len(arg))
        for key in sorted(arg.keys(), key=lambda k: (type(k).__name__, str(k))):
            _hash_str(h, type(key).__name__)
            _hash_str(h, str(key))
            _hash_arg(h, arg[key])
        h.update(b"\x00")
    else:
        raise TypeError(f"Unsupported hash argument type: {type(arg).__name__}")


def _hash_setting_value(h: "hashlib._Hash", val: Any) -> None:
    """Hash a Settings value (from Settings.get())."""
    _hash_arg(h, val)


def _item_content_hash(item: Any) -> str:
    """Return a 16-char hex content hash for a single value.

    Uses the object's ``content_hash()`` if available (DataClass),
    otherwise hashes the value via SHA-256.
    """
    if hasattr(item, "content_hash"):
        return item.content_hash()
    h = hashlib.sha256()
    _hash_arg(h, item)
    return h.hexdigest()[:16]


def _type_tag(item: Any) -> str:  # noqa: PLR0911
    """Return a short type tag for an item (used in output_hashes)."""
    if isinstance(item, list):
        if item and hasattr(item[0], "_data_type_name"):
            return f"list[{item[0]._data_type_name}]"  # noqa: SLF001
        return "list"
    if hasattr(item, "_data_type_name"):
        return item._data_type_name  # noqa: SLF001
    if isinstance(item, bool):
        return "bool"
    if isinstance(item, int):
        return "int"
    if isinstance(item, float):
        return "float"
    if isinstance(item, str):
        return "str"
    if item is None:
        return "none"
    return type(item).__name__


def _is_primitive(type_tag: str) -> bool:
    """Return True if the type tag represents a JSON-safe primitive."""
    return type_tag in ("bool", "int", "float", "str", "none")


def collect_content_hashes(result: Any) -> list[dict[str, Any]]:
    """Collect per-item content hashes from an algorithm result.

    For tuple results (e.g. ``(energy, wavefunction)``), returns one
    entry per item.  For a single value, returns a one-element list.

    Each entry is a dict with keys:
    - ``hash``: 16-character hex content hash
    - ``type``: short type tag (e.g. ``"float"``, ``"wavefunction"``)
    - ``value`` *(only for primitives)*: the actual value, so it can be
      reconstructed from the Job JSON without a cache backend.

    Args:
        result: The algorithm result (single value or tuple).

    Returns:
        List of output-hash descriptors.

    """
    items = result if isinstance(result, tuple) else (result,)
    entries: list[dict[str, Any]] = []
    for item in items:
        tag = _type_tag(item)
        entry: dict[str, Any] = {
            "hash": _item_content_hash(item),
            "type": tag,
        }
        if _is_primitive(tag):
            entry["value"] = item
        entries.append(entry)
    return entries
