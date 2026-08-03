"""Folder-based cache backend for QDK/Chemistry.

Stores job metadata and DataClass blobs as plain files in a directory::

    cache_dir/
        <run_hash>.job.json              # Job metadata
        <content_hash>.<type_name>.h5    # DataClass blobs

Primitives (floats, ints, strings, …) are stored inline in the Job JSON
and never written as separate files.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

from qdk_chemistry.data._hashing import _item_content_hash
from qdk_chemistry.remote.cache.base import CacheBackend

if TYPE_CHECKING:
    from qdk_chemistry.data.base import DataClass
    from qdk_chemistry.remote.job import Job


_dataclass_type_cache: dict[str, type[DataClass]] = {}


def _resolve_dataclass_type(type_name: str) -> type[DataClass] | None:
    """Find the DataClass subclass whose ``_data_type_name`` matches *type_name*."""
    cached = _dataclass_type_cache.get(type_name)
    if cached is not None:
        return cached

    import qdk_chemistry.data  # noqa: PLC0415, F401 — ensure all subclasses are imported
    from qdk_chemistry._core.data import DataClass as _CppBase  # noqa: PLC0415
    from qdk_chemistry.data.base import DataClass as _PyBase  # noqa: PLC0415

    # Walk both the Python and C++ DataClass hierarchies since C++ types
    # (Orbitals, Wavefunction, …) inherit from the pybind11 base, not the
    # Python base.
    stack = list(_PyBase.__subclasses__()) + list(_CppBase.__subclasses__())
    seen: set[int] = set()
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if getattr(cls, "_data_type_name", None) == type_name:
            _dataclass_type_cache[type_name] = cls  # type: ignore[assignment]
            return cls  # type: ignore[return-value]
        stack.extend(cls.__subclasses__())
    return None


class FolderCache(CacheBackend):
    """Content-addressed folder cache.

    Args:
        path: Directory to use as the cache root.  Created on first write.
        is_shared: ``True`` when this directory is a network mount
            reachable from remote compute nodes.

    """

    name = "folder"

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        is_shared: bool = False,
        **_kwargs: Any,
    ):
        """Initialise with the cache directory path."""
        super().__init__(is_shared=is_shared)
        self._root = pathlib.Path(path)

    # ── Job metadata ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_key(key: str, label: str = "key") -> None:
        """Reject keys containing path separators or glob metacharacters."""
        if not key or any(c in key for c in ("/", "\\", "..", "*", "?", "[", "]")):
            raise ValueError(f"Invalid cache {label}: {key!r}")

    def _job_path(self, run_hash: str) -> pathlib.Path:
        self._validate_key(run_hash, "run_hash")
        return self._root / f"{run_hash}.job.json"

    def get_job(self, run_hash: str) -> Job | None:
        """Retrieve job metadata by *run_hash*, or ``None`` on miss."""
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        p = self._job_path(run_hash)
        if not p.exists():
            return None
        try:
            return Job.load(p)
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            return None

    def put_job(self, run_hash: str, job: Job) -> None:
        """Store (or update) job metadata keyed by *run_hash*."""
        self._root.mkdir(parents=True, exist_ok=True)
        p = self._job_path(run_hash)
        self._atomic_write_text(p, json.dumps(job.to_dict(), indent=2))

    # ── DataClass blobs ──────────────────────────────────────────────────

    def get_data(self, content_hash: str) -> DataClass | list | None:
        """Retrieve a DataClass object (or list) by its content hash, or ``None``."""
        self._validate_key(content_hash, "content_hash")
        generic_list_path = self._root / f"{content_hash}.list.json"
        if generic_list_path.exists():
            return self._get_generic_data_list(generic_list_path)
        # Check for list manifest first — escape literal brackets so glob
        # doesn't treat them as a character class.
        list_matches = sorted(self._root.glob(f"{content_hash}.list[[]*].json"))
        if list_matches:
            return self._get_data_list(list_matches[0])
        # Glob for <content_hash>.*.h5 — the type name is in the filename
        matches = sorted(self._root.glob(f"{content_hash}.*.h5"))
        if not matches:
            return None
        filepath = matches[0]
        # Extract type name from filename: <hash>.<type_name>.h5
        type_name = filepath.name.removeprefix(f"{content_hash}.").removesuffix(".h5")
        dataclass_type = _resolve_dataclass_type(type_name)
        if dataclass_type is None:
            return None
        return dataclass_type.from_hdf5_file(str(filepath))  # type: ignore[attr-defined]

    def _get_data_list(self, manifest_path: pathlib.Path) -> list | None:
        """Reconstruct a list of DataClass objects from a manifest."""
        try:
            manifest = json.loads(manifest_path.read_text())
            dataclass_type = _resolve_dataclass_type(manifest["type"])
            item_hashes = manifest["items"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None
        if dataclass_type is None:
            return None
        items = []
        for item_hash in item_hashes:
            matches = sorted(self._root.glob(f"{item_hash}.*.h5"))
            if not matches:
                return None
            items.append(dataclass_type.from_hdf5_file(str(matches[0])))  # type: ignore[attr-defined]
        return items  # type: ignore[return-value]

    def _get_generic_data_list(self, manifest_path: pathlib.Path) -> list | None:
        """Reconstruct a nested list/tuple result from a generic manifest."""
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("kind") != "sequence" or manifest.get("sequence_type") != "list":
                return None
            data = self._node_to_data(manifest)
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None
        return data if isinstance(data, list) else None

    def put_data(self, content_hash: str, data: DataClass | list) -> None:
        """Store a DataClass object (or list of them) by content hash."""
        self._validate_key(content_hash, "content_hash")
        if isinstance(data, list):
            return self._put_data_list(content_hash, data)
        type_name = data._data_type_name  # noqa: SLF001
        filepath = self._root / f"{content_hash}.{type_name}.h5"
        if filepath.exists():
            return None  # already cached
        self._root.mkdir(parents=True, exist_ok=True)
        self._atomic_write_hdf5(filepath, data)
        return None

    def _put_data_list(self, content_hash: str, data_list: list) -> None:
        """Store a list of DataClass objects as individual files."""
        if not self._is_homogeneous_dataclass_list(data_list):
            return self._put_generic_data_list(content_hash, data_list)

        type_name = data_list[0]._data_type_name  # noqa: SLF001
        manifest_path = self._root / f"{content_hash}.list[{type_name}].json"
        if manifest_path.exists():
            return None
        self._root.mkdir(parents=True, exist_ok=True)
        item_hashes = []
        for item in data_list:
            item_hash = item.content_hash()[:16]
            self.put_data(item_hash, item)
            item_hashes.append(item_hash)
        self._atomic_write_text(manifest_path, json.dumps({"type": type_name, "items": item_hashes}))
        return None

    def _put_generic_data_list(self, content_hash: str, data_list: list) -> None:
        """Store a list containing nested tuples/lists, DataClass objects, and primitives."""
        manifest_path = self._root / f"{content_hash}.list.json"
        if manifest_path.exists():
            return
        self._root.mkdir(parents=True, exist_ok=True)
        manifest = self._data_to_node(data_list)
        self._atomic_write_text(manifest_path, json.dumps(manifest))

    @staticmethod
    def _is_homogeneous_dataclass_list(data_list: list) -> bool:
        """Return whether *data_list* can use the legacy homogeneous-list manifest."""
        if not data_list:
            return False
        if not hasattr(data_list[0], "_data_type_name"):
            return False
        type_name = data_list[0]._data_type_name  # noqa: SLF001
        for item in data_list:
            if not hasattr(item, "_data_type_name"):
                return False
            if item._data_type_name != type_name:  # noqa: SLF001
                return False
        return True

    def _data_to_node(self, data: Any) -> dict[str, Any]:
        """Convert supported cached data into a JSON manifest node."""
        if isinstance(data, list | tuple):
            return {
                "kind": "sequence",
                "sequence_type": "tuple" if isinstance(data, tuple) else "list",
                "items": [self._data_to_node(item) for item in data],
            }
        if data is None or isinstance(data, bool | int | float | str):
            return {"kind": "primitive", "value": data}
        if hasattr(data, "_data_type_name"):
            item_hash = _item_content_hash(data)
            self.put_data(item_hash, data)
            return {
                "kind": "dataclass",
                "hash": item_hash,
                "type": data._data_type_name,  # noqa: SLF001
            }
        raise TypeError(
            "FolderCache only supports caching DataClass objects, primitives, and nested lists/tuples containing them"
        )

    def _node_to_data(self, node: dict[str, Any]) -> Any:
        """Reconstruct supported cached data from a JSON manifest node."""
        kind = node["kind"]
        if kind == "primitive":
            return node.get("value")
        if kind == "sequence":
            items = [self._node_to_data(item) for item in node["items"]]
            sequence_type = node["sequence_type"]
            if sequence_type == "list":
                return items
            if sequence_type == "tuple":
                return tuple(items)
            raise ValueError(f"Unknown cached sequence type: {sequence_type!r}")
        if kind == "dataclass":
            dataclass_type = _resolve_dataclass_type(node["type"])
            if dataclass_type is None:
                raise ValueError(f"Unknown cached data type: {node['type']!r}")
            path = self._root / f"{node['hash']}.{node['type']}.h5"
            if not path.exists():
                raise FileNotFoundError(path)
            return dataclass_type.from_hdf5_file(str(path))  # type: ignore[attr-defined]
        raise ValueError(f"Unknown cached manifest node kind: {kind!r}")

    def has_data(self, content_hash: str) -> bool:
        """Fast existence check via glob (no deserialization)."""
        self._validate_key(content_hash, "content_hash")
        return (
            bool(list(self._root.glob(f"{content_hash}.*.h5")))
            or bool(list(self._root.glob(f"{content_hash}.list[[]*].json")))
            or (self._root / f"{content_hash}.list.json").exists()
        )

    def to_config(self) -> dict:
        """Return kwargs to reconstruct this FolderCache."""
        return {"path": str(self._root), "is_shared": self.is_shared}

    # ── Deletion ────────────────────────────────────────────────────────────

    def delete_job(self, run_hash: str) -> bool:
        """Remove job metadata by *run_hash*.

        Only the ``.job.json`` file is deleted.  Data blobs are left intact
        because they may be referenced by other jobs.
        """
        p = self._job_path(run_hash)
        if not p.exists():
            return False
        p.unlink()
        return True

    def delete_data(self, content_hash: str) -> bool:
        """Remove a DataClass blob (or list manifest and its items) by content hash."""
        self._validate_key(content_hash, "content_hash")
        import json as _json  # noqa: PLC0415

        deleted = False

        # Remove list manifests and their per-item blobs
        generic_manifest = self._root / f"{content_hash}.list.json"
        if generic_manifest.exists():
            try:
                manifest = _json.loads(generic_manifest.read_text())
                self._delete_data_nodes(manifest)
            except (_json.JSONDecodeError, OSError, KeyError, TypeError):
                pass
            try:
                generic_manifest.unlink()
                deleted = True
            except OSError:
                pass

        list_matches = list(self._root.glob(f"{content_hash}.list[[]*].json"))
        for manifest_path in list_matches:
            try:
                manifest = _json.loads(manifest_path.read_text())
            except (_json.JSONDecodeError, OSError):
                try:
                    manifest_path.unlink()
                    deleted = True
                except OSError:
                    pass
                continue

            for item_hash in manifest.get("items", []):
                for f in self._root.glob(f"{item_hash}.*.h5"):
                    deleted = self._unlink_existing(f) or deleted
            try:
                manifest_path.unlink()
                deleted = True
            except OSError:
                pass

        # Remove single-object blobs
        for f in self._root.glob(f"{content_hash}.*.h5"):
            deleted = self._unlink_existing(f) or deleted

        return deleted

    @staticmethod
    def _unlink_existing(path: pathlib.Path) -> bool:
        """Unlink *path*, tolerating concurrent deletion."""
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def _delete_data_nodes(self, node: dict[str, Any]) -> None:
        """Delete DataClass blobs referenced by a generic manifest node."""
        kind = node.get("kind")
        if kind == "dataclass":
            for f in self._root.glob(f"{node['hash']}.*.h5"):
                self._unlink_existing(f)
            return
        if kind == "sequence":
            for item in node.get("items", []):
                self._delete_data_nodes(item)

    def clear(self) -> None:
        """Remove all cached jobs and data blobs."""
        if not self._root.exists():
            return
        import shutil  # noqa: PLC0415

        shutil.rmtree(self._root)

    # ── Atomic write helpers ────────────────────────────────────────────────

    def _atomic_write_text(self, path: pathlib.Path, text: str) -> None:
        """Write *text* to *path* atomically via temp file + os.replace."""
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            os.write(fd, text.encode())
            os.close(fd)
            fd = -1
            os.replace(tmp, path)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    def _atomic_write_hdf5(self, path: pathlib.Path, data: DataClass) -> None:
        """Write *data* to *path* atomically via temp file + os.replace."""
        # Temp file must match the <hash>.<type>.h5 naming convention
        # expected by DataClass.to_hdf5_file.
        type_name = data._data_type_name  # noqa: SLF001
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix="tmp_", suffix=f".{type_name}.h5")
        os.close(fd)
        try:
            data.to_hdf5_file(tmp)
            os.replace(tmp, path)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise
