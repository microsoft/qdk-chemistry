"""Folder-based cache backend for QDK/Chemistry.

Stores job metadata and content-addressed data as plain files in a directory::

    cache_dir/
        <run_hash>.job.json              # Job metadata
        <content_hash>.<type_name>.h5    # DataClass blobs
        <content_hash>.ndarray.npy       # NumPy arrays
        <content_hash>.list.json         # Supported nested lists

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

import numpy as np

from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data._hashing import _item_content_hash, _numpy_scalar_to_python
from qdk_chemistry.data._type_name import instance_data_type_name
from qdk_chemistry.data.registry import get_dataclass_type
from qdk_chemistry.remote.cache.base import CacheBackend, is_cacheable

if TYPE_CHECKING:
    from qdk_chemistry.data.base import DataClass
    from qdk_chemistry.remote.job import Job


def _resolve_dataclass_type(type_name: str) -> type[DataClass] | None:
    """Find the DataClass loader whose wire-format identifier matches *type_name*."""
    return get_dataclass_type(type_name)


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

    # ── Data blobs ───────────────────────────────────────────────────────

    def get_data(self, content_hash: str) -> Any | None:  # noqa: PLR0911
        """Retrieve cached data by its content hash, or ``None``."""
        self._validate_key(content_hash, "content_hash")
        generic_list_path = self._root / f"{content_hash}.list.json"
        if generic_list_path.exists():
            return self._get_generic_data_list(generic_list_path)
        # Check for list manifest first — escape literal brackets so glob
        # doesn't treat them as a character class.
        list_matches = sorted(self._root.glob(f"{content_hash}.list[[]*].json"))
        if list_matches:
            return self._get_data_list(list_matches[0])
        array_path = self._root / f"{content_hash}.ndarray.npy"
        if array_path.exists():
            try:
                return np.load(array_path, allow_pickle=False)
            except (OSError, ValueError):
                return None
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

    def _get_generic_data_list(self, manifest_path: pathlib.Path) -> list | tuple | None:
        """Reconstruct a nested list or tuple result from a generic manifest."""
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("kind") != "sequence" or manifest.get("sequence_type") not in {"list", "tuple"}:
                return None
            data = self._node_to_data(manifest)
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None
        return data if isinstance(data, list | tuple) else None

    def put_data(self, content_hash: str, data: Any, *, shared_only: bool = False) -> None:
        """Store data by content hash unless shared storage is required but unavailable."""
        if shared_only and not self.is_shared:
            return None
        self._validate_key(content_hash, "content_hash")
        if not is_cacheable(data):
            raise TypeError(f"FolderCache does not support caching values of type {type(data).__name__}")
        if isinstance(data, list | tuple):
            return self._put_data_list(content_hash, data)
        if isinstance(data, np.ndarray):
            filepath = self._root / f"{content_hash}.ndarray.npy"
            if filepath.exists():
                return None
            self._root.mkdir(parents=True, exist_ok=True)
            self._atomic_write_array(filepath, data)
            return None
        type_name = instance_data_type_name(data)
        filepath = self._root / f"{content_hash}.{type_name}.h5"
        if filepath.exists():
            return None  # already cached
        self._root.mkdir(parents=True, exist_ok=True)
        self._atomic_write_hdf5(filepath, data)
        return None

    def _put_data_list(self, content_hash: str, data_list: list | tuple) -> None:
        """Store a sequence of DataClass objects as individual files."""
        if isinstance(data_list, tuple):
            return self._put_generic_data_list(content_hash, data_list)
        if not self._is_homogeneous_dataclass_list(data_list):
            return self._put_generic_data_list(content_hash, data_list)

        type_name = instance_data_type_name(data_list[0])
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

    def _put_generic_data_list(self, content_hash: str, data_list: list | tuple) -> None:
        """Store a list containing nested tuples/lists, DataClass objects, and primitives."""
        manifest_path = self._root / f"{content_hash}.list.json"
        if manifest_path.exists():
            return
        self._root.mkdir(parents=True, exist_ok=True)
        manifest = self._data_to_node(data_list)
        self._atomic_write_text(manifest_path, json.dumps(manifest))

    @staticmethod
    def _is_homogeneous_dataclass_list(data_list: list | tuple) -> bool:
        """Return whether *data_list* can use the legacy homogeneous-list manifest."""
        if not data_list:
            return False
        if not isinstance(data_list[0], CoreDataClass):
            return False
        type_name = instance_data_type_name(data_list[0])
        for item in data_list:
            if not isinstance(item, CoreDataClass):
                return False
            if instance_data_type_name(item) != type_name:
                return False
        return True

    def _data_to_node(self, data: Any) -> dict[str, Any]:
        """Convert supported cached data into a JSON manifest node."""
        if isinstance(data, np.generic):
            data = _numpy_scalar_to_python(data)
        if isinstance(data, list | tuple):
            return {
                "kind": "sequence",
                "sequence_type": "tuple" if isinstance(data, tuple) else "list",
                "items": [self._data_to_node(item) for item in data],
            }
        if data is None or isinstance(data, bool | int | float | str):
            return {"kind": "primitive", "value": data}
        if isinstance(data, CoreDataClass):
            item_hash = _item_content_hash(data)
            self.put_data(item_hash, data)
            return {
                "kind": "dataclass",
                "hash": item_hash,
                "type": instance_data_type_name(data),
            }
        if isinstance(data, np.ndarray):
            item_hash = _item_content_hash(data)
            self.put_data(item_hash, data)
            return {"kind": "ndarray", "hash": item_hash}
        raise TypeError(
            "FolderCache does not support this value graph; expected DataClass objects, NumPy arrays, primitives, "
            "or nested lists/tuples containing them"
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
        if kind == "ndarray":
            path = self._root / f"{node['hash']}.ndarray.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            return np.load(path, allow_pickle=False)
        raise ValueError(f"Unknown cached manifest node kind: {kind!r}")

    def has_data(self, content_hash: str, *, shared_only: bool = False) -> bool:
        """Fast existence check via glob (no deserialization)."""
        self._validate_key(content_hash, "content_hash")
        if shared_only and not self.is_shared:
            return False
        return (
            bool(list(self._root.glob(f"{content_hash}.*.h5")))
            or (self._root / f"{content_hash}.ndarray.npy").exists()
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
        deleted = False
        child_nodes: set[tuple[str, str]] = set()

        # Remove list manifests and their per-item blobs
        generic_manifest = self._root / f"{content_hash}.list.json"
        if generic_manifest.exists():
            try:
                manifest = json.loads(generic_manifest.read_text())
                self._collect_data_nodes(manifest, child_nodes)
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                pass
            try:
                generic_manifest.unlink()
                deleted = True
            except OSError:
                pass

        list_matches = list(self._root.glob(f"{content_hash}.list[[]*].json"))
        for manifest_path in list_matches:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                try:
                    manifest_path.unlink()
                    deleted = True
                except OSError:
                    pass
                continue

            for item_hash in manifest.get("items", []):
                child_nodes.add(("dataclass", item_hash))
            try:
                manifest_path.unlink()
                deleted = True
            except OSError:
                pass

        if child_nodes:
            referenced_nodes = self._referenced_data_nodes()
            for kind, item_hash in child_nodes - referenced_nodes:
                if kind == "dataclass":
                    for f in self._root.glob(f"{item_hash}.*.h5"):
                        deleted = self._unlink_existing(f) or deleted
                elif kind == "ndarray":
                    deleted = self._unlink_existing(self._root / f"{item_hash}.ndarray.npy") or deleted

        # Remove single-object blobs
        for f in self._root.glob(f"{content_hash}.*.h5"):
            deleted = self._unlink_existing(f) or deleted
        array_path = self._root / f"{content_hash}.ndarray.npy"
        if array_path.exists():
            deleted = self._unlink_existing(array_path) or deleted

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

    @staticmethod
    def _collect_data_nodes(node: dict[str, Any], nodes: set[tuple[str, str]]) -> None:
        """Collect data blobs referenced by a generic manifest node."""
        kind = node.get("kind")
        if kind == "dataclass":
            nodes.add((kind, node["hash"]))
            return
        if kind == "ndarray":
            nodes.add((kind, node["hash"]))
            return
        if kind == "sequence":
            for item in node.get("items", []):
                FolderCache._collect_data_nodes(item, nodes)

    def _referenced_data_nodes(self) -> set[tuple[str, str]]:
        """Return data blobs referenced by sequence manifests in the cache."""
        nodes: set[tuple[str, str]] = set()
        for manifest_path in self._root.glob("*.list.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                self._collect_data_nodes(manifest, nodes)
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                continue
        for manifest_path in self._root.glob("*.list[[]*].json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                nodes.update(("dataclass", item_hash) for item_hash in manifest["items"])
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                continue
        return nodes

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
        type_name = instance_data_type_name(data)
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix="tmp_", suffix=f".{type_name}.h5")
        os.close(fd)
        try:
            data.to_hdf5_file(tmp)
            os.replace(tmp, path)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    def _atomic_write_array(self, path: pathlib.Path, data: np.ndarray) -> None:
        """Write a NumPy array atomically via temp file + os.replace."""
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".ndarray.npy")
        try:
            with os.fdopen(fd, "wb") as file:
                np.save(file, data, allow_pickle=False)
            os.replace(tmp, path)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise
