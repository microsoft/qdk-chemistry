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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdk_chemistry.remote.cache.base import CacheBackend

if TYPE_CHECKING:
    from qdk_chemistry.data.base import DataClass
    from qdk_chemistry.remote.job import Job


def _resolve_dataclass_type(type_name: str) -> type[DataClass] | None:
    """Find the DataClass subclass whose ``_data_type_name`` matches *type_name*."""
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
        path: str | Path,
        *,
        is_shared: bool = False,
        **_kwargs: Any,
    ):
        """Initialise with the cache directory path."""
        super().__init__(is_shared=is_shared)
        self._root = Path(path)

    # ── Job metadata ─────────────────────────────────────────────────────

    def _job_path(self, run_hash: str) -> Path:
        return self._root / f"{run_hash}.job.json"

    def get_job(self, run_hash: str) -> Job | None:
        """Retrieve job metadata by *run_hash*, or ``None`` on miss."""
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        p = self._job_path(run_hash)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return Job(
            job_id=data["job_id"],
            backend=data["backend"],
            backend_config=data.get("backend_config", {}),
            backend_state=data.get("backend_state", {}),
            algorithm_info=data.get("algorithm_info", {}),
            status=data.get("status", "unknown"),
            submitted_at=data.get("submitted_at"),
            file_path=p,
            run_hash=data.get("run_hash"),
            input_hashes=data.get("input_hashes"),
            output_hashes=data.get("output_hashes"),
        )

    def put_job(self, run_hash: str, job: Job) -> None:
        """Store (or update) job metadata keyed by *run_hash*."""
        self._root.mkdir(parents=True, exist_ok=True)
        p = self._job_path(run_hash)
        p.write_text(json.dumps(job.to_dict(), indent=2))

    # ── DataClass blobs ──────────────────────────────────────────────────

    def get_data(self, content_hash: str) -> DataClass | list | None:
        """Retrieve a DataClass object (or list) by its content hash, or ``None``."""
        # Check for list manifest first
        list_matches = list(self._root.glob(f"{content_hash}.list[*].json"))
        if list_matches:
            return self._get_data_list(list_matches[0])
        # Glob for <content_hash>.*.h5 — the type name is in the filename
        matches = list(self._root.glob(f"{content_hash}.*.h5"))
        if not matches:
            return None
        filepath = matches[0]
        # Extract type name from filename: <hash>.<type_name>.h5
        type_name = filepath.name.removeprefix(f"{content_hash}.").removesuffix(".h5")
        dataclass_type = _resolve_dataclass_type(type_name)
        if dataclass_type is None:
            return None
        return dataclass_type.from_hdf5_file(str(filepath))  # type: ignore[attr-defined]

    def _get_data_list(self, manifest_path: Path) -> list | None:
        """Reconstruct a list of DataClass objects from a manifest."""
        import json  # noqa: PLC0415

        manifest = json.loads(manifest_path.read_text())
        dataclass_type = _resolve_dataclass_type(manifest["type"])
        if dataclass_type is None:
            return None
        items = []
        for item_hash in manifest["items"]:
            matches = list(self._root.glob(f"{item_hash}.*.h5"))
            if not matches:
                return None
            items.append(dataclass_type.from_hdf5_file(str(matches[0])))  # type: ignore[attr-defined]
        return items  # type: ignore[return-value]

    def put_data(self, content_hash: str, data: DataClass | list) -> None:
        """Store a DataClass object (or list of them) by content hash."""
        if isinstance(data, list):
            return self._put_data_list(content_hash, data)
        type_name = data._data_type_name  # noqa: SLF001
        filepath = self._root / f"{content_hash}.{type_name}.h5"
        if filepath.exists():
            return None  # already cached
        self._root.mkdir(parents=True, exist_ok=True)
        data.to_hdf5_file(str(filepath))
        return None

    def _put_data_list(self, content_hash: str, data_list: list) -> None:
        """Store a list of DataClass objects as individual files."""
        if not data_list:
            return
        import json  # noqa: PLC0415

        type_name = data_list[0]._data_type_name  # noqa: SLF001
        manifest_path = self._root / f"{content_hash}.list[{type_name}].json"
        if manifest_path.exists():
            return
        self._root.mkdir(parents=True, exist_ok=True)
        item_hashes = []
        for item in data_list:
            item_hash = item.content_hash()[:16]
            self.put_data(item_hash, item)
            item_hashes.append(item_hash)
        manifest_path.write_text(json.dumps({"type": type_name, "items": item_hashes}))

    def has_data(self, content_hash: str) -> bool:
        """Fast existence check via glob (no deserialization)."""
        return bool(list(self._root.glob(f"{content_hash}.*.h5")))

    def to_config(self) -> dict:
        """Return kwargs to reconstruct this FolderCache."""
        return {"path": str(self._root)}

    # ── Deletion ────────────────────────────────────────────────────────────

    def delete_job(self, run_hash: str) -> bool:
        """Remove job metadata and its associated data blobs."""
        p = self._job_path(run_hash)
        if not p.exists():
            return False
        # Remove associated data blobs
        job = self.get_job(run_hash)
        if job and job.output_hashes:
            for entry in job.output_hashes:
                if "value" not in entry:
                    self.delete_data(entry["hash"])
        p.unlink()
        return True

    def delete_data(self, content_hash: str) -> bool:
        """Remove a DataClass blob by content hash."""
        matches = list(self._root.glob(f"{content_hash}.*.h5"))
        if not matches:
            return False
        for f in matches:
            f.unlink()
        return True

    def clear(self) -> None:
        """Remove all cached jobs and data blobs."""
        if not self._root.exists():
            return
        import shutil  # noqa: PLC0415

        shutil.rmtree(self._root)
