"""File-based serialization for remote execution of QDK/Chemistry.

This module provides serialization for all QDK Chemistry data classes,
enabling efficient transfer of algorithm inputs and outputs between local
and remote systems. Each DataClass object is serialized to its own HDF5 file.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Type  # noqa: UP035

from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data._hashing import _item_content_hash
from qdk_chemistry.data._type_name import instance_data_type_name
from qdk_chemistry.data.registry import (
    available_dataclasses,
    get_dataclass_type,
)
from qdk_chemistry.data.registry import (
    register_dataclass as _register_dataclass,
)

__all__ = [
    "FileSerializer",
    "deserialize_inputs",
    "deserialize_outputs",
    "get_input_files",
    "get_output_files",
    "get_serialized_file_names",
    "serialize_inputs",
    "serialize_outputs",
]


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON to *path* atomically via a sibling temporary file."""
    file_descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = -1
            json.dump(value, file, indent=2)
        os.replace(temporary_path, path)
    except BaseException:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        Path(temporary_path).unlink(missing_ok=True)
        raise


def _jsonable_settings_value(value: Any) -> Any:
    """Convert a settings value to the tagged JSON form understood by Settings."""
    if isinstance(value, AlgorithmRef):
        settings = value.settings
        return {
            "__type__": "algorithm_ref",
            "algorithm_type": value.algorithm_type,
            "algorithm_name": value.algorithm_name,
            "settings": _jsonable_settings(settings) if settings is not None else None,
        }
    if isinstance(value, dict):
        return {str(key): _jsonable_settings_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_settings_value(item) for item in value]
    return value


def _jsonable_settings(settings: Settings) -> dict[str, Any]:
    """Serialize setting values without relying on native nested-ref JSON support."""
    setting_keys = settings.keys()
    return {key: _jsonable_settings_value(settings.get(key)) for key in setting_keys}


# ─────────────────────────────────────────────────────────────────────────────
# Serializer for individual files
# ─────────────────────────────────────────────────────────────────────────────


class FileSerializer:
    """Handles file-based serialization of QDK Chemistry objects for remote transport.

    Each DataClass object is serialized to its own .{type_name}.h5 file.
    Primitives and simple types are stored in a JSON manifest file. Algorithm
    references embed plain tagged settings dictionaries, recursively preserving
    further nested algorithm references.

    Directory structure for inputs::

        job_dir/
            manifest.json          # Metadata and primitive values
            arg_0.structure.h5     # First arg (if Structure)
            kwarg_0.basis_set.h5   # First keyword arg (if BasisSet)
            ...

    Directory structure for outputs::

        job_dir/
            manifest.json          # Metadata
            result_0.wavefunction.h5   # First result
            ...

    """

    @classmethod
    def register_dataclass(cls, dataclass_type: Type) -> type:  # noqa: UP006
        """Register a DataClass subclass for deserialization.

        Args:
            dataclass_type: A DataClass loader with a static data type name.

        Returns:
            The registered class (allows use as decorator).

        """
        return _register_dataclass(dataclass_type)

    @classmethod
    def _get_dataclass_type(cls, type_name: str) -> Type | None:  # noqa: UP006
        """Get the DataClass type for a type name, with lazy loading."""
        return get_dataclass_type(type_name)

    @classmethod
    def _lazy_load_dataclasses(cls) -> None:
        """Discover and register imported DataClass types."""
        available_dataclasses()

    @classmethod
    def _get_dataclass_extension(cls, type_name: str) -> str:
        """Get the required file extension for a DataClass type.

        Args:
            type_name: The wire-format identifier of the DataClass.

        Returns:
            The extension pattern (e.g., ".structure.h5").

        """
        return f".{type_name}.h5"

    @classmethod
    def is_dataclass(cls, value: Any) -> bool:
        """Check if a value is a QDK Chemistry DataClass."""
        return isinstance(value, CoreDataClass)

    @classmethod
    def is_cacheable(cls, value: Any) -> bool:
        """Check if a value can be stored in a shared cache."""
        return cls.is_dataclass(value) or isinstance(value, list)

    @classmethod
    def serialize_value(  # noqa: PLR0911
        cls,
        directory: Path,
        name: str,
        value: Any,
        *,
        cache: Any = None,
        content_hash: str | None = None,
        seed_cache: bool = False,
    ) -> dict[str, Any]:
        """Serialize a single value, returning manifest entry.

        Args:
            directory: Directory to write files to.
            name: Base name for the file.
            value: Value to serialize.
            cache: Optional shared ``CacheBackend``.  When the cache
                reports ``is_shared`` and already contains the blob, the
                file is **not** written and a ``"cached"`` reference is
                emitted instead.
            content_hash: Optional content hash for *value*.  Used for
                the cache existence check.
            seed_cache: Whether to write a missing cacheable value into the
                shared cache instead of serializing it to a file.

        Returns:
            Manifest entry describing the serialized value.

        """
        # Handle None
        if value is None:
            return {"type": "none", "value": None}

        if isinstance(value, AlgorithmRef):
            settings = value.settings
            return {
                "type": "algorithm_ref",
                "algorithm_type": value.algorithm_type,
                "algorithm_name": value.algorithm_name,
                "settings": _jsonable_settings(settings) if settings is not None else None,
            }

        # A shared cache can replace file transfer for cacheable values.
        if (
            cls.is_cacheable(value)
            and cache is not None
            and content_hash is not None
            and getattr(cache, "is_shared", False)
        ):
            cached = cache.has_data(content_hash)
            if not cached and seed_cache:
                cache.put_data(content_hash, value)
                cached = True
            if cached:
                return {
                    "type": "cached",
                    "dataclass_type": instance_data_type_name(value) if cls.is_dataclass(value) else "list",
                    "content_hash": content_hash,
                }

        # Handle DataClass objects - serialize to individual file
        if cls.is_dataclass(value):
            type_name = instance_data_type_name(value)

            ext = cls._get_dataclass_extension(type_name)
            safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
            filename = f"{safe_name}{ext}"
            filepath = directory / filename
            value.to_hdf5_file(str(filepath))
            entry: dict[str, Any] = {
                "type": "dataclass",
                "dataclass_type": type_name,
                "file": filename,
            }
            if content_hash is not None:
                entry["content_hash"] = content_hash
            return entry

        # Handle primitives
        if isinstance(value, bool):
            return {"type": "bool", "value": value}

        if isinstance(value, int):
            return {"type": "int", "value": value}

        if isinstance(value, float):
            return {"type": "float", "value": value}

        if isinstance(value, str):
            return {"type": "str", "value": value}

        # Handle lists
        if isinstance(value, list):
            items = []
            for i, item in enumerate(value):
                items.append(
                    cls.serialize_value(
                        directory,
                        f"{name}_item_{i}",
                        item,
                        cache=cache,
                        content_hash=_item_content_hash(item) if cls.is_cacheable(item) else None,
                        seed_cache=seed_cache,
                    )
                )
            return {"type": "list", "items": items}

        # Handle tuples
        if isinstance(value, tuple):
            items = []
            for i, item in enumerate(value):
                items.append(
                    cls.serialize_value(
                        directory,
                        f"{name}_item_{i}",
                        item,
                        cache=cache,
                        content_hash=_item_content_hash(item) if cls.is_cacheable(item) else None,
                        seed_cache=seed_cache,
                    )
                )
            return {"type": "tuple", "items": items}

        # Handle dicts
        if isinstance(value, dict):
            entries = []
            for index, (key, item) in enumerate(value.items()):
                entries.append(
                    {
                        "key": cls.serialize_value(
                            directory,
                            f"{name}_entry_{index}_key",
                            key,
                            cache=cache,
                            content_hash=_item_content_hash(key) if cls.is_cacheable(key) else None,
                            seed_cache=seed_cache,
                        ),
                        "value": cls.serialize_value(
                            directory,
                            f"{name}_entry_{index}_value",
                            item,
                            cache=cache,
                            content_hash=_item_content_hash(item) if cls.is_cacheable(item) else None,
                            seed_cache=seed_cache,
                        ),
                    }
                )
            return {"type": "dict", "entries": entries}

        raise TypeError(f"Cannot serialize object of type {type(value).__name__}")

    @classmethod
    def deserialize_value(  # noqa: PLR0911
        cls,
        directory: Path,
        entry: dict[str, Any],
        *,
        cache: Any = None,
    ) -> Any:
        """Deserialize a value from a manifest entry.

        Args:
            directory: Directory containing the files.
            entry: Manifest entry describing the value.
            cache: Optional ``CacheBackend`` used to resolve
                ``"cached"`` entries that were not uploaded as files.

        Returns:
            The deserialized value.

        """
        type_tag = entry["type"]

        if type_tag == "none":
            return None

        if type_tag == "algorithm_ref":
            serialized_settings = entry["settings"]
            settings = Settings.from_json(json.dumps(serialized_settings)) if serialized_settings is not None else None
            return AlgorithmRef(
                entry["algorithm_type"],
                entry["algorithm_name"],
                settings=settings,
            )

        if type_tag == "cached":
            if cache is None:
                raise TypeError("Manifest contains a 'cached' entry but no cache backend was provided to resolve it")
            data = cache.get_data(entry["content_hash"])
            if data is None:
                raise LookupError(
                    f"Cache miss for content_hash={entry['content_hash']!r} (type={entry.get('dataclass_type', '?')})"
                )
            return data

        if type_tag == "dataclass":
            dataclass_type_name = entry["dataclass_type"]
            dataclass_type = cls._get_dataclass_type(dataclass_type_name)
            if dataclass_type is None:
                raise TypeError(f"Unknown DataClass type: {dataclass_type_name}")
            filepath = directory / entry["file"]
            # All QDK Chemistry DataClass types have from_hdf5_file
            return dataclass_type.from_hdf5_file(str(filepath))  # type: ignore[attr-defined]

        if type_tag == "bool":
            return bool(entry["value"])

        if type_tag == "int":
            return int(entry["value"])

        if type_tag == "float":
            return float(entry["value"])

        if type_tag == "str":
            return str(entry["value"])

        if type_tag == "list":
            return [cls.deserialize_value(directory, item, cache=cache) for item in entry["items"]]

        if type_tag == "tuple":
            return tuple(cls.deserialize_value(directory, item, cache=cache) for item in entry["items"])

        if type_tag == "dict":
            return {
                cls.deserialize_value(directory, item["key"], cache=cache): cls.deserialize_value(
                    directory, item["value"], cache=cache
                )
                for item in entry["entries"]
            }

        raise TypeError(f"Unknown type tag: {type_tag}")


def get_serialized_file_names(entry: dict[str, Any]) -> list[str]:
    """Return every file referenced by a serialized manifest entry."""
    type_tag = entry["type"]

    if type_tag == "dataclass":
        return [entry["file"]]

    if type_tag in ("list", "tuple"):
        return [filename for item in entry["items"] for filename in get_serialized_file_names(item)]

    if type_tag == "dict":
        return [
            filename
            for item in entry["entries"]
            for value in (item["key"], item["value"])
            for filename in get_serialized_file_names(value)
        ]

    if type_tag in ("none", "algorithm_ref", "cached", "bool", "int", "float", "str"):
        return []

    raise TypeError(f"Unknown type tag: {type_tag}")


def serialize_inputs(
    directory: str | Path,
    args: tuple,
    kwargs: dict,
    algorithm_type: str,
    algorithm_name: str,
    settings: dict,
    *,
    run_hash: str | None = None,
    input_hashes: dict[str, str] | None = None,
    remote_cache: dict[str, str] | None = None,
    remote_cache_backend: Any = None,
    remote_cache_transport: bool = False,
) -> list[Path]:
    """Serialize algorithm inputs to a directory of files.

    Args:
        directory: Directory to write files to.
        args: Positional arguments for the algorithm.
        kwargs: Keyword arguments for the algorithm.
        algorithm_type: Type of algorithm (e.g., "scf_solver").
        algorithm_name: Name of algorithm implementation.
        settings: Algorithm settings dictionary.
        run_hash: Optional pre-computed algorithm run hash.
        input_hashes: Optional dict mapping input names to their content hashes.
        remote_cache: Optional cache backend coordinates (``{"name": ..., ...}``)
            passed to the remote so it can instantiate the same cache via
            ``get_cache()``.
        remote_cache_backend: Optional ``CacheBackend`` instance.  When
            ``is_shared`` is true and a DataClass blob already exists in
            this cache, the HDF5 file is **not** written and a ``"cached"``
            reference is emitted in the manifest instead.
        remote_cache_transport: Whether the shared cache is the job's artifact
            transport rather than an optional cache optimization.

    Returns:
        List of all files created (for upload).

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest: dict[str, Any] = {
        "algorithm_type": algorithm_type,
        "algorithm_name": algorithm_name,
        "settings": {},
        "args": [],
        "kwargs": {},
    }
    if run_hash is not None:
        manifest["run_hash"] = run_hash
    if input_hashes is not None:
        manifest["input_hashes"] = input_hashes
    if remote_cache is not None:
        manifest["remote_cache"] = remote_cache
    if remote_cache_transport:
        manifest["remote_cache_transport"] = True

    # Serialize settings
    for index, (key, value) in enumerate(settings.items()):
        manifest["settings"][key] = FileSerializer.serialize_value(
            directory,
            f"setting_{index}",
            value,
            cache=remote_cache_backend,
            content_hash=_item_content_hash(value) if FileSerializer.is_cacheable(value) else None,
            seed_cache=remote_cache_transport,
        )

    # Serialize positional arguments
    for i, arg in enumerate(args):
        chash = input_hashes.get(f"arg_{i}") if input_hashes else None
        entry = FileSerializer.serialize_value(
            directory,
            f"arg_{i}",
            arg,
            cache=remote_cache_backend,
            content_hash=chash,
            seed_cache=remote_cache_transport,
        )
        if chash and "content_hash" not in entry:
            entry["content_hash"] = chash
        manifest["args"].append(entry)

    # Serialize keyword arguments
    for index, (key, value) in enumerate(kwargs.items()):
        chash = input_hashes.get(key) if input_hashes else None
        entry = FileSerializer.serialize_value(
            directory,
            f"kwarg_{index}",
            value,
            cache=remote_cache_backend,
            content_hash=chash,
            seed_cache=remote_cache_transport,
        )
        if chash and "content_hash" not in entry:
            entry["content_hash"] = chash
        manifest["kwargs"][key] = entry

    files_created = [
        directory / filename
        for entries in (manifest["settings"].values(), manifest["args"], manifest["kwargs"].values())
        for entry in entries
        for filename in get_serialized_file_names(entry)
    ]

    manifest_path = directory / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    files_created.insert(0, manifest_path)

    return files_created


def deserialize_inputs(directory: str | Path, *, cache: Any = None) -> dict:
    """Deserialize algorithm inputs from a directory.

    Args:
        directory: Directory containing the input files.
        cache: Optional ``CacheBackend`` used to resolve ``"cached"``
            manifest entries that were not uploaded as files.

    Returns:
        Deserialized inputs containing ``algorithm_type``, ``algorithm_name``,
        ``settings``, ``args``, and ``kwargs``.

    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Deserialize settings
    settings = {}
    for key, entry in manifest["settings"].items():
        settings[key] = FileSerializer.deserialize_value(directory, entry, cache=cache)

    # Deserialize positional arguments
    args = tuple(FileSerializer.deserialize_value(directory, entry, cache=cache) for entry in manifest["args"])

    # Deserialize keyword arguments
    kwargs = {}
    for key, entry in manifest["kwargs"].items():
        kwargs[key] = FileSerializer.deserialize_value(directory, entry, cache=cache)

    return {
        "algorithm_type": manifest["algorithm_type"],
        "algorithm_name": manifest["algorithm_name"],
        "settings": settings,
        "args": args,
        "kwargs": kwargs,
        "run_hash": manifest.get("run_hash"),
        "input_hashes": manifest.get("input_hashes"),
        "remote_cache": manifest.get("remote_cache"),
    }


def serialize_outputs(directory: str | Path, result: Any) -> list[Path]:
    """Serialize algorithm outputs to a directory.

    Args:
        directory: Directory to write files to.
        result: The result from algorithm.run() (may be a tuple or single value).

    Returns:
        List of all files created (for download).

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest: dict[str, Any] = {"is_tuple": False, "results": []}

    # Handle tuple results (common pattern: (energy, wavefunction))
    if isinstance(result, tuple):
        manifest["is_tuple"] = True
        for i, item in enumerate(result):
            entry = FileSerializer.serialize_value(directory, f"result_{i}", item)
            if hasattr(item, "content_hash"):
                entry["content_hash"] = item.content_hash()
            manifest["results"].append(entry)
    else:
        entry = FileSerializer.serialize_value(directory, "result", result)
        if hasattr(result, "content_hash"):
            entry["content_hash"] = result.content_hash()
        manifest["results"].append(entry)

    files_created = [
        directory / filename for entry in manifest["results"] for filename in get_serialized_file_names(entry)
    ]

    manifest_path = directory / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    files_created.insert(0, manifest_path)

    return files_created


def deserialize_outputs(directory: str | Path) -> Any:
    """Deserialize algorithm outputs from a directory.

    Args:
        directory: Directory containing the output files.

    Returns:
        The deserialized result (tuple or single value).

    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    results = [FileSerializer.deserialize_value(directory, entry) for entry in manifest["results"]]

    if manifest["is_tuple"]:
        return tuple(results)
    return results[0] if results else None


def get_input_files(directory: str | Path) -> list[Path]:
    """Get list of all input files in a directory.

    Args:
        directory: Directory containing input files.

    Returns:
        List of all files that should be uploaded.

    """
    directory = Path(directory)
    files = [directory / "manifest.json"]
    files.extend(directory.glob("*.h5"))
    return [f for f in files if f.exists()]


def get_output_files(directory: str | Path) -> list[Path]:
    """Get list of all output files in a directory.

    Args:
        directory: Directory containing output files.

    Returns:
        List of all files that should be downloaded.

    """
    directory = Path(directory)
    files = [directory / "manifest.json"]
    files.extend(directory.glob("*.h5"))
    return [f for f in files if f.exists()]
