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
from contextlib import suppress
from pathlib import Path
from typing import Any, Type  # noqa: UP035

import numpy as np

from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data._hashing import _item_content_hash, _numpy_scalar_to_python
from qdk_chemistry.data._type_name import instance_data_type_name
from qdk_chemistry.data.registry import (
    available_dataclasses,
    get_dataclass_type,
)
from qdk_chemistry.data.registry import (
    register_dataclass as _register_dataclass,
)
from qdk_chemistry.remote.cache.base import CacheBackend, is_cacheable

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

_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_VERSION = 1


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest and validate its schema version."""
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    version = manifest.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version {version!r}; expected {_MANIFEST_VERSION}")
    return manifest


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON to *path* atomically via a sibling temporary file."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(value, file, indent=2)
        if temporary_path is None:
            raise RuntimeError("Temporary serialization file was not created")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_dataclass(path: Path, value: CoreDataClass, type_name: str) -> None:
    """Write a data class atomically unless its content-addressed file already exists."""
    if path.is_file():
        return
    if path.exists():
        raise FileExistsError(f"Serialization file path is not a file: {path}")

    file_descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, suffix=f".{type_name}.h5")
    os.close(file_descriptor)
    try:
        value.to_hdf5_file(temporary_path)
        if path.is_file():
            return
        if path.exists():
            raise FileExistsError(f"Serialization file path is not a file: {path}")
        os.replace(temporary_path, path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _atomic_write_array(path: Path, value: np.ndarray) -> None:
    """Write a NumPy array atomically unless its content-addressed file already exists."""
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise TypeError("Cannot serialize NumPy arrays with object or structured dtype")
    if path.is_file():
        return
    if path.exists():
        raise FileExistsError(f"Serialization file path is not a file: {path}")

    file_descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, suffix=".ndarray.npy")
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            np.save(file, value, allow_pickle=False)
        if path.is_file():
            return
        if path.exists():
            raise FileExistsError(f"Serialization file path is not a file: {path}")
        os.replace(temporary_path, path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _manifest_file_path(directory: Path, filename: str) -> Path:
    """Resolve a manifest file name while enforcing serialization-directory containment."""
    if not isinstance(filename, str) or not filename or "\\" in filename:
        raise ValueError(f"Invalid serialization file name in manifest: {filename!r}")
    resolved_directory = directory.resolve()
    filepath = (resolved_directory / filename).resolve()
    if not filepath.is_relative_to(resolved_directory):
        raise ValueError(f"Manifest file path resolves outside the serialization directory: {filename}")
    return filepath


def _commit_staged_serialization(
    directory: Path,
    staging_directory: Path,
    manifest: dict[str, Any],
    filenames: list[str],
) -> list[Path]:
    """Commit staged artifacts before atomically replacing the live manifest."""
    staged_manifest_path = staging_directory / _MANIFEST_FILENAME
    _atomic_write_json(staged_manifest_path, manifest)

    committed_files: list[Path] = []
    new_files: list[Path] = []
    try:
        for filename in dict.fromkeys(filenames):
            staged_path = _manifest_file_path(staging_directory, filename)
            if not staged_path.is_file():
                raise FileNotFoundError(f"Staged serialization file does not exist: {staged_path}")
            committed_path = _manifest_file_path(directory, filename)
            existed = committed_path.exists()
            os.replace(staged_path, committed_path)
            committed_files.append(committed_path)
            if not existed:
                new_files.append(committed_path)

        manifest_path = directory / _MANIFEST_FILENAME
        os.replace(staged_manifest_path, manifest_path)
    except BaseException:
        for new_file in new_files:
            with suppress(OSError):
                new_file.unlink()
        raise

    return [manifest_path, *committed_files]


def _jsonable_settings_value(value: Any, setting_type: str | None = None) -> Any:
    """Convert a settings value to the tagged JSON form understood by Settings."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
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
        empty_array_element_types = {
            "vector<int64_t>": "int64",
            "vector<double>": "double",
            "vector<string>": "string",
        }
        if not value and setting_type in empty_array_element_types:
            return {
                "__type__": "array",
                "__element_type__": empty_array_element_types[setting_type],
                "__value__": [],
            }
        return [_jsonable_settings_value(item) for item in value]
    return value


def _jsonable_settings(settings: Settings) -> dict[str, Any]:
    """Serialize setting values without relying on native nested-ref JSON support."""
    setting_keys = settings.keys()
    return {key: _jsonable_settings_value(settings.get(key), settings.get_type_name(key)) for key in setting_keys}


class FileSerializer:
    """Handles file-based serialization of QDK Chemistry objects for remote transport.

    Each DataClass object is serialized to its own .{type_name}.h5 file, and
    each NumPy array to its own .ndarray.npy file. Primitives and simple types
    are stored in a JSON manifest file. Algorithm references embed plain tagged
    settings dictionaries, recursively preserving further nested algorithm
    references.

    Directory structure for inputs::

        job_dir/
            manifest.json          # Metadata and primitive values
            <content_hash>.structure.h5
            <content_hash>.basis_set.h5
            ...

    Directory structure for outputs::

        job_dir/
            manifest.json          # Metadata
            <content_hash>.wavefunction.h5
            <content_hash>.ndarray.npy
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
        return is_cacheable(value)

    @classmethod
    def serialize_value(  # noqa: PLR0911
        cls,
        directory: Path,
        name: str,
        value: Any,
        *,
        cache: CacheBackend | None = None,
        content_hash: str | None = None,
        seed_cache: bool = False,
    ) -> dict[str, Any]:
        """Serialize a single value, returning manifest entry.

        Args:
            directory: Directory to write files to.
            name: Logical name used to derive opaque DataClass file names while traversing nested values.
            value: Value to serialize.
            cache: Shared cache backend used to replace existing blobs with ``"cached"`` manifest references.
            content_hash: Optional hash used to check whether *value* is cached.
            seed_cache: Whether to write a cacheable value to shared storage before emitting a reference.

        Returns:
            Manifest entry describing the serialized value.

        """
        # Handle None
        if value is None:
            return {"type": "none", "value": None}

        if isinstance(value, np.generic):
            value = _numpy_scalar_to_python(value)

        if isinstance(value, AlgorithmRef):
            settings = value.settings
            return {
                "type": "algorithm_ref",
                "algorithm_type": value.algorithm_type,
                "algorithm_name": value.algorithm_name,
                "settings": _jsonable_settings(settings) if settings is not None else None,
            }

        if cls.is_cacheable(value) and cache is not None and content_hash is not None and cache.is_shared:
            cached = cache.has_data(content_hash, shared_only=True)
            if not cached and seed_cache:
                cache.put_data(content_hash, value, shared_only=True)
                cached = cache.has_data(content_hash, shared_only=True)
            if cached:
                cached_type = (
                    instance_data_type_name(value)
                    if cls.is_dataclass(value)
                    else "ndarray"
                    if isinstance(value, np.ndarray)
                    else "list"
                )
                return {
                    "type": "cached",
                    "dataclass_type": cached_type,
                    "content_hash": content_hash,
                }

        # Handle DataClass objects - serialize to individual file
        if cls.is_dataclass(value):
            type_name = instance_data_type_name(value)
            ext = cls._get_dataclass_extension(type_name)
            filename = f"{_item_content_hash(value)}{ext}"
            filepath = directory / filename
            _atomic_write_dataclass(filepath, value, type_name)
            entry: dict[str, Any] = {
                "type": "dataclass",
                "dataclass_type": type_name,
                "file": filename,
            }
            if content_hash is not None:
                entry["content_hash"] = content_hash
            return entry

        if isinstance(value, np.ndarray):
            filename = f"{_item_content_hash(value)}.ndarray.npy"
            _atomic_write_array(directory / filename, value)
            entry = {"type": "ndarray", "file": filename}
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
            entry = {"type": "list", "items": items}
            if content_hash is not None:
                entry["content_hash"] = content_hash
            return entry

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
        cache: CacheBackend | None = None,
    ) -> Any:
        """Deserialize a value from a manifest entry.

        Args:
            directory: Directory containing the files.
            entry: Manifest entry describing the value.
            cache: Optional cache backend used to resolve ``"cached"`` entries omitted from uploaded files.

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

        if type_tag in ("dataclass", "ndarray", "list"):
            content_hash = entry.get("content_hash")
            if cache is not None and content_hash is not None:
                data = cache.get_data(content_hash)
                if data is not None:
                    return data

        if type_tag == "dataclass":
            dataclass_type_name = entry["dataclass_type"]
            dataclass_type = cls._get_dataclass_type(dataclass_type_name)
            if dataclass_type is None:
                raise TypeError(f"Unknown DataClass type: {dataclass_type_name}")
            filepath = _manifest_file_path(directory, entry["file"])
            # All QDK Chemistry DataClass types have from_hdf5_file
            return dataclass_type.from_hdf5_file(str(filepath))  # type: ignore[attr-defined]

        if type_tag == "ndarray":
            filepath = _manifest_file_path(directory, entry["file"])
            return np.load(filepath, allow_pickle=False)

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
    """Return every file referenced by a serialized manifest entry.

    Args:
        entry: Serialized value entry from an input or output manifest.

    Returns:
        Artifact file names in manifest order; primitive and cached entries contribute no names.

    """
    type_tag = entry["type"]

    if type_tag in ("dataclass", "ndarray"):
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


def _get_manifest_files(directory: Path, entries: list[dict[str, Any]]) -> list[Path]:
    """Return the manifest and each unique data file it references."""
    files = [directory / _MANIFEST_FILENAME]
    seen: set[Path] = set()
    for entry in entries:
        for filename in get_serialized_file_names(entry):
            filepath = _manifest_file_path(directory, filename)
            if filepath in seen:
                continue
            if not filepath.is_file():
                raise FileNotFoundError(f"Manifest references a missing serialization file: {filepath}")
            files.append(filepath)
            seen.add(filepath)
    return files


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
    force_rerun: bool = False,
    remote_cache: dict[str, Any] | None = None,
    remote_cache_backend: CacheBackend | None = None,
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
        force_rerun: Whether the compute node must skip its cache lookup.
        remote_cache: Optional coordinates passed to the remote cache factory, ``get_cache()``.
        remote_cache_backend: Shared cache backend; existing cacheable values become ``"cached"`` manifest references.
        remote_cache_transport: Whether to seed shared-cache misses and use the cache as artifact transport.

    Returns:
        List of all files created (for upload).

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest: dict[str, Any] = {
        "version": _MANIFEST_VERSION,
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
    if force_rerun:
        manifest["force_rerun"] = True
    if remote_cache is not None:
        manifest["remote_cache"] = remote_cache
    if remote_cache_transport:
        manifest["remote_cache_transport"] = True

    with tempfile.TemporaryDirectory(dir=directory, prefix=".serialization-") as staging_directory_name:
        staging_directory = Path(staging_directory_name)

        for index, (key, value) in enumerate(settings.items()):
            serialization_name = (
                f"setting_{index}_{_item_content_hash(value)}"
                if FileSerializer.is_cacheable(value)
                else f"setting_{index}"
            )
            manifest["settings"][key] = FileSerializer.serialize_value(
                staging_directory,
                serialization_name,
                value,
                cache=remote_cache_backend,
                content_hash=_item_content_hash(value) if FileSerializer.is_cacheable(value) else None,
                seed_cache=remote_cache_transport,
            )

        for i, arg in enumerate(args):
            content_hash = input_hashes.get(f"args.arg_{i}") if input_hashes else None
            serialization_name = (
                f"arg_{i}_{_item_content_hash(arg)}" if FileSerializer.is_cacheable(arg) else f"arg_{i}"
            )
            entry = FileSerializer.serialize_value(
                staging_directory,
                serialization_name,
                arg,
                cache=remote_cache_backend,
                content_hash=content_hash,
                seed_cache=remote_cache_transport,
            )
            if content_hash and "content_hash" not in entry:
                entry["content_hash"] = content_hash
            manifest["args"].append(entry)

        for index, (key, value) in enumerate(kwargs.items()):
            content_hash = input_hashes.get(f"kwargs.{key}") if input_hashes else None
            serialization_name = (
                f"kwarg_{index}_{_item_content_hash(value)}" if FileSerializer.is_cacheable(value) else f"kwarg_{index}"
            )
            entry = FileSerializer.serialize_value(
                staging_directory,
                serialization_name,
                value,
                cache=remote_cache_backend,
                content_hash=content_hash,
                seed_cache=remote_cache_transport,
            )
            if content_hash and "content_hash" not in entry:
                entry["content_hash"] = content_hash
            manifest["kwargs"][key] = entry

        filenames = [
            filename
            for entries in (manifest["settings"].values(), manifest["args"], manifest["kwargs"].values())
            for entry in entries
            for filename in get_serialized_file_names(entry)
        ]
        return _commit_staged_serialization(directory, staging_directory, manifest, filenames)


def deserialize_inputs(directory: str | Path, *, cache: CacheBackend | None = None) -> dict:
    """Deserialize algorithm inputs from a directory.

    Args:
        directory: Directory containing the input files.
        cache: Optional cache backend used to resolve ``"cached"`` entries omitted from uploaded files.

    Returns:
        Deserialized inputs containing algorithm metadata, settings, arguments, and cache metadata.

    """
    directory = Path(directory)
    manifest_path = directory / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

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
        "force_rerun": manifest.get("force_rerun", False),
        "remote_cache": manifest.get("remote_cache"),
        "remote_cache_transport": manifest.get("remote_cache_transport", False),
    }


def serialize_outputs(
    directory: str | Path,
    result: Any,
    *,
    cache: CacheBackend | None = None,
    cache_transport: bool = False,
) -> list[Path]:
    """Serialize algorithm outputs to a directory.

    Args:
        directory: Directory to write files to.
        result: The result from algorithm.run() (may be a tuple or single value).
        cache: Optional shared cache backend used for cacheable result values.
        cache_transport: Whether to seed shared-cache misses and use the cache as artifact transport.

    Returns:
        List of all files created (for download).

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest: dict[str, Any] = {"version": _MANIFEST_VERSION, "is_tuple": False, "results": []}

    with tempfile.TemporaryDirectory(dir=directory, prefix=".serialization-") as staging_directory_name:
        staging_directory = Path(staging_directory_name)

        if isinstance(result, tuple):
            manifest["is_tuple"] = True
            for i, item in enumerate(result):
                content_hash = _item_content_hash(item) if FileSerializer.is_cacheable(item) else None
                serialization_name = f"result_{i}_{content_hash}" if content_hash is not None else f"result_{i}"
                entry = FileSerializer.serialize_value(
                    staging_directory,
                    serialization_name,
                    item,
                    cache=cache,
                    content_hash=content_hash,
                    seed_cache=cache_transport,
                )
                manifest["results"].append(entry)
        else:
            content_hash = _item_content_hash(result) if FileSerializer.is_cacheable(result) else None
            serialization_name = f"result_{content_hash}" if content_hash is not None else "result"
            entry = FileSerializer.serialize_value(
                staging_directory,
                serialization_name,
                result,
                cache=cache,
                content_hash=content_hash,
                seed_cache=cache_transport,
            )
            manifest["results"].append(entry)

        filenames = [filename for entry in manifest["results"] for filename in get_serialized_file_names(entry)]
        return _commit_staged_serialization(directory, staging_directory, manifest, filenames)


def deserialize_outputs(directory: str | Path, *, cache: CacheBackend | None = None) -> Any:
    """Deserialize algorithm outputs from a directory.

    Args:
        directory: Directory containing the output files.
        cache: Optional cache backend used to resolve or reuse cacheable result values.

    Returns:
        The deserialized result (tuple or single value).

    """
    directory = Path(directory)
    manifest_path = directory / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    results = [FileSerializer.deserialize_value(directory, entry, cache=cache) for entry in manifest["results"]]

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
    manifest_path = directory / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    manifest = _load_manifest(manifest_path)
    entries = [
        entry
        for values in (manifest["settings"].values(), manifest["args"], manifest["kwargs"].values())
        for entry in values
    ]
    return _get_manifest_files(directory, entries)


def get_output_files(directory: str | Path) -> list[Path]:
    """Get list of all output files in a directory.

    Args:
        directory: Directory containing output files.

    Returns:
        List of all files that should be downloaded.

    """
    directory = Path(directory)
    manifest_path = directory / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    manifest = _load_manifest(manifest_path)
    return _get_manifest_files(directory, manifest["results"])
