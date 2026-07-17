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
from pathlib import Path
from typing import Any, ClassVar

# ─────────────────────────────────────────────────────────────────────────────
# Serializer for individual files
# ─────────────────────────────────────────────────────────────────────────────


class FileSerializer:
    """Handles file-based serialization of QDK Chemistry objects for remote transport.

    Each DataClass object is serialized to its own .{type_name}.h5 file.
    Primitives and simple types are stored in a JSON manifest file.

    Directory structure for inputs::

        job_dir/
            manifest.json          # Metadata and primitive values
            arg_0.structure.h5     # First arg (if Structure)
            kwarg_basis.basis_set.h5   # Keyword arg (if BasisSet)
            ...

    Directory structure for outputs::

        job_dir/
            manifest.json          # Metadata
            result_0.wavefunction.h5   # First result
            ...

    """

    # Registry mapping type names to DataClass subclasses
    _dataclass_registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register_dataclass(cls, dataclass_type: type) -> type:
        """Register a DataClass subclass for deserialization.

        Args:
            dataclass_type: A DataClass subclass with _data_type_name attribute.

        Returns:
            The registered class (allows use as decorator).

        """
        type_name = getattr(dataclass_type, "_data_type_name", None)
        if type_name:
            cls._dataclass_registry[type_name] = dataclass_type
        return dataclass_type

    @classmethod
    def _get_dataclass_type(cls, type_name: str) -> type | None:
        """Get the DataClass type for a type name, with lazy loading."""
        if type_name in cls._dataclass_registry:
            return cls._dataclass_registry[type_name]

        # Lazy load common QDK Chemistry data types
        cls._lazy_load_dataclasses()
        return cls._dataclass_registry.get(type_name)

    @classmethod
    def _lazy_load_dataclasses(cls) -> None:
        """Lazily load and register common DataClass types."""
        if cls._dataclass_registry:
            return  # Already loaded

        try:
            from qdk_chemistry.data import (  # noqa: PLC0415
                Ansatz,
                BasisSet,
                Circuit,
                Configuration,
                ConfigurationSet,
                Hamiltonian,
                Orbitals,
                QubitHamiltonian,
                Settings,
                Structure,
                Wavefunction,
            )

            # Register known types
            for dataclass_type in [
                Ansatz,
                BasisSet,
                Circuit,
                Configuration,
                ConfigurationSet,
                Hamiltonian,
                Orbitals,
                QubitHamiltonian,
                Settings,
                Structure,
                Wavefunction,
            ]:
                cls.register_dataclass(dataclass_type)
        except ImportError:
            pass  # Some types may not be available

    @classmethod
    def _get_dataclass_extension(cls, type_name: str) -> str:
        """Get the required file extension for a DataClass type.

        Args:
            type_name: The _data_type_name of the DataClass.

        Returns:
            The extension pattern (e.g., ".structure.h5").

        """
        return f".{type_name}.h5"

    @classmethod
    def is_dataclass(cls, value: Any) -> bool:
        """Check if a value is a QDK Chemistry DataClass."""
        return hasattr(value, "to_hdf5_file") and hasattr(value, "_data_type_name")

    @classmethod
    def serialize_value(  # noqa: PLR0911
        cls,
        directory: Path,
        name: str,
        value: Any,
        *,
        cache: Any = None,
        content_hash: str | None = None,
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

        Returns:
            Manifest entry describing the serialized value.

        """
        # Handle None
        if value is None:
            return {"type": "none", "value": None}

        # Handle DataClass objects - serialize to individual file
        if cls.is_dataclass(value):
            type_name = value._data_type_name  # noqa: SLF001

            # If a shared cache already has this blob, skip the file
            if (
                cache is not None
                and content_hash is not None
                and getattr(cache, "is_shared", False)
                and cache.has_data(content_hash)
            ):
                return {
                    "type": "cached",
                    "dataclass_type": type_name,
                    "content_hash": content_hash,
                }

            ext = cls._get_dataclass_extension(type_name)
            filename = f"{name}{ext}"
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
                items.append(cls.serialize_value(directory, f"{name}_item_{i}", item))
            return {"type": "list", "items": items}

        # Handle tuples
        if isinstance(value, tuple):
            items = []
            for i, item in enumerate(value):
                items.append(cls.serialize_value(directory, f"{name}_item_{i}", item))
            return {"type": "tuple", "items": items}

        # Handle dicts
        if isinstance(value, dict):
            entries = {}
            for k, v in value.items():
                entries[str(k)] = cls.serialize_value(directory, f"{name}_{k}", v)
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
            return {k: cls.deserialize_value(directory, v, cache=cache) for k, v in entry["entries"].items()}

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

    Returns:
        List of all files created (for upload).

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    files_created: list[Path] = []

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
    if remote_cache is not None:
        manifest["remote_cache"] = remote_cache

    # Serialize settings
    for key, value in settings.items():
        manifest["settings"][key] = FileSerializer.serialize_value(directory, f"setting_{key}", value)

    # Serialize positional arguments
    for i, arg in enumerate(args):
        chash = input_hashes.get(f"arg_{i}") if input_hashes else None
        entry = FileSerializer.serialize_value(
            directory,
            f"arg_{i}",
            arg,
            cache=remote_cache_backend,
            content_hash=chash,
        )
        if chash and "content_hash" not in entry:
            entry["content_hash"] = chash
        manifest["args"].append(entry)
        if entry.get("file"):
            files_created.append(directory / entry["file"])

    # Serialize keyword arguments
    for key, value in kwargs.items():
        chash = input_hashes.get(key) if input_hashes else None
        entry = FileSerializer.serialize_value(
            directory,
            f"kwarg_{key}",
            value,
            cache=remote_cache_backend,
            content_hash=chash,
        )
        if chash and "content_hash" not in entry:
            entry["content_hash"] = chash
        manifest["kwargs"][key] = entry
        if entry.get("file"):
            files_created.append(directory / entry["file"])

    # Write manifest
    manifest_path = directory / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    files_created.insert(0, manifest_path)

    return files_created


def deserialize_inputs(directory: str | Path, *, cache: Any = None) -> dict:
    """Deserialize algorithm inputs from a directory.

    Args:
        directory: Directory containing the input files.
        cache: Optional ``CacheBackend`` used to resolve ``"cached"``
            manifest entries that were not uploaded as files.

    Returns:
        Dictionary with keys: algorithm_type, algorithm_name, settings, args, kwargs

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

    files_created: list[Path] = []

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
            if entry.get("file"):
                files_created.append(directory / entry["file"])
    else:
        entry = FileSerializer.serialize_value(directory, "result", result)
        if hasattr(result, "content_hash"):
            entry["content_hash"] = result.content_hash()
        manifest["results"].append(entry)
        if entry.get("file"):
            files_created.append(directory / entry["file"])

    # Write manifest
    manifest_path = directory / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
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
