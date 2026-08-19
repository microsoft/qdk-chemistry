"""Tests for file-based algorithm input and output serialization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import qdk_chemistry.remote.serialization as serialization_module
from qdk_chemistry.data import AlgorithmRef, Orbitals, Settings, Structure
from qdk_chemistry.remote.cache import FolderCache, TieredCache
from qdk_chemistry.remote.serialization import (
    FileSerializer,
    deserialize_inputs,
    deserialize_outputs,
    get_input_files,
    get_output_files,
    serialize_inputs,
    serialize_outputs,
)

from .test_helpers import create_test_orbitals


@pytest.fixture
def sample_orbitals():
    """Create a small Orbitals fixture."""
    return create_test_orbitals(3)


@pytest.fixture
def h2_structure():
    """Create a two-atom Structure fixture."""
    return Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))


@pytest.mark.parametrize(
    ("value", "type_tag"),
    [
        (None, "none"),
        (True, "bool"),
        (42, "int"),
        (-3.14, "float"),
        ("hello", "str"),
    ],
)
def test_primitive_round_trip(tmp_path, value, type_tag):
    """Round-trip each supported primitive type."""
    entry = FileSerializer.serialize_value(tmp_path, "test", value)

    assert entry["type"] == type_tag
    result = FileSerializer.deserialize_value(tmp_path, entry)
    assert result == value
    assert type(result) is type(value)


@pytest.mark.parametrize(
    "value",
    [
        [1, 2.5, "three"],
        (10, "abc"),
        {"a": 1, "b": 2.0},
        [{"x": [1, 2]}, (True, None)],
        {1: "integer", "1": "string", (2, "two"): "tuple"},
    ],
)
def test_container_round_trip(tmp_path, value):
    """Round-trip nested list, tuple, and dictionary values."""
    entry = FileSerializer.serialize_value(tmp_path, "value", value)

    result = FileSerializer.deserialize_value(tmp_path, entry)

    assert result == value
    assert type(result) is type(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.bool_(True), True),
        (np.int64(42), 42),
        (np.float32(1.25), 1.25),
        (np.longdouble(1.25), 1.25),
        (np.str_("value"), "value"),
    ],
)
def test_numpy_scalar_round_trip_as_python_primitive(tmp_path, value, expected):
    """Round-trip NumPy scalars as their corresponding Python primitives."""
    entry = FileSerializer.serialize_value(tmp_path, "value", value)

    result = FileSerializer.deserialize_value(tmp_path, entry)

    assert result == expected
    assert type(result) is type(expected)


def test_numpy_array_round_trip(tmp_path):
    """Round-trip a NumPy array without changing its shape or dtype."""
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    entry = FileSerializer.serialize_value(tmp_path, "value", value)
    result = FileSerializer.deserialize_value(tmp_path, entry)

    assert entry["type"] == "ndarray"
    np.testing.assert_array_equal(result, value)
    assert result.dtype == value.dtype


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, (np.float32(2.0), np.array([3.0]))], True),
        ([{"x": 1}], False),
        ([object()], False),
        (np.array([1.0]), True),
        (np.array([object()], dtype=object), False),
        ((1, 2), False),
    ],
)
def test_cacheable_matches_supported_cache_value_graph(value, expected):
    """Match cache eligibility to the recursive value graph accepted by cache backends."""
    assert FileSerializer.is_cacheable(value) is expected


def test_unsupported_value_and_type_tag_raise(tmp_path):
    """Reject unsupported Python values and manifest type tags."""
    with pytest.raises(TypeError, match="Cannot serialize"):
        FileSerializer.serialize_value(tmp_path, "bad", object())

    with pytest.raises(TypeError, match="Unknown type tag"):
        FileSerializer.deserialize_value(tmp_path, {"type": "unknown"})


def test_dataclass_round_trip(tmp_path, sample_orbitals):
    """Round-trip a DataClass through its HDF5 artifact."""
    entry = FileSerializer.serialize_value(tmp_path, "orbitals", sample_orbitals)

    assert entry["type"] == "dataclass"
    assert (tmp_path / entry["file"]).exists()
    restored = FileSerializer.deserialize_value(tmp_path, entry)
    assert isinstance(restored, Orbitals)
    np.testing.assert_array_equal(restored.get_coefficients(), sample_orbitals.get_coefficients())


@pytest.mark.parametrize(
    "name",
    ["", "../outside", "..\\outside", "/absolute", "CON", 'bad:name*?"', "line\nbreak", "caf\u00e9", "x" * 512],
)
def test_dataclass_filename_uses_content_hash(tmp_path, h2_structure, name):
    """Use a portable content hash regardless of the caller-provided name."""
    entry = FileSerializer.serialize_value(tmp_path, name, h2_structure)

    assert entry["file"] == f"{h2_structure.content_hash()}.structure.h5"
    assert (tmp_path / entry["file"]).is_file()


@pytest.mark.parametrize("path_type", ["relative", "absolute", "symlink"])
def test_dataclass_path_must_stay_within_directory(tmp_path, h2_structure, path_type):
    """Reject manifest paths that resolve outside the serialization directory."""
    directory = tmp_path / "job"
    directory.mkdir()
    outside_file = tmp_path / "outside.structure.h5"
    h2_structure.to_hdf5_file(str(outside_file))
    if path_type == "relative":
        manifest_file = f"../{outside_file.name}"
    elif path_type == "absolute":
        manifest_file = str(outside_file)
    else:
        symlink = directory / "linked.structure.h5"
        symlink.symlink_to(outside_file)
        manifest_file = symlink.name
    entry = {"type": "dataclass", "dataclass_type": "structure", "file": manifest_file}

    with pytest.raises(ValueError, match=r"Invalid serialization file name|outside the serialization directory"):
        FileSerializer.deserialize_value(directory, entry)


def test_nested_dataclass_filenames_are_unique(tmp_path, h2_structure):
    """Give distinct nested DataClass values distinct content-addressed files."""
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    entry = FileSerializer.serialize_value(tmp_path, "structures", {"/": h2_structure, "_": helium})

    filenames = [item["value"]["file"] for item in entry["entries"]]
    assert set(filenames) == {
        f"{h2_structure.content_hash()}.structure.h5",
        f"{helium.content_hash()}.structure.h5",
    }
    assert all((tmp_path / filename).exists() for filename in filenames)


def test_algorithm_ref_round_trip(tmp_path):
    """Round-trip nested AlgorithmRef settings."""
    energy = AlgorithmRef("scf_solver", "qdk")
    energy.settings.set("method", "pbe")
    derivative = AlgorithmRef("nuclear_derivative_calculator", "qdk")
    derivative.settings.set("energy_calculator", energy)

    entry = FileSerializer.serialize_value(tmp_path, "algorithm", derivative)
    restored = FileSerializer.deserialize_value(tmp_path, entry)

    assert isinstance(restored, AlgorithmRef)
    assert restored.algorithm_type == "nuclear_derivative_calculator"
    restored_energy = restored.settings.get("energy_calculator")
    assert isinstance(restored_energy, AlgorithmRef)
    assert restored_energy.algorithm_type == "scf_solver"
    assert restored_energy.settings.get("method") == "pbe"


def test_algorithm_ref_round_trip_preserves_empty_double_vector_type(tmp_path):
    """Preserve the declared element type of an empty settings vector."""
    settings = Settings.from_json(
        json.dumps(
            {
                "coefficients": {
                    "__type__": "array",
                    "__element_type__": "double",
                    "__value__": [],
                }
            }
        )
    )
    algorithm = AlgorithmRef("test_algorithm", "plugin", settings)

    entry = FileSerializer.serialize_value(tmp_path, "algorithm", algorithm)
    restored = FileSerializer.deserialize_value(tmp_path, entry)

    restored.settings.set("coefficients", [1.0])
    assert restored.settings.get_type_name("coefficients") == "vector<double>"


def test_input_round_trip(tmp_path, h2_structure):
    """Round-trip a complete algorithm input manifest."""
    files = serialize_inputs(
        tmp_path,
        args=(h2_structure, 0, 1),
        kwargs={"basis": "cc-pvdz"},
        algorithm_type="scf_solver",
        algorithm_name="qdk",
        settings={"max_iterations": 100},
        run_hash="run-hash",
        input_hashes={"arg_0": "structure-hash"},
        remote_cache={"name": "shared"},
    )

    assert set(files) == set(get_input_files(tmp_path))
    restored = deserialize_inputs(tmp_path)
    assert restored["algorithm_type"] == "scf_solver"
    assert restored["algorithm_name"] == "qdk"
    assert isinstance(restored["args"][0], Structure)
    assert restored["args"][1:] == (0, 1)
    assert restored["kwargs"] == {"basis": "cc-pvdz"}
    assert restored["settings"] == {"max_iterations": 100}
    assert restored["run_hash"] == "run-hash"
    assert restored["input_hashes"] == {"arg_0": "structure-hash"}
    assert restored["remote_cache"] == {"name": "shared"}
    assert restored["remote_cache_transport"] is False


def test_output_round_trip_survives_file_transfer(tmp_path, sample_orbitals):
    """Round-trip outputs after copying only the reported artifacts."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()

    files = serialize_outputs(source, (-75.5, [sample_orbitals, sample_orbitals]))
    assert set(files) == set(get_output_files(source))
    for path in files:
        shutil.copy2(path, destination / path.name)

    energy, orbitals = deserialize_outputs(destination)
    assert energy == -75.5
    assert all(isinstance(item, Orbitals) for item in orbitals)


def test_numpy_algorithm_output_survives_file_transfer(tmp_path):
    """Round-trip a NumPy scalar and array algorithm result after file transfer."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    result = np.float64(-1.5), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    files = serialize_outputs(source, result)
    for path in files:
        shutil.copy2(path, destination / path.name)
    energy, state = deserialize_outputs(destination)

    assert {path.suffix for path in files} == {".json", ".npy"}
    assert energy == -1.5
    assert type(energy) is float
    np.testing.assert_array_equal(state, result[1])


class SharedCache:
    """Minimal cache fake for serialization transport tests."""

    def __init__(self, value=None, *, is_shared=True):
        self.is_shared = is_shared
        self.value = value
        self.values = {}

    def has_data(self, content_hash, *, shared_only=False):
        """Report whether a requested value is available."""
        if shared_only and not self.is_shared:
            return False
        return content_hash == "cached-hash" or content_hash in self.values

    def get_data(self, content_hash):
        """Return a cached value by content hash."""
        if content_hash == "cached-hash":
            return self.value
        return self.values.get(content_hash)

    def put_data(self, content_hash, value, *, shared_only=False):
        """Store a value when the requested sharing policy permits it."""
        if shared_only and not self.is_shared:
            return
        self.values[content_hash] = value


def test_output_round_trip_uses_shared_cache_transport(tmp_path, sample_orbitals):
    """Transfer an output through shared cache using only its manifest."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    cache = SharedCache()

    files = serialize_outputs(source, sample_orbitals, cache=cache, cache_transport=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")

    assert files == [source / "manifest.json"]
    assert deserialize_outputs(destination, cache=cache) is sample_orbitals


def test_numpy_array_output_uses_shared_cache_transport(tmp_path):
    """Transfer a NumPy array output through a shared folder cache."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    cache = FolderCache(path=tmp_path / "shared", is_shared=True)
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    files = serialize_outputs(source, value, cache=cache, cache_transport=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    restored = deserialize_outputs(destination, cache=cache)

    assert files == [source / "manifest.json"]
    np.testing.assert_array_equal(restored, value)
    assert restored.dtype == value.dtype


def test_unsupported_cache_value_graph_uses_manifest_transport(tmp_path):
    """Serialize unsupported cache graphs without attempting a cache write."""
    source = tmp_path / "source"
    cache_path = tmp_path / "shared"
    cache = FolderCache(path=cache_path, is_shared=True)
    value = [{"x": 1}]

    files = serialize_outputs(source, value, cache=cache, cache_transport=True)

    assert files == [source / "manifest.json"]
    assert deserialize_outputs(source) == value
    assert not cache_path.exists()


def test_input_round_trip_uses_shared_cache_transport(tmp_path, h2_structure):
    """Seed and resolve an input through shared-cache transport."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    cache = SharedCache()
    content_hash = h2_structure.content_hash()

    files = serialize_inputs(
        source,
        args=(h2_structure,),
        kwargs={},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={},
        input_hashes={"arg_0": content_hash},
        remote_cache={"name": "shared"},
        remote_cache_backend=cache,
        remote_cache_transport=True,
    )
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    restored = deserialize_inputs(destination, cache=cache)

    assert files == [source / "manifest.json"]
    assert restored["args"][0] is h2_structure
    assert restored["remote_cache_transport"] is True


def test_deserialize_output_prefers_cache_to_data_file(tmp_path, sample_orbitals):
    """Resolve a file-backed output from cache when its HDF5 file is absent."""
    cache = SharedCache()
    cache.put_data(sample_orbitals.content_hash(), sample_orbitals)
    serialize_outputs(tmp_path, sample_orbitals)
    next(tmp_path.glob("*.h5")).unlink()

    assert deserialize_outputs(tmp_path, cache=cache) is sample_orbitals


def test_nested_output_values_use_shared_cache(tmp_path, sample_orbitals):
    """Use cached references for DataClass values nested in an output list."""
    cache = SharedCache()
    cache.put_data(sample_orbitals.content_hash(), sample_orbitals)

    files = serialize_outputs(tmp_path, [sample_orbitals, sample_orbitals], cache=cache)
    restored = deserialize_outputs(tmp_path, cache=cache)

    assert files == [tmp_path / "manifest.json"]
    assert restored == [sample_orbitals, sample_orbitals]


@pytest.mark.parametrize("container_type", ["list", "tuple", "dict"])
def test_nested_containers_propagate_shared_cache(tmp_path, sample_orbitals, container_type):
    """Propagate cache access through lists, tuples, and dictionaries."""
    cache = SharedCache()
    cache.put_data(sample_orbitals.content_hash(), sample_orbitals)
    value = {
        "list": [sample_orbitals],
        "tuple": (sample_orbitals,),
        "dict": {"orbitals": sample_orbitals},
    }[container_type]

    entry = FileSerializer.serialize_value(tmp_path, "value", value, cache=cache)
    restored = FileSerializer.deserialize_value(tmp_path, entry, cache=cache)
    restored_orbitals = restored["orbitals"] if container_type == "dict" else restored[0]

    assert not list(tmp_path.glob("*.h5"))
    assert restored_orbitals is sample_orbitals


def test_tiered_local_cache_hit_does_not_replace_output_file(tmp_path, sample_orbitals):
    """Keep an output artifact when only a local cache tier contains it."""
    local = FolderCache(path=tmp_path / "local")
    shared = FolderCache(path=tmp_path / "shared", is_shared=True)
    cache = TieredCache([local, shared])
    content_hash = sample_orbitals.content_hash()
    local.put_data(content_hash, sample_orbitals)

    files = serialize_outputs(tmp_path / "output", sample_orbitals, cache=cache)

    assert any(path.suffix == ".h5" for path in files)
    assert not shared.has_data(content_hash)


def test_shared_cache_entry_avoids_data_file(tmp_path, h2_structure):
    """Replace a shared-cache hit with a cached manifest entry."""
    entry = FileSerializer.serialize_value(
        tmp_path,
        "structure",
        h2_structure,
        cache=SharedCache(),
        content_hash="cached-hash",
    )

    assert entry == {
        "type": "cached",
        "dataclass_type": "structure",
        "content_hash": "cached-hash",
    }
    assert not list(tmp_path.glob("*.h5"))
    assert FileSerializer.deserialize_value(tmp_path, entry, cache=SharedCache(h2_structure)) is h2_structure


def test_non_shared_cache_entry_includes_data_file(tmp_path, h2_structure):
    """Retain an HDF5 artifact when a cache is not shared."""
    entry = FileSerializer.serialize_value(
        tmp_path,
        "structure",
        h2_structure,
        cache=SharedCache(h2_structure, is_shared=False),
        content_hash="cached-hash",
    )

    expected_file = f"{h2_structure.content_hash()}.structure.h5"
    assert entry == {
        "type": "dataclass",
        "dataclass_type": "structure",
        "file": expected_file,
        "content_hash": "cached-hash",
    }
    assert (tmp_path / entry["file"]).exists()
    assert isinstance(FileSerializer.deserialize_value(tmp_path, entry), Structure)


def test_cached_entry_requires_available_cache_value(tmp_path):
    """Fail clearly when a cached manifest entry cannot be resolved."""
    entry = {"type": "cached", "dataclass_type": "structure", "content_hash": "missing"}

    with pytest.raises(TypeError, match="no cache backend"):
        FileSerializer.deserialize_value(tmp_path, entry)

    with pytest.raises(LookupError, match="Cache miss"):
        FileSerializer.deserialize_value(tmp_path, entry, cache=SharedCache())


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_manifest_write_is_atomic(tmp_path, monkeypatch, serializer):
    """Preserve the live manifest when writing its replacement fails."""
    manifest_path = tmp_path / "manifest.json"
    original_manifest = '{"status": "complete"}'
    manifest_path.write_text(original_manifest)

    def fail_after_partial_write(_value, file, **_kwargs):
        file.write('{"partial"')
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(serialization_module.json, "dump", fail_after_partial_write)

    def write_manifest():
        if serializer == "inputs":
            serialize_inputs(tmp_path, (), {}, "test_algorithm", "plugin", {})
        else:
            serialize_outputs(tmp_path, 42)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        write_manifest()

    assert manifest_path.read_text() == original_manifest
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_failed_rerun_preserves_committed_files(tmp_path, monkeypatch, serializer, h2_structure):
    """Preserve committed artifacts when a staged rerun fails."""
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    def serialize(value):
        if serializer == "inputs":
            return serialize_inputs(tmp_path, (value,), {}, "test_algorithm", "plugin", {})
        return serialize_outputs(tmp_path, value)

    serialize(h2_structure)
    manifest_before = (tmp_path / "manifest.json").read_bytes()
    hdf5_before = {path.name: path.read_bytes() for path in tmp_path.glob("*.h5")}

    def fail_after_partial_write(_value, file, **_kwargs):
        file.write('{"partial"')
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(serialization_module.json, "dump", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        serialize(helium)

    assert (tmp_path / "manifest.json").read_bytes() == manifest_before
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.h5")} == hdf5_before
    assert not list(tmp_path.glob(".serialization-*"))


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_file_discovery_uses_only_current_manifest(tmp_path, serializer, h2_structure):
    """Discover only artifacts referenced by the current manifest."""
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    if serializer == "inputs":
        first_files = serialize_inputs(tmp_path, (h2_structure,), {}, "test_algorithm", "plugin", {})
        current_files = serialize_inputs(tmp_path, (helium,), {}, "test_algorithm", "plugin", {})
        discovered_files = get_input_files(tmp_path)
    else:
        first_files = serialize_outputs(tmp_path, h2_structure)
        current_files = serialize_outputs(tmp_path, helium)
        discovered_files = get_output_files(tmp_path)

    stale_file = tmp_path / "stale.structure.h5"
    h2_structure.to_hdf5_file(str(stale_file))

    assert set(discovered_files) == set(current_files)
    assert stale_file not in discovered_files
    assert any(path.suffix == ".h5" and path not in discovered_files for path in first_files)


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_failed_rerun_preserves_previous_serialized_bundle(tmp_path, monkeypatch, serializer, h2_structure):
    """Leave the previous serialized bundle intact after a failed rerun."""
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    if serializer == "inputs":
        original_files = serialize_inputs(tmp_path, (h2_structure,), {}, "test_algorithm", "plugin", {})
    else:
        original_files = serialize_outputs(tmp_path, h2_structure)
    original_contents = {path.name: path.read_bytes() for path in original_files}

    def fail_after_partial_write(_value, file, **_kwargs):
        file.write('{"partial"')
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(serialization_module.json, "dump", fail_after_partial_write)

    def write_helium():
        if serializer == "inputs":
            serialize_inputs(tmp_path, (helium,), {}, "test_algorithm", "plugin", {})
        else:
            serialize_outputs(tmp_path, helium)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        write_helium()

    assert {path.name: path.read_bytes() for path in original_files} == original_contents
    assert {path.name for path in tmp_path.iterdir()} == set(original_contents)


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_get_files_excludes_hdf5_files_not_referenced_by_manifest(tmp_path, serializer, h2_structure):
    """Exclude stale HDF5 artifacts from transfer file discovery."""
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    if serializer == "inputs":
        serialize_inputs(tmp_path, (h2_structure, helium), {}, "test_algorithm", "plugin", {})
        current_files = serialize_inputs(tmp_path, (h2_structure,), {}, "test_algorithm", "plugin", {})
        discovered_files = get_input_files(tmp_path)
    else:
        serialize_outputs(tmp_path, [h2_structure, helium])
        current_files = serialize_outputs(tmp_path, h2_structure)
        discovered_files = get_output_files(tmp_path)

    assert set(discovered_files) == set(current_files)


def test_settings_json_does_not_include_internal_descriptions(tmp_path):
    """Exclude internal Settings descriptions from the wire manifest."""
    algorithm = AlgorithmRef("scf_solver", "qdk")
    algorithm.settings.set("method", "pbe")

    serialize_inputs(
        tmp_path,
        args=(),
        kwargs={},
        algorithm_type="geometry_optimizer",
        algorithm_name="geometric",
        settings={"energy_calculator": algorithm},
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    serialized_settings = manifest["settings"]["energy_calculator"]["settings"]
    assert "_descriptions" not in serialized_settings
    assert serialized_settings["method"] == "pbe"
