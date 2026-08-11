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
    return create_test_orbitals(3)


@pytest.fixture
def h2_structure():
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
    entry = FileSerializer.serialize_value(tmp_path, "value", value)

    result = FileSerializer.deserialize_value(tmp_path, entry)

    assert result == value
    assert type(result) is type(value)


def test_unsupported_value_and_type_tag_raise(tmp_path):
    with pytest.raises(TypeError, match="Cannot serialize"):
        FileSerializer.serialize_value(tmp_path, "bad", object())

    with pytest.raises(TypeError, match="Unknown type tag"):
        FileSerializer.deserialize_value(tmp_path, {"type": "unknown"})


def test_dataclass_round_trip(tmp_path, sample_orbitals):
    entry = FileSerializer.serialize_value(tmp_path, "orbitals", sample_orbitals)

    assert entry["type"] == "dataclass"
    assert (tmp_path / entry["file"]).exists()
    restored = FileSerializer.deserialize_value(tmp_path, entry)
    assert isinstance(restored, Orbitals)
    np.testing.assert_array_equal(restored.get_coefficients(), sample_orbitals.get_coefficients())


def test_nested_dataclass_filenames_are_unique(tmp_path, h2_structure):
    helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))

    entry = FileSerializer.serialize_value(tmp_path, "structures", {"/": h2_structure, "_": helium})

    filenames = [item["value"]["file"] for item in entry["entries"]]
    assert len(set(filenames)) == 2
    assert all((tmp_path / filename).exists() for filename in filenames)


def test_algorithm_ref_round_trip(tmp_path):
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


def test_input_round_trip(tmp_path, h2_structure):
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


def test_output_round_trip_survives_file_transfer(tmp_path, sample_orbitals):
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


class SharedCache:
    is_shared = True

    def __init__(self, value=None):
        self.value = value

    def has_data(self, content_hash):
        return content_hash == "cached-hash"

    def get_data(self, content_hash):
        return self.value if content_hash == "cached-hash" else None


def test_shared_cache_entry_avoids_data_file(tmp_path, h2_structure):
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


def test_cached_entry_requires_available_cache_value(tmp_path):
    entry = {"type": "cached", "dataclass_type": "structure", "content_hash": "missing"}

    with pytest.raises(TypeError, match="no cache backend"):
        FileSerializer.deserialize_value(tmp_path, entry)

    with pytest.raises(LookupError, match="Cache miss"):
        FileSerializer.deserialize_value(tmp_path, entry, cache=SharedCache())


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_manifest_write_is_atomic(tmp_path, monkeypatch, serializer):
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


def test_settings_json_does_not_include_internal_descriptions(tmp_path):
    settings = Settings()
    settings.set("method", "pbe")
    algorithm = AlgorithmRef("scf_solver", "qdk", settings=settings)

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
