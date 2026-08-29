"""Tests for the remote execution system (serialization, Job, backends, proxy, run)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import gc
import json
import logging
import shutil
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import qdk_chemistry.algorithms as algorithms_module
import qdk_chemistry.remote.backends as remote_backends
import qdk_chemistry.remote.backends.base as remote_backend_registry
import qdk_chemistry.remote.backends.local as local_backend_module
import qdk_chemistry.remote.cache as remote_cache_module
import qdk_chemistry.remote.job as job_module
import qdk_chemistry.remote.serialization as serialization_module
import qdk_chemistry.remote.worker as remote_worker
from qdk_chemistry.data import AlgorithmRef, Orbitals, Settings, Structure
from qdk_chemistry.data._hashing import _item_content_hash, collect_content_hashes
from qdk_chemistry.plugins import DuplicateRegistrationError
from qdk_chemistry.remote.backends import available_backends, get_backend
from qdk_chemistry.remote.backends.base import JobState, JobStatus, RemoteBackend, register_backend
from qdk_chemistry.remote.backends.local import LocalBackend
from qdk_chemistry.remote.cache.folder import FolderCache
from qdk_chemistry.remote.cache.tiered import TieredCache
from qdk_chemistry.remote.job import Job
from qdk_chemistry.remote.proxy import _build_payload_for, run, submit
from qdk_chemistry.remote.serialization import (
    FileSerializer,
    deserialize_inputs,
    deserialize_outputs,
    serialize_inputs,
    serialize_outputs,
)
from qdk_chemistry.remote.worker import execute_job

from .test_helpers import create_test_orbitals


@pytest.fixture
def sample_orbitals():
    return create_test_orbitals(3)


@pytest.fixture
def h2_structure():
    return Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))


class TestFileSerializerPrimitives:
    def test_build_payload_hashes_algorithm_ref_argument(self):
        """A serializable algorithm reference can be submitted as an input."""

        class FakeSettings:
            def to_dict(self):
                return {}

        class FakeAlgorithm:
            def type_name(self):
                return "test_algorithm"

            def name(self):
                return "test"

            def settings(self):
                return FakeSettings()

            def hash(self, *_args, **_kwargs):
                return "run_hash"

        reference = AlgorithmRef()
        payload = _build_payload_for(FakeAlgorithm(), (reference,), {})

        assert payload["input_hashes"] == {"args.arg_0": _item_content_hash(reference)}

    @pytest.mark.parametrize(
        ("value", "type_tag"),
        [
            (None, "none"),
            (True, "bool"),
            (False, "bool"),
            (42, "int"),
            (-3.14, "float"),
            ("hello", "str"),
        ],
    )
    def test_primitive_round_trip(self, tmp_path, value, type_tag):
        entry = FileSerializer.serialize_value(tmp_path, "test", value)
        assert entry["type"] == type_tag
        result = FileSerializer.deserialize_value(tmp_path, entry)
        assert result == value
        assert type(result) is type(value)

    def test_list_round_trip(self, tmp_path):
        value = [1, 2.5, "three"]
        entry = FileSerializer.serialize_value(tmp_path, "lst", value)
        assert entry["type"] == "list"
        assert FileSerializer.deserialize_value(tmp_path, entry) == value

    def test_tuple_round_trip(self, tmp_path):
        value = (10, "abc")
        entry = FileSerializer.serialize_value(tmp_path, "tup", value)
        assert entry["type"] == "tuple"
        result = FileSerializer.deserialize_value(tmp_path, entry)
        assert result == value
        assert isinstance(result, tuple)

    def test_nested_structures(self, tmp_path):
        value = [[1, 2], (True, None)]
        entry = FileSerializer.serialize_value(tmp_path, "nested", value)
        result = FileSerializer.deserialize_value(tmp_path, entry)
        assert result[0] == [1, 2]
        assert result[1] == (True, None)

    def test_unsupported_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Cannot serialize"):
            FileSerializer.serialize_value(tmp_path, "bad", object())

    def test_unknown_type_tag_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Unknown type tag"):
            FileSerializer.deserialize_value(tmp_path, {"type": "unknown_xyz"})


class TestFileSerializerDataClass:
    def test_orbitals_round_trip(self, tmp_path, sample_orbitals):
        entry = FileSerializer.serialize_value(tmp_path, "orb", sample_orbitals)
        assert entry["type"] == "dataclass"
        assert (tmp_path / entry["file"]).exists()

        loaded = FileSerializer.deserialize_value(tmp_path, entry)
        assert isinstance(loaded, Orbitals)
        np.testing.assert_array_equal(loaded.get_coefficients(), sample_orbitals.get_coefficients())

    def test_structure_round_trip(self, tmp_path, h2_structure):
        entry = FileSerializer.serialize_value(tmp_path, "struct", h2_structure)
        loaded = FileSerializer.deserialize_value(tmp_path, entry)
        assert isinstance(loaded, Structure)
        np.testing.assert_array_almost_equal(loaded.get_coordinates(), h2_structure.get_coordinates())

    def test_nested_blob_filenames_are_unique(self, tmp_path, h2_structure):
        """Distinct nested values cannot resolve to the same blob file."""
        helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))
        entry = FileSerializer.serialize_value(tmp_path, "structures", [h2_structure, helium])

        filenames = [item["file"] for item in entry["items"]]
        assert len(set(filenames)) == 2
        assert all((tmp_path / filename).exists() for filename in filenames)

        result = FileSerializer.deserialize_value(tmp_path, entry)
        np.testing.assert_array_equal(result[0].get_coordinates(), h2_structure.get_coordinates())
        np.testing.assert_array_equal(result[1].get_coordinates(), helium.get_coordinates())

    def test_is_dataclass(self, sample_orbitals):
        assert FileSerializer.is_dataclass(sample_orbitals)
        assert not FileSerializer.is_dataclass(42)
        assert not FileSerializer.is_dataclass("hello")


class TestInputSerialization:
    def test_round_trip_primitives(self, tmp_path):
        serialize_inputs(
            tmp_path / "job",
            args=(42, "hello"),
            kwargs={"flag": True},
            algorithm_type="scf_solver",
            algorithm_name="qdk",
            settings={"max_iterations": 100},
        )
        result = deserialize_inputs(tmp_path / "job")
        assert result["algorithm_type"] == "scf_solver"
        assert result["algorithm_name"] == "qdk"
        assert result["args"] == (42, "hello")
        assert result["kwargs"] == {"flag": True}

    def test_round_trip_with_dataclass(self, tmp_path, h2_structure):
        files = serialize_inputs(
            tmp_path / "job",
            args=(h2_structure, 0, 1, "cc-pvdz"),
            kwargs={},
            algorithm_type="scf_solver",
            algorithm_name="qdk",
            settings={},
        )
        assert len(files) >= 2  # manifest + structure HDF5

        result = deserialize_inputs(tmp_path / "job")
        assert isinstance(result["args"][0], Structure)
        assert result["args"][1:] == (0, 1, "cc-pvdz")

    def test_map_blob_filenames_do_not_depend_on_keys(self, tmp_path, h2_structure):
        """Top-level setting and keyword names cannot collide as blob paths."""
        helium = Structure(["He"], np.array([[1.0, 0.0, 0.0]]))
        files = serialize_inputs(
            tmp_path / "job",
            args=(),
            kwargs={"/": h2_structure, "_": helium},
            algorithm_type="test_algorithm",
            algorithm_name="plugin",
            settings={"/": h2_structure, "_": helium},
        )

        assert {path.name for path in files} == {
            "manifest.json",
            f"{h2_structure.content_hash()}.structure.h5",
            f"{helium.content_hash()}.structure.h5",
        }
        result = deserialize_inputs(tmp_path / "job")
        for values in (result["settings"], result["kwargs"]):
            np.testing.assert_array_equal(values["/"].get_coordinates(), h2_structure.get_coordinates())
            np.testing.assert_array_equal(values["_"].get_coordinates(), helium.get_coordinates())

    def test_round_trip_with_nested_algorithm_refs(self, tmp_path):
        """Nested algorithm settings survive remote input serialization."""
        derivative_settings = Settings.from_json(
            json.dumps(
                {
                    "compute_hessian": False,
                    "energy_calculator": {
                        "__type__": "algorithm_ref",
                        "algorithm_type": "scf_solver",
                        "algorithm_name": "qdk",
                        "settings": {
                            "method": "b3lyp",
                            "scf_type": "auto",
                            "max_iterations": 100,
                        },
                    },
                }
            )
        )
        derivative_calculator = AlgorithmRef(
            "nuclear_derivative_calculator",
            "qdk",
            settings=derivative_settings,
        )

        serialize_inputs(
            tmp_path / "job",
            args=(),
            kwargs={},
            algorithm_type="geometry_optimizer",
            algorithm_name="geometric",
            settings={"derivative_calculator": derivative_calculator},
        )

        result = deserialize_inputs(tmp_path / "job")
        restored = result["settings"]["derivative_calculator"]
        assert isinstance(restored, AlgorithmRef)
        assert restored.algorithm_type == "nuclear_derivative_calculator"
        assert restored.algorithm_name == "qdk"
        assert restored.settings.get("compute_hessian") is False
        energy_calculator = restored.settings.get("energy_calculator")
        assert isinstance(energy_calculator, AlgorithmRef)
        assert energy_calculator.algorithm_type == "scf_solver"
        assert energy_calculator.algorithm_name == "qdk"
        assert energy_calculator.settings.get("method") == "b3lyp"
        assert energy_calculator.settings.get("scf_type") == "auto"
        assert energy_calculator.settings.get("max_iterations") == 100

    def test_round_trip_with_live_pbe_optimizer_settings(self, tmp_path):
        """Discovery manifests encode live nested PBE settings as plain JSON."""
        energy_calculator = AlgorithmRef("scf_solver", "qdk")
        energy_calculator.settings.set("method", "pbe")
        derivative_calculator = AlgorithmRef("nuclear_derivative_calculator", "qdk")
        derivative_calculator.settings.set("energy_calculator", energy_calculator)

        serialize_inputs(
            tmp_path / "job",
            args=(),
            kwargs={},
            algorithm_type="geometry_optimizer",
            algorithm_name="geometric",
            settings={"derivative_calculator": derivative_calculator},
        )

        manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
        serialized = manifest["settings"]["derivative_calculator"]["settings"]
        assert "_descriptions" not in serialized
        assert serialized["energy_calculator"]["__type__"] == "algorithm_ref"
        assert serialized["energy_calculator"]["settings"]["method"] == "pbe"

        restored = deserialize_inputs(tmp_path / "job")["settings"]["derivative_calculator"]
        restored_energy = restored.settings.get("energy_calculator")
        assert isinstance(restored_energy, AlgorithmRef)
        assert restored_energy.settings.get("method") == "pbe"

    def test_round_trip_with_three_algorithm_ref_levels(self, tmp_path):
        """Three AlgorithmRef levels and their leaf settings survive serialization."""
        circuit_builder_settings = Settings.from_json(
            json.dumps(
                {
                    "num_bits": 12,
                    "unitary_builder": {
                        "__type__": "algorithm_ref",
                        "algorithm_type": "hamiltonian_unitary_builder",
                        "algorithm_name": "zassenhaus",
                        "settings": {
                            "order": 4,
                            "weight_threshold": 1e-10,
                            "term_grouper": {
                                "__type__": "algorithm_ref",
                                "algorithm_type": "term_grouper",
                                "algorithm_name": "commuting",
                                "settings": {"leaf_marker": "preserved"},
                            },
                        },
                    },
                }
            )
        )
        circuit_builder = AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            settings=circuit_builder_settings,
        )

        serialize_inputs(
            tmp_path / "job",
            args=(),
            kwargs={},
            algorithm_type="phase_estimation",
            algorithm_name="qdk_iterative",
            settings={"qpe_circuit_builder": circuit_builder},
        )

        result = deserialize_inputs(tmp_path / "job")
        restored_builder = result["settings"]["qpe_circuit_builder"]
        assert isinstance(restored_builder, AlgorithmRef)
        assert restored_builder.algorithm_type == "qpe_circuit_builder"
        assert restored_builder.algorithm_name == "qdk_iterative"
        assert restored_builder.settings.get("num_bits") == 12

        unitary_builder = restored_builder.settings.get("unitary_builder")
        assert isinstance(unitary_builder, AlgorithmRef)
        assert unitary_builder.algorithm_type == "hamiltonian_unitary_builder"
        assert unitary_builder.algorithm_name == "zassenhaus"
        assert unitary_builder.settings.get("order") == 4
        assert unitary_builder.settings.get("weight_threshold") == pytest.approx(1e-10)

        term_grouper = unitary_builder.settings.get("term_grouper")
        assert isinstance(term_grouper, AlgorithmRef)
        assert term_grouper.algorithm_type == "term_grouper"
        assert term_grouper.algorithm_name == "commuting"
        assert term_grouper.settings.get("leaf_marker") == "preserved"

    def test_run_hash_persisted(self, tmp_path):
        serialize_inputs(
            tmp_path / "job",
            args=(),
            kwargs={},
            algorithm_type="scf_solver",
            algorithm_name="qdk",
            settings={},
            run_hash="deadbeef12345678",
        )
        manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
        assert manifest["run_hash"] == "deadbeef12345678"

    def test_input_hashes_persisted(self, tmp_path):
        serialize_inputs(
            tmp_path / "job",
            args=(1,),
            kwargs={},
            algorithm_type="scf_solver",
            algorithm_name="qdk",
            settings={},
            input_hashes={"args.arg_0": "hash_of_arg0"},
        )
        manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
        assert manifest["input_hashes"] == {"args.arg_0": "hash_of_arg0"}
        assert manifest["args"][0]["content_hash"] == "hash_of_arg0"
        assert deserialize_inputs(tmp_path / "job")["input_hashes"] == {"args.arg_0": "hash_of_arg0"}


class TestOutputSerialization:
    def test_single_primitive(self, tmp_path):
        serialize_outputs(tmp_path, -75.5)
        assert deserialize_outputs(tmp_path) == -75.5

    def test_tuple_result(self, tmp_path):
        serialize_outputs(tmp_path, (-75.5, "converged"))
        result = deserialize_outputs(tmp_path)
        assert result == (-75.5, "converged")
        assert isinstance(result, tuple)

    def test_dataclass_result(self, tmp_path, sample_orbitals):
        serialize_outputs(tmp_path, sample_orbitals)
        assert isinstance(deserialize_outputs(tmp_path), Orbitals)

    def test_mixed_tuple(self, tmp_path, sample_orbitals):
        serialize_outputs(tmp_path, (-75.5, sample_orbitals))
        result = deserialize_outputs(tmp_path)
        assert result[0] == -75.5
        assert isinstance(result[1], Orbitals)

    def test_nested_dataclass_result_survives_transfer(self, tmp_path, sample_orbitals):
        source_dir = tmp_path / "source"
        destination_dir = tmp_path / "destination"
        destination_dir.mkdir()

        files = serialize_outputs(source_dir, [sample_orbitals, sample_orbitals])

        assert {path.name for path in files} == {
            "manifest.json",
            f"{sample_orbitals.content_hash()}.orbitals.h5",
        }
        for path in files:
            shutil.copy2(path, destination_dir / path.name)

        result = deserialize_outputs(destination_dir)
        assert all(isinstance(item, Orbitals) for item in result)


@pytest.mark.parametrize("serializer", ["inputs", "outputs"])
def test_manifest_write_is_atomic(tmp_path, monkeypatch, serializer):
    """A failed temporary write must not replace a complete manifest."""
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


def test_worker_executes_serialized_algorithm(tmp_path, monkeypatch):
    """The compute-node worker reconstructs and executes an algorithm."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    serialize_inputs(
        input_dir,
        args=(2,),
        kwargs={"scale": 3},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={"threshold": 0.5},
    )
    algorithm = MagicMock()
    algorithm.run.return_value = 6
    create_algorithm = MagicMock(return_value=algorithm)
    monkeypatch.setattr(algorithms_module, "create", create_algorithm)

    result = execute_job(input_dir, output_dir)

    assert result == 6
    assert deserialize_outputs(output_dir) == 6
    create_algorithm.assert_called_once_with("test_algorithm", "plugin")
    algorithm.settings.return_value.set.assert_called_once_with("threshold", 0.5)
    algorithm.run.assert_called_once_with(2, scale=3)


def test_worker_force_rerun_bypasses_remote_cache(tmp_path, monkeypatch):
    """The compute node ignores a shared-cache result when forced to rerun."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    serialize_inputs(
        input_dir,
        args=(),
        kwargs={},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={},
        run_hash="testhash",
        force_rerun=True,
    )
    monkeypatch.setattr(remote_worker, "_load_remote_cache", MagicMock(return_value=(MagicMock(), "testhash", True)))
    get_cached_result = MagicMock(return_value=-75.5)
    monkeypatch.setattr(remote_worker, "_get_cached_result", get_cached_result)
    algorithm = MagicMock()
    algorithm.run.return_value = 6
    monkeypatch.setattr(algorithms_module, "create", MagicMock(return_value=algorithm))

    assert execute_job(input_dir, output_dir) == 6
    get_cached_result.assert_not_called()
    algorithm.run.assert_called_once_with()


def test_worker_cache_transport_skips_output_serialization(tmp_path, monkeypatch):
    """Shared-cache transport does not create unused output artifacts."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    serialize_inputs(
        input_dir,
        args=(),
        kwargs={},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={},
        run_hash="testhash",
        remote_cache={"name": "shared"},
        remote_cache_transport=True,
    )
    result = 6
    monkeypatch.setattr(remote_worker, "_load_remote_cache", MagicMock(return_value=(MagicMock(), "testhash", False)))
    monkeypatch.setattr(remote_worker, "_get_cached_result", MagicMock(return_value=result))
    serialize = MagicMock()
    monkeypatch.setattr(serialization_module, "serialize_outputs", serialize)

    assert execute_job(input_dir, output_dir) == result
    serialize.assert_not_called()
    assert not output_dir.exists()


def test_worker_logs_remote_cache_load_failure(tmp_path, monkeypatch, caplog):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "run_hash": "testhash", "remote_cache": {"name": "unavailable"}})
    )
    monkeypatch.setattr(remote_cache_module, "get_cache", MagicMock(side_effect=RuntimeError("cache unavailable")))

    with caplog.at_level(logging.WARNING, logger=remote_worker.__name__):
        cache, run_hash, force_rerun = remote_worker._load_remote_cache(input_dir)

    assert cache is None
    assert run_hash == "testhash"
    assert force_rerun is False
    record = next(record for record in caplog.records if record.name == remote_worker.__name__)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert "Failed to load remote cache" in record.message


def test_worker_validates_manifest_before_loading_remote_cache(tmp_path, monkeypatch, caplog):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "run_hash": "testhash", "remote_cache": {"name": "shared"}})
    )
    get_cache = MagicMock()
    monkeypatch.setattr(remote_cache_module, "get_cache", get_cache)

    with caplog.at_level(logging.WARNING, logger=remote_worker.__name__):
        cache, run_hash, force_rerun = remote_worker._load_remote_cache(input_dir)

    assert cache is None
    assert run_hash is None
    assert force_rerun is False
    get_cache.assert_not_called()
    record = next(record for record in caplog.records if record.name == remote_worker.__name__)
    assert "Unsupported manifest version 2; expected 1" in record.exc_text


def test_worker_logs_cache_read_failure(caplog):
    cache = MagicMock()
    cache.get_job.side_effect = RuntimeError("cache read failed")

    with caplog.at_level(logging.WARNING, logger=remote_worker.__name__):
        result = remote_worker._get_cached_result(cache, "testhash")

    assert result is remote_worker._CACHE_MISS
    record = next(record for record in caplog.records if record.name == remote_worker.__name__)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert "Failed to read cached result for run testhash" in record.message


def test_worker_cache_hit_skips_input_deserialization(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    serialize_inputs(
        input_dir,
        args=(),
        kwargs={},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={},
        run_hash="testhash",
    )
    monkeypatch.setattr(remote_worker, "_load_remote_cache", MagicMock(return_value=(MagicMock(), "testhash", False)))
    monkeypatch.setattr(remote_worker, "_get_cached_result", MagicMock(return_value=6))
    deserialize = MagicMock()
    monkeypatch.setattr(serialization_module, "deserialize_inputs", deserialize)

    assert execute_job(input_dir, output_dir) == 6
    deserialize.assert_not_called()
    assert deserialize_outputs(output_dir) == 6


@pytest.mark.parametrize(
    ("result", "output_is_tuple"),
    [
        pytest.param(None, False, id="none"),
        pytest.param((42,), True, id="singleton-tuple"),
        pytest.param((), True, id="empty-tuple"),
        pytest.param((None,), True, id="singleton-none-tuple"),
        pytest.param((1, (2, 3)), True, id="nested-tuple"),
    ],
)
def test_worker_cache_preserves_result_shape(tmp_path, result, output_is_tuple):
    cache = FolderCache(path=tmp_path / "cache")
    inputs = {
        "algorithm_type": "test_algorithm",
        "algorithm_name": "plugin",
        "settings": {},
    }

    remote_worker._store_cached_result(cache, "testhash", inputs, result)

    assert remote_worker._get_cached_result(cache, "testhash") == result
    job = cache.get_job("testhash")
    assert job is not None
    assert job.output_is_tuple is output_is_tuple


def test_worker_logs_cache_write_failure(caplog):
    cache = MagicMock()
    cache.put_job.side_effect = RuntimeError("cache write failed")
    inputs = {
        "algorithm_type": "test_algorithm",
        "algorithm_name": "plugin",
        "settings": {},
    }

    with caplog.at_level(logging.WARNING, logger=remote_worker.__name__):
        remote_worker._store_cached_result(cache, "testhash", inputs, 42)

    record = next(record for record in caplog.records if record.name == remote_worker.__name__)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert "Failed to store cached result for run testhash" in record.message


def test_worker_executes_transferred_nested_dataclasses(tmp_path, monkeypatch, h2_structure):
    """The upload list includes nested arguments and file-backed settings."""
    source_dir = tmp_path / "source"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    files = serialize_inputs(
        source_dir,
        args=([h2_structure, h2_structure],),
        kwargs={},
        algorithm_type="test_algorithm",
        algorithm_name="plugin",
        settings={"reference": h2_structure},
    )

    assert {path.name for path in files} == {
        "manifest.json",
        f"{h2_structure.content_hash()}.structure.h5",
    }
    for path in files:
        shutil.copy2(path, input_dir / path.name)

    algorithm = MagicMock()
    algorithm.run.return_value = 2
    monkeypatch.setattr(algorithms_module, "create", MagicMock(return_value=algorithm))

    assert execute_job(input_dir, output_dir) == 2
    nested_arg = algorithm.run.call_args.args[0]
    assert len(nested_arg) == 2
    assert all(isinstance(item, Structure) for item in nested_arg)
    algorithm.settings.return_value.set.assert_called_once()


class TestJob:
    def test_save_load_round_trip(self, tmp_path):
        job = Job(
            job_id="j1",
            backend="local",
            backend_config={"timeout": 60},
            backend_state={"pid": 1234},
            algorithm_info={"type": "scf_solver"},
            status="submitted",
            run_hash="aaaa",
            input_hashes={"args.arg_0": "hash0"},
            owner={"workspace_root": "/workspace", "project_name": "project-a"},
        )
        loaded = Job.load(job.save(tmp_path / "job_j1.json"))
        assert loaded.job_id == "j1"
        assert loaded.backend == "local"
        assert loaded.backend_config == {"timeout": 60}
        assert loaded.backend_state == {"pid": 1234}
        assert loaded.run_hash == "aaaa"
        assert loaded.input_hashes == {"args.arg_0": "hash0"}
        assert loaded.owner == {"workspace_root": "/workspace", "project_name": "project-a"}

    def test_load_legacy_job_without_owner(self, tmp_path):
        job = Job(job_id="j1", backend="local", backend_config={}, backend_state={})
        data = job.to_dict()
        data["version"] = 2
        data.pop("owner", None)
        path = tmp_path / "job_j1.json"
        path.write_text(json.dumps(data))

        assert Job.load(path).owner is None

    def test_load_requires_status(self, tmp_path):
        path = tmp_path / "job_j1.json"
        path.write_text(json.dumps({"job_id": "j1", "backend": "local"}))

        with pytest.raises(ValueError, match="missing required field 'status'"):
            Job.load(path)

    def test_to_dict_normalizes_paths_and_algorithm_refs(self, tmp_path):
        """Job metadata uses the same durable representation as remote settings."""
        energy_calculator = AlgorithmRef("scf_solver", "qdk")
        job = Job(
            job_id="j1",
            backend="local",
            backend_config={"workdir": tmp_path},
            backend_state={"output_dir": tmp_path / "output"},
            algorithm_info={"settings": {"energy_calculator": energy_calculator}},
        )

        persisted = job.to_dict()

        assert persisted["backend_config"]["workdir"] == str(tmp_path)
        assert persisted["backend_state"]["output_dir"] == str(tmp_path / "output")
        serialized_ref = persisted["algorithm_info"]["settings"]["energy_calculator"]
        assert serialized_ref["__type__"] == "algorithm_ref"
        assert serialized_ref["algorithm_type"] == "scf_solver"
        json.dumps(persisted)

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        """A failed temporary write must preserve the existing job handle."""
        path = tmp_path / "job_x.json"
        original_job = '{"job_id": "x", "status": "submitted"}'
        path.write_text(original_job)

        def fail_after_partial_write(_value, file, **_kwargs):
            file.write('{"partial"')
            raise OSError("simulated job write failure")

        monkeypatch.setattr(job_module.json, "dump", fail_after_partial_write)
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status="running")

        with pytest.raises(OSError, match="simulated job write failure"):
            job.save(path)

        assert path.read_text() == original_job
        assert not list(tmp_path.glob("*.tmp"))

    @pytest.mark.parametrize(
        "status",
        ["succeeded", "Succeeded", "FAILED", "canceled", "CANCELLED", "retrieved"],
    )
    def test_is_terminal(self, status):
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status=status)
        assert job.is_terminal

    def test_terminal_statuses_are_public(self):
        assert frozenset({"succeeded", "failed", "canceled", "cancelled", "retrieved"}) == JobStatus.TERMINAL_STATUSES

    def test_job_states_are_case_insensitive(self):
        assert remote_backends.JobState("RUNNING") is remote_backends.JobState.RUNNING
        assert remote_backends.JobState("Succeeded") is remote_backends.JobState.SUCCEEDED

    @pytest.mark.parametrize("status", ["succeeded", "Succeeded", "SUCCEEDED"])
    def test_is_successful(self, status):
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status=status)
        assert job.is_successful

    @pytest.mark.parametrize("status", ["submitted", "running", "pending"])
    def test_is_not_terminal(self, status):
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status=status)
        assert not job.is_terminal

    def test_wait_times_out_with_backend_error(self, monkeypatch):
        job = Job(
            job_id="x",
            backend="test-remote",
            backend_config={"timeout": 0},
            backend_state={},
        )
        job.check = MagicMock(return_value=JobStatus(job_id="x", status="unknown", error="Invalid PID file"))
        job.attach_backend(MagicMock())
        sleep = MagicMock()
        monkeypatch.setattr(time, "sleep", sleep)

        with pytest.raises(TimeoutError, match="Invalid PID file"):
            job.wait()

        job.check.assert_called_once_with()
        sleep.assert_not_called()

    def test_wait_uses_backend_poll_interval(self, monkeypatch):
        job = Job(
            job_id="x",
            backend="test-remote",
            backend_config={"poll_interval": 0.25, "timeout": 1},
            backend_state={},
            status="running",
        )
        job.check = MagicMock(
            side_effect=[
                JobStatus(job_id="x", status="running"),
                JobStatus(job_id="x", status="succeeded"),
            ]
        )
        job.attach_backend(MagicMock())
        sleep = MagicMock()
        monkeypatch.setattr(time, "sleep", sleep)

        job.wait()

        sleep.assert_called_once_with(0.25)

    def test_wait_reuses_reconstructed_backend(self, monkeypatch):
        backend = MagicMock()
        backend.check.side_effect = [
            JobStatus(job_id="x", status="running"),
            JobStatus(job_id="x", status="succeeded"),
        ]
        get_backend = MagicMock(return_value=backend)
        monkeypatch.setattr(remote_backends, "get_backend", get_backend)
        job = Job(
            job_id="x",
            backend="test-remote",
            backend_config={"poll_interval": 0, "timeout": 1},
            backend_state={"operation_id": "operation"},
        )

        status = job.wait()

        assert status.status == "succeeded"
        get_backend.assert_called_once_with("test-remote", poll_interval=0, timeout=1)
        backend.connect.assert_called_once_with()
        assert backend.check.call_count == 2
        backend.disconnect.assert_called_once_with()
        assert job._active_backend is None

    def test_discover_finds_jobs(self, tmp_path):
        for i in range(3):
            Job(job_id=f"j{i}", backend="local", backend_config={}, backend_state={}).save(tmp_path / f"j{i}.job.json")
        jobs = Job.discover(tmp_path)
        assert {j.job_id for j in jobs} == {"j0", "j1", "j2"}

    def test_discover_skips_corrupt(self, tmp_path):
        (tmp_path / "bad.job.json").write_text("not json")
        Job(job_id="good", backend="local", backend_config={}, backend_state={}).save(tmp_path / "good.job.json")
        assert [j.job_id for j in Job.discover(tmp_path)] == ["good"]

    def test_save_without_path_raises(self):
        with pytest.raises(ValueError, match="No file path"):
            Job(job_id="x", backend="local", backend_config={}, backend_state={}).save()

    def test_check_delegates_to_backend_and_persists_status(self, tmp_path, monkeypatch):
        backend = MagicMock()
        backend.check.return_value = MagicMock(status="Succeeded", logs="done")
        get_backend = MagicMock(return_value=backend)
        monkeypatch.setattr(remote_backends, "get_backend", get_backend)
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={"project_name": "project"},
            backend_state={"operation_id": "operation"},
            file_path=tmp_path / "job_x.json",
        )

        status = job.check()

        get_backend.assert_called_once_with("discovery", project_name="project")
        backend.connect.assert_called_once_with()
        backend.check.assert_called_once_with({"operation_id": "operation"})
        backend.disconnect.assert_called_once_with()
        assert status.status == "Succeeded"
        assert Job.load(tmp_path / "job_x.json").status == "Succeeded"

    def test_check_restores_retrieved_status_from_output_hashes(self, tmp_path, monkeypatch):
        backend = MagicMock()
        backend.check.return_value = JobStatus(job_id="x", status="Succeeded")
        monkeypatch.setattr(remote_backends, "get_backend", MagicMock(return_value=backend))
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
            status="Succeeded",
            file_path=tmp_path / "job_x.json",
            output_hashes=[{"hash": "wavefunction-hash", "type": "wavefunction"}],
        )

        status = job.check()

        assert status.status == "retrieved"
        persisted = Job.load(tmp_path / "job_x.json")
        assert persisted.status == "retrieved"
        assert persisted.output_hashes == job.output_hashes

    def test_cancel_delegates_to_backend_and_persists_status(self, tmp_path, monkeypatch):
        backend = MagicMock()
        monkeypatch.setattr(remote_backends, "get_backend", MagicMock(return_value=backend))
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
            file_path=tmp_path / "job_x.json",
        )

        job.cancel()

        backend.connect.assert_called_once_with()
        backend.cancel.assert_called_once_with({"operation_id": "operation"})
        backend.disconnect.assert_called_once_with()
        assert Job.load(tmp_path / "job_x.json").status == "canceled"

    @pytest.mark.parametrize("cleanup", [False, True])
    def test_fetch_delegates_to_backend_and_persists_results(self, tmp_path, monkeypatch, cleanup):
        backend = MagicMock()
        backend.fetch.return_value = (-1.0, "wavefunction.h5")
        monkeypatch.setattr(remote_backends, "get_backend", MagicMock(return_value=backend))
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
            file_path=tmp_path / "job_x.json",
        )
        result_dir = tmp_path / "results"

        def assert_persisted_before_cleanup(backend_state):
            assert backend_state == {"operation_id": "operation"}
            persisted = Job.load(tmp_path / "job_x.json")
            assert persisted.status == "retrieved"
            assert persisted.output_hashes is not None
            assert persisted.output_is_tuple is True

        backend.cleanup_job.side_effect = assert_persisted_before_cleanup

        result = job.fetch(local_dir=result_dir, cleanup=cleanup)

        backend.connect.assert_called_once_with()
        backend.fetch.assert_called_once_with({"operation_id": "operation"}, local_dir=result_dir)
        if cleanup:
            backend.cleanup_job.assert_called_once_with({"operation_id": "operation"})
        else:
            backend.cleanup_job.assert_not_called()
        backend.disconnect.assert_called_once_with()
        assert result == (-1.0, "wavefunction.h5")
        persisted = Job.load(tmp_path / "job_x.json")
        assert persisted.status == "retrieved"
        assert persisted.output_hashes is not None
        assert persisted.output_is_tuple is True

    def test_fetch_failure_preserves_backend_artifacts(self, monkeypatch):
        backend = MagicMock()
        backend.fetch.side_effect = ValueError("invalid output")
        monkeypatch.setattr(remote_backends, "get_backend", MagicMock(return_value=backend))
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
        )

        with pytest.raises(ValueError, match="invalid output"):
            job.fetch(cleanup=True)

        backend.cleanup_job.assert_not_called()
        backend.disconnect.assert_called_once_with()

    def test_cleanup_delegates_for_terminal_job(self, monkeypatch):
        backend = MagicMock()
        monkeypatch.setattr(remote_backends, "get_backend", MagicMock(return_value=backend))
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
            status="succeeded",
        )

        job.cleanup()

        backend.connect.assert_called_once_with()
        backend.cleanup_job.assert_called_once_with({"operation_id": "operation"})
        backend.disconnect.assert_called_once_with()

    def test_cleanup_rejects_nonterminal_job(self):
        job = Job(
            job_id="x",
            backend="discovery",
            backend_config={},
            backend_state={"operation_id": "operation"},
        )

        with pytest.raises(RuntimeError, match="terminal state"):
            job.cleanup()

    def test_output_hashes_round_trip(self, tmp_path):
        hashes = [{"hash": "h1", "type": "float", "value": -75.5}, {"hash": "h2", "type": "wavefunction"}]
        job = Job(
            job_id="x",
            backend="local",
            backend_config={},
            backend_state={},
            output_hashes=hashes,
            output_is_tuple=True,
        )
        loaded = Job.load(job.save(tmp_path / "job_x.json"))
        assert loaded.output_hashes == hashes
        assert loaded.output_is_tuple is True


class TestBackendRegistry:
    def test_builtin_backends_registered(self):
        registered = available_backends()
        assert "local" in registered

    def test_get_local_backend(self):
        assert isinstance(get_backend("local"), LocalBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="No backend"):
            get_backend("does_not_exist")

    def test_register_duplicate_backend_name_raises(self, monkeypatch):
        """Two remote backends cannot silently claim the same name."""

        class FirstBackend(RemoteBackend):
            """First remote backend claiming the test name."""

        class SecondBackend(RemoteBackend):
            """Second remote backend claiming the test name."""

        monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {})
        register_backend("duplicate-backend")(FirstBackend)

        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            register_backend("duplicate-backend")(SecondBackend)

        with pytest.raises(DuplicateRegistrationError, match=r"already registered.*duplicate-backend"):
            register_backend("backend-alias")(FirstBackend)

        assert remote_backend_registry._BACKENDS["duplicate-backend"] is FirstBackend
        assert "backend-alias" not in remote_backend_registry._BACKENDS
        assert FirstBackend.name == "duplicate-backend"

    def test_register_custom_backend(self):
        class StubBackend(RemoteBackend):
            name = "_test_stub"

            def connect(self):
                pass

            def disconnect(self):
                pass

            def upload(self, local_path, remote_path):
                pass

            def download(self, remote_path, local_path):
                pass

            def _submit(self, payload):
                raise NotImplementedError

            def check(self, backend_state):
                raise NotImplementedError

            def fetch(self, backend_state, local_dir=None):
                raise NotImplementedError

        register_backend("_test_stub")(StubBackend)

        assert "_test_stub" in available_backends()
        assert isinstance(get_backend("_test_stub"), StubBackend)

    @pytest.mark.parametrize(
        ("declaration", "match"),
        [
            (["endpoint"], "frozenset"),
            (frozenset({""}), "non-empty strings"),
            (frozenset({"missing"}), "not named constructor parameters"),
        ],
    )
    def test_register_backend_validates_mcp_safe_config_options(self, monkeypatch, declaration, match):
        class InvalidBackend(RemoteBackend):
            mcp_safe_config_options = declaration

            def __init__(self, *, endpoint=None):
                super().__init__(endpoint=endpoint)

        monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {})

        with pytest.raises(TypeError, match=rf"{match}"):
            register_backend("invalid-mcp-config")(InvalidBackend)

        assert remote_backend_registry._BACKENDS == {}

    def test_registered_backend_does_not_inherit_mcp_safe_config(self, monkeypatch):
        class ParentBackend(RemoteBackend):
            mcp_safe_config_options = frozenset({"endpoint"})

            def __init__(self, *, endpoint=None):
                super().__init__(endpoint=endpoint)

        class ChildBackend(ParentBackend):
            pass

        monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {})
        register_backend("child-backend")(ChildBackend)

        assert "mcp_safe_config_options" not in ChildBackend.__dict__


@pytest.fixture(params=["local"])
def backend(request):
    """Yield a connected backend instance; disconnect after use.

    Add new backend names to `params` to run the contract tests against them.
    """
    if request.param == "local":
        b = LocalBackend()
    else:
        pytest.skip(f"Backend {request.param!r} not available in CI")
    b.connect()
    yield b
    b.disconnect()


class TestBackendContract:
    """Shared tests every backend must satisfy. Parameterized via the `backend` fixture."""

    @staticmethod
    def _mock_backend(*, backend_config=None, backend_state=None):
        """Create a backend double with configurable persisted metadata."""
        backend = MagicMock(spec=RemoteBackend)
        backend.name = "test-remote"
        backend._backend_args = backend_config or {}
        backend._submit.return_value = ("job-id", backend_state or {})
        return backend

    def test_submit_normalizes_persisted_metadata(self, tmp_path):
        """Supported rich values are normalized before a job is returned."""
        backend = self._mock_backend(
            backend_config={"workdir": tmp_path},
            backend_state={"output_dir": tmp_path / "output"},
        )
        energy_calculator = AlgorithmRef("scf_solver", "qdk")

        job = RemoteBackend.submit(
            backend,
            {
                "algorithm_type": "test_algorithm",
                "algorithm_name": "plugin",
                "settings": {"energy_calculator": energy_calculator},
                "owner": {"workspace_root": "/workspace", "project_name": "project-a"},
            },
            job_dir=tmp_path,
        )

        loaded = Job.load(tmp_path / "job-id.job.json")
        assert loaded.backend_config == {"workdir": str(tmp_path)}
        assert loaded.backend_state == {"output_dir": str(tmp_path / "output")}
        assert loaded.algorithm_info["settings"]["energy_calculator"]["__type__"] == "algorithm_ref"
        assert loaded.owner == {"workspace_root": "/workspace", "project_name": "project-a"}
        assert job.to_dict() == loaded.to_dict()
        backend._submit.assert_called_once()

    @pytest.mark.parametrize(
        ("backend_config", "payload", "field"),
        [
            ({"client": object()}, {"settings": {}}, "backend_config"),
            ({}, {"settings": {"client": object()}}, "algorithm_info"),
            ({}, {"settings": {}, "run_hash": object()}, "run_hash"),
            ({}, {"settings": {}, "input_hashes": {"arg": object()}}, "input_hashes"),
        ],
    )
    def test_submit_rejects_unpersistable_metadata_before_launch(self, backend_config, payload, field):
        """Known unpersistable metadata prevents backend submission."""
        backend = self._mock_backend(backend_config=backend_config)

        with pytest.raises(TypeError, match=field):
            RemoteBackend.submit(backend, payload)

        backend._submit.assert_not_called()

    def test_submit_discards_job_with_unpersistable_backend_state(self):
        """Invalid state triggers best-effort cancellation and cleanup."""
        backend_state = {"client": object()}
        backend = self._mock_backend(backend_state=backend_state)

        with pytest.raises(TypeError, match="backend_state"):
            RemoteBackend.submit(backend, {"settings": {}})

        backend.cancel.assert_called_once_with(backend_state)
        backend.cleanup_job.assert_called_once_with(backend_state)

    def test_upload_download_round_trip(self, backend, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("hello remote")

        remote = f"{backend.remote_workdir}/sub/input.txt"
        backend.upload(src, remote)

        dest = tmp_path / "downloaded.txt"
        backend.download(remote, dest)
        assert dest.read_text() == "hello remote"

    def test_async_submit_reaches_terminal(self, backend):
        payload = {
            "algorithm_type": "scf_solver",
            "algorithm_name": "qdk",
            "settings": {},
            "args": (42,),
            "kwargs": {"flag": True},
        }
        job_id, state = backend._submit(payload)
        assert isinstance(job_id, str)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = backend.check(state)
            if status.is_terminal:
                break
            time.sleep(0.2)
        assert status.is_terminal


class TestLocalBackendSpecific:
    def test_constructor_arguments_are_persisted(self, monkeypatch):
        backend = LocalBackend(
            poll_interval=0.25,
            timeout=60,
            python_path="/custom/python",
        )
        monkeypatch.setattr(backend, "_submit", MagicMock(return_value=("job-id", {})))

        job = backend.submit({})

        assert job.backend_config == {
            "poll_interval": 0.25,
            "timeout": 60,
            "python_path": "/custom/python",
        }
        assert backend._backend_args == job.backend_config

    def test_constructor_rejects_unknown_arguments(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            get_backend("local", timeot=60)

    def test_connect_creates_workdir(self):
        backend = LocalBackend()
        backend.connect()
        assert Path(backend.remote_workdir).exists()
        backend.disconnect()

    def test_disconnect_removes_workdir(self):
        backend = LocalBackend()
        backend.connect()
        workdir = backend.remote_workdir
        backend.disconnect()
        assert not Path(workdir).exists()

    def test_disconnect_preserves_nonempty_workdir(self):
        backend = LocalBackend()
        backend.connect()
        workdir = Path(backend.remote_workdir)
        (workdir / "job_running").mkdir()

        backend.disconnect()

        assert Path(workdir).exists()
        shutil.rmtree(workdir, ignore_errors=True)

    def test_cleanup_job_is_idempotent_and_removes_empty_workdir(self, tmp_path):
        workdir = tmp_path / "qdk_local"
        job_workdir = workdir / "job_x"
        output_dir = job_workdir / "output"
        output_dir.mkdir(parents=True)
        state = {
            "workdir": str(workdir),
            "job_workdir": str(job_workdir),
            "output_dir": str(output_dir),
        }
        backend = LocalBackend()

        backend.cleanup_job(state)
        backend.cleanup_job(state)

        assert not job_workdir.exists()
        assert not workdir.exists()

    def test_cleanup_job_preserves_sibling_jobs(self, tmp_path):
        workdir = tmp_path / "qdk_local"
        job_workdir = workdir / "job_x"
        output_dir = job_workdir / "output"
        output_dir.mkdir(parents=True)
        sibling = workdir / "job_y"
        sibling.mkdir()
        backend = LocalBackend()

        backend.cleanup_job(
            {
                "workdir": str(workdir),
                "job_workdir": str(job_workdir),
                "output_dir": str(output_dir),
            }
        )

        assert not job_workdir.exists()
        assert sibling.exists()

    def test_cleanup_job_rejects_inconsistent_paths(self, tmp_path):
        backend = LocalBackend()

        with pytest.raises(ValueError, match="inconsistent"):
            backend.cleanup_job(
                {
                    "workdir": str(tmp_path / "workdir"),
                    "job_workdir": str(tmp_path / "other" / "job_x"),
                    "output_dir": str(tmp_path / "other" / "job_x" / "output"),
                }
            )

    def test_submit_launches_remote_worker_module(self, monkeypatch):
        backend = LocalBackend()
        backend.connect()
        popen = MagicMock(return_value=MagicMock(pid=1234))
        monkeypatch.setattr("subprocess.Popen", popen)
        try:
            backend._submit(
                {
                    "algorithm_type": "test_algorithm",
                    "algorithm_name": "plugin",
                    "settings": {},
                    "args": (),
                    "kwargs": {},
                }
            )
        finally:
            backend.disconnect()

        command = popen.call_args.args[0]
        assert command[1:3] == ["-m", "qdk_chemistry.remote.worker"]

    def test_check_polls_submitted_process_handle(self, monkeypatch):
        backend = LocalBackend()
        backend.connect()
        process = MagicMock(pid=1234)
        process.poll.side_effect = [None, 1]
        monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=process))
        try:
            job_id, state = backend._submit(
                {
                    "algorithm_type": "test_algorithm",
                    "algorithm_name": "plugin",
                    "settings": {},
                    "args": (),
                    "kwargs": {},
                }
            )

            assert state["job_id"] == job_id
            running_status = backend.check(state)
            failed_status = backend.check(state)
            assert running_status.status is JobState.RUNNING
            assert running_status.job_id == job_id
            assert failed_status.status is JobState.FAILED
            assert failed_status.job_id == job_id
        finally:
            backend.disconnect()

    def test_check_falls_back_to_persisted_pid(self, tmp_path, monkeypatch):
        process_is_running = MagicMock(return_value=False)
        monkeypatch.setattr(local_backend_module, "_process_is_running", process_is_running)
        backend = LocalBackend()
        state = {
            "job_id": "job-id",
            "pid": 1234,
            "output_dir": str(tmp_path / "output"),
            "job_workdir": str(tmp_path),
        }

        status = backend.check(state)
        assert status.status is JobState.FAILED
        assert status.job_id == "job-id"
        process_is_running.assert_called_once_with(1234)

    def test_check_rehydrated_job_prefers_completed_manifest(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "manifest.json").write_text("{}")
        process_is_running = MagicMock(return_value=True)
        process_identity = MagicMock(return_value="linux:789")
        monkeypatch.setattr(local_backend_module, "_process_is_running", process_is_running)
        monkeypatch.setattr(local_backend_module, "_process_identity", process_identity)
        backend = LocalBackend()
        state = {
            "job_id": "job-id",
            "pid": 1234,
            "process_identity": "linux:456",
            "output_dir": str(output_dir),
            "job_workdir": str(tmp_path),
        }

        status = backend.check(state)

        assert status.status is JobState.SUCCEEDED
        process_is_running.assert_not_called()
        process_identity.assert_not_called()

    def test_check_rehydrated_job_rejects_reused_pid(self, tmp_path, monkeypatch):
        process_is_running = MagicMock(return_value=True)
        process_identity = MagicMock(return_value="linux:789")
        monkeypatch.setattr(local_backend_module, "_process_is_running", process_is_running)
        monkeypatch.setattr(local_backend_module, "_process_identity", process_identity)
        backend = LocalBackend()
        state = {
            "job_id": "job-id",
            "pid": 1234,
            "process_identity": "linux:456",
            "output_dir": str(tmp_path / "output"),
            "job_workdir": str(tmp_path),
        }

        status = backend.check(state)

        assert status.status is JobState.FAILED
        process_identity.assert_called_once_with(1234)
        process_is_running.assert_not_called()

    def test_cancel_rehydrated_job_requires_matching_process_identity(self, monkeypatch):
        backend = LocalBackend()
        kill = MagicMock()
        monkeypatch.setattr("os.kill", kill)
        monkeypatch.setattr(local_backend_module, "_process_identity", MagicMock(return_value="linux:456"))

        backend.cancel({"pid": 1234, "process_identity": "linux:456"})

        kill.assert_called_once_with(1234, signal.SIGTERM)

    def test_cancel_rehydrated_job_refuses_unverified_process(self, monkeypatch):
        backend = LocalBackend()
        kill = MagicMock()
        monkeypatch.setattr("os.kill", kill)
        monkeypatch.setattr(local_backend_module, "_process_identity", MagicMock(return_value="linux:789"))

        with pytest.raises(RuntimeError, match="Cannot verify local job process identity"):
            backend.cancel({"pid": 1234, "process_identity": "linux:456"})

        kill.assert_not_called()


class TestRunWithCache:
    @staticmethod
    def _mock_algorithm(result=-75.5):
        algo = MagicMock()
        algo.type_name.return_value = "scf_solver"
        algo.name.return_value = "qdk"
        algo.settings.return_value.to_dict.return_value = {}
        algo.hash.return_value = "testhash1234abcd"
        algo.run.return_value = result
        return algo

    def test_input_hash_namespaces_do_not_collide(self, tmp_path):
        """Keep a keyword named arg_0 distinct from positional argument zero."""
        positional = np.array([1.0])
        keyword = np.array([2.0])
        payload = _build_payload_for(self._mock_algorithm(), (positional,), {"arg_0": keyword})
        input_hashes = payload["input_hashes"]
        assert input_hashes == {
            "args.arg_0": _item_content_hash(positional),
            "kwargs.arg_0": _item_content_hash(keyword),
        }

        shared_cache = FolderCache(path=tmp_path / "shared", is_shared=True)
        shared_cache.put_data(input_hashes["args.arg_0"], positional)
        shared_cache.put_data(input_hashes["kwargs.arg_0"], keyword)
        serialize_inputs(
            tmp_path / "input",
            args=payload["args"],
            kwargs=payload["kwargs"],
            algorithm_type=payload["algorithm_type"],
            algorithm_name=payload["algorithm_name"],
            settings=payload["settings"],
            input_hashes=input_hashes,
            remote_cache_backend=shared_cache,
        )
        restored = deserialize_inputs(tmp_path / "input", cache=shared_cache)

        np.testing.assert_array_equal(restored["args"][0], positional)
        np.testing.assert_array_equal(restored["kwargs"]["arg_0"], keyword)

    def test_run_no_cache(self):
        algo = self._mock_algorithm()
        assert run(algo, "arg1", cache=None, remote=None) == -75.5
        algo.run.assert_called_once_with("arg1")

    def test_named_remote_without_cache_disconnects_owned_backend(self, monkeypatch):
        algo = self._mock_algorithm()
        backend = MagicMock(spec=RemoteBackend)
        backend.name = "test"
        backend._backend_args = {"poll_interval": 0, "timeout": 1}
        backend._submit.return_value = ("remote-job", {})
        backend.submit.side_effect = lambda payload, job_dir=None: RemoteBackend.submit(
            backend, payload, job_dir=job_dir
        )
        backend.check.side_effect = [
            JobStatus(job_id="remote-job", status="running"),
            JobStatus(job_id="remote-job", status="succeeded"),
        ]
        backend.fetch.return_value = -75.5
        get_backend = MagicMock(return_value=backend)
        monkeypatch.setattr(remote_backends, "get_backend", get_backend)

        assert run(algo, "arg1", remote="test") == -75.5

        get_backend.assert_called_once_with("test")
        backend.submit.assert_called_once()
        assert backend.check.call_count == 2
        backend.fetch.assert_called_once_with({}, local_dir=None)
        backend.cleanup_job.assert_not_called()
        backend.connect.assert_called_once_with()
        backend.disconnect.assert_called_once_with()

    def test_injected_remote_without_cache_remains_connected(self):
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="succeeded",
        )
        job.fetch = MagicMock(return_value=-75.5)
        job.cleanup = MagicMock()
        backend.submit.return_value = job

        assert run(algo, "arg1", remote=backend) == -75.5
        gc.collect()

        backend.submit.assert_called_once()
        job.fetch.assert_called_once_with()
        job.cleanup.assert_not_called()
        backend.disconnect.assert_not_called()

    def test_injected_unregistered_backend_runs_to_completion(self):
        class UnregisteredBackend(RemoteBackend):
            name = "unregistered"

            def __init__(self):
                super().__init__(poll_interval=0, timeout=1)
                self.statuses = iter(["running", "succeeded"])

            def connect(self):
                pass

            def disconnect(self):
                pass

            def upload(self, local_path, remote_path):
                pass

            def download(self, remote_path, local_path):
                pass

            def _submit(self, _payload):
                return "remote-job", {}

            def check(self, _backend_state):
                return JobStatus(job_id="remote-job", status=next(self.statuses))

            def fetch(self, _backend_state, local_dir=None):
                del local_dir
                return -75.5

            def cleanup_job(self, _backend_state):
                pass

        assert run(self._mock_algorithm(), remote=UnregisteredBackend()) == -75.5

    def test_injected_backend_without_cleanup_runs_to_completion(self):
        class UnregisteredBackend(RemoteBackend):
            name = "unregistered"

            def __init__(self):
                super().__init__(poll_interval=0, timeout=1)
                self.statuses = iter(["running", "succeeded"])

            def connect(self):
                pass

            def disconnect(self):
                pass

            def upload(self, local_path, remote_path):
                pass

            def download(self, remote_path, local_path):
                pass

            def _submit(self, _payload):
                return "remote-job", {}

            def check(self, _backend_state):
                return JobStatus(job_id="remote-job", status=next(self.statuses))

            def fetch(self, _backend_state, local_dir=None):
                del local_dir
                return -75.5

        assert run(self._mock_algorithm(), remote=UnregisteredBackend()) == -75.5

    def test_submit_returns_job_for_injected_backend(self):
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = MagicMock(spec=Job)
        backend.submit.return_value = job

        assert submit(algo, "arg1", remote=backend) is job
        backend.submit.assert_called_once()

    def test_named_remote_with_cache_disconnects_owned_backend(self, tmp_path, monkeypatch):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="Succeeded",
        )
        job.fetch = MagicMock(return_value=-75.5)

        job.cleanup = MagicMock()
        backend.submit.return_value = job
        get_backend = MagicMock(return_value=backend)
        monkeypatch.setattr(remote_backends, "get_backend", get_backend)

        assert run(algo, "arg1", cache=cache, remote="test") == -75.5

        get_backend.assert_called_once_with("test")
        backend.connect.assert_called_once_with()
        backend.disconnect.assert_called_once_with()
        job.cleanup.assert_not_called()
        payload = backend.submit.call_args.args[0]
        assert "remote_cache" not in payload
        assert "remote_cache_backend" not in payload

    def test_remote_failure_includes_backend_diagnostics(self, tmp_path):
        """A terminal remote failure reports backend error details and logs."""
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={"poll_interval": 0},
            backend_state={},
        )

        def fail_job():
            job.status = "failed"
            return JobStatus(
                job_id=job.job_id,
                status="failed",
                error="Remote worker exited with code 1",
                logs="Traceback (most recent call last): ...",
            )

        job.check = MagicMock(side_effect=fail_job)
        backend.submit.return_value = job

        with pytest.raises(RuntimeError) as error:
            run(algo, "arg1", cache=cache, remote=backend)

        assert "failed" in str(error.value)
        assert "Remote worker exited with code 1" in str(error.value)
        assert "Traceback (most recent call last): ..." in str(error.value)

    def test_cached_poll_accepts_lowercase_success(self, tmp_path, monkeypatch):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={"poll_interval": 0},
            backend_state={},
            status="running",
            run_hash="testhash1234abcd",
        )
        cache.put_job("testhash1234abcd", job)

        def complete_job(cached_job):
            cached_job.status = "succeeded"
            return JobStatus(job_id=cached_job.job_id, status="succeeded")

        monkeypatch.setattr(Job, "check", complete_job)
        monkeypatch.setattr(Job, "fetch", MagicMock(return_value=-75.5))
        cleanup = MagicMock()
        monkeypatch.setattr(Job, "cleanup", cleanup)
        sleep = MagicMock()
        monkeypatch.setattr(time, "sleep", sleep)

        assert run(algo, "arg1", cache=cache, remote=backend) == -75.5

        backend.submit.assert_not_called()
        sleep.assert_not_called()
        cleanup.assert_not_called()

    def test_foreign_inflight_cache_job_is_not_resumed(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        foreign_job = Job(
            job_id="foreign-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="running",
            run_hash="testhash1234abcd",
            owner={"workspace_root": "/workspace", "project_name": "project-a"},
        )
        foreign_job.wait = MagicMock()
        cache.put_job("testhash1234abcd", foreign_job)
        submitted_job = Job(
            job_id="current-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="succeeded",
        )
        submitted_job.wait = MagicMock(return_value=JobStatus(job_id="current-job", status="succeeded"))
        submitted_job.fetch = MagicMock(return_value=-75.5)
        backend = MagicMock()
        backend.submit.return_value = submitted_job
        owner = {"workspace_root": "/workspace", "project_name": "project-b"}

        assert run(algo, cache=cache, remote=backend, _owner=owner) == -75.5

        foreign_job.wait.assert_not_called()
        backend.submit.assert_called_once()
        assert backend.submit.call_args.args[0]["owner"] == owner
        assert submitted_job.owner == owner

    def test_cached_success_without_outputs_retries_fetch(self, tmp_path, monkeypatch):
        """A persisted success is fetched again instead of being resubmitted."""
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="succeeded",
            run_hash="testhash1234abcd",
        )
        cache.put_job("testhash1234abcd", job)
        fetch = MagicMock(return_value=-75.5)
        monkeypatch.setattr(Job, "fetch", fetch)
        cleanup = MagicMock()
        monkeypatch.setattr(Job, "cleanup", cleanup)

        assert run(algo, "arg1", cache=cache, remote=backend) == -75.5

        fetch.assert_called_once_with()
        cleanup.assert_not_called()
        backend.submit.assert_not_called()

    def test_remote_uses_shared_tier_from_cache(self, tmp_path):
        local_cache = FolderCache(path=tmp_path / "local")
        shared_cache = FolderCache(path=tmp_path / "shared", is_shared=True)
        cache = TieredCache([local_cache, shared_cache])
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="Succeeded",
        )
        job.fetch = MagicMock(return_value=-75.5)
        job.cleanup = MagicMock()
        backend.submit.return_value = job

        assert run(algo, "arg1", cache=cache, remote=backend) == -75.5

        payload = backend.submit.call_args.args[0]
        assert payload["remote_cache"] == {
            "name": "folder",
            "path": str(tmp_path / "shared"),
            "is_shared": True,
        }
        assert payload["remote_cache_backend"] is shared_cache
        job.cleanup.assert_not_called()

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(None, id="none"),
            pytest.param((42,), id="singleton-tuple"),
            pytest.param((), id="empty-tuple"),
        ],
    )
    def test_remote_reconstructs_result_from_shared_cache(self, tmp_path, result):
        local_cache = FolderCache(path=tmp_path / "local")
        shared_cache = FolderCache(path=tmp_path / "shared", is_shared=True)
        cache = TieredCache([local_cache, shared_cache])
        algo = self._mock_algorithm(result=result)
        backend = MagicMock()
        submitted_job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={"poll_interval": 0},
            backend_state={},
            status="running",
        )
        submitted_job.fetch = MagicMock(return_value="fetched")
        submitted_job.cleanup = MagicMock()
        remote_job = Job(
            job_id="remote-job",
            backend="remote",
            backend_config={},
            backend_state={},
            status="retrieved",
            output_hashes=collect_content_hashes(result),
            output_is_tuple=isinstance(result, tuple),
        )

        def complete_job():
            shared_cache.put_job("testhash1234abcd", remote_job)
            submitted_job.status = "succeeded"
            return JobStatus(job_id=submitted_job.job_id, status="succeeded")

        submitted_job.check = MagicMock(side_effect=complete_job)
        backend.submit.return_value = submitted_job

        assert run(algo, cache=cache, remote=backend) == result
        submitted_job.fetch.assert_not_called()
        submitted_job.cleanup.assert_not_called()

    def test_run_stores_in_cache(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()

        assert run(algo, "arg1", cache=cache, remote=None) == -75.5
        job = cache.get_job("testhash1234abcd")
        assert job is not None
        assert job.status == "retrieved"

    def test_cache_hit_skips_execution(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()

        run(algo, "arg1", cache=cache, remote=None)
        algo.run.reset_mock()

        assert run(algo, "arg1", cache=cache, remote=None) == -75.5
        algo.run.assert_not_called()

    @pytest.mark.parametrize(
        ("result", "output_is_tuple"),
        [
            pytest.param(None, False, id="none"),
            pytest.param((42,), True, id="singleton-tuple"),
            pytest.param((), True, id="empty-tuple"),
            pytest.param((None,), True, id="singleton-none-tuple"),
        ],
    )
    def test_cache_hit_preserves_result_shape(self, tmp_path, result, output_is_tuple):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm(result=result)

        assert run(algo, cache=cache, remote=None) == result
        algo.run.reset_mock()

        assert run(algo, cache=cache, remote=None) == result
        algo.run.assert_not_called()
        job = cache.get_job("testhash1234abcd")
        assert job is not None
        assert job.output_is_tuple is output_is_tuple

    def test_force_rerun(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()

        run(algo, "arg1", cache=cache, remote=None)
        algo.run.reset_mock()

        run(algo, "arg1", cache=cache, remote=None, force_rerun=True)
        algo.run.assert_called_once()

    def test_force_rerun_reaches_remote_worker(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="Succeeded",
        )
        job.fetch = MagicMock(return_value=-75.5)
        job.cleanup = MagicMock()
        backend.submit.return_value = job

        assert run(algo, "arg1", cache=cache, remote=backend, force_rerun=True) == -75.5

        assert backend.submit.call_args.args[0]["force_rerun"] is True
        job.cleanup.assert_not_called()

    def test_remote_submission_callback_runs_after_handle_is_persisted(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()
        backend = MagicMock()
        job = Job(
            job_id="remote-job",
            backend="test",
            backend_config={},
            backend_state={},
            status="Succeeded",
        )
        job.fetch = MagicMock(return_value=-75.5)
        job.cleanup = MagicMock()
        backend.submit.return_value = job
        persisted_jobs = []

        def on_job_submitted(_job):
            persisted_jobs.append(cache.get_job("testhash1234abcd"))

        result = run(
            algo,
            "arg1",
            cache=cache,
            remote=backend,
            _on_job_submitted=on_job_submitted,
        )

        assert result == -75.5
        assert persisted_jobs[0] is not None
        assert persisted_jobs[0].job_id == "remote-job"
        job.cleanup.assert_not_called()

    def test_string_cache_path(self, tmp_path):
        algo = self._mock_algorithm(result=42)
        assert run(algo, cache=str(tmp_path / "str_cache"), remote=None) == 42
        assert (tmp_path / "str_cache").exists()


class TestHashingUtilities:
    def test_primitive_hash_deterministic(self):
        h1 = _item_content_hash(42)
        assert h1 == _item_content_hash(42)
        assert h1 != _item_content_hash(43)
        assert len(h1) == 16

    def test_dataclass_hash(self, sample_orbitals):
        h = _item_content_hash(sample_orbitals)
        assert isinstance(h, str)
        assert len(h) > 0
        assert h == _item_content_hash(sample_orbitals)

    def test_collect_content_hashes_tuple(self, sample_orbitals):
        entries = collect_content_hashes((-75.5, sample_orbitals))
        assert len(entries) == 2
        assert entries[0]["type"] == "float"
        assert "value" in entries[0]
        assert entries[1]["type"] == "orbitals"
        assert "value" not in entries[1]

    def test_collect_content_hashes_single(self):
        entries = collect_content_hashes(-75.5)
        assert len(entries) == 1
        assert entries[0]["value"] == -75.5
