"""Tests for the remote execution system (serialization, Job, backends, proxy, run)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from qdk_chemistry.data import Orbitals, Structure
from qdk_chemistry.data._hashing import _item_content_hash, collect_content_hashes
from qdk_chemistry.remote.backends import available_backends, get_backend
from qdk_chemistry.remote.backends.base import RemoteBackend, register_backend
from qdk_chemistry.remote.backends.local import LocalBackend
from qdk_chemistry.remote.cache.folder import FolderCache
from qdk_chemistry.remote.job import Job
from qdk_chemistry.remote.proxy import run
from qdk_chemistry.remote.serialization import (
    FileSerializer,
    deserialize_inputs,
    deserialize_outputs,
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


class TestFileSerializerPrimitives:
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

    def test_dict_round_trip(self, tmp_path):
        value = {"a": 1, "b": 2.0, "c": "three"}
        entry = FileSerializer.serialize_value(tmp_path, "dct", value)
        assert entry["type"] == "dict"
        assert FileSerializer.deserialize_value(tmp_path, entry) == value

    def test_nested_structures(self, tmp_path):
        value = [{"x": [1, 2]}, (True, None)]
        entry = FileSerializer.serialize_value(tmp_path, "nested", value)
        result = FileSerializer.deserialize_value(tmp_path, entry)
        assert result[0] == {"x": [1, 2]}
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
            input_hashes={"arg_0": "hash_of_arg0"},
        )
        manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
        assert manifest["args"][0]["content_hash"] == "hash_of_arg0"


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
            input_hashes={"arg_0": "hash0"},
        )
        loaded = Job.load(job.save(tmp_path / "job_j1.json"))
        assert loaded.job_id == "j1"
        assert loaded.backend == "local"
        assert loaded.backend_config == {"timeout": 60}
        assert loaded.backend_state == {"pid": 1234}
        assert loaded.run_hash == "aaaa"
        assert loaded.input_hashes == {"arg_0": "hash0"}

    @pytest.mark.parametrize("status", ["Succeeded", "Failed", "canceled", "retrieved"])
    def test_is_terminal(self, status):
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status=status)
        assert job.is_terminal

    @pytest.mark.parametrize("status", ["submitted", "running", "pending"])
    def test_is_not_terminal(self, status):
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, status=status)
        assert not job.is_terminal

    def test_discover_finds_jobs(self, tmp_path):
        for i in range(3):
            Job(job_id=f"j{i}", backend="local", backend_config={}, backend_state={}).save(tmp_path / f"job_j{i}.json")
        jobs = Job.discover(tmp_path)
        assert {j.job_id for j in jobs} == {"j0", "j1", "j2"}

    def test_discover_skips_corrupt(self, tmp_path):
        (tmp_path / "job_bad.json").write_text("not json")
        Job(job_id="good", backend="local", backend_config={}, backend_state={}).save(tmp_path / "job_good.json")
        assert [j.job_id for j in Job.discover(tmp_path)] == ["good"]

    def test_save_without_path_raises(self):
        with pytest.raises(ValueError, match="No file path"):
            Job(job_id="x", backend="local", backend_config={}, backend_state={}).save()

    def test_output_hashes_round_trip(self, tmp_path):
        hashes = [{"hash": "h1", "type": "float", "value": -75.5}, {"hash": "h2", "type": "wavefunction"}]
        job = Job(job_id="x", backend="local", backend_config={}, backend_state={}, output_hashes=hashes)
        assert Job.load(job.save(tmp_path / "job_x.json")).output_hashes == hashes


class TestBackendRegistry:
    def test_builtin_backends_registered(self):
        registered = available_backends()
        assert "local" in registered
        assert "ssh" in registered

    def test_get_local_backend(self):
        assert isinstance(get_backend("local"), LocalBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="No backend"):
            get_backend("does_not_exist")

    def test_register_custom_backend(self):
        class StubBackend(RemoteBackend):
            name = "_test_stub"

            def connect(self):
                pass

            def disconnect(self):
                pass

            def upload(self, local_path, remote_path):
                pass

            def execute(self, script_content, remote_workdir):
                pass

            def download(self, remote_path, local_path):
                pass

        register_backend("_test_stub")(StubBackend)

        assert "_test_stub" in available_backends()
        assert isinstance(get_backend("_test_stub"), StubBackend)


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

    def test_upload_download_round_trip(self, backend, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("hello remote")

        remote = f"{backend.remote_workdir}/sub/input.txt"
        backend.upload(src, remote)

        dest = tmp_path / "downloaded.txt"
        backend.download(remote, dest)
        assert dest.read_text() == "hello remote"

    def test_execute_success(self, backend):
        backend.execute("print('ok')", backend.remote_workdir)

    def test_execute_failure_raises(self, backend):
        with pytest.raises(RuntimeError):
            backend.execute("raise SystemExit(1)", backend.remote_workdir)

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

        for _ in range(100):
            status = backend.check(state)
            if status.status != "running":
                break
            time.sleep(0.2)
        assert status.status in ("Succeeded", "Failed")


class TestLocalBackendSpecific:
    def test_connect_creates_workdir(self):
        b = LocalBackend()
        b.connect()
        assert Path(b.remote_workdir).exists()
        b.disconnect()

    def test_disconnect_removes_workdir(self):
        b = LocalBackend()
        b.connect()
        workdir = b.remote_workdir
        b.disconnect()
        assert not Path(workdir).exists()

    def test_keep_workdir_option(self):
        b = LocalBackend(keep_workdir=True)
        b.connect()
        workdir = b.remote_workdir
        b.disconnect()
        assert Path(workdir).exists()
        shutil.rmtree(workdir, ignore_errors=True)


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

    def test_run_no_cache(self):
        algo = self._mock_algorithm()
        assert run(algo, "arg1", cache=None, remote=None) == -75.5
        algo.run.assert_called_once_with("arg1")

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

    def test_force_rerun(self, tmp_path):
        cache = FolderCache(path=tmp_path / "cache")
        algo = self._mock_algorithm()

        run(algo, "arg1", cache=cache, remote=None)
        algo.run.reset_mock()

        run(algo, "arg1", cache=cache, remote=None, force_rerun=True)
        algo.run.assert_called_once()

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
