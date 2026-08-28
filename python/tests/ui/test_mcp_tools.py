"""Tests for new MCP tools and behavioral changes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from qdk_chemistry import data
from qdk_chemistry.remote.job import Job
from qdk_chemistry.ui import tools as srv
from qdk_chemistry.ui.cli import _VERSION_PLACEHOLDER, _copy_with_version
from qdk_chemistry.ui.config import QDKMCPConfig, config


@pytest.fixture
def _dirs():
    """Temp projects + jobs dirs, restored after test."""
    with tempfile.TemporaryDirectory() as t:
        orig_p, orig_j, orig_c = config.projects_dir, config.jobs_dir, config.cache_dir
        config.projects_dir = Path(t) / "projects"
        config.jobs_dir = Path(t) / "jobs"
        config.cache_dir = Path(t) / "cache"
        config.projects_dir.mkdir()
        config.jobs_dir.mkdir()
        config.cache_dir.mkdir()
        yield
        config.projects_dir, config.jobs_dir, config.cache_dir = orig_p, orig_j, orig_c


@pytest.fixture
def h2_proj(_dirs):
    """Project with an H2 structure file."""
    p = config.projects_dir / "h2"
    p.mkdir()
    s = data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.4]]), symbols=["H", "H"])
    s.to_json_file(str(p / "h2.structure.json"))
    return "h2"


# ── Unit conversion ──────────────────────────────────────────────────────


def test_convert_coordinates_angstrom_to_bohr():
    r = srv.convert_coordinates(coordinates_json="[[0,0,0],[0.74,0,0]]", to_unit="bohr")
    assert r["status"] == "ok"
    assert abs(r["result"]["coordinates"][1][0] - 1.398) < 0.01


def test_convert_energy_hartree_to_ev():
    r = srv.convert_energy(value=1.0, from_unit="hartree", to_unit="ev")
    assert r["status"] == "ok"
    assert abs(r["result"]["output"]["value"] - 27.211) < 0.01


def test_convert_coordinates_bad_input():
    assert srv.convert_coordinates(coordinates_json="bad", to_unit="bohr")["status"] == "error"


def test_convert_energy_bad_unit():
    assert srv.convert_energy(value=1.0, from_unit="hartree", to_unit="furlongs")["status"] == "error"


# ── Project management ───────────────────────────────────────────────────


@pytest.mark.usefixtures("_dirs")
def test_create_and_list_projects():
    srv.create_project(project_name="alpha")
    srv.create_project(project_name="beta")
    r = srv.list_projects()
    assert set(r["result"]["projects"]) == {"alpha", "beta"}


def test_list_project_files(h2_proj):
    r = srv.list_project_files(project_name=h2_proj)
    names = [f["filename"] for f in r["result"]["files"]]
    assert "h2.structure.json" in names


@pytest.mark.usefixtures("_dirs")
def test_list_project_files_nonexistent():
    assert srv.list_project_files(project_name="nope")["status"] == "error"


# ── get_summary ──────────────────────────────────────────────────────────


def test_get_summary_structure(h2_proj):
    r = srv.get_summary(project_name=h2_proj, filename="h2.structure.json")
    assert r["status"] == "ok"
    assert r["result"]["data_type"] == "Structure"


def test_get_summary_missing_file(h2_proj):
    assert srv.get_summary(project_name=h2_proj, filename="nope.json")["status"] == "error"


# ── list_tools ───────────────────────────────────────────────────────────


def test_list_tools_all():
    r = srv.list_tools()
    cats = r["result"]["categories"]
    assert "project" in cats
    assert "classical_calculation" in cats


def test_list_tools_filter():
    r = srv.list_tools(category="utility")
    assert list(r["result"]["categories"].keys()) == ["utility"]


# ── Algorithm discovery ─────────────────────────────────────────────────


def test_list_algorithms_filters_by_type():
    result = srv.list_algorithms(algorithm_type="nuclear_derivative_calculator")

    assert result["status"] == "ok"
    algorithm_types = result["result"]["algorithm_types"]
    assert list(algorithm_types) == ["nuclear_derivative_calculator"]
    assert algorithm_types["nuclear_derivative_calculator"]["default"] == "qdk"
    assert "qdk" in algorithm_types["nuclear_derivative_calculator"]["implementations"]


def test_describe_algorithm_returns_settings_schema():
    result = srv.describe_algorithm(algorithm_type="nuclear_derivative_calculator", algorithm_name="qdk")

    assert result["status"] == "ok"
    description = result["result"]
    assert description["name"] == "qdk"
    assert {"analytical_gradient", "qdk"} <= set(description["aliases"])
    assert description["is_default"] is True
    assert description["default_settings"]["compute_hessian"] is False
    settings = {setting["name"]: setting for setting in description["settings"]}
    assert settings["compute_hessian"]["type"] == "bool"
    assert settings["compute_hessian"]["default"] is False


def test_list_algorithms_rejects_unknown_type():
    result = srv.list_algorithms(algorithm_type="unknown")

    assert result["status"] == "error"
    assert "Unknown algorithm type" in result["message"]


# ── Overwrite ────────────────────────────────────────────────────────────


def test_overwrite_bypasses_exists(h2_proj):
    r1 = srv.create_structure(
        project_name=h2_proj,
        coordinates_json="[[0,0,0],[0,0,1.4]]",
        symbols=["H", "H"],
        filename_to_save="h2.structure.json",
    )
    assert r1["status"] == "exists"
    r2 = srv.create_structure(
        project_name=h2_proj,
        coordinates_json="[[0,0,0],[0,0,1.4]]",
        symbols=["H", "H"],
        filename_to_save="h2.structure.json",
        overwrite=True,
    )
    assert r2["status"] == "ok"


# ── _run_algorithm ───────────────────────────────────────────────────────


def test_run_algorithm_local():
    m = MagicMock()
    with patch("qdk_chemistry.ui.tools._remote_run", return_value="ok") as mock_run:
        assert srv._run_algorithm(m, "a") == "ok"
    assert mock_run.call_args.args[:2] == (m, "a")
    assert mock_run.call_args.kwargs["local_cache"] is not None
    assert mock_run.call_args.kwargs["remote"] is None


def test_run_algorithm_remote_auto_cache():
    m = MagicMock()
    cache = MagicMock()
    with (
        patch("qdk_chemistry.ui.tools._REMOTE_AVAILABLE", True),
        patch("qdk_chemistry.ui.tools.FolderCache") as mock_fc,
        patch("qdk_chemistry.ui.tools._remote_run", return_value="ok") as mock_run,
    ):
        mock_fc.return_value = cache
        assert srv._run_algorithm(m, cache=None, remote="disc", remote_timeout=None) == "ok"
    mock_fc.assert_called_once_with(path=config.cache_dir)
    assert mock_run.call_args.args[:2] == (m,)
    assert mock_run.call_args.kwargs["local_cache"] is cache
    assert mock_run.call_args.kwargs["remote"] == "disc"


def test_timed_remote_run_rejects_missing_hash_before_submission():
    algorithm = MagicMock()
    algorithm.hash.side_effect = AttributeError("hash is not bound")

    with (
        patch("qdk_chemistry.ui.tools._REMOTE_AVAILABLE", True),
        patch("qdk_chemistry.ui.tools._remote_run") as mock_run,
        pytest.raises(RuntimeError, match=r"algorithm.hash\(\) failed"),
    ):
        srv._run_algorithm(algorithm, "input", remote="test-remote", remote_timeout=120)

    mock_run.assert_not_called()


@pytest.mark.usefixtures("_dirs")
def test_timed_remote_run_timeout_starts_after_handle_is_persisted():
    algorithm = MagicMock()
    algorithm.hash.return_value = "run-hash"
    job = Job(job_id="remote-job", backend="test-remote", backend_config={}, backend_state={})
    cache = MagicMock()
    submission_started = threading.Event()
    persist_handle = threading.Event()
    finish_remote_run = threading.Event()

    def remote_run(*_args, **kwargs):
        submission_started.set()
        persist_handle.wait()
        callback = kwargs.get("_on_job_submitted")
        if callback is not None:
            callback(job)
        finish_remote_run.wait()
        return "finished"

    @srv._structured
    def run():
        return srv._run_algorithm(algorithm, "input", remote="test-remote", remote_timeout=0)

    try:
        with (
            patch("qdk_chemistry.ui.tools._REMOTE_AVAILABLE", True),
            patch("qdk_chemistry.ui.tools.FolderCache", return_value=cache),
            patch("qdk_chemistry.ui.tools._remote_run", side_effect=remote_run),
            ThreadPoolExecutor(max_workers=1) as caller_pool,
        ):
            result_future = caller_pool.submit(run)
            assert submission_started.wait(timeout=1)
            assert not result_future.done()

            persist_handle.set()
            result = result_future.result(timeout=1)
    finally:
        finish_remote_run.set()

    assert result["status"] == "submitted"
    assert result["job"]["job_id"] == "remote-job"
    assert Job.load(config.jobs_dir / "run-hash.job.json").job_id == "remote-job"


def test_run_nuclear_derivative_calculator_tool(h2_proj):
    gradients = data.NuclearGradients(
        data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.4]]), symbols=["H", "H"]),
        np.zeros(6),
    )
    algorithm = MagicMock()
    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=algorithm),
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=(-1.0, gradients, None, None)) as run_algorithm,
    ):
        r = srv.run_nuclear_derivative_calculator(
            project_name=h2_proj,
            structure_filename="h2.structure.json",
            out_gradients_filename="h2-gradients.nuclear_gradients.json",
            charge=0,
            spin_multiplicity=1,
            seed_or_basis="sto-3g",
            n_inactive_orbitals=2,
            cache="folder",
            remote="local",
        )

    assert r["status"] == "ok"
    assert r["result"]["energy"] == -1.0
    assert r["result"]["gradients_filename"] == "h2-gradients.nuclear_gradients.json"
    assert run_algorithm.call_args.args[1:] == (ANY, 0, 1, "sto-3g", 2)
    assert run_algorithm.call_args.kwargs["cache"] == "folder"
    assert run_algorithm.call_args.kwargs["remote"] == "local"


def test_run_geometry_optimization_tool(h2_proj):
    optimized = data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.35]]), symbols=["H", "H"])
    algorithm = MagicMock()
    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=algorithm),
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=(-1.1, optimized, None, None)) as run_algorithm,
    ):
        r = srv.run_geometry_optimization(
            project_name=h2_proj,
            structure_filename="h2.structure.json",
            out_structure_filename="h2-opt.structure.json",
            charge=0,
            spin_multiplicity=1,
            seed_or_basis="sto-3g",
            n_inactive_orbitals=2,
            cache="folder",
            remote="local",
        )

    assert r["status"] == "ok"
    assert r["result"]["energy"] == -1.1
    assert r["result"]["structure_filename"] == "h2-opt.structure.json"
    assert run_algorithm.call_args.args[1:] == (ANY, 0, 1, "sto-3g", 2)
    assert run_algorithm.call_args.kwargs["cache"] == "folder"
    assert run_algorithm.call_args.kwargs["remote"] == "local"


def test_run_geometry_optimization_routes_derivative_settings(h2_proj):
    optimized = data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.35]]), symbols=["H", "H"])
    optimizer = MagicMock()
    settings = {"derivative_calculator": {"algorithm_name": "custom_derivatives", "accuracy": 0.5}}

    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=optimizer) as create,
        patch("qdk_chemistry.ui.tools._apply_settings") as apply_settings,
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=(-1.1, optimized, None, None)) as run_algorithm,
    ):
        result = srv.run_geometry_optimization(
            project_name=h2_proj,
            structure_filename="h2.structure.json",
            out_structure_filename="h2-custom-opt.structure.json",
            charge=0,
            spin_multiplicity=1,
            seed_or_basis="sto-3g",
            algorithm_name="geometric",
            settings=settings,
        )

    assert result["status"] == "ok"
    create.assert_called_once_with("geometry_optimizer", "geometric")
    apply_settings.assert_called_once_with(optimizer, settings)
    assert run_algorithm.call_args.args[0] is optimizer


def test_run_geometry_optimization_explains_derivative_name_category(h2_proj):
    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", side_effect=KeyError("not a geometry optimizer")),
        patch("qdk_chemistry.ui.tools.algorithms.available", return_value=["custom_derivatives"]),
    ):
        result = srv.run_geometry_optimization(
            project_name=h2_proj,
            structure_filename="h2.structure.json",
            out_structure_filename="h2-custom-opt.structure.json",
            charge=0,
            spin_multiplicity=1,
            seed_or_basis="sto-3g",
            algorithm_name="custom_derivatives",
        )

    assert result["status"] == "error"
    assert "nuclear derivative calculator, not a geometry optimizer" in result["message"]
    assert "settings['derivative_calculator']" in result["message"]


def test_run_population_analysis_tool(h2_proj):
    algorithm = MagicMock()
    algorithm.name.return_value = "qdk"
    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=algorithm),
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=[0.1, -0.1]) as run_algorithm,
    ):
        r = srv.run_population_analysis(
            project_name=h2_proj,
            input_filename="h2.structure.json",
            charge=0,
            spin_multiplicity=1,
            n_inactive_orbitals=2,
            cache="folder",
            remote="local",
        )

    assert r["status"] == "ok"
    assert r["result"]["algorithm"] == "qdk"
    assert r["result"]["populations"] == [0.1, -0.1]
    assert r["result"]["population_sum"] == 0.0
    assert run_algorithm.call_args.args[1:] == (ANY, 0, 1, 2)
    assert run_algorithm.call_args.kwargs["cache"] == "folder"
    assert run_algorithm.call_args.kwargs["remote"] == "local"


# ── _JobSubmittedError ───────────────────────────────────────────────────


def test_structured_catches_job_submitted():
    job = Job(job_id="x", backend="b", backend_config={}, backend_state={}, status="submitted")

    @srv._structured
    def boom():
        raise srv._JobSubmittedError(job)

    r = boom()
    assert r["status"] == "submitted"
    assert r["job"]["job_id"] == "x"


# ── Job discovery ────────────────────────────────────────────────────────


@pytest.mark.usefixtures("_dirs")
def test_discover_and_load_job():
    j = Job(job_id="abc", backend="local", backend_config={}, backend_state={}, run_hash="h1")
    j.save(config.jobs_dir / "h1.job.json")

    found = srv._discover_cached_jobs()
    assert any(x.job_id == "abc" for x in found)

    loaded, err = srv._load_remote_job("abc")
    assert err is None
    assert loaded.job_id == "abc"

    _, err2 = srv._load_remote_job("missing")
    assert err2 is not None


@pytest.mark.usefixtures("_dirs")
def test_discover_prefers_jobs_dir_over_cache_copy():
    Job(job_id="abc", backend="local", backend_config={}, backend_state={}, status="running").save(
        config.cache_dir / "h1.job.json"
    )
    Job(job_id="abc", backend="local", backend_config={}, backend_state={}, status="Succeeded").save(
        config.jobs_dir / "h1.job.json"
    )

    found = [job for job in srv._discover_cached_jobs() if job.job_id == "abc"]
    loaded, err = srv._load_remote_job("abc")

    assert len(found) == 1
    assert found[0].status == "Succeeded"
    assert err is None
    assert loaded.file_path.parent == config.jobs_dir


# ── Config ───────────────────────────────────────────────────────────────


def test_config_jobs_dir_env():
    with tempfile.TemporaryDirectory() as t:
        os.environ["QDK_SCRATCH_DIR"] = t
        os.environ["QDK_JOBS_DIR"] = t
        try:
            assert QDKMCPConfig().jobs_dir == Path(t)
        finally:
            del os.environ["QDK_JOBS_DIR"]
            del os.environ["QDK_SCRATCH_DIR"]


# ── Version injection ────────────────────────────────────────────────────


def test_version_injection():
    with tempfile.TemporaryDirectory() as t:
        src, dst = Path(t) / "in.md", Path(t) / "out.md"
        src.write_text(f"v={_VERSION_PLACEHOLDER}")
        _copy_with_version(src, dst)
        assert _VERSION_PLACEHOLDER not in dst.read_text()
        assert dst.read_text().startswith("v=v")
