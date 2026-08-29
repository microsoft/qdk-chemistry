"""Tests for new MCP tools and behavioral changes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import ast
import errno
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from qdk_chemistry import data
from qdk_chemistry.remote.backends import base as remote_backend_registry
from qdk_chemistry.remote.backends.base import RemoteBackend, get_mcp_safe_config_options, register_backend
from qdk_chemistry.remote.job import Job
from qdk_chemistry.ui import tools as srv
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


def test_get_summary_loads_registered_data_class(h2_proj):
    result = srv.get_summary(project_name=h2_proj, filename="h2.structure.json")

    assert result["status"] == "ok"
    assert result["result"]["data_type"] == "Structure"


@pytest.mark.usefixtures("_dirs")
def test_create_structure_rejects_output_paths_outside_project():
    coordinates = "[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]"
    outside_paths = [
        "../victim.structure.json",
        str(config.projects_dir.parent / "victim.structure.json"),
    ]

    for filename in outside_paths:
        result = srv.create_structure(
            project_name="safe",
            coordinates_json=coordinates,
            symbols=["H", "H"],
            filename_to_save=filename,
        )

        assert result["status"] == "error"
        assert "outside project directory" in result["message"]

    assert not (config.projects_dir / "victim.structure.json").exists()
    assert not (config.projects_dir.parent / "victim.structure.json").exists()


@pytest.mark.usefixtures("_dirs")
def test_create_structure_checks_exact_nested_output_path():
    project_dir = config.projects_dir / "safe"
    nested_dir = project_dir / "nested"
    nested_dir.mkdir(parents=True)
    existing = data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.4]]), symbols=["H", "H"])
    existing.to_json_file(str(nested_dir / "victim.structure.json"))

    result = srv.create_structure(
        project_name="safe",
        coordinates_json="[[0.0, 0.0, 0.0]]",
        symbols=["He"],
        filename_to_save="nested/victim.structure.json",
    )

    assert result["status"] == "exists"
    assert "already exists with valid data" in result["message"]
    unchanged = data.Structure.from_json_file(str(nested_dir / "victim.structure.json"))
    assert unchanged.get_atomic_symbols() == ["H", "H"]


@pytest.mark.usefixtures("_dirs")
def test_list_project_files_nonexistent():
    assert srv.list_project_files(project_name="nope")["status"] == "error"


@pytest.mark.usefixtures("_dirs")
@pytest.mark.parametrize("tool", [srv.create_project, srv.list_project_files])
@pytest.mark.parametrize("project_name", ["../outside", "nested/project", r"..\outside"])
def test_project_management_rejects_non_component_names(tool, project_name):
    result = tool(project_name=project_name)

    assert result["status"] == "error"
    assert "single path component" in result["message"]


@pytest.mark.usefixtures("_dirs")
@pytest.mark.parametrize("tool", [srv.create_project, srv.list_project_files])
def test_project_management_rejects_absolute_name(tool):
    result = tool(project_name=str(config.projects_dir.parent / "outside"))

    assert result["status"] == "error"
    assert "single path component" in result["message"]


@pytest.mark.usefixtures("_dirs")
@pytest.mark.parametrize("tool", [srv.create_project, srv.list_project_files])
def test_project_management_rejects_symlink_escape(tool):
    outside = config.projects_dir.parent / "outside"
    outside.mkdir()
    (config.projects_dir / "escape").symlink_to(outside, target_is_directory=True)

    result = tool(project_name="escape")

    assert result["status"] == "error"
    assert "outside projects directory" in result["message"]


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
    assert mock_run.call_args.kwargs["cache"] is not None
    assert "local_cache" not in mock_run.call_args.kwargs
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
    assert mock_run.call_args.kwargs["cache"] is cache
    assert "local_cache" not in mock_run.call_args.kwargs
    assert mock_run.call_args.kwargs["remote"] == "disc"


def test_run_algorithm_allows_safe_remote_config():
    """Safe MCP backend options are forwarded to the constructor."""
    algorithm = MagicMock()
    backend = MagicMock()
    with (
        patch("qdk_chemistry.remote.backends.get_backend", return_value=backend) as get_backend,
        patch("qdk_chemistry.ui.tools._remote_run", return_value="ok") as remote_run,
    ):
        result = srv._run_algorithm(
            algorithm,
            remote="local",
            remote_config={"poll_interval": 2.0, "timeout": 30.0},
            remote_timeout=None,
        )

    assert result == "ok"
    get_backend.assert_called_once_with("local", poll_interval=2.0, timeout=30.0)
    backend.connect.assert_called_once_with()
    backend.disconnect.assert_called_once_with()
    assert remote_run.call_args.kwargs["remote"] is backend


def test_timed_remote_run_disconnects_configured_backend():
    """MCP-owned configured backends are cleaned up by the worker thread."""
    algorithm = MagicMock()
    algorithm.hash.return_value = "run-hash"
    backend = MagicMock()

    with (
        patch("qdk_chemistry.remote.backends.get_backend", return_value=backend),
        patch("qdk_chemistry.ui.tools._remote_run", return_value="ok"),
    ):
        result = srv._run_algorithm(
            algorithm,
            remote="local",
            remote_config={"poll_interval": 2.0},
            remote_timeout=120,
        )

    assert result == "ok"
    backend.connect.assert_called_once_with()
    backend.disconnect.assert_called_once_with()


@pytest.mark.parametrize(
    ("remote", "remote_config"),
    [
        ("local", {"python_path": "/workspace/evil"}),
        ("discovery", {"image": "untrusted-image"}),
        ("plugin-remote", {"endpoint": "untrusted.example.com"}),
    ],
)
def test_run_algorithm_rejects_unsafe_remote_config(remote, remote_config):
    """MCP backend options that can redirect execution are rejected."""
    algorithm = MagicMock()
    with (
        patch("qdk_chemistry.remote.backends.get_backend") as get_backend,
        patch("qdk_chemistry.ui.tools._remote_run") as remote_run,
        pytest.raises(ValueError, match="cannot control"),
    ):
        srv._run_algorithm(
            algorithm,
            remote=remote,
            remote_config=remote_config,
            remote_timeout=None,
        )

    get_backend.assert_not_called()
    remote_run.assert_not_called()


def test_describe_remote_backend_only_lists_safe_mcp_config():
    """Backend discovery advertises only MCP-safe remote options."""
    result = srv.describe_backend(backend_type="remote", name="local")

    assert result["status"] == "ok"
    parameter_names = {parameter["name"] for parameter in result["result"]["parameters"]}
    assert parameter_names == {"poll_interval", "timeout"}


def test_plugin_backend_declares_safe_mcp_config(monkeypatch):
    """A registered plugin class controls which of its constructor options MCP may set."""

    class PluginBackend(RemoteBackend):
        mcp_safe_config_options = frozenset({"poll_interval"})

        def __init__(self, *, endpoint=None, poll_interval=5.0):
            super().__init__(endpoint=endpoint, poll_interval=poll_interval)

    backend = MagicMock()
    monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {})
    register_backend("plugin-remote")(PluginBackend)

    with (
        patch("qdk_chemistry.remote.backends.get_backend", return_value=backend) as get_backend,
        patch("qdk_chemistry.ui.tools._remote_run", return_value="ok"),
    ):
        result = srv._run_algorithm(
            MagicMock(),
            remote="plugin-remote",
            remote_config={"poll_interval": 2.0},
            remote_timeout=None,
        )

    assert result == "ok"
    get_backend.assert_called_once_with("plugin-remote", poll_interval=2.0)

    description = srv.describe_backend(backend_type="remote", name="plugin-remote")
    assert description["status"] == "ok"
    assert {parameter["name"] for parameter in description["result"]["parameters"]} == {"poll_interval"}


def test_undeclared_backend_cannot_spoof_safe_options_by_registry_name(monkeypatch):
    """An undeclared class remains deny-all even under a built-in registry name."""

    class SpoofedLocalBackend(RemoteBackend):
        def __init__(self, *, poll_interval=5.0):
            super().__init__(poll_interval=poll_interval)

    monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {"local": SpoofedLocalBackend})

    assert get_mcp_safe_config_options("local") == frozenset()
    with pytest.raises(ValueError, match="cannot control"):
        srv._validate_mcp_remote_config("local", {"poll_interval": 2.0})


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


def test_run_scf_accepts_wavefunction_output(h2_proj):
    algorithm = MagicMock()
    wavefunction = MagicMock()
    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=algorithm),
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=(-1.0, wavefunction)) as run_algorithm,
        patch("qdk_chemistry.ui.tools.save_data_object") as save_data_object,
    ):
        result = srv.run_scf(
            project_name=h2_proj,
            structure_filename="h2.structure.json",
            out_wavefunction_filename="result.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

    assert result == {"status": "ok", "result": [-1.0, "result.wavefunction.json"]}
    run_algorithm.assert_called_once()
    save_data_object.assert_called_once_with(wavefunction, "result.wavefunction.json")


def test_controlled_evolution_mapper_uses_common_execution_path(h2_proj):
    algorithm = MagicMock()
    unitary = MagicMock()
    circuit = MagicMock()

    with (
        patch("qdk_chemistry.ui.tools.algorithms.create", return_value=algorithm),
        patch("qdk_chemistry.ui.tools.ensure_filename_format", return_value="controlled.circuit.json"),
        patch("qdk_chemistry.ui.tools.load_data_object", return_value=unitary),
        patch("qdk_chemistry.ui.tools.save_data_object") as save_data_object,
        patch("qdk_chemistry.ui.tools._run_algorithm", return_value=circuit) as run_algorithm,
    ):
        result = srv.run_controlled_evolution_circuit_mapper(
            project_name=h2_proj,
            time_evolution_unitary_filename="evolution.unitary.json",
            out_circuit_filename="controlled.circuit.json",
            cache="folder",
            remote="local",
            remote_config={"keep_workdir": True},
            remote_timeout=30,
            overwrite=True,
        )

    assert result["status"] == "ok", result
    assert run_algorithm.call_args.args == (algorithm, unitary)
    assert run_algorithm.call_args.kwargs == {
        "cache": "folder",
        "remote": "local",
        "remote_config": {"keep_workdir": True},
        "remote_timeout": 30,
        "overwrite": True,
    }
    save_data_object.assert_called_once_with(circuit, "controlled.circuit.json")


def test_algorithm_runs_are_centralized():
    source = Path(srv.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    direct_runs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "run":
            continue
        containing_node = parents.get(node)
        while containing_node is not None and not isinstance(containing_node, ast.FunctionDef | ast.AsyncFunctionDef):
            containing_node = parents.get(containing_node)
        function_name = (
            containing_node.name if isinstance(containing_node, ast.FunctionDef | ast.AsyncFunctionDef) else None
        )
        direct_runs.append((function_name, node.lineno))

    assert len(direct_runs) == 1
    assert direct_runs[0][0] == "_run_algorithm"


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
    owner = srv._current_job_owner("project-a")
    j = Job(job_id="abc", backend="local", backend_config={}, backend_state={}, run_hash="h1", owner=owner)
    j.save(config.jobs_dir / "h1.job.json")

    found = srv._discover_cached_jobs(owner)
    assert any(x.job_id == "abc" for x in found)

    loaded, err = srv._load_remote_job("abc", owner)
    assert err is None
    assert loaded.job_id == "abc"

    _, err2 = srv._load_remote_job("missing", owner)
    assert err2 is not None


@pytest.mark.usefixtures("_dirs")
def test_discover_prefers_jobs_dir_over_cache_copy():
    owner = srv._current_job_owner("project-a")
    Job(job_id="abc", backend="local", backend_config={}, backend_state={}, status="running", owner=owner).save(
        config.cache_dir / "h1.job.json"
    )
    Job(job_id="abc", backend="local", backend_config={}, backend_state={}, status="Succeeded", owner=owner).save(
        config.jobs_dir / "h1.job.json"
    )

    found = [job for job in srv._discover_cached_jobs(owner) if job.job_id == "abc"]
    loaded, err = srv._load_remote_job("abc", owner)

    assert len(found) == 1
    assert found[0].status == "Succeeded"
    assert err is None
    assert loaded.file_path.parent == config.jobs_dir


@pytest.mark.usefixtures("_dirs")
def test_job_record_paths_are_scoped_by_owner():
    first = Job(
        job_id="first",
        backend="local",
        backend_config={},
        backend_state={},
        owner=srv._current_job_owner("project-a"),
    )
    second = Job(
        job_id="second",
        backend="local",
        backend_config={},
        backend_state={},
        owner=srv._current_job_owner("project-b"),
    )

    assert srv._job_record_path(first, "same-run") != srv._job_record_path(second, "same-run")


@pytest.mark.usefixtures("_dirs")
def test_list_remote_jobs_only_returns_current_project_jobs():
    current_owner = srv._current_job_owner("project-a")
    other_owner = srv._current_job_owner("project-b")
    Job(job_id="current", backend="local", backend_config={}, backend_state={}, owner=current_owner).save(
        config.jobs_dir / "current.job.json"
    )
    Job(job_id="other", backend="local", backend_config={}, backend_state={}, owner=other_owner).save(
        config.jobs_dir / "other.job.json"
    )
    Job(job_id="legacy", backend="local", backend_config={}, backend_state={}).save(config.jobs_dir / "legacy.job.json")

    result = srv.list_remote_jobs(project_name="project-a")

    assert result["status"] == "ok"
    assert [job["job_id"] for job in result["result"]["jobs"]] == ["current"]


@pytest.mark.usefixtures("_dirs")
@pytest.mark.parametrize(
    ("tool", "operation"),
    [
        (srv.check_remote_job, "check"),
        (srv.retrieve_remote_results, "fetch"),
        (srv.cancel_remote_job, "cancel"),
    ],
)
def test_remote_job_operations_hide_jobs_owned_by_other_projects(tool, operation):
    Job(
        job_id="other",
        backend="local",
        backend_config={},
        backend_state={},
        owner=srv._current_job_owner("project-b"),
    ).save(config.jobs_dir / "other.job.json")

    with patch.object(Job, operation) as backend_operation:
        result = tool(project_name="project-a", job_id="other")

    assert result["status"] == "error"
    assert "No remote job found" in result["message"]
    backend_operation.assert_not_called()


@pytest.mark.usefixtures("_dirs")
@pytest.mark.parametrize(
    ("tool", "operation"),
    [
        (srv.check_remote_job, "check"),
        (srv.retrieve_remote_results, "fetch"),
        (srv.cancel_remote_job, "cancel"),
    ],
)
def test_remote_job_operations_hide_ownerless_jobs(tool, operation):
    Job(job_id="legacy", backend="local", backend_config={}, backend_state={}).save(config.jobs_dir / "legacy.job.json")

    with patch.object(Job, operation) as backend_operation:
        result = tool(project_name="project-a", job_id="legacy")

    assert result["status"] == "error"
    assert "No remote job found" in result["message"]
    backend_operation.assert_not_called()


@pytest.mark.usefixtures("_dirs")
def test_remote_job_lookup_filters_owner_before_duplicate_job_ids():
    current_owner = srv._current_job_owner("project-a")
    Job(
        job_id="shared-id",
        backend="local",
        backend_config={},
        backend_state={"operation": "foreign"},
        run_hash="foreign-run",
        owner=srv._current_job_owner("project-b"),
    ).save(config.jobs_dir / "foreign.job.json")
    Job(
        job_id="shared-id",
        backend="local",
        backend_config={},
        backend_state={"operation": "current"},
        run_hash="current-run",
        owner=current_owner,
    ).save(config.cache_dir / "current.job.json")

    job, error = srv._load_remote_job("shared-id", current_owner)

    assert error is None
    assert job.backend_state == {"operation": "current"}


@pytest.mark.usefixtures("_dirs")
def test_remote_job_operations_hide_jobs_owned_by_other_workspaces():
    owner = {"workspace_root": "/workspace-a", "project_name": "project-a"}
    Job(job_id="other", backend="local", backend_config={}, backend_state={}, owner=owner).save(
        config.jobs_dir / "other.job.json"
    )

    with (
        patch.object(srv, "current_workspace_root", return_value=Path("/workspace-b")),
        patch.object(Job, "cancel") as cancel,
    ):
        result = srv.cancel_remote_job(project_name="project-a", job_id="other")

    assert result["status"] == "error"
    assert "No remote job found" in result["message"]
    cancel.assert_not_called()


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


def test_config_falls_back_from_read_only_default_scratch(monkeypatch, tmp_path):
    def read_only_default_scratch(self, *args, **kwargs):
        if self == Path("/scratch"):
            raise OSError(errno.EROFS, "Read-only file system")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.delenv("QDK_SCRATCH_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    original_mkdir = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", read_only_default_scratch)

    assert QDKMCPConfig().scratch_dir == tmp_path / ".qdk_chem" / "scratch"
