"""Tests for new MCP tools and behavioral changes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        orig_p, orig_j = config.projects_dir, config.jobs_dir
        config.projects_dir = Path(t) / "projects"
        config.jobs_dir = Path(t) / "jobs"
        config.projects_dir.mkdir()
        config.jobs_dir.mkdir()
        yield
        config.projects_dir, config.jobs_dir = orig_p, orig_j


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
    m.run.return_value = "ok"
    assert srv._run_algorithm(m, "a") == "ok"
    m.run.assert_called_once_with("a")


def test_run_algorithm_remote_auto_cache():
    m = MagicMock()
    m.run.return_value = "ok"
    with (
        patch("qdk_chemistry.ui.tools._REMOTE_AVAILABLE", True),
        patch("qdk_chemistry.ui.tools.FolderCache") as mock_fc,
    ):
        mock_fc.return_value = MagicMock()
        srv._run_algorithm(m, cache=None, remote="disc", remote_timeout=None)
    mock_fc.assert_called_once_with(path=config.jobs_dir)
    assert m.run.call_args[1]["cache"] is mock_fc.return_value
    assert m.run.call_args[1]["remote"] == "disc"


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
