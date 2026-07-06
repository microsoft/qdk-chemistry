"""Tests for new MCP tools and behavioral changes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qdk_chemistry import data
from qdk_chemistry.ui import tools as srv
from qdk_chemistry.ui.cli import _VERSION_PLACEHOLDER, _copy_with_version
from qdk_chemistry.ui.config import config


@pytest.fixture
def _dirs():
    """Temp projects dir, restored after test."""
    with tempfile.TemporaryDirectory() as t:
        orig_p = config.projects_dir
        config.projects_dir = Path(t) / "projects"
        config.projects_dir.mkdir()
        yield
        config.projects_dir = orig_p


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


def test_run_algorithm_auto_cache():
    m = MagicMock()
    m.run.return_value = "ok"
    with patch("qdk_chemistry.ui.tools.FolderCache") as mock_fc:
        mock_fc.return_value = MagicMock()
        srv._run_algorithm(m, cache=None)
    mock_fc.assert_called_once_with(path=config.cache_dir)
    assert m.run.call_args[1]["cache"] is mock_fc.return_value


# ── Version injection ────────────────────────────────────────────────────


def test_version_injection():
    with tempfile.TemporaryDirectory() as t:
        src, dst = Path(t) / "in.md", Path(t) / "out.md"
        src.write_text(f"v={_VERSION_PLACEHOLDER}")
        _copy_with_version(src, dst)
        assert _VERSION_PLACEHOLDER not in dst.read_text()
        assert dst.read_text().startswith("v=v")
