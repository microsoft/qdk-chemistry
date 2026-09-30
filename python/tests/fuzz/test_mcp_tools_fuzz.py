"""Property-based tests for filesystem-facing MCP tools."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st

from qdk_chemistry.ui import tools
from qdk_chemistry.ui.config import config

_PATH_TEXT = st.text(
    alphabet=st.characters(categories=("L", "N")) | st.sampled_from("./\\:_- \x00"),
    max_size=80,
)


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@given(project_name=_PATH_TEXT)
def test_create_project_never_writes_outside_workspace(project_name: str) -> None:
    """Arbitrary project names cannot create directories outside the workspace."""
    original_projects_dir = config.projects_dir
    attempted_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        workspace = Path(temporary_directory)
        projects_dir = workspace / "projects"
        projects_dir.mkdir()
        config.projects_dir = projects_dir

        def record_mkdir(path: Path, *_args, **_kwargs) -> None:
            attempted_paths.append(path)

        try:
            with patch.object(Path, "mkdir", record_mkdir):
                result = tools.create_project(project_name=project_name)
        finally:
            config.projects_dir = original_projects_dir

        assert result["status"] in {"ok", "error"}
        assert all(_is_within(path, projects_dir) for path in attempted_paths)


@given(filename=_PATH_TEXT)
def test_create_structure_never_writes_outside_workspace(filename: str) -> None:
    """Arbitrary output paths remain contained or return a structured error."""
    original_projects_dir = config.projects_dir
    attempted_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        workspace = Path(temporary_directory)
        projects_dir = workspace / "projects"
        (projects_dir / "safe").mkdir(parents=True)
        config.projects_dir = projects_dir
        structure = MagicMock()
        structure.to_json_file.side_effect = lambda path: attempted_paths.append(Path(path))
        structure.to_hdf5_file.side_effect = lambda path: attempted_paths.append(Path(path))
        try:
            with patch.object(tools.data, "Structure", return_value=structure):
                result = tools.create_structure(
                    project_name="safe",
                    coordinates_json="[[0.0, 0.0, 0.0]]",
                    symbols=["H"],
                    filename_to_save=filename,
                )
        finally:
            config.projects_dir = original_projects_dir

        assert result["status"] in {"ok", "exists", "error"}
        assert all(_is_within(path, projects_dir) for path in attempted_paths)


@given(coordinates_json=st.text(max_size=300), to_unit=st.text(max_size=40))
def test_convert_coordinates_always_returns_a_structured_result(coordinates_json: str, to_unit: str) -> None:
    """Arbitrary coordinate text is converted or rejected without escaping the envelope."""
    result = tools.convert_coordinates(coordinates_json=coordinates_json, to_unit=to_unit)

    assert result["status"] in {"ok", "error"}
