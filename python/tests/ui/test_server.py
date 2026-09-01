"""Tests for the supported QDK Chemistry UI validation contracts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import threading
from pathlib import Path

import pytest

from qdk_chemistry.ui.validation import (
    FilenameFormatError,
    current_project_dir,
    ensure_filename_format,
    is_project_valid,
    resolve_project_file,
    validate_project,
)


class TestProjectValidation:
    """Test project directory validation."""

    def test_is_project_valid_creates_directory(self, tmp_path: Path):
        is_valid, message = is_project_valid("test_project", tmp_path / "projects")

        assert is_valid is True
        assert "exists" in message
        assert (tmp_path / "projects" / "test_project").is_dir()

    def test_is_project_valid_rejects_invalid_projects_directory(self):
        is_valid, message = is_project_valid("test_project", 42)

        assert is_valid is False
        assert "Path or string" in message

    @pytest.mark.parametrize("project_name", ["../outside", "nested/project", r"..\outside"])
    def test_is_project_valid_rejects_non_component_names(self, tmp_path: Path, project_name: str):
        is_valid, message = is_project_valid(project_name, tmp_path / "projects")

        assert is_valid is False
        assert "single path component" in message

    def test_is_project_valid_rejects_absolute_name(self, tmp_path: Path):
        is_valid, message = is_project_valid(str(tmp_path / "outside"), tmp_path / "projects")

        assert is_valid is False
        assert "single path component" in message

    def test_is_project_valid_rejects_symlink_escape(self, tmp_path: Path):
        projects_dir = tmp_path / "projects"
        outside = tmp_path / "outside"
        projects_dir.mkdir()
        outside.mkdir()
        (projects_dir / "escape").symlink_to(outside, target_is_directory=True)

        is_valid, message = is_project_valid("escape", projects_dir)

        assert is_valid is False
        assert "symbolic link" in message

    def test_is_project_valid_rejects_symlink_alias_inside_projects(self, tmp_path: Path):
        projects_dir = tmp_path / "projects"
        target = projects_dir / "target"
        target.mkdir(parents=True)
        (projects_dir / "alias").symlink_to(target, target_is_directory=True)

        is_valid, message = is_project_valid("alias", projects_dir)

        assert is_valid is False
        assert "symbolic link" in message

    def test_validate_project_does_not_change_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decorated tool exposes its project without changing process CWD."""
        original_cwd = tmp_path / "original"
        projects_dir = tmp_path / "projects"
        original_cwd.mkdir()
        monkeypatch.chdir(original_cwd)
        monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)

        @validate_project
        def tool(project_name: str) -> tuple[Path | None, Path]:
            """Return the validated project and process working directories."""
            del project_name
            return current_project_dir(), Path.cwd()

        assert tool("test_project") == (projects_dir / "test_project", original_cwd)
        assert Path.cwd() == original_cwd

    def test_validate_project_restores_working_directory_after_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decorated tool restores the caller's directory after an error."""
        original_cwd = tmp_path / "original"
        projects_dir = tmp_path / "projects"
        original_cwd.mkdir()
        monkeypatch.chdir(original_cwd)
        monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)

        @validate_project
        def tool(project_name: str) -> None:
            """Raise an error while running in the project directory."""
            del project_name
            raise RuntimeError("tool failed")

        with pytest.raises(RuntimeError, match="tool failed"):
            tool("test_project")

        assert Path.cwd() == original_cwd

    def test_resolve_project_file_rejects_escape_and_foreign_absolute_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        projects_dir = tmp_path / "projects"
        monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)

        @validate_project
        def tool(project_name: str, filename: str) -> Path:
            del project_name
            return resolve_project_file(filename, allow_nested=True)

        with pytest.raises(ValueError, match="outside project|invalid path component"):
            tool("safe", "../outside.json")
        with pytest.raises(ValueError, match="outside project"):
            tool("safe", str(tmp_path / "outside.json"))

    def test_validate_project_contexts_are_concurrent_and_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        projects_dir = tmp_path / "projects"
        monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)
        entered = threading.Barrier(2)
        results: dict[str, Path] = {}

        @validate_project
        def tool(project_name: str) -> None:
            entered.wait(timeout=2)
            results[project_name] = resolve_project_file("result.json")

        threads = [threading.Thread(target=tool, args=(name,)) for name in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert results == {
            "first": projects_dir / "first" / "result.json",
            "second": projects_dir / "second" / "result.json",
        }


class TestFilenameFormat:
    """Test invalid filename format handling."""

    def test_accepts_known_data_type(self):
        assert ensure_filename_format("wavefunction.json", "Wavefunction") == "wavefunction.wavefunction.json"

    def test_rejects_unknown_data_type(self):
        with pytest.raises(FilenameFormatError, match="Unrecognized data type 'UnknownDataType'"):
            ensure_filename_format("output.json", "UnknownDataType")
