"""Tests for UI file I/O helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import pytest

from qdk_chemistry.ui.io import load_data_object, save_data_object
from qdk_chemistry.ui.validation import validate_project


def test_load_data_object_uses_validated_project_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pass an absolute project-confined path to the data class loader."""
    projects_dir = tmp_path / "projects"
    monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)
    path = projects_dir / "project" / "result.json"

    class Data:
        """Minimal data class loader."""

        @classmethod
        def from_json_file(cls, filename):
            """Return the filename received from the loader."""
            return filename

    @validate_project
    def load(project_name: str):
        del project_name
        return load_data_object("result.json", Data)

    assert load("project") == str(path)


def test_save_data_object_uses_validated_project_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pass an absolute project-confined path to the data object writer."""
    projects_dir = tmp_path / "projects"
    monkeypatch.setattr("qdk_chemistry.ui.validation.config.projects_dir", projects_dir)
    path = projects_dir / "project" / "result.hdf5"
    saved_paths = []

    class Data:
        """Minimal data object writer."""

        def to_hdf5_file(self, filename):
            """Record the filename received from the writer."""
            saved_paths.append(filename)

    @validate_project
    def save(project_name: str):
        del project_name
        return save_data_object(Data(), "result.hdf5")

    assert save("project") == str(path)
    assert saved_paths == [str(path)]
