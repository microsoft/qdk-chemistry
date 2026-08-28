"""Tests for the supported QDK Chemistry UI validation contracts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import pytest

from qdk_chemistry.ui.validation import FilenameFormatError, ensure_filename_format, is_project_valid


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


class TestFilenameFormat:
    """Test invalid filename format handling."""

    def test_rejects_unknown_data_type(self):
        with pytest.raises(FilenameFormatError, match="Unrecognized data type 'Wavefunction'"):
            ensure_filename_format("wavefunction.json", "Wavefunction")
