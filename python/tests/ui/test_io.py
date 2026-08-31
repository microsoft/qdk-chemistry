"""Tests for UI file I/O helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.ui.io import load_data_object, save_data_object


def test_load_data_object_preserves_directory_components(tmp_path):
    """Pass the exact nested path to the data class loader."""
    path = tmp_path / "nested" / "result.json"

    class Data:
        """Minimal data class loader."""

        @classmethod
        def from_json_file(cls, filename):
            """Return the filename received from the loader."""
            return filename

    assert load_data_object(path, Data) == str(path)


def test_save_data_object_preserves_directory_components(tmp_path):
    """Pass the exact nested path to the data object writer."""
    path = tmp_path / "nested" / "result.hdf5"
    saved_paths = []

    class Data:
        """Minimal data object writer."""

        def to_hdf5_file(self, filename):
            """Record the filename received from the writer."""
            saved_paths.append(filename)

    assert save_data_object(Data(), path) == str(path)
    assert saved_paths == [str(path)]
