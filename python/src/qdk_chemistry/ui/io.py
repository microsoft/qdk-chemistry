"""File I/O utilities for QDK Chemistry data objects."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
from typing import Any


def check_output_exists(filename: str, data_class: Any | None = None) -> str | None:
    """Check if an output file already exists with valid content.

    Args:
        filename (str): The output filename to check
        data_class: Optional data class to validate file content. If provided,
                   attempts to load the file to verify it contains valid data.

    Returns:
        Optional[str]: A message if the file exists with valid content, None otherwise

    """
    filename = filename.rsplit("/", maxsplit=1)[-1]  # Strip path if provided

    if not os.path.exists(filename):
        return None

    if data_class is not None:
        try:
            existing_obj = load_data_object(filename, data_class)
            if existing_obj is not None:
                return (
                    f"EXISTS: Output file '{filename}' already exists with valid data. "
                    "Do you want to run again and overwrite it? If this was an expensive "
                    "calculation and you trust the contents of the older file, you should use the older file."
                )
        except (RuntimeError, ValueError, FileNotFoundError, OSError):
            return None  # File exists but is invalid, proceed with calculation
    else:
        return (
            f"EXISTS: Output file '{filename}' already exists. "
            "Do you want to run again and overwrite it? If this was an expensive "
            "calculation and you trust the contents of the older file, you should use the older file."
        )

    return None


def load_data_object(filename: str, data_class):
    """Load a data object from either json or hdf5 file based on extension.

    Args:
        filename (str): Filename with extension (.json or .hdf5/.h5)
        data_class: The qdk_chemistry.data class to instantiate

    Returns:
        The loaded data object

    Raises:
        ValueError: If file extension is not supported

    """
    filename = filename.rsplit("/", maxsplit=1)[-1]  # Strip path if provided

    if filename.endswith(".json"):
        return data_class.from_json_file(filename)
    if filename.endswith((".hdf5", ".h5")):
        return data_class.from_hdf5_file(filename)
    raise ValueError(f"Unsupported file extension for {filename}. Must be .json, .hdf5, or .h5")


def save_data_object(data_obj, filename: str):
    """Save a data object to either json or hdf5 file based on extension.

    Args:
        data_obj: The qdk_chemistry data object to save
        filename (str): Filename with extension (.json or .hdf5/.h5)

    Returns:
        str: The filename where data was saved

    Raises:
        ValueError: If file extension is not supported

    """
    filename = filename.rsplit("/", maxsplit=1)[-1]  # Strip path if provided

    if filename.endswith(".json"):
        data_obj.to_json_file(filename)
    elif filename.endswith((".hdf5", ".h5")):
        data_obj.to_hdf5_file(filename)
    else:
        raise ValueError(f"Unsupported file extension for {filename}. Must be .json, .hdf5, or .h5")

    return filename
