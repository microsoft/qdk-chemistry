"""Input validation and project management for QDK Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import inspect
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from qdk_chemistry import data

from .config import config


class FilenameFormatError(Exception):
    """Raised when a filename has an invalid format for the expected data type."""


def _build_data_type_markers() -> dict[str, str]:
    """Build the data type to filename marker mapping from qdk_chemistry.data classes.

    This function discovers all data classes that have a `_data_type_name` attribute
    and builds a mapping from class name to the filename marker (e.g., ".structure.").

    Returns:
        dict[str, str]: Mapping from class name to filename marker

    """
    markers = {}
    for name in data.__all__:
        obj = getattr(data, name, None)
        if obj is None or not inspect.isclass(obj):
            continue

        # Get _data_type_name if it exists and is not None
        type_name = getattr(obj, "_data_type_name", None)
        if type_name is None and name == "MajoranaMapping":
            type_name = "majorana_mapping"
        if type_name is not None:
            markers[name] = f".{type_name}."

    return markers


# Data type to filename marker mapping (auto-discovered from qdk_chemistry.data)
_DATA_TYPE_MARKERS = _build_data_type_markers()


def ensure_filename_format(filename: str, data_type: str) -> str:
    """Ensure filename contains the correct type marker for the given data type.

    Args:
        filename: The filename to check/correct
        data_type: The data type name (e.g., "Wavefunction", "QubitHamiltonian")

    Returns:
        The corrected filename with proper type marker

    Raises:
        FilenameFormatError: If the data type is unrecognized or the file extension is invalid

    """
    marker = _DATA_TYPE_MARKERS.get(data_type)
    if marker is None:
        raise FilenameFormatError(
            f"Unrecognized data type '{data_type}' for filename '{filename}'. "
            f"Valid types are: {', '.join(_DATA_TYPE_MARKERS.keys())}"
        )

    # Check if marker is already present
    if marker in filename:
        return filename

    # Find the extension and insert the marker before it
    for ext in [".json", ".hdf5", ".h5"]:
        if filename.endswith(ext):
            base = filename[: -len(ext)]
            # Remove trailing dot if present (e.g., "file." -> "file")
            base = base.rstrip(".")
            return f"{base}{marker[:-1]}{ext}"  # marker already has dots, remove trailing

    # No recognized extension - raise error
    raise FilenameFormatError(f"Unrecognized file extension for '{filename}'. Must end with .json, .hdf5, or .h5")


def validate_project(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate project before executing the function.

    Validates that a project exists, is properly structured, and is the
    current working directory before executing the decorated function.

    It expects the decorated function to have ``project_name`` as its
    first parameter after ``self`` (if applicable).

    Args:
        func: The function to decorate. Must have ``project_name: str`` as a parameter.

    Returns:
        Callable[..., Any]: The decorated function with project validation logic.

    Example::

        @validate_project
        @app.tool()
        def my_function(project_name: str, other_param: int) -> str:
            # This function will only execute if project_name is valid
            return "success"

    """

    @wraps(func)
    def wrapper(project_name: str, *args: Any, **kwargs: Any) -> Any:
        """Wrap function with project validation logic.

        Args:
            project_name: Project name inside default project directory
            *args: Additional positional arguments passed to the decorated function
            **kwargs: Additional keyword arguments passed to the decorated function

        Returns:
            String with error information if validation fails, otherwise
            returns the result of the decorated function

        """
        original_cwd = Path.cwd()
        is_valid, message = is_project_valid(project_name, config.projects_dir)
        if not is_valid:
            return f"Project validation failed: {message} for project_name: {project_name}"

        try:
            # Proceed with the original function while relative file access resolves inside the project.
            return func(project_name, *args, **kwargs)
        finally:
            os.chdir(original_cwd)

    return wrapper


def is_project_valid(  # noqa: PLR0911
    project_name: str, projects_dir: str | Path
) -> tuple[bool, str]:
    """Checks validity of base project dir/name combination.

    Tries to make directory if it doesn't exist yet.

    Args:
        project_name: Name of specific project
        projects_dir: Path to all projects directories (can be string or Path)

    Returns:
        Tuple[bool, str] that states whether the project is valid, and if not, an explanation

    """
    # Convert projects_dir to Path if it's a string
    if isinstance(projects_dir, str):
        projects_dir = Path(projects_dir)
    elif not isinstance(projects_dir, Path):
        return False, f"Projects dir should be a Path or string but it's {type(projects_dir)}"

    # Check if project exists
    project_path = projects_dir / project_name

    try:
        project_exists = project_path.exists()
    except PermissionError:
        return False, f"No read permissions to access {projects_dir}"
    except OSError as e:
        return False, f"Cannot access project directory {project_path}: {e}"

    if not project_exists:
        # try to create project
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return False, f"No write permissions to {project_path}"
        except OSError as e:
            return False, f"Cannot create project directory {project_path}: {e}"

    # change to current working directory
    try:
        os.chdir(project_path)
    except PermissionError as e:
        return False, f"Cannot move to {project_path} : {e}"
    except OSError as e:
        return False, f"Cannot access project directory {project_path} : {e}"

    return True, f"Project with path {project_path} exists and is the current working directory"
