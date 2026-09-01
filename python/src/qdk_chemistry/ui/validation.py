"""Input validation and project management for QDK Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar, cast

from qdk_chemistry import data
from qdk_chemistry.data._type_name import class_data_type_name

from .config import config

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

_CURRENT_PROJECT_NAME: ContextVar[str | None] = ContextVar("qdk_current_project_name", default=None)
_CURRENT_PROJECT_DIR: ContextVar[Path | None] = ContextVar("qdk_current_project_dir", default=None)


def current_project_name() -> str | None:
    """Return the project currently validated for this execution context."""
    return _CURRENT_PROJECT_NAME.get()


def current_project_dir() -> Path | None:
    """Return the absolute project directory validated for this execution context."""
    return _CURRENT_PROJECT_DIR.get()


def resolve_project_file(
    filename: str | Path,
    *,
    allow_nested: bool = False,
    allow_absolute: bool = False,
) -> Path:
    """Resolve a client-provided filename inside the current project sandbox.

    Absolute paths, traversal components, Windows drive or UNC paths, and
    symlinks that resolve outside the project are rejected.

    Args:
        filename: Project-relative filename to resolve.
        allow_nested: Whether ordinary nested path components are accepted.
        allow_absolute: Whether an internal absolute path may be revalidated.

    Returns:
        An absolute path contained by the current project directory.

    Raises:
        RuntimeError: If called outside a validated project context.
        ValueError: If the filename is invalid or escapes the project.

    """
    project_dir = current_project_dir()
    if project_dir is None:
        raise RuntimeError("Project file access requires a validated project context")

    value = str(filename)
    if not value or not value.strip() or "\x00" in value:
        raise ValueError("Project filename must be a non-empty string")

    native_path = Path(value)
    if native_path.is_absolute():
        if not allow_absolute:
            raise ValueError(f"Project filename must be relative: {value!r}")
        try:
            candidate = native_path.resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Cannot resolve project filename {value!r}: {error}") from error
        if not candidate.is_relative_to(project_dir):
            raise ValueError(f"Project filename resolves outside project directory: {value!r}")
        return candidate

    path_flavors = (PurePosixPath(value), PureWindowsPath(value))
    if any(path.is_absolute() or path.anchor or path.drive for path in path_flavors):
        raise ValueError(f"Project filename must be relative: {value!r}")
    if any(part in {"", ".", ".."} for path in path_flavors for part in path.parts):
        raise ValueError(f"Project filename contains an invalid path component: {value!r}")
    if not allow_nested and any(len(path.parts) != 1 for path in path_flavors):
        raise ValueError(f"Project filename must be a single path component: {value!r}")

    try:
        candidate = (project_dir / value).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve project filename {value!r}: {error}") from error
    if not candidate.is_relative_to(project_dir):
        raise ValueError(f"Project filename resolves outside project directory: {value!r}")
    return candidate


class FilenameFormatError(Exception):
    """Raised when a filename has an invalid format for the expected data type."""


def resolve_project_path(project_name: str, projects_dir: str | Path) -> tuple[Path | None, str]:  # noqa: PLR0911
    """Resolve a single-component project name beneath the projects directory.

    Args:
        project_name: Name of the project directory.
        projects_dir: Root directory containing projects.

    Returns:
        The resolved project path and an empty error message, or ``None`` and
        an explanation when the path is invalid.

    """
    if isinstance(projects_dir, str):
        projects_dir = Path(projects_dir)
    elif not isinstance(projects_dir, Path):
        return None, f"Projects dir should be a Path or string but it's {type(projects_dir)}"

    if not isinstance(project_name, str) or not project_name.strip():
        return None, "Project name must be a non-empty string"

    path_flavors = (PurePosixPath(project_name), PureWindowsPath(project_name))
    if any(path.is_absolute() or path.parts != (project_name,) or project_name in {".", ".."} for path in path_flavors):
        return None, "Project name must be a single path component"

    try:
        projects_root = projects_dir.resolve()
        unresolved_project_path = projects_root / project_name
        if unresolved_project_path.is_symlink():
            return None, f"Project path {unresolved_project_path} must not be a symbolic link"
        project_path = unresolved_project_path.resolve()
    except (OSError, RuntimeError) as e:
        return None, f"Cannot resolve project directory: {e}"

    if not project_path.is_relative_to(projects_root):
        return None, f"Project path {project_path} is outside projects directory {projects_root}"

    return project_path, ""


def _build_data_type_markers() -> dict[str, str]:
    """Build the data type to filename marker mapping from qdk_chemistry.data classes.

    This function discovers public names for registered data classes and builds a
    mapping from class name to the filename marker (e.g., ".structure.").

    Returns:
        dict[str, str]: Mapping from class name to filename marker

    """
    registered_classes = set(data.available_dataclasses().values())
    markers = {}
    for name in data.__all__:
        obj = getattr(data, name, None)
        if not isinstance(obj, type) or obj not in registered_classes:
            continue

        type_name = class_data_type_name(obj)
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


def validate_project(func: F) -> F:
    """Decorator to validate project before executing the function.

    Validates that a project exists and exposes its absolute directory through
    the current execution context. The process working directory is unchanged.

    It expects the decorated function to have ``project_name`` as its
    first parameter after ``self`` (if applicable).

    Args:
        func: The function to decorate. Must have ``project_name: str`` as a parameter.

    Returns:
        F: The decorated function with project validation logic,
        or str: a JSON string with error information.

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
        is_valid, message = is_project_valid(project_name, config.projects_dir)
        if not is_valid:
            return f"Project validation failed: {message} for project_name: {project_name}"

        project_dir, error = resolve_project_path(project_name, config.projects_dir)
        if project_dir is None:
            return f"Project validation failed: {error} for project_name: {project_name}"

        name_token = _CURRENT_PROJECT_NAME.set(project_name)
        dir_token = _CURRENT_PROJECT_DIR.set(project_dir)
        try:
            return func(project_name, *args, **kwargs)
        finally:
            _CURRENT_PROJECT_DIR.reset(dir_token)
            _CURRENT_PROJECT_NAME.reset(name_token)

    return cast("F", wrapper)


def is_project_valid(  # noqa: PLR0911
    project_name: str, projects_dir: str | Path
) -> tuple[bool, str]:
    """Checks validity of base project dir/name combination.

    Tries to make the directory if it doesn't exist yet. This function does
    not change the process working directory.

    Args:
        project_name: Name of specific project
        projects_dir: Path to all projects directories (can be string or Path)

    Returns:
        Tuple[bool, str] that states whether the project is valid, and if not, an explanation

    """
    project_path, error = resolve_project_path(project_name, projects_dir)
    if project_path is None:
        return False, error

    try:
        project_exists = project_path.exists()
    except PermissionError:
        return False, f"No read permissions to access {project_path.parent}"
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

    if not project_path.is_dir():
        return False, f"Project path {project_path} is not a directory"

    return True, f"Project with path {project_path} exists"
