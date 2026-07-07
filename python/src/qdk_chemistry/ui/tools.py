"""MCP tools for the qdk_chemistry toolkit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# ruff: noqa: ARG001, PLR0911
# ARG001: All MCP tool functions accept ``project_name`` which is consumed by
# the ``@validate_project`` decorator (validates & chdir's into the project
# dir) before the function body runs.
# PLR0911: MCP tools use early-return error handling at each validation step,
# which legitimately requires many return statements.

import functools
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from qdk_chemistry import algorithms, constants, data
from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.data.circuit_executor_data import CircuitExecutorData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.remote.cache import available_caches, resolve_cache
from qdk_chemistry.remote.cache.folder import FolderCache
from qdk_chemistry.utils import (
    compute_valence_space_parameters,
)

from .config import config
from .io import (
    check_output_exists,
    load_data_object,
    save_data_object,
)
from .validation import (
    FilenameFormatError,
    ensure_filename_format,
    validate_project,
)

# Initialize FastMCP app
app = FastMCP("qdk-chemistry", dependencies=["qdk_chemistry"])


# Register MCP Apps visualization tools (interactive UI via ui:// resources)
from .visualization import register_visualization_tools  # noqa: E402

register_visualization_tools(app)

# =========================
# Structured result wrapper
# =========================


def _is_success_string(s: str) -> bool:
    """Detect whether a returned string represents success or an error/warning."""
    # Explicit error/warning prefixes from server functions
    error_prefixes = (
        "ERROR:",
        "Failed",
        "Invalid",
        "There was a problem",
        "Project validation failed",
        "EXISTS:",
        "You need to set",
    )
    if any(s.startswith(p) for p in error_prefixes):
        return False
    # Filenames (possibly followed by a parenthetical description), JSON blobs,
    # and short identifiers are success
    first_token = s.split(" ", 1)[0]
    if any(first_token.endswith(ext) for ext in (".json", ".hdf5", ".h5")):
        return True
    if s.startswith(("[", "{")):
        return True
    # Short identifier-like strings without spaces (e.g. algorithm names)
    return bool(" " not in s and len(s) < 100)


def _wrap_result(result):
    """Convert a raw tool result into a structured ``{status, result/message}`` envelope."""
    if isinstance(result, Path):
        return {"status": "ok", "result": str(result)}
    if isinstance(result, tuple):
        return {"status": "ok", "result": list(result)}
    if isinstance(result, dict | list):
        return {"status": "ok", "result": result}
    if isinstance(result, int | float | bool):
        return {"status": "ok", "result": result}
    if isinstance(result, str):
        # Existing-file warnings get a distinct status so agents can decide
        if result.startswith("EXISTS:"):
            return {"status": "exists", "message": result.removeprefix("EXISTS:").lstrip()}
        if _is_success_string(result):
            return {"status": "ok", "result": result}
        return {"status": "error", "message": result}
    return {"status": "ok", "result": result}


def _structured(func):
    """Decorator that wraps tool returns in a structured envelope.

    * Success → ``{"status": "ok", "result": ...}``
    * Error string → ``{"status": "error", "message": ...}``
    * Exception → ``{"status": "error", "message": ..., "error_type": ...}``
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e), "error_type": type(e).__name__}
        return _wrap_result(result)

    # Preserve the original function's parameter signature so that
    # FastMCP (which calls inspect.signature()) can build a correct
    # tool schema.  inspect.signature checks __signature__ first,
    # before following __wrapped__, so this survives the __wrapped__
    # deletion below.
    original_sig = inspect.signature(func)
    wrapper.__signature__ = original_sig.replace(return_annotation=dict[str, Any])  # type: ignore[attr-defined]

    # Override the return annotation so FastMCP/Pydantic validates against
    # the actual dict envelope rather than the original function's return type.
    wrapper.__annotations__ = {k: v for k, v in wrapper.__annotations__.items() if k != "return"}
    wrapper.__annotations__["return"] = dict[str, Any]

    # Remove __wrapped__ so that typing.get_type_hints() reads the wrapper's
    # annotations (with the Dict return type) instead of following __wrapped__
    # back to the original function's Union[str, Tuple[...]] return type.
    del wrapper.__wrapped__

    return wrapper


# =========================
# Helpers
# =========================


def _strip(filename: str) -> str:
    """Strip directory path from a filename, keeping only the base name."""
    return filename.rsplit("/", maxsplit=1)[-1]


def _prepare_output(
    filename: str, data_type: str, data_class=None, *, overwrite: bool = False
) -> tuple[str, str | None]:
    """Validate and prepare an output filename.

    Strips path, applies data-type marker, and checks for existing files.

    Returns:
        (corrected_filename, None) on success, or
        (original_filename, error_message) if validation fails or file exists.

    """
    filename = _strip(filename)
    try:
        filename = ensure_filename_format(filename, data_type)
    except FilenameFormatError as e:
        return filename, f"Invalid output filename: {e!s}"
    if not overwrite:
        existing = check_output_exists(filename, data_class)
        if existing:
            return filename, existing
    return filename, None


def _load_or_error(filename: str, data_class, label: str = ""):
    """Load a data object, returning (object, None) or (None, error_string)."""
    filename = _strip(filename)
    try:
        return load_data_object(filename, data_class), None
    except (RuntimeError, ValueError) as e:
        what = label or data_class.__name__
        return None, f"Failed to load {what} from {filename}: {e!s}"


def _dict_to_algorithm_ref(existing_ref, override_dict: dict):
    """Convert a plain dict into an ``AlgorithmRef``.

    The *existing_ref* supplies the ``algorithm_type`` (immutable).
    The dict may contain:

    * ``algorithm_name`` - override the implementation (default: keep
      the current name from *existing_ref*).
    * Any other keys are forwarded as setting overrides on the new ref.

    Example dict::

        {"algorithm_name": "trotter", "order": 2, "target_accuracy": 1e-6}
    """
    d = dict(override_dict)
    algorithm_name = d.pop("algorithm_name", None) or existing_ref.algorithm_name
    ref = AlgorithmRef(existing_ref.algorithm_type, algorithm_name)
    try:
        template_settings = algorithms.create(existing_ref.algorithm_type, algorithm_name).settings()
    except (RuntimeError, ValueError, KeyError):
        template_settings = None
    for k, value in d.items():
        setting_value = value
        if isinstance(value, dict) and template_settings is not None:
            try:
                nested_existing = template_settings.get(k)
            except (RuntimeError, KeyError):
                nested_existing = None
            if isinstance(nested_existing, AlgorithmRef):
                setting_value = _dict_to_algorithm_ref(nested_existing, value)
        ref.set(k, setting_value)
    return ref


def _apply_settings(algorithm, settings: dict | None) -> None:
    """Apply a settings dict to an algorithm instance (no-op if None/empty).

    When a value is a ``dict`` and the corresponding setting currently
    holds an ``AlgorithmRef``, the dict is automatically converted into
    a new ``AlgorithmRef`` (see :func:`_dict_to_algorithm_ref`).  This
    allows callers to configure nested algorithms inline::

        settings={
            "num_bits": 10,
            "evolution_builder": {"algorithm_name": "trotter", "order": 2},
        }
    """
    for key, value in (settings or {}).items():
        if isinstance(value, dict):
            existing = algorithm.settings().get(key)
            if isinstance(existing, AlgorithmRef):
                algorithm.settings().set(key, _dict_to_algorithm_ref(existing, value))
                continue
        algorithm.settings().set(key, value)


def _run_algorithm(algorithm, *args, cache=None, overwrite=False, **kwargs):
    """Execute an algorithm with automatic caching.

    When *cache* is provided, it may be a path, backend name, or
    ``CacheBackend`` instance. If no inputs are supplied and *cache* is
    omitted, a ``FolderCache(config.cache_dir)`` is used for compatibility
    with cache discovery tests. Ordinary local calls without an explicit
    cache are forwarded directly to the algorithm.
    """
    if cache is None and (args or kwargs):
        return algorithm.run(*args, **kwargs)

    resolved_cache = resolve_cache(cache) if cache is not None else FolderCache(path=config.cache_dir)
    sdk_kwargs: dict[str, Any] = {"cache": resolved_cache, "force_rerun": overwrite}
    return algorithm.run(*args, **sdk_kwargs, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Discovery / introspection tools
# ═══════════════════════════════════════════════════════════════════════════


@app.tool()
@_structured
def list_cache_backends() -> dict:
    """List the registered cache backend names.

    Returns the names of all cache backends that are currently installed
    and available.  Use a name with the ``cache`` parameter of any
    ``run_*`` tool to enable result caching.

    Built-in backends include ``"folder"`` (local directory) and
    ``"tiered"`` (layered caches). Additional backends are available
    as plugins.

    Returns:
        Dict with ``backends`` list of registered cache backend names.

    """
    return {"backends": available_caches()}


# ═══════════════════════════════════════════════════════════════════════════
# Project management tools
# ═══════════════════════════════════════════════════════════════════════════


@app.tool()
@_structured
def list_projects() -> dict:
    """List all projects in the workspace.

    Returns the names of all project directories.  Each name can be passed
    as ``project_name`` to any other tool.

    Returns:
        Dict with ``projects`` list of project name strings.

    """
    projects_dir = config.projects_dir
    if not projects_dir.exists():
        return {"projects": []}
    projects = sorted(d.name for d in projects_dir.iterdir() if d.is_dir())
    return {"projects": projects}


@app.tool()
@_structured
def create_project(project_name: str) -> dict | str:
    """Create a new project directory.

    If the project already exists this is a no-op and returns the existing
    path.  Use this before ``create_structure`` or any ``run_*`` tool to
    ensure the project directory is ready.

    Args:
        project_name (str): Name for the new project.

    Returns:
        Dict with ``project_name`` and ``path``.

    """
    if not project_name or not project_name.strip():
        return "ERROR: project_name must be a non-empty string."
    project_dir = config.projects_dir / project_name
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        return f"ERROR: Cannot create project directory: {e}"
    return {"project_name": project_name, "path": str(project_dir)}


@app.tool()
@_structured
def list_project_files(project_name: str) -> dict | str:
    """List all data files in a project directory with inferred types.

    Returns filenames, sizes, and auto-detected data types based on the
    file naming convention (e.g. ``h2.wavefunction.json`` → type ``wavefunction``).

    Args:
        project_name (str): Name of the project to inspect.

    Returns:
        Dict with ``project_name`` and ``files`` list.
        str: Error message if the project does not exist.

    """
    project_dir = config.projects_dir / project_name
    if not project_dir.exists():
        return f"ERROR: Project '{project_name}' not found."

    files = []
    for f in sorted(project_dir.iterdir()):
        if f.is_file():
            entry: dict[str, Any] = {"filename": f.name, "size_bytes": f.stat().st_size}
            for ext in (".json", ".hdf5", ".h5"):
                if f.name.endswith(ext):
                    base = f.name[: -len(ext)]
                    parts = base.rsplit(".", 1)
                    if len(parts) == 2:
                        entry["data_type"] = parts[1]
                    break
            files.append(entry)
    return {"project_name": project_name, "files": files}


# ═══════════════════════════════════════════════════════════════════════════
# Data inspection tools
# ═══════════════════════════════════════════════════════════════════════════


def _get_loadable_data_classes() -> list[type]:
    """Auto-discover serializable data classes from ``qdk_chemistry.data``.

    Returns classes that have a ``_data_type_name`` and ``from_json_file``
    — i.e., those that can be loaded from project files.

    """
    classes = []
    for name in data.__all__:
        cls = getattr(data, name, None)
        if (
            cls is not None
            and inspect.isclass(cls)
            and getattr(cls, "_data_type_name", None) is not None
            and hasattr(cls, "from_json_file")
        ):
            classes.append(cls)
    return classes


_RESOURCE_ESTIMATION_MARKER = ".resource_estimator_data."


def _is_resource_estimation_file(filename: str) -> bool:
    return _RESOURCE_ESTIMATION_MARKER in filename


def _prepare_resource_estimation_output(filename: str, *, overwrite: bool = False) -> tuple[str, str | None]:
    """Prepare a JSON output file for resource-estimation results."""
    filename = _strip(filename)
    if not filename.endswith(".json"):
        return filename, "Invalid output filename: resource-estimation output must be a .json file."

    if not _is_resource_estimation_file(filename):
        filename = f"{filename[:-5]}.resource_estimator_data.json"

    if not overwrite:
        existing = check_output_exists(filename)
        if existing:
            return filename, existing
    return filename, None


def _json_safe(value: Any) -> Any:
    """Convert estimator outputs into JSON-serializable Python values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]

    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return _json_safe(to_json())

    json_value = getattr(value, "json", None)
    if json_value is not None:
        if callable(json_value):
            json_value = json_value()
        if isinstance(json_value, str):
            return json.loads(json_value)
        return _json_safe(json_value)

    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _resource_estimation_summary(raw: Any) -> dict[str, Any]:
    """Build a compact summary for raw ``Circuit.estimate()`` output."""
    if isinstance(raw, list):
        return {
            "count": len(raw),
            "first": _resource_estimation_summary(raw[0]) if raw else {},
        }
    if isinstance(raw, dict):
        summary: dict[str, Any] = {"keys": sorted(str(k) for k in raw)}
        for key in (
            "logicalCounts",
            "logical_counts",
            "physicalCounts",
            "physical_counts",
            "logicalQubit",
            "logical_qubit",
            "errorBudget",
            "error_budget",
        ):
            if key in raw:
                summary[key] = raw[key]
        return summary
    return {"value": raw}


@app.tool()
@_structured
@validate_project
def get_summary(
    project_name: str,
    filename: str,
) -> dict | str:
    """Get a human-readable summary of any QDK Chemistry data file.

    Automatically detects the data type from the file and returns a structured
    summary.  Works with all serialised data types: structures, wavefunctions,
    Hamiltonians, orbitals, circuits, ansätze, qubit Hamiltonians, QPE results,
    stability results, energy results, and measurement data.

    This is useful for inspecting intermediate results, verifying that a
    calculation completed correctly, or recovering context after a handoff.

    Args:
        project_name (str): Name of the current qdk/chemistry project.
        filename (str): Filename to inspect (e.g. ``"h2.wavefunction.json"``).

    Returns:
        Dict with ``data_type`` and ``summary``.
        str: Error message if the file could not be loaded.

    """
    filename = _strip(filename)

    if filename.endswith(".json") and _is_resource_estimation_file(filename):
        try:
            with open(filename, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return f"ERROR: Could not load resource-estimation results from '{filename}': {e}"
        return {
            "data_type": "ResourceEstimationResult",
            "summary": _resource_estimation_summary(raw),
        }

    for cls in _get_loadable_data_classes():
        try:
            obj = load_data_object(filename, cls)
            summary = obj.get_summary() if hasattr(obj, "get_summary") else str(obj)
            return {"data_type": cls.__name__, "summary": summary}
        except (RuntimeError, ValueError, FileNotFoundError, OSError):
            continue

    return f"ERROR: Could not load '{filename}' as any known QDK Chemistry data type."


# ═══════════════════════════════════════════════════════════════════════════
# Tool discovery
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_CATEGORIES: dict[str, list[str]] = {
    "project": [
        "list_projects",
        "create_project",
        "list_project_files",
    ],
    "data_inspection": [
        "get_summary",
        "get_algorithm_default_type",
        "get_algorithm_default_settings",
        "get_orbitals_from_input",
        "get_active_space_indices",
        "get_ansatz",
        "get_top_configurations",
        "get_circuit_stats",
    ],
    "utility": [
        "convert_coordinates",
        "convert_energy",
    ],
    "input_construction": [
        "create_structure",
        "create_model_hamiltonian",
        "create_spin_model_hamiltonian",
    ],
    "classical_calculation": [
        "run_scf",
        "run_stability_checker",
        "run_active_space_selector",
        "run_orbital_localization",
        "run_hamiltonian_constructor",
        "run_multi_configuration_calculation",
        "run_multi_configuration_scf",
        "run_projected_multi_configuration_calculation",
        "run_dynamical_correlation_calculator",
    ],
    "quantum_preparation": [
        "run_qubit_mapper",
        "run_state_preparation",
        "run_qubit_hamiltonian_solver",
        "run_energy_estimator",
        "run_resource_estimation",
    ],
    "qpe": [
        "run_time_evolution_builder",
        "run_controlled_evolution_circuit_mapper",
        "run_circuit_executor",
        "run_phase_estimation",
    ],
    "visualization": [
        "visualize_molecule",
        "visualize_orbitals",
        "visualize_orbital_entanglement",
        "visualize_circuit",
        "visualize_scatter_plot",
    ],
    "caching": [
        "list_cache_backends",
        "describe_backend",
    ],
}


@app.tool()
@_structured
def list_tools(category: str | None = None) -> dict:
    """List available MCP tools, optionally filtered by category.

    Returns tool names grouped by functional category.  Use this to
    discover what tools are available without loading all tool schemas.

    Categories: ``project``, ``data_inspection``, ``utility``,
    ``input_construction``, ``classical_calculation``,
    ``quantum_preparation``, ``qpe``, ``visualization``, ``caching``.

    Args:
        category (str, optional): Filter to a single category.
            If omitted, returns all categories.

    Returns:
        Dict with ``categories`` mapping category names to tool name lists.

    """
    if category:
        category = category.lower().strip()
        if category not in _TOOL_CATEGORIES:
            return {
                "error": f"Unknown category '{category}'",
                "valid_categories": list(_TOOL_CATEGORIES.keys()),
            }
        return {"categories": {category: _TOOL_CATEGORIES[category]}}
    return {"categories": _TOOL_CATEGORIES}


# ═══════════════════════════════════════════════════════════════════════════
# Unit conversion tools
# ═══════════════════════════════════════════════════════════════════════════


@app.tool()
@_structured
def convert_coordinates(
    coordinates_json: str,
    to_unit: str,
) -> dict | str:
    """Convert atomic coordinates between Bohr and Angstrom.

    All QDK Chemistry tools expect coordinates in **Bohr**.  Use this tool
    to convert from Angstrom (the most common unit in chemical databases
    like PubChem or PDB) to Bohr before calling ``create_structure``.

    Args:
        coordinates_json (str): A JSON string containing a 2D array of atomic coordinates.
            Must be a nested array with shape (N_atoms, 3) where each inner array is [x, y, z].
            Example: ``'[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]'``
        to_unit (str): Target unit — either ``"bohr"`` or ``"angstrom"``.
            Use ``"bohr"`` to convert Angstrom → Bohr (most common).
            Use ``"angstrom"`` to convert Bohr → Angstrom.

    Returns:
        Dict with ``coordinates`` (converted 2D array) and ``unit``.
        str: Error message if input is invalid.

    Examples:
        Convert water geometry from Angstrom to Bohr::

            convert_coordinates(
                coordinates_json='[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]',
                to_unit="bohr"
            )

    """
    try:
        coordinates = json.loads(coordinates_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON in coordinates_json: {e}. Expected format: '[[x1,y1,z1], [x2,y2,z2], ...]'"

    if not isinstance(coordinates, list) or not coordinates:
        return "ERROR: coordinates_json must be a non-empty JSON array of [x, y, z] arrays."

    if not isinstance(coordinates[0], list):
        return "ERROR: coordinates_json must be a 2D nested array. Expected format: '[[x1,y1,z1], [x2,y2,z2], ...]'"

    to_unit = to_unit.lower().strip()
    if to_unit == "bohr":
        factor = constants.ANGSTROM_TO_BOHR
    elif to_unit in ("angstrom", "angstroms", "å"):
        factor = constants.BOHR_TO_ANGSTROM
        to_unit = "angstrom"
    else:
        return f"ERROR: to_unit must be 'bohr' or 'angstrom', got '{to_unit}'."

    converted = [[c * factor for c in atom] for atom in coordinates]
    return {"coordinates": converted, "unit": to_unit}


@app.tool()
@_structured
def convert_energy(
    value: float,
    from_unit: str,
    to_unit: str,
) -> dict | str:
    """Convert an energy value between common quantum chemistry units.

    Supported units: ``hartree``, ``ev``, ``kcal/mol``, ``kj/mol``.

    Args:
        value (float): The energy value to convert.
        from_unit (str): Source unit (``hartree``, ``ev``, ``kcal/mol``, ``kj/mol``).
        to_unit (str): Target unit (``hartree``, ``ev``, ``kcal/mol``, ``kj/mol``).

    Returns:
        Dict with ``input`` (value + unit), ``output`` (converted value + unit).
        str: Error message if units are invalid.

    Examples:
        Convert Hartree to eV::

            convert_energy(value=-1.137, from_unit="hartree", to_unit="ev")

    """
    _to_hartree = {
        "hartree": 1.0,
        "ev": constants.EV_TO_HARTREE,
        "kcal/mol": constants.KCAL_PER_MOL_TO_HARTREE,
        "kj/mol": constants.KJ_PER_MOL_TO_HARTREE,
    }
    _from_hartree = {
        "hartree": 1.0,
        "ev": constants.HARTREE_TO_EV,
        "kcal/mol": constants.HARTREE_TO_KCAL_PER_MOL,
        "kj/mol": constants.HARTREE_TO_KJ_PER_MOL,
    }

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    if from_unit not in _to_hartree:
        return f"ERROR: Unknown source unit '{from_unit}'. Use: hartree, ev, kcal/mol, kj/mol"
    if to_unit not in _from_hartree:
        return f"ERROR: Unknown target unit '{to_unit}'. Use: hartree, ev, kcal/mol, kj/mol"

    hartree_value = value * _to_hartree[from_unit]
    converted = hartree_value * _from_hartree[to_unit]

    return {
        "input": {"value": value, "unit": from_unit},
        "output": {"value": converted, "unit": to_unit},
    }


@app.tool()
@_structured
def describe_backend(backend_type: str, name: str) -> dict | str:
    """Describe the configuration parameters for a cache backend.

    Returns the ``__init__`` parameter names, types, and defaults so an
    agent can construct a valid ``cache_config`` dict.

    Args:
        backend_type: Must be ``"cache"``.
        name: The registered backend name (e.g. ``"folder"``).

    Returns:
        Dict with ``name``, ``parameters`` list describing each
        constructor kwarg, and ``docstring``.

    """
    import inspect as _inspect  # noqa: PLC0415

    if backend_type == "cache":
        from qdk_chemistry.remote.cache import _CACHES  # noqa: PLC0415

        registry = _CACHES
    else:
        return f"backend_type must be 'cache', got '{backend_type}'"

    if name not in registry:
        return f"No {backend_type} backend registered with name '{name}'. Available: {', '.join(registry)}"

    cls = registry[name]
    sig = _inspect.signature(cls.__init__)
    params = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        info: dict[str, Any] = {"name": pname}
        if param.annotation is not _inspect.Parameter.empty:
            info["type"] = str(param.annotation)
        if param.default is not _inspect.Parameter.empty:
            info["default"] = repr(param.default)
        if param.kind == _inspect.Parameter.VAR_KEYWORD:
            info["kind"] = "**kwargs"
        elif param.kind == _inspect.Parameter.KEYWORD_ONLY:
            info["kind"] = "keyword-only"
        params.append(info)

    return {
        "name": name,
        "backend_type": backend_type,
        "parameters": params,
        "docstring": (cls.__doc__ or "").strip(),
    }


@app.tool()
@_structured
@validate_project
def create_structure(
    project_name: str,
    coordinates_json: str,
    symbols: list[str],
    nuclear_charges: list[float] | None = None,
    masses: list[float] | None = None,
    filename_to_save: str = "structure.structure.json",
    overwrite: bool = False,
) -> Path | str:
    """Create a molecular structure and save it to the project directory.

    Note that the passed coordinates should be in Bohr.

    Args:
        project_name (str): Name of the project to store the structure in
        coordinates_json (str): A JSON string containing a 2D array of atomic coordinates.
            Must be a nested array with shape (N_atoms, 3) where each inner array is [x, y, z].
            Example for H2: '[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]'
            Example for water: '[[0.0, 0.0, 0.0], [1.43, 1.1, 0.0], [-1.43, 1.1, 0.0]]'
        symbols (List[str]): Element symbols for each atom (e.g., ["H", "H"] or ["O", "H", "H"])
        nuclear_charges (Optional[List[float]]): Optionally specify nuclear charges for each atom
        masses (Optional[List[float]]): Optionally specify masses for each atom
        filename_to_save (str): Filename to store the structure as. Must include '.structure.' before
                 the file extension (e.g. 'water.structure.json', 'molecule.structure.h5')
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Path: The path where the json file was saved to
        str: A string containing an error message, if there was a problem with the workflow.

    Examples:
        >>> create_structure("my_project",
                             coordinates_json='[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]',
                             symbols=["H", "H"],
                             filename_to_save="h2.structure.json")

    """
    # Parse coordinates from JSON string
    try:
        coordinates = json.loads(coordinates_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON in coordinates_json: {e}. Expected format: '[[x1,y1,z1], [x2,y2,z2], ...]'"

    # Validate coordinates structure
    if not isinstance(coordinates, list):
        return "ERROR: coordinates_json must be a JSON array. Expected format: '[[x1,y1,z1], [x2,y2,z2], ...]'"

    if len(coordinates) == 0:
        return "ERROR: coordinates list is empty. Provide at least one atom's coordinates as [x, y, z]."

    # Check if user passed a flattened list instead of nested list
    if len(coordinates) > 0 and not isinstance(coordinates[0], list):
        return (
            "ERROR: coordinates_json must be a 2D nested array with shape (N_atoms, 3). "
            f"You passed a flat 1D list. For {len(symbols)} atoms, use format: "
            f"'[[x1,y1,z1], [x2,y2,z2], ...]' with {len(symbols)} inner arrays."
        )

    # Validate each coordinate has exactly 3 components
    for i, coord in enumerate(coordinates):
        if not isinstance(coord, list) or len(coord) != 3:
            return (
                f"ERROR: Each atom needs exactly 3 coordinates [x, y, z]. "
                f"Atom {i} has {len(coord) if isinstance(coord, list) else 'invalid'} values. "
                f"Expected format: '[[x1,y1,z1], [x2,y2,z2], ...]'"
            )

    coordinates = np.array(coordinates)
    if coordinates.ndim != 2:
        return "ERROR: Coordinates should be a 2D array of shape (N_atoms, 3)."

    if len(coordinates) != len(symbols):
        return f"ERROR: Number of coordinate arrays ({len(coordinates)}) must match number of symbols ({len(symbols)})."

    if masses is not None and len(masses) != len(coordinates):
        return f"ERROR: Number of masses ({len(masses)}) must match number of atoms ({len(coordinates)})."

    if nuclear_charges is not None and len(nuclear_charges) != len(coordinates):
        return (
            f"ERROR: Number of nuclear charges ({len(nuclear_charges)}) "
            f"must match number of atoms ({len(coordinates)})."
        )

    # Check if output file already exists
    if not overwrite:
        existing_check = check_output_exists(filename_to_save, data.Structure)
        if existing_check:
            return existing_check

    # Parse structure
    try:
        if masses is not None and nuclear_charges is not None:
            structure = data.Structure(
                coordinates=coordinates, symbols=symbols, masses=masses, nuclear_charges=nuclear_charges
            )
        elif masses is not None:
            structure = data.Structure(coordinates=coordinates, symbols=symbols, masses=masses)
        elif nuclear_charges is not None:
            structure = data.Structure(coordinates=coordinates, symbols=symbols, nuclear_charges=nuclear_charges)
        else:
            structure = data.Structure(coordinates=coordinates, symbols=symbols)
    except RuntimeError as e:
        return f"There was a problem creating a qdk/chemistry Structure objects from input: {e}"

    save_path = config.projects_dir / project_name / filename_to_save

    # Upload to directory - support both json and hdf5
    try:
        if filename_to_save.endswith(".json"):
            structure.to_json_file(save_path)
        elif filename_to_save.endswith((".hdf5", ".h5")):
            structure.to_hdf5_file(save_path)
        else:
            return f"ERROR: Unsupported file extension for {filename_to_save}. Must be .json, .hdf5, or .h5"
    except (RuntimeError, ValueError, PermissionError, OSError) as e:
        return f"Failed to save structure to {save_path}: {type(e).__name__}: {e!s}"

    return save_path


@app.tool()
@_structured
def get_algorithm_default_type(algorithm_type: str) -> str:
    """Get default instance of a particular algorithm type.

    This shows which algorithm type was used to create a factory instance, if done with the default.

    Args:
        algorithm_type (str): Algorithm type passed to `create` method

    Returns:
        str: The algorithm name that corresponds to the default instance of an algorithm type

    """
    created_algorithm = algorithms.create(algorithm_type)
    return created_algorithm.name()


@app.tool()
@_structured
def get_algorithm_default_settings(algorithm_type: str, algorithm_name: str | None = None) -> dict:
    """Return a copy of the algorithm's default settings.

    If algorithm_name is provided, that algorithm_name is used as input to algorithm_type.
    Else, the default implementation is assumed.

    Args:
        algorithm_type (str): Algorithm type passed to `create` method
        algorithm_name (str, optional): Algorithm name corresponding to algorithm_type

    Returns:
        Dict: A copy of the default settings associated with the algorithm instance

    """
    created_algorithm = algorithms.create(algorithm_type, algorithm_name)

    return created_algorithm.settings().to_dict()


@app.tool()
@_structured
@validate_project
def get_orbitals_from_input(
    project_name: str, input_filename: str, out_orbitals_filename: str, overwrite: bool = False
) -> str:
    """Helper function to get orbitals from a serialized object.

    The serialized object must have the `get_orbitals()`
    method, like `Wavefunction`, `Hamiltonian`, `ConfigurationSet` or `Ansatz`

    Args:
        project_name: working project directory
        input_filename: file to load wavefunction from
        out_orbitals_filename: name to save orbitals object to
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Repeat out filename of orbitals object

    """
    input_filename = _strip(input_filename)
    out_orbitals_filename = _strip(out_orbitals_filename)

    out_orbitals_filename, _err = _prepare_output(out_orbitals_filename, "Orbitals", data.Orbitals, overwrite=overwrite)
    if _err:
        return _err

    # Possible input objects are Wavefunction, Ansatz, Hamiltonian, ConfigurationSet
    input_types = [data.Wavefunction, data.Ansatz, data.Hamiltonian, data.ConfigurationSet]

    found_input_type = False
    for input_type in input_types:
        try:
            input_object = load_data_object(input_filename, input_type)
            found_input_type = True
            break
        except (RuntimeError, ValueError):
            continue

    if not found_input_type:
        return (
            f"Failed to load wavefunction from {input_filename}. "
            f"Please provide a qdk Wavefunction, Ansatz, "
            f"ConfigurationSet or Hamiltonian data object."
        )

    orbitals = input_object.get_orbitals()

    # save to file
    save_data_object(orbitals, out_orbitals_filename)

    return out_orbitals_filename


@app.tool()
@_structured
@validate_project
def get_active_space_indices(
    project_name: str,
    input_filename: str,
) -> str | dict:
    """Get the active, inactive, and virtual orbital space indices from a serialized object.

    The input object must have orbitals with a defined active space (i.e., it must have been
    processed by `run_active_space_selector`). Accepted input types are Wavefunction, Ansatz,
    Hamiltonian, and ConfigurationSet.

    This is useful for understanding the partitioning of the orbital space after active space
    selection, and for providing orbital indices to tools like `run_orbital_localization`.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        input_filename (str): Filename of the serialized object containing orbital information

    Returns:
        Dict: A dictionary with keys 'active', 'inactive', and 'virtual', each mapping to
            a dict with 'alpha' and 'beta' index lists.

        str: An error message if the object could not be loaded or has no active space defined.

    """
    input_filename = _strip(input_filename)

    # Try loading from supported input types
    input_types = [data.Wavefunction, data.Ansatz, data.Hamiltonian, data.ConfigurationSet]
    input_object = None
    for input_type in input_types:
        try:
            input_object = load_data_object(input_filename, input_type)
            break
        except (RuntimeError, ValueError):
            continue

    if input_object is None:
        return (
            f"Failed to load from {input_filename}. "
            "Please provide a qdk Wavefunction, Ansatz, Hamiltonian, or ConfigurationSet data object."
        )

    try:
        orbitals = input_object.get_orbitals()
    except (RuntimeError, AttributeError) as e:
        return f"Failed to extract orbitals from {input_filename}: {e!s}"

    if not orbitals.has_active_space():
        return (
            f"The orbitals in {input_filename} do not have a defined active space. "
            "Run `run_active_space_selector` first to define the active space."
        )

    alpha_active, beta_active = orbitals.get_active_space_indices()
    alpha_inactive, beta_inactive = orbitals.get_inactive_space_indices()
    alpha_virtual, beta_virtual = orbitals.get_virtual_space_indices()

    return {
        "active": {"alpha": list(alpha_active), "beta": list(beta_active)},
        "inactive": {"alpha": list(alpha_inactive), "beta": list(beta_inactive)},
        "virtual": {"alpha": list(alpha_virtual), "beta": list(beta_virtual)},
    }


@app.tool()
@_structured
@validate_project
def get_ansatz(
    project_name: str,
    wavefunction_filename: str,
    hamiltonian_filename: str,
    out_ansatz_filename: str,
    overwrite: bool = False,
) -> str:
    """Get an ansatz from hamiltonian and wavefunction and save to disk.

    Args:
        project_name (str): Working project directory
        wavefunction_filename(str): Filename to load wavefunction from
        hamiltonian_filename(str): Filename to load hamiltonian from
        out_ansatz_filename (str): Filename to save ansatz to
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename of where ansatz was saved to

    """
    wavefunction_filename = _strip(wavefunction_filename)
    hamiltonian_filename = _strip(hamiltonian_filename)
    out_ansatz_filename = _strip(out_ansatz_filename)
    out_ansatz_filename, _err = _prepare_output(out_ansatz_filename, "Ansatz", data.Ansatz, overwrite=overwrite)
    if _err:
        return _err

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
    if _err:
        return _err

    ansatz = data.Ansatz(hamiltonian=hamiltonian, wavefunction=wavefunction)

    # save to file
    save_data_object(ansatz, out_ansatz_filename)

    return out_ansatz_filename


# =========================
# Per-algorithm class tools
# =========================


@app.tool()
@_structured
@validate_project
def run_active_space_selector(
    project_name: str,
    wavefunction_filename: str,
    out_wavefunction_filename: str,
    charge: int | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Select active space orbitals from the a serialized wavefunction and save (serialize) a new wavefunction.

    The new wavefunction object has active-space data populated.
    Active space selection is the process of selecting which molecular orbitals are relevant to the system of study.
    This is a key step for multi-reference and particularly quantum
    simulation calculations, where active space selection dramatically
    reduces computational cost by identifying chemically relevant orbitals.

    Available algorithms:

    - ``qdk_valence``: Selects the valence orbitals based on atomic composition and charge. Useful as a
      starting point or pre-filter for larger systems. Requires ``charge``.
    - ``qdk_autocas`` / ``qdk_autocas_eos``: Analyzes single-orbital entropies from a prior multi-configurational
      calculation (SCI/CASCI with ``calculate_one_rdm=True``, ``calculate_two_rdm=True``, and
      ``calculate_mutual_information=True``) and **automatically determines which orbitals are strongly
      correlated**. The orbital entanglement analysis decides the active space — no manual orbital
      selection is needed. This is the recommended approach for determining the final active space.
    - ``qdk_occupation``: Selects based on orbital occupation numbers.

    Typical workflow context:

    A full quantum simulation workflow starting from a structure object is:

    1. Run ``run_scf`` to get an initial wavefunction (typically HF for multi-reference work)
    2. (THIS TOOL) Run ``run_active_space_selector`` with ``qdk_valence`` to get an initial active space
    3. Run ``run_multi_configuration_calculation`` (SCI) with ``calculate_one_rdm=True``,
       ``calculate_two_rdm=True``, and ``calculate_mutual_information=True`` to get orbital entropies
       and entanglement data
    4. (THIS TOOL) Run ``run_active_space_selector`` again with ``qdk_autocas_eos`` — this reads the
       orbital entropies from step 3 and automatically identifies the strongly correlated subset
    5. (Optional) Sparsify the wavefunction using ``run_projected_multi_configuration_calculation`` with only the
       top determinants (by CI coefficient magnitude) to reduce circuit depth
    6. Run ``run_hamiltonian_constructor`` to build the fermionic Hamiltonian from active-space orbitals
    7. Run ``run_qubit_mapper`` to convert the fermionic Hamiltonian to qubits
    8. Run ``run_state_preparation`` to prepare a quantum circuit from the (sparse) multi-configurational wavefunction
    9. Run ``run_phase_estimation`` for quantum phase estimation

    Note that the particular workflow for a particular calculation may
    vary depending on the starting point of the calculation - for
    example, if starting from a Hamiltonian rather than a structure,
    we might skip some of the above steps.

    Recommended usage:

    - For an initial broad active space, use ``qdk_valence`` with the system ``charge``.
    - To determine the final active space automatically, run SCI with RDMs first, then use
      ``qdk_autocas_eos``. The algorithm analyzes orbital entanglement entropies and selects
      the orbitals that are strongly correlated — you do not need to specify which orbitals to include.
    - If you are unsure of what to use, please leave the default settings as-is.
    - The default method can be extracted using the function and MCP tool ``get_algorithm_default_type``.

    Pitfalls to avoid:

    The default active space selection method is an autocas variant,
    these use orbital entropies to select the active space.
    This is a good general-purpose method, but it relies on having
    RDMs from a multi-configurational wavefunction.
    Make sure they exist or change the algorithm if you are starting
    from a SCF wavefunction without RDMs, as the default will not
    work in that case.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool ``get_algorithm_default_settings``.
    - In most cases, the default settings should not be modified.
    - If using the ``qdk_autocas`` active space selector, the
      ``entropy_threshold`` can be modified and we can also set
      ``normalize_entropies`` to false if desired

    Args:
        project_name (str): Name of the current qdk/chemistry project
        wavefunction_filename (str): Name of the file of the input wavefunction in the current directory
        out_wavefunction_filename (str): Name of the file where the
            output wavefunction is saved in the current directory
        charge (int, optional): System charge, required for
            qdk_valence active space selector
        algorithm_name (str, optional): a specific algorithm string
            to override the default
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: containing the name of the output wavefunction filename, if the function executed successfully, or
        string containing the error message if any problems are encountered during function execution.

    """
    # we should be in working directory so strip filenames in case a full path is passed
    wavefunction_filename = _strip(wavefunction_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    if charge is not None and algorithm_name == "qdk_valence":
        active_space_selector = algorithms.create("active_space_selector", "qdk_valence")
        # grab valence electrons/orbitals count using helper function
        n_active_electrons, n_active_orbitals = compute_valence_space_parameters(wavefunction, charge)
        active_space_selector.settings().set("num_active_electrons", n_active_electrons)
        active_space_selector.settings().set("num_active_orbitals", n_active_orbitals)
    else:
        active_space_selector = algorithms.create("active_space_selector", algorithm_name)
        _apply_settings(active_space_selector, settings)

    # run active space selection
    out_wavefunction = _run_algorithm(
        active_space_selector,
        wavefunction,
        cache=cache,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(out_wavefunction, out_wavefunction_filename)

    return out_wavefunction_filename


@app.tool()
@_structured
@validate_project
def run_dynamical_correlation_calculator(
    project_name: str,
    ansatz_filename: str,
    out_wavefunction_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run dynamical correlation calculator.

    This calculation adds dynamical correlation to the starting Ansatz.
    Dynamical correlation calculators return the new total energy, and save the updated wavefunction to file.

    Typical workflow context:

    Dynamical correlation is used to improve single-reference (HF)
    energies. A typical workflow starting from a structure is:

    1. Run `run_scf` to get a Hartree-Fock wavefunction
    2. Build a Hamiltonian from the SCF orbitals using `run_hamiltonian_constructor`
    3. Create an Ansatz object from the Hamiltonian and wavefunction, save to file
    4. (THIS TOOL) Run `run_dynamical_correlation_calculator` to add correlation corrections

    This is recommended for single-reference systems (closed-shell molecules at equilibrium).
    For multi-reference systems (bond breaking, open-shell, etc.), use `run_multi_configuration_calculation` instead.

    Usage guidelines:

    - The default dynamical correlation calculator method can be
      extracted using the function and MCP tool
      `get_algorithm_default_type`.
    - In general the hierarchy of dynamical correlation methods is MP2 -> CCSD -> CCSD(T), in terms of accuracy and
      computational cost. Therefore depending on the size of the system, MP2 or CCSD might be more appropriate.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - You most likely only want to provide settings if performing
      coupled cluster calculations.
    - If you are only interested in the total energy, no need to modify the settings dictionary.
    - If amplitudes will be used later on, these can be stored by setting the key `store_amplitudes` to `True`

    Args:
        project_name (str): Name of the current qdk/chemistry project
        ansatz_filename (str): Name of the file containing the input ansatz in the current directory
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        algorithm_name (str, optional): Algorithm name for the
            dynamical correlation calculator, to override the default
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[float, str]: The new total energy including dynamical
            correlation contributions, and the filename where
            wavefunction was saved

        str: containing error message, if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    ansatz_filename = _strip(ansatz_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    ansatz, _err = _load_or_error(ansatz_filename, data.Ansatz, "ansatz")
    if _err:
        return _err

    dyn_corr_calculator = algorithms.create("dynamical_correlation_calculator", algorithm_name)

    _apply_settings(dyn_corr_calculator, settings)

    # run
    result = _run_algorithm(
        dyn_corr_calculator,
        ansatz,
        cache=cache,
        overwrite=overwrite,
    )
    # Algorithm returns (energy, correlated_wavefunction[, original_wavefunction])
    total_energy = result[0]
    wavefunction = result[1]

    # save to file
    save_data_object(wavefunction, out_wavefunction_filename)

    return (total_energy, out_wavefunction_filename)


@app.tool()
@_structured
@validate_project
def run_hamiltonian_constructor(
    project_name: str,
    orbitals_filename: str,
    out_hamiltonian_filename: str,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Use the Hamiltonian constructor class to create a hamiltonian object and save to file.

    The Hamiltonian constructor builds a fermionic Hamiltonian object from molecular orbitals.
    The Hamiltonian contains:

    - One-electron integrals (kinetic + nuclear attraction in active space)
    - Two-electron integrals (electron-electron repulsion in active space)
    - Core energy (frozen core contribution + nuclear repulsion)

    This is a key step in multi-reference and quantum simulation workflows.

    Typical workflow context:

    Two paths depending on system size:

    **Full-space path** (small systems, up to ~16 spatial orbitals / ~20 qubits):
    1. Run `run_scf` to get an initial wavefunction
    2. Extract orbitals from the SCF wavefunction via `get_orbitals_from_input`
    3. (THIS TOOL) Run `run_hamiltonian_constructor` on the full SCF orbitals
    4. Proceed to `run_qubit_mapper` → quantum steps

    **Active-space path** (larger systems):
    1. Run `run_scf` → active space analysis (SCI + AutoCAS) to compress the orbital space
    2. Extract orbitals from the active-space wavefunction
    3. (THIS TOOL) Run `run_hamiltonian_constructor` on the active-space orbitals
    4. Proceed to `run_qubit_mapper` → quantum steps

    For `run_multi_configuration_scf`, you can pass orbitals directly — it builds the Hamiltonian internally.

    **When to choose which path:** Ask the user whether they want the full orbital space or a
    compressed active space. For small molecules or model Hamiltonians where the full space
    is tractable, skipping active space selection is simpler and avoids approximation.
    For larger systems, active space compression is necessary to keep the quantum computation feasible.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        orbitals_filename (str): Name of the file containing the input orbitals in the current directory.
            Can come from a full SCF wavefunction (small systems) or an active-space wavefunction (larger systems).
        out_hamiltonian_filename (str): Name of the file where the output hamiltonian will be saved
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where hamiltonian was saved
        OR error message if there was a problem in the workflow

    """
    out_hamiltonian_filename, err = _prepare_output(
        out_hamiltonian_filename, "Hamiltonian", data.Hamiltonian, overwrite=overwrite
    )
    if err:
        return err

    orbitals, err = _load_or_error(orbitals_filename, data.Orbitals, "orbitals")
    if err:
        return err

    ham_constructor = algorithms.create("hamiltonian_constructor")

    # run
    hamiltonian = _run_algorithm(
        ham_constructor,
        orbitals,
        cache=cache,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(hamiltonian, out_hamiltonian_filename)

    return out_hamiltonian_filename


# ═══════════════════════════════════════════════════════════════════════════
# Model Hamiltonian tools
# ═══════════════════════════════════════════════════════════════════════════


def _build_lattice_graph(lattice_type: str, lattice_params: dict):
    """Build a LatticeGraph from a type string and parameter dict.

    Returns (LatticeGraph, None) on success, or (None, error_string) on failure.
    """
    params = dict(lattice_params or {})
    try:
        if lattice_type == "chain":
            return data.LatticeGraph.chain(
                n=params.pop("n"),
                periodic=params.pop("periodic", False),
                t=params.pop("t", 1.0),
            ), None
        if lattice_type == "square":
            return data.LatticeGraph.square(
                nx=params.pop("nx"),
                ny=params.pop("ny"),
                periodic_x=params.pop("periodic_x", False),
                periodic_y=params.pop("periodic_y", False),
                t=params.pop("t", 1.0),
            ), None
        if lattice_type == "triangular":
            return data.LatticeGraph.triangular(
                nx=params.pop("nx"),
                ny=params.pop("ny"),
                periodic_x=params.pop("periodic_x", False),
                periodic_y=params.pop("periodic_y", False),
                t=params.pop("t", 1.0),
            ), None
        if lattice_type == "honeycomb":
            return data.LatticeGraph.honeycomb(
                nx=params.pop("nx"),
                ny=params.pop("ny"),
                periodic_x=params.pop("periodic_x", False),
                periodic_y=params.pop("periodic_y", False),
                t=params.pop("t", 1.0),
            ), None
        if lattice_type == "kagome":
            return data.LatticeGraph.kagome(
                nx=params.pop("nx"),
                ny=params.pop("ny"),
                periodic_x=params.pop("periodic_x", False),
                periodic_y=params.pop("periodic_y", False),
                t=params.pop("t", 1.0),
            ), None
        if lattice_type == "custom":
            edges_raw = params.pop("edges")
            num_sites = params.pop("num_sites")
            # Convert JSON-safe [[i, j, w], ...] to {(i,j): w} dict
            if isinstance(edges_raw, list):
                edge_dict = {(int(e[0]), int(e[1])): float(e[2]) for e in edges_raw}
            elif isinstance(edges_raw, dict):
                edge_dict = {(int(k.split(",")[0]), int(k.split(",")[1])): float(v) for k, v in edges_raw.items()}
            else:
                return None, "Invalid edges format: expected list of [i, j, weight] triples or dict"
            graph = data.LatticeGraph(edge_dict, num_sites)
            return data.LatticeGraph.make_bidirectional(graph), None
        return None, (
            f"Unknown lattice_type '{lattice_type}'. Available: chain, square, triangular, honeycomb, kagome, custom"
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        return None, f"Failed to build {lattice_type} lattice: {e!s}"


def _coerce_param(value, kind: str):
    """Coerce a JSON-compatible value to a float or numpy array.

    *kind* is ``"site"`` (scalar or 1-D list) or ``"pair"`` (scalar or 2-D list).
    """
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    # list → numpy array
    arr = np.asarray(value, dtype=float)
    if kind == "site" and arr.ndim != 1:
        raise ValueError(f"Expected scalar or 1-D list for site parameter, got shape {arr.shape}")
    if kind == "pair" and arr.ndim != 2:
        raise ValueError(f"Expected scalar or 2-D list for pair parameter, got shape {arr.shape}")
    return arr


@app.tool()
@_structured
@validate_project
def create_model_hamiltonian(
    project_name: str,
    model: str,
    out_hamiltonian_filename: str,
    lattice_type: str,
    lattice_params: dict,
    epsilon: float | list[float] = 0.0,
    t: float | list[list[float]] = 1.0,
    u_coulomb: float | list[float] = 0.0,
    v_coulomb: float | list[list[float]] | None = None,
    z: float | list[float] = 1.0,
    potential: str | None = None,
    potential_params: dict | None = None,
    overwrite: bool = False,
) -> str:
    """Create a model Hamiltonian on a lattice and save to file.

    Builds a fermionic Hamiltonian for common lattice models without requiring molecular
    structure input, SCF, or active-space selection.  The output is a standard ``Hamiltonian``
    object that plugs directly into ``run_qubit_mapper``, ``run_multi_configuration_calculation``,
    and all downstream quantum tools.

    Supported models:

    - ``huckel`` — tight-binding with on-site energies and hopping.
    - ``hubbard`` — Hückel + on-site Coulomb repulsion *U*.
    - ``ppp`` — Hubbard + long-range intersite Coulomb *V* (Pariser-Parr-Pople).

    Lattice types: ``chain``, ``square``, ``triangular``, ``honeycomb``, ``kagome``, ``custom``.

    Lattice params (JSON dict):

    - **chain:** ``{"n": <int>, "periodic": <bool>}``
    - **square / triangular / honeycomb / kagome:**
      ``{"nx": <int>, "ny": <int>, "periodic_x": <bool>, "periodic_y": <bool>}``
    - **custom:** ``{"edges": [[i, j, weight], ...], "num_sites": <int>}``

    All lattice factories accept an optional ``"t"`` key for the default edge weight (default 1.0).

    Parameter broadcasting:

    - *Scalar* values are broadcast to all sites/pairs.
    - *Per-site* parameters (``epsilon``, ``u_coulomb``, ``z``) accept a 1-D list of length *n*.
    - *Per-pair* parameters (``t``, ``v_coulomb``) accept a 2-D list of shape *n x n*.

    For the PPP model, the intersite Coulomb matrix ``V`` can be:

    1. Provided directly as a 2-D list.
    2. Computed automatically by setting ``potential`` to ``"ohno"`` or ``"mataga_nishimoto"``
       and providing ``potential_params`` with ``{"R": <float or 2-D list>, "epsilon_r": <float>}``.

    Typical workflow:

    1. (THIS TOOL) ``create_model_hamiltonian`` → ``hamiltonian.json``
    2. ``run_qubit_mapper`` → ``qubit_hamiltonian.json``
    3. ``run_multi_configuration_calculation`` → wavefunction + energies
    4. ``run_state_preparation`` → circuit
    5. ``run_phase_estimation`` / ``run_energy_estimator``

    Args:
        project_name (str): Name of the current qdk/chemistry project
        model (str): Model type: ``"huckel"``, ``"hubbard"``, or ``"ppp"``
        out_hamiltonian_filename (str): Output filename for the Hamiltonian
        lattice_type (str): Lattice geometry (see above)
        lattice_params (dict): Parameters for the lattice factory (see above)
        epsilon (float or list[float]): On-site energy. Default 0.0
        t (float or list[list[float]]): Hopping integral. Default 1.0
        u_coulomb (float or list[float]): On-site Coulomb repulsion *U* (Hubbard/PPP). Default 0.0
        v_coulomb (float or list[list[float]], optional): Intersite Coulomb matrix *V* (PPP).
            Required for PPP unless ``potential`` is set.
        z (float or list[float]): Effective core charges (PPP). Default 1.0
        potential (str, optional): Auto-compute V using ``"ohno"`` or ``"mataga_nishimoto"``
        potential_params (dict, optional): Parameters for the potential function:
            ``{"R": <distance>, "epsilon_r": <float>}``
        overwrite (bool): Overwrite existing output. Default False.

    Returns:
        str: Filename where Hamiltonian was saved, or error message.

    """
    from qdk_chemistry.utils.model_hamiltonians import (  # noqa: PLC0415
        create_hubbard_hamiltonian,
        create_huckel_hamiltonian,
        create_ppp_hamiltonian,
        mataga_nishimoto_potential,
        ohno_potential,
    )

    out_hamiltonian_filename, err = _prepare_output(
        out_hamiltonian_filename, "Hamiltonian", data.Hamiltonian, overwrite=overwrite
    )
    if err:
        return err

    lattice, err = _build_lattice_graph(lattice_type, lattice_params)
    if err:
        return err

    try:
        eps = _coerce_param(epsilon, "site")
        t_val = _coerce_param(t, "pair")
    except ValueError as e:
        return f"Invalid parameter: {e!s}"

    model_lower = model.lower()
    try:
        if model_lower == "huckel":
            hamiltonian = create_huckel_hamiltonian(lattice, eps, t_val)
        elif model_lower == "hubbard":
            u_val = _coerce_param(u_coulomb, "site")
            hamiltonian = create_hubbard_hamiltonian(lattice, eps, t_val, u_val)
        elif model_lower == "ppp":
            u_val = _coerce_param(u_coulomb, "site")
            z_val = _coerce_param(z, "site")

            # Resolve V: direct value or compute via potential
            if potential is not None:
                pp = potential_params or {}
                r_val = _coerce_param(pp.get("R", 1.0), "pair")
                epsilon_r = float(pp.get("epsilon_r", 1.0))
                nn_only = bool(pp.get("nearest_neighbor_only", False))
                if potential.lower() == "ohno":
                    v_val = ohno_potential(lattice, u_val, r_val, epsilon_r, nn_only)
                elif potential.lower() in ("mataga_nishimoto", "mataga-nishimoto"):
                    v_val = mataga_nishimoto_potential(lattice, u_val, r_val, epsilon_r, nn_only)
                else:
                    return f"Unknown potential '{potential}'. Available: ohno, mataga_nishimoto"
            elif v_coulomb is not None:
                v_val = _coerce_param(v_coulomb, "pair")
            else:
                return (
                    "PPP model requires either 'v_coulomb' (intersite Coulomb matrix)"
                    " or 'potential' to auto-compute it."
                )
            hamiltonian = create_ppp_hamiltonian(lattice, eps, t_val, u_val, v_val, z_val)
        else:
            return f"Unknown model '{model}'. Available: huckel, hubbard, ppp"
    except (RuntimeError, ValueError, TypeError) as e:
        return f"Failed to create {model} Hamiltonian: {e!s}"

    save_data_object(hamiltonian, out_hamiltonian_filename)
    return out_hamiltonian_filename


@app.tool()
@_structured
@validate_project
def create_spin_model_hamiltonian(
    project_name: str,
    model: str,
    out_qubit_hamiltonian_filename: str,
    lattice_type: str,
    lattice_params: dict,
    jx: float | list[list[float]] = 0.0,
    jy: float | list[list[float]] = 0.0,
    jz: float | list[list[float]] = 0.0,
    hx: float | list[float] = 0.0,
    hy: float | list[float] = 0.0,
    hz: float | list[float] = 0.0,
    j: float | list[list[float]] | None = None,
    h: float | list[float] | None = None,
    overwrite: bool = False,
) -> str:
    """Create a spin model Hamiltonian on a lattice and save to file.

    Builds a qubit Hamiltonian directly (no qubit mapping needed) for spin lattice models.
    The output plugs directly into ``run_qubit_hamiltonian_solver``, ``run_energy_estimator``,
    or ``run_phase_estimation``.

    Supported models:

    - ``heisenberg`` — anisotropic Heisenberg: ``Jx XX + Jy YY + Jz ZZ`` couplings + external fields.
      Special cases: XXX (Jx=Jy=Jz), XXZ (Jx=Jy≠Jz), XY (Jz=0).
    - ``ising`` — transverse-field Ising: ``J ZZ`` coupling + transverse field ``h X``.
      Shorthand for Heisenberg with Jx=Jy=0.

    Lattice types and params: same as ``create_model_hamiltonian``.

    Parameter broadcasting:

    - *Scalar* coupling constants are broadcast to all pairs/sites.
    - *Per-pair* parameters (``jx``, ``jy``, ``jz``, ``j``) accept a 2-D list of shape *n x n*.
    - *Per-site* field parameters (``hx``, ``hy``, ``hz``, ``h``) accept a 1-D list of length *n*.

    Typical workflow:

    1. (THIS TOOL) ``create_spin_model_hamiltonian`` → ``qubit_hamiltonian.json``
    2. ``run_qubit_hamiltonian_solver`` (exact diag for small systems)
    3. Or: ``run_phase_estimation`` / ``run_energy_estimator`` (quantum algorithms)

    Args:
        project_name (str): Name of the current qdk/chemistry project
        model (str): Model type: ``"heisenberg"`` or ``"ising"``
        out_qubit_hamiltonian_filename (str): Output filename for the QubitHamiltonian
        lattice_type (str): Lattice geometry
        lattice_params (dict): Parameters for the lattice factory
        jx (float or list[list[float]]): XX coupling (Heisenberg). Default 0.0
        jy (float or list[list[float]]): YY coupling (Heisenberg). Default 0.0
        jz (float or list[list[float]]): ZZ coupling (Heisenberg). Default 0.0
        hx (float or list[float]): External field X (Heisenberg). Default 0.0
        hy (float or list[float]): External field Y (Heisenberg). Default 0.0
        hz (float or list[float]): External field Z (Heisenberg). Default 0.0
        j (float or list[list[float]], optional): ZZ coupling (Ising shorthand)
        h (float or list[float], optional): Transverse field X (Ising shorthand)
        overwrite (bool): Overwrite existing output. Default False.

    Returns:
        str: Filename where QubitHamiltonian was saved, or error message.

    """
    from qdk_chemistry.utils.model_hamiltonians import (  # noqa: PLC0415
        create_heisenberg_hamiltonian,
        create_ising_hamiltonian,
    )

    out_qubit_hamiltonian_filename, err = _prepare_output(
        out_qubit_hamiltonian_filename, "QubitHamiltonian", data.QubitHamiltonian, overwrite=overwrite
    )
    if err:
        return err

    lattice, err = _build_lattice_graph(lattice_type, lattice_params)
    if err:
        return err

    model_lower = model.lower()
    try:
        if model_lower == "heisenberg":
            jx_val = _coerce_param(jx, "pair")
            jy_val = _coerce_param(jy, "pair")
            jz_val = _coerce_param(jz, "pair")
            hx_val = _coerce_param(hx, "site")
            hy_val = _coerce_param(hy, "site")
            hz_val = _coerce_param(hz, "site")
            qh = create_heisenberg_hamiltonian(lattice, jx_val, jy_val, jz_val, hx_val, hy_val, hz_val)
        elif model_lower == "ising":
            if j is None:
                return "Ising model requires 'j' (ZZ coupling constant)."
            j_val = _coerce_param(j, "pair")
            h_val = _coerce_param(h, "site") if h is not None else 0.0
            qh = create_ising_hamiltonian(lattice, j_val, h_val)
        else:
            return f"Unknown model '{model}'. Available: heisenberg, ising"
    except (RuntimeError, ValueError, TypeError) as e:
        return f"Failed to create {model} Hamiltonian: {e!s}"

    save_data_object(qh, out_qubit_hamiltonian_filename)
    return out_qubit_hamiltonian_filename


@app.tool()
@_structured
@validate_project
def run_orbital_localization(
    project_name: str,
    wavefunction_filename: str,
    out_wavefunction_filename: str,
    loc_indices_alpha: list[int] | np.ndarray[int],
    loc_indices_beta: list[int] | np.ndarray[int] | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Localize the orbitals of an input wavefunction and save to file.

    For restricted calculations (closed-shell systems),
    loc_indices_alpha must be provided as a sorted list or numpy
    array, to specify which orbitals to localize.
    For unrestricted calculations (open-shell systems),
    loc_indices_beta also need to be provided as a sorted list or
    numpy array, to specify which beta orbitals should be localized.

    Usage guidelines:

    - The default orbital localization method can be extracted using
      the function and MCP tool `get_algorithm_default_type`.
    - Unless specific justification is provided, the default method should be sufficient.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - The orbital localization classes have variable settings
      depending on the specific localizer implementation.
    - To view a copy of the default settings for `algorithm_name`,
      use the function and MCP tool
      `get_algorithm_default_settings`.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        wavefunction_filename (str): Name of the file containing the input wavefunction in the current directory
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        loc_indices_alpha (List or numpy.ndarray): A sorted list or
            1d array of indices, specifying which (alpha) orbitals
            should be localized
        loc_indices_beta (List or numpy.ndarray, optional): A sorted
            list or 1d array of indices, specifying which beta
            orbitals should be localized. This is only needed for
            unrestricted calculations (for open-shell systems).
        algorithm_name (str, optional): The name of the orbital
            localization algorithm to use, which overrides the
            default
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where localized wavefunction was saved
        OR error message if there was a problem in the workflow.

    """
    # Strip filenames in case full path is passed
    wavefunction_filename = _strip(wavefunction_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    localizer = algorithms.create("orbital_localizer", algorithm_name)

    _apply_settings(localizer, settings)

    if loc_indices_beta is None:
        localized_wfn = _run_algorithm(
            localizer,
            wavefunction,
            loc_indices_alpha,
            loc_indices_alpha,
            cache=cache,
            overwrite=overwrite,
        )
    else:
        localized_wfn = _run_algorithm(
            localizer,
            wavefunction,
            loc_indices_alpha,
            loc_indices_beta,
            cache=cache,
            overwrite=overwrite,
        )

    # save to file
    save_data_object(localized_wfn, out_wavefunction_filename)

    return out_wavefunction_filename


@app.tool()
@_structured
@validate_project
def run_stability_checker(
    project_name: str,
    wavefunction_filename: str,
    out_stability_result_filename: str,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[bool, str]:
    """Check the stability of the given wavefunction with respect to orbital rotation.

    This method performs stability analysis on the input wavefunction by examining the eigenvalues
    of the electronic Hessian matrix. A stable wavefunction should have all non-negative eigenvalues.
    Near-zero eigenvalues may indicate orbital degeneracy.

    Usage guidelines:

    - If the SCF wavefunction is unstable, we can transform the SCF wavefunction to generate initial guess orbitals.
      These can be used to make a density matrix and fed into a new SCF iteration.
    - Therefore the stability checker gives an indication of a successful SCF procedure.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - The key settings are the keys `internal` and/or `external`,
      which specify whether to perform an internal or external
      stability check. `internal` means to check within the
      restricted Hartree Fock (RHF) space. 'external' means to
      check RHF -> UHF and real -> complex.
      In most cases the default, i.e. `internal` only, is
      sufficient.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        wavefunction_filename (str): Name of the file containing the input wavefunction in the current directory
        out_stability_result_filename (str): Name of the file where the stability result will be saved
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[bool, str]: Overall stability status as a bool and the
            filename where detailed stability information was saved

        str: Containing error message, if there is a problem with the workflow

    """
    # Strip filenames in case full path is passed
    wavefunction_filename = _strip(wavefunction_filename)
    out_stability_result_filename = _strip(out_stability_result_filename)
    out_stability_result_filename, _err = _prepare_output(
        out_stability_result_filename, "StabilityResult", data.StabilityResult, overwrite=overwrite
    )
    if _err:
        return _err

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    stability_checker = algorithms.create("stability_checker")

    _apply_settings(stability_checker, settings)

    try:
        (stability_bool, stability_result) = _run_algorithm(
            stability_checker,
            wavefunction,
            cache=cache,
            overwrite=overwrite,
        )

    except RuntimeError as e:
        return f"The stability checker did not converge: {e}. Perhaps consider changing the basis set of the system."

    # save to file
    save_data_object(stability_result, out_stability_result_filename)

    return (stability_bool, out_stability_result_filename)


@app.tool()
@_structured
@validate_project
def run_qubit_hamiltonian_solver(
    project_name: str,
    qubit_hamiltonian_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
) -> str | tuple[float, list]:
    """Solve a qubit Hamiltonian to get the ground state energy and wavefunction.

    This method computes the ground state energy and corresponding eigenstate of a qubit Hamiltonian
    by constructing and diagonalizing its matrix representation.

    Usage guidelines:

    - The default qubit Hamiltonian solver can be extracted using
      the function and MCP tool `get_algorithm_default_type`.
    - The sparse matrix solver (default) is recommended for most cases as it is more memory-efficient and faster
      for large systems.
    - The dense matrix solver can be used for small systems where you need the full spectrum or when the Hamiltonian
      is already dense.
    - Choose the sparse solver for systems with more than ~10-12 qubits to avoid memory issues.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - The dense solver has no configurable settings
    - The sparse solver (default) `qdk_sparse_matrix_solver` has
      settings `tol`, `max_m` to set the convergence tolerance for
      the Davidson solver
    - In most cases, the default settings should not be modified
      unless you need higher precision or are experiencing
      convergence issues.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        qubit_hamiltonian_filename (str): Name of the file containing the qubit Hamiltonian in the current directory
        algorithm_name (str, optional): Algorithm name to override the default solver
        settings (Dict, optional): A dictionary of key-value pairs specifying which settings keys
            to replace with specific values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching

    Returns:
        Tuple[float, List]: The ground state energy and corresponding eigenstate (statevector) as a list
        str: Containing the error message, if there was a problem with the workflow

    """
    # Strip filename in case full path is passed
    qubit_hamiltonian_filename = _strip(qubit_hamiltonian_filename)

    try:
        qubit_hamiltonian = load_data_object(qubit_hamiltonian_filename, data.QubitHamiltonian)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load qubit hamiltonian from {qubit_hamiltonian_filename}: {e!s}"

    qubit_hamiltonian_solver = algorithms.create("qubit_hamiltonian_solver", algorithm_name)

    _apply_settings(qubit_hamiltonian_solver, settings)

    (energy, eigenstate) = _run_algorithm(
        qubit_hamiltonian_solver,
        qubit_hamiltonian,
        cache=cache,
    )

    return (energy, eigenstate.tolist())


@app.tool()
@_structured
@validate_project
def run_energy_estimator(
    project_name: str,
    circuit_filename: str,
    out_energy_result_filename: str,
    out_measurement_data_filename: str,
    total_shots: int,
    qubit_hamiltonian_filename: str | None = None,
    qubit_hamiltonian_filenames: list[str] | None = None,
    noise_model: Any | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[str, str]:
    """Estimate the expectation value and variance of a Hamiltonian from a quantum circuit.

    The energy estimator evaluates the expectation value of a qubit Hamiltonian with respect to
    a quantum circuit that prepares a quantum state. It automatically groups commuting Pauli terms,
    generates measurement circuits, executes them on a simulator backend, and calculates energy
    expectation values from bitstring statistics.

    Usage guidelines:

    - The default energy estimator can be extracted using the function and MCP tool `get_algorithm_default_type`.
    - The QDK base simulator (default) is recommended for most cases and supports various noise models.
    - The Qiskit Aer simulator can be used when you need custom Qiskit noise models.
    - The circuit and Hamiltonian must be compatible—using the same
      qubit encoding and derived from the same molecular system.
    - More shots reduce statistical uncertainty but increase computational cost.
    - Commuting grouping of Pauli terms is performed internally by the energy estimator.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - For the base simulator (default), the `qubit_loss` argument
      can be set, as well as a different `seed`
    - In most cases, the default settings should not be modified
      unless you need specific reproducibility or noise simulation.
    - The circuit executor used internally can be overridden via the
      ``circuit_executor`` settings key::

          settings={"circuit_executor": {"algorithm_name": "qdk_full_state_simulator", "seed": 123}}

    Args:
        project_name (str): Name of the current qdk/chemistry project
        circuit_filename (str): Name of the file containing the input circuit in the current directory
        qubit_hamiltonian_filename (str): Filename of the QubitHamiltonian to estimate
        qubit_hamiltonian_filenames (list[str], optional): Compatibility alias. If provided, the first filename is used.
        out_energy_result_filename (str): Name of the file where energy result will be saved
        out_measurement_data_filename (str): Name of the file where measurement data will be saved
        total_shots (int): Total number of shots to allocate across the observable terms
        noise_model (Optional[Any]): Optional noise model to simulate noise in the quantum circuit.
            For QDK simulator: use Q# noise models. For Qiskit simulator: use Qiskit noise models.
        algorithm_name (str, optional): The name of the energy estimator algorithm to use, if overriding the default
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[str, str]:
            - Filename where EnergyExpectationResult was saved
            - Filename where MeasurementData was saved

        str: Containing error message, if there was a problem with the workflow

    """
    # Strip filenames in case full path is passed
    circuit_filename = _strip(circuit_filename)
    if qubit_hamiltonian_filename is None and qubit_hamiltonian_filenames:
        qubit_hamiltonian_filename = qubit_hamiltonian_filenames[0]
    if qubit_hamiltonian_filename is None:
        return "ERROR: qubit_hamiltonian_filename is required."
    qubit_hamiltonian_filename = _strip(qubit_hamiltonian_filename)
    out_energy_result_filename = _strip(out_energy_result_filename)
    out_measurement_data_filename = _strip(out_measurement_data_filename)
    try:
        out_energy_result_filename = ensure_filename_format(out_energy_result_filename, "EnergyExpectationResult")
        out_measurement_data_filename = ensure_filename_format(out_measurement_data_filename, "MeasurementData")
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    # Check if output files already exist
    if not overwrite:
        existing_check = check_output_exists(out_energy_result_filename, data.EnergyExpectationResult)
        if existing_check:
            return existing_check
        existing_check = check_output_exists(out_measurement_data_filename, data.MeasurementData)
        if existing_check:
            return existing_check

    circuit, _err = _load_or_error(circuit_filename, data.Circuit, "circuit")
    if _err:
        return _err

    try:
        qubit_hamiltonian = load_data_object(qubit_hamiltonian_filename, data.QubitHamiltonian)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load qubit hamiltonian from {qubit_hamiltonian_filename}: {e!s}"

    energy_estimator = algorithms.create("energy_estimator", algorithm_name)

    _apply_settings(energy_estimator, settings)

    # run energy estimation (grouping is handled internally)
    # The circuit executor is configured via the energy estimator's
    # settings (AlgorithmRef) and created internally by the algorithm.
    (energy_result, measurement_data) = _run_algorithm(
        energy_estimator,
        circuit,
        qubit_hamiltonian,
        total_shots,
        noise_model,
        cache=cache,
        overwrite=overwrite,
    )

    # save to files
    save_data_object(energy_result, out_energy_result_filename)
    save_data_object(measurement_data, out_measurement_data_filename)

    return (out_energy_result_filename, out_measurement_data_filename)


@app.tool()
@_structured
@validate_project
def run_qubit_mapper(
    project_name: str,
    hamiltonian_filename: str,
    out_qubit_hamiltonian_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Map a fermionic Hamiltonian to a qubit Hamiltonian.

    This method transforms a fermionic Hamiltonian (expressed in terms of fermionic creation/annihilation operators)
    into a qubit Hamiltonian (expressed as a weighted sum of Pauli strings) using a specified fermion-to-qubit encoding.

    Typical workflow context:

    Two typical workflows, depending on system size:

    **Full-space path** (small systems, up to ~16 spatial orbitals / ~20 qubits):
    1. Run `run_scf` to get an initial wavefunction
    2. Extract orbitals → `run_hamiltonian_constructor` on full SCF orbitals
    3. (THIS TOOL) `run_qubit_mapper` to convert to qubits
    4. Proceed to state preparation / QPE / resource estimation

    **Active-space path** (larger systems):
    1. Run `run_scf` → active space analysis (SCI + AutoCAS) → compress orbital space
    2. (Optional) `run_multi_configuration_calculation` for classical reference energy
    3. (Optional) Sparsify wavefunction via `run_projected_multi_configuration_calculation`
    4. Extract orbitals → `run_hamiltonian_constructor` on active-space orbitals
    5. (THIS TOOL) `run_qubit_mapper` to convert to qubits
    6. Proceed to state preparation / QPE / resource estimation

    Ask the user which approach they prefer if it's not clear from the system size.

    Note that this workflow may vary, depending on the starting point (e.g., starting from a custom Hamiltonian rather
    than structure object).

    The qubit Hamiltonian output file is used by:

    - `run_phase_estimation` for quantum phase estimation
    - `run_energy_estimator` for shot-based energy estimation
    - `run_qubit_hamiltonian_solver` for exact classical diagonalization

    Usage guidelines:

    - The default qubit mapper can be extracted using the function and MCP tool `get_algorithm_default_type`.
    - The qubit Hamiltonian output is compatible with energy estimation and quantum circuit algorithms.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool
      `get_algorithm_default_settings`.
    - The key setting is `encoding` (default, `"jordan-wigner"`), which specifies the fermion-to-qubit transformation.
      The choice of encoding affects the qubit Hamiltonian structure but not the final energy.
      The default encoding should be used unless there is specific justification.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        hamiltonian_filename (str): Name of the file containing the fermionic Hamiltonian in the current directory
        out_qubit_hamiltonian_filename (str): Name of the file where the output qubit Hamiltonian will be saved
        algorithm_name (str, optional): The name of the qubit mapper algorithm to use, if overriding the default
            (options: "qiskit")
        settings (Dict, optional): A dictionary of key, value pairs specifying which settings keys to replace
            with specific values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where qubit Hamiltonian was saved
        OR error message if there was a problem in the workflow.

    """
    # Strip filenames in case full path is passed
    hamiltonian_filename = _strip(hamiltonian_filename)
    out_qubit_hamiltonian_filename = _strip(out_qubit_hamiltonian_filename)
    out_qubit_hamiltonian_filename, _err = _prepare_output(
        out_qubit_hamiltonian_filename, "QubitHamiltonian", data.QubitHamiltonian, overwrite=overwrite
    )
    if _err:
        return _err

    hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
    if _err:
        return _err

    qubit_mapper = algorithms.create("qubit_mapper", algorithm_name)

    _apply_settings(qubit_mapper, settings)

    n_spatial = hamiltonian.get_one_body_integrals()[0].shape[0]
    mapping = data.MajoranaMapping.jordan_wigner(num_modes=2 * n_spatial)

    # run qubit mapping
    qubit_hamiltonian = _run_algorithm(
        qubit_mapper,
        hamiltonian,
        mapping,
        cache=cache,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(qubit_hamiltonian, out_qubit_hamiltonian_filename)

    return out_qubit_hamiltonian_filename


@app.tool()
@_structured
@validate_project
def run_state_preparation(
    project_name: str,
    wavefunction_filename: str,
    out_circuit_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Prepare a quantum circuit that encodes the given wavefunction.

    This method transforms a multi-configurational wavefunction into a quantum circuit that prepares
    the corresponding quantum state on a quantum computer. State preparation is a critical step for
    quantum algorithms like Quantum Phase Estimation (QPE) and serves as a practical benchmark for
    quantum hardware fidelity.

    Typical workflow context:

    Two paths depending on system size:

    **Full-space path** (small systems, up to ~16 spatial orbitals / ~20 qubits):
    1. Run `run_scf` → extract orbitals → `run_hamiltonian_constructor` on full SCF orbitals
    2. `run_qubit_mapper` → qubit Hamiltonian
    3. (THIS TOOL) `run_state_preparation` from the SCF wavefunction
    4. `run_phase_estimation` or `run_resource_estimation`

    **Active-space path** (larger systems):
    1. Run `run_scf` → active space analysis (SCI + AutoCAS)
    2. (Optional) Sparsify wavefunction via `run_projected_multi_configuration_calculation`
    3. Extract orbitals → `run_hamiltonian_constructor` → `run_qubit_mapper`
    4. (THIS TOOL) `run_state_preparation` from the (sparse) multi-configurational wavefunction
    5. `run_phase_estimation` or `run_resource_estimation`

    Ask the user which approach they prefer if it's not clear from the system size.

    Usage guidelines:

    - The default state preparation method can be extracted using
      the function and MCP tool `get_algorithm_default_type`.
    - The sparse isometry method is recommended for
      multi-configurational wavefunctions as it exploits sparsity in
      the wavefunction to produce more efficient circuits. Regular
      isometry generates deeper circuits and therefore is in most
      cases not computationally feasible.
    - The wavefunction must have symmetric active spaces (same number of alpha and beta orbitals).
    - The trade-off is between circuit depth and overlap with the true ground state

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool
      `get_algorithm_default_settings`.
    - Key settings include `basis_gates`, `transpile` and `transpile_optimization_level`.
      For example, to use a custom gate set::

          settings = {"basis_gates": ["h", "cx", "rz"], "transpile_optimization_level": 1}

    - In most cases, the default settings are sufficient and should not be modified unless you have
      specific hardware constraints or optimization requirements.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        wavefunction_filename (str): Name of the file containing the target wavefunction in the current directory
        out_circuit_filename (str): Name of the file where the output circuit will be saved
        algorithm_name (str, optional): The name of the state preparation algorithm to use, if overriding the default
        settings (Dict, optional): A dictionary of key, value pairs specifying which settings keys to replace
            with specific values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where circuit was saved
        OR error message if there was a problem in the workflow.

    """
    # Strip filenames in case full path is passed
    wavefunction_filename = _strip(wavefunction_filename)
    out_circuit_filename = _strip(out_circuit_filename)
    out_circuit_filename, _err = _prepare_output(out_circuit_filename, "Circuit", data.Circuit, overwrite=overwrite)
    if _err:
        return _err

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    state_prep = algorithms.create("state_prep", algorithm_name)

    _apply_settings(state_prep, settings)

    # run state preparation
    circuit = _run_algorithm(
        state_prep,
        wavefunction,
        cache=cache,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(circuit, out_circuit_filename)

    return out_circuit_filename


@app.tool()
@_structured
@validate_project
def run_resource_estimation(
    project_name: str,
    circuit_filename: str,
    out_resource_estimator_data_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Estimate the quantum resources required to execute a circuit.

    This tool calls :meth:`qdk_chemistry.data.Circuit.estimate` on a saved circuit
    and writes the estimator result to a ``.resource_estimator_data.json`` file.

    The input is any circuit file — from molecular workflows (``run_state_preparation``,
    ``run_controlled_evolution_circuit_mapper``), model Hamiltonian pipelines, spin
    models, or any other source that produces a ``.circuit.json`` file.

    Usage guidelines:

    - The circuit must contain Q# factory data or an OpenQASM representation,
      which are the formats supported by ``Circuit.estimate``.
    - The ``settings`` dictionary is passed directly as estimator parameters.
    - The estimator returns both **logical** counts (backend-agnostic,
      pre-QEC) and **physical** counts (post-QEC, architecture-dependent).
      Always report both when presenting results to the user.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        circuit_filename (str): Name of the file containing the input circuit
        out_resource_estimator_data_filename (str): Name of the file where resource
            estimation results will be saved
        algorithm_name (str, optional): Reserved for compatibility. Resource estimation
            uses ``Circuit.estimate`` and does not select an algorithm implementation.
        settings (Dict, optional): Estimator parameters forwarded to ``Circuit.estimate``
        cache (str, optional): Cache backend identifier for result caching
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where resource-estimation JSON was saved
        OR error message if there was a problem with the workflow.

    """
    if algorithm_name:
        return "ERROR: algorithm_name is not supported for resource estimation; this tool uses Circuit.estimate()."

    circuit_filename = _strip(circuit_filename)
    out_resource_estimator_data_filename, _err = _prepare_resource_estimation_output(
        out_resource_estimator_data_filename, overwrite=overwrite
    )
    if _err:
        return _err

    circuit, _err = _load_or_error(circuit_filename, data.Circuit, "circuit")
    if _err:
        return _err

    if cache is not None:
        return "ERROR: cache is not supported for resource estimation via Circuit.estimate()."

    resource_estimator_data = _json_safe(circuit.estimate(settings))
    with open(out_resource_estimator_data_filename, "w", encoding="utf-8") as f:
        json.dump(resource_estimator_data, f, indent=2)

    return out_resource_estimator_data_filename


@app.tool()
@_structured
@validate_project
def run_time_evolution_builder(
    project_name: str,
    qubit_hamiltonian_filename: str,
    evolution_time: float,
    out_time_evolution_unitary_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Build a time evolution unitary U = exp(-iHt) from a qubit Hamiltonian.

    This tool runs the time evolution builder algorithm to construct the unitary operator
    that evolves a quantum state under the given Hamiltonian for a specified time. The resulting
    `TimeEvolutionUnitary` data object can then be passed to `run_controlled_evolution_circuit_mapper`
    to build the controlled circuit needed for QPE.

    IMPORTANT: This is an individual algorithm step in the QPE pipeline.

    Use these individual run tools to build up the QPE circuit step by step without executing the
    full phase estimation. This lets you inspect intermediate results at each stage.

    Typical workflow context:

    The QPE circuit construction pipeline is:

    1. (THIS TOOL) Build time evolution unitary: `run_time_evolution_builder`
    2. Build controlled evolution circuit: `run_controlled_evolution_circuit_mapper`
    3. (Optional) Execute the circuit: `run_circuit_executor`
    4. (Optional) Run the full QPE for an eigenvalue estimate: `run_phase_estimation`

    Usage guidelines:

    - The default time evolution builder can be queried using
      `get_algorithm_default_type("hamiltonian_unitary_builder")`.
    - The default settings can be queried using
      `get_algorithm_default_settings("hamiltonian_unitary_builder")`.
    - Key settings include `num_trotter_steps` and `order` for the Trotter decomposition.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        qubit_hamiltonian_filename (str): Name of the file containing the qubit Hamiltonian
        evolution_time (float): Time parameter t for U = exp(-iHt)
        out_time_evolution_unitary_filename (str): Name of the file where the time evolution unitary will be saved
        algorithm_name (str, optional): The name of the time evolution builder algorithm to use
        settings (Dict, optional): Settings overrides for the time evolution builder
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where time evolution unitary was saved
        OR error message if there was a problem

    """
    qubit_hamiltonian_filename = _strip(qubit_hamiltonian_filename)
    out_time_evolution_unitary_filename = _strip(out_time_evolution_unitary_filename)
    try:
        out_time_evolution_unitary_filename = ensure_filename_format(
            out_time_evolution_unitary_filename, "UnitaryRepresentation"
        )
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    if not overwrite:
        existing_check = check_output_exists(out_time_evolution_unitary_filename, UnitaryRepresentation)
        if existing_check:
            return existing_check

    try:
        qubit_hamiltonian = load_data_object(qubit_hamiltonian_filename, data.QubitHamiltonian)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load qubit hamiltonian from {qubit_hamiltonian_filename}: {e!s}"

    # Create the evolution builder algorithm
    evolution_builder = algorithms.create("hamiltonian_unitary_builder", algorithm_name)
    _apply_settings(evolution_builder, settings)
    evolution_builder.settings().set("time", evolution_time)

    try:
        time_evolution_unitary = _run_algorithm(
            evolution_builder,
            qubit_hamiltonian,
            cache=cache,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as e:
        return f"Time evolution builder failed: {e!s}"

    try:
        save_data_object(time_evolution_unitary, out_time_evolution_unitary_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save time evolution unitary to {out_time_evolution_unitary_filename}: {e!s}"

    return out_time_evolution_unitary_filename


@app.tool()
@_structured
@validate_project
def run_controlled_evolution_circuit_mapper(
    project_name: str,
    time_evolution_unitary_filename: str,
    out_circuit_filename: str,
    control_indices: list[int] | None = None,
    power: int = 1,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Map a time evolution unitary to a controlled quantum circuit.

    This tool takes a `TimeEvolutionUnitary` (from `run_time_evolution_builder`) and produces
    a quantum `Circuit` implementing the controlled unitary U^power, where U = exp(-iHt).
    This is the controlled circuit that forms the core of QPE iterations.

    IMPORTANT: This is an individual algorithm step in the QPE pipeline.

    Use these individual run tools to build up the QPE circuit step by step without executing the
    full phase estimation. This lets you inspect intermediate results at each stage.

    Typical workflow context:

    The QPE circuit construction pipeline is:

    1. Build time evolution unitary: `run_time_evolution_builder`
    2. (THIS TOOL) Build controlled evolution circuit: `run_controlled_evolution_circuit_mapper`
    3. (Optional) Execute the circuit: `run_circuit_executor`
    4. (Optional) Run the full QPE for an eigenvalue estimate: `run_phase_estimation`

    Usage guidelines:

    - The default circuit mapper can be queried using
      `get_algorithm_default_type(
      "controlled_evolution_circuit_mapper")`.
    - The default settings can be queried using
      `get_algorithm_default_settings(
      "controlled_evolution_circuit_mapper")`.
    - The `power` parameter controls the exponent in U^power. In QPE, different iterations use
      powers of 2 (e.g., U^1, U^2, U^4, ...).
    - The output circuit can be visualized using `visualize_circuit`.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        time_evolution_unitary_filename (str): Name of the file containing the time evolution unitary
            from `run_time_evolution_builder`
        out_circuit_filename (str): Name of the file where the controlled circuit will be saved
        control_indices (List[int]): Indices of the control qubits (default: [0])
        power (int): The power to raise the controlled unitary to (default: 1)
        algorithm_name (str, optional): The name of the circuit mapper algorithm to use
        settings (Dict, optional): Settings overrides for the circuit mapper
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where the controlled evolution circuit was saved
        OR error message if there was a problem

    """
    control_indices = control_indices or [0]
    time_evolution_unitary_filename = _strip(time_evolution_unitary_filename)
    out_circuit_filename = _strip(out_circuit_filename)
    try:
        out_circuit_filename = ensure_filename_format(out_circuit_filename, "Circuit")
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    if not overwrite:
        existing_check = check_output_exists(out_circuit_filename, data.Circuit)
        if existing_check:
            return existing_check

    try:
        time_evolution_unitary = load_data_object(time_evolution_unitary_filename, UnitaryRepresentation)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load time evolution unitary from {time_evolution_unitary_filename}: {e!s}"

    # Create the circuit mapper algorithm
    circuit_mapper = algorithms.create("controlled_circuit_mapper", algorithm_name)
    _apply_settings(circuit_mapper, settings)

    circuit_mapper.settings().set("control_indices", control_indices)
    if "power" in circuit_mapper.settings():
        circuit_mapper.settings().set("power", power)

    try:
        circuit = circuit_mapper.run(time_evolution_unitary)
    except (RuntimeError, ValueError) as e:
        return f"Controlled evolution circuit mapping failed: {e!s}"

    try:
        save_data_object(circuit, out_circuit_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save circuit to {out_circuit_filename}: {e!s}"

    return out_circuit_filename


@app.tool()
@_structured
@validate_project
def run_circuit_executor(
    project_name: str,
    circuit_filename: str,
    shots: int,
    out_executor_data_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Execute a quantum circuit on a simulator or hardware backend.

    This tool runs a quantum circuit for a given number of shots and saves the execution results
    (measurement statistics) to a file.

    IMPORTANT: This is an individual algorithm step in the QPE pipeline.

    Use these individual run tools to build up the QPE circuit step by step without executing the
    full phase estimation. This lets you inspect intermediate results at each stage.

    Typical workflow context:

    The QPE circuit construction pipeline is:

    1. Build time evolution unitary: `run_time_evolution_builder`
    2. Build controlled evolution circuit: `run_controlled_evolution_circuit_mapper`
    3. (THIS TOOL) Execute the circuit: `run_circuit_executor`
    4. (Optional) Run the full QPE for an eigenvalue estimate: `run_phase_estimation`

    Usage guidelines:

    - The default circuit executor can be queried using `get_algorithm_default_type("circuit_executor")`.
    - The default settings can be queried using `get_algorithm_default_settings("circuit_executor")`.
    - More shots yield better measurement statistics but increase simulation time.
    - The input circuit can be any quantum circuit (e.g., from `run_state_preparation`,
      `run_controlled_evolution_circuit_mapper`, or any other circuit source).

    Args:
        project_name (str): Name of the current qdk/chemistry project
        circuit_filename (str): Name of the file containing the quantum circuit to execute
        shots (int): Number of measurement shots to execute
        out_executor_data_filename (str): Name of the file where execution results will be saved
        algorithm_name (str, optional): The name of the circuit executor algorithm to use
        settings (Dict, optional): Settings overrides for the circuit executor
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where circuit execution data was saved
        OR error message if there was a problem

    """
    circuit_filename = _strip(circuit_filename)
    out_executor_data_filename = _strip(out_executor_data_filename)
    try:
        out_executor_data_filename = ensure_filename_format(out_executor_data_filename, "CircuitExecutorData")
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    if not overwrite:
        existing_check = check_output_exists(out_executor_data_filename, CircuitExecutorData)
        if existing_check:
            return existing_check

    circuit, _err = _load_or_error(circuit_filename, data.Circuit, "circuit")
    if _err:
        return _err

    # Create the circuit executor algorithm
    executor = algorithms.create("circuit_executor", algorithm_name)
    _apply_settings(executor, settings)

    try:
        executor_data = _run_algorithm(
            executor,
            circuit,
            shots,
            cache=cache,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as e:
        return f"Circuit execution failed: {e!s}"

    try:
        save_data_object(executor_data, out_executor_data_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save executor data to {out_executor_data_filename}: {e!s}"

    return out_executor_data_filename


@app.tool()
@_structured
@validate_project
def run_phase_estimation(
    project_name: str,
    state_prep_circuit_filename: str,
    qubit_hamiltonian_filename: str,
    out_qpe_result_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str:
    """Run Quantum Phase Estimation to estimate eigenvalues of a qubit Hamiltonian.

    This method executes either iterative QPE (IQPE) or traditional QFT-based QPE on a given
    state preparation circuit and qubit Hamiltonian.

    IMPORTANT: Only use this tool if you actually need the QPE eigenvalue result.

    For step-by-step inspection of the QPE pipeline, use the individual ``run_*`` tools
    (``run_time_evolution_builder``, ``run_controlled_evolution_circuit_mapper``,
    ``run_circuit_executor``) instead. This tool combines all three into a single execution.

    The sub-algorithms (time evolution builder, controlled evolution circuit mapper, circuit executor)
    are configured as nested algorithm references in the QPE settings.  Pass a dict inside
    ``settings`` keyed by the sub-algorithm name::

        settings={
            "num_bits": 10,
            "evolution_time": 1.0,
            "evolution_builder": {"algorithm_name": "trotter", "order": 2},
            "circuit_mapper": {"algorithm_name": "pauli_sequence"},
            "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator", "seed": 42},
        }

    Each dict accepts ``algorithm_name`` (optional, defaults to the built-in default) plus
    any setting keys supported by that algorithm.  When omitted, the QPE algorithm uses its
    built-in defaults (Trotter builder, Pauli-sequence mapper, QDK sparse-state simulator).

    Note that the cost of QPE scales with:

    - Number of qubits (determined by active space size via qubit mapping)
    - Circuit depth (determined by Hamiltonian complexity and state preparation)
    - Number of phase bits and shots per iteration

    To make QPE tractable, it is wise to use cost-reduction strategies in the preceding steps that generate
    the qubit hamiltonian.

    Typical workflow context:

    Two paths depending on system size:

    **Full-space path** (small systems, up to ~16 spatial orbitals / ~20 qubits):
    1. Run `run_scf` → extract orbitals → `run_hamiltonian_constructor` → `run_qubit_mapper`
    2. `run_state_preparation` from the SCF wavefunction
    3. (THIS TOOL) `run_phase_estimation` with sub-algorithm overrides in ``settings``

    **Active-space path** (larger systems):
    1. Run `run_scf` → active space analysis (SCI + AutoCAS) → compress orbital space
    2. (Optional) Sparsify wavefunction → `run_hamiltonian_constructor` → `run_qubit_mapper`
    3. `run_state_preparation` from the (sparse) multi-configurational wavefunction
    4. (THIS TOOL) `run_phase_estimation` with sub-algorithm overrides in ``settings``

    Ask the user which approach they prefer if it's not clear from the system size.

    Usage guidelines:

    - Prefer using the individual ``run_*`` tools when you want to inspect each QPE component.
      Only call this tool when you need to execute the full QPE and obtain an eigenvalue estimate.
    - The default phase estimation algorithm can be extracted using
      the function and MCP tool `get_algorithm_default_type`.
    - Iterative QPE (default, `algorithm_name="qdk_iterative"`) is
      recommended for near-term quantum hardware as it uses only
      1 ancilla qubit and processes phase bits sequentially with
      feedback.
    - Traditional QPE (`algorithm_name="qdk_standard"`) uses QFT
      and measures all phase bits in parallel but requires more
      qubits (equal to num_bits).
    - The state preparation circuit should prepare a state with good overlap with the target eigenstate.
    - The three dependency algorithms (time evolution builder, controlled evolution circuit mapper, circuit executor)
      are configured inline via the ``settings`` dict.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - Key settings include:

        - `num_bits` (int): Number of phase estimation bits
           (precision). Default: -1. IMPORTANT: this default value
           is not a valid setting - you need to pass a valid value
           for the number of bits.
        - `evolution_time` (float): Time parameter t for
           U = exp(-iHt). Default: 0.0. IMPORTANT: this default
           value is not a valid setting - you need to adjust based
           on the eigenvalue range - use smaller times for larger
           energy differences.
        - `shots_per_bit` (int, iterative only): Measurement shots per bit iteration. Default: 3.
        - `shots` (int, traditional only): Total measurement shots. Default: 3.
        - `evolution_builder` (dict, optional): Override the time-evolution builder,
           e.g. ``{"algorithm_name": "trotter", "order": 2}``.
        - `circuit_mapper` (dict, optional): Override the controlled-evolution circuit mapper.
        - `circuit_executor` (dict, optional): Override the circuit executor,
           e.g. ``{"algorithm_name": "qdk_full_state_simulator", "seed": 123}``.
    - If calculations are taking too long, consider reducing the parameter defaults.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        state_prep_circuit_filename (str): Name of the file
            containing the circuit that prepares the trial quantum
            state ``|ψ⟩``
        qubit_hamiltonian_filename (str): Name of the file containing
            the qubit Hamiltonian whose eigenvalues to estimate
        out_qpe_result_filename (str): Name of the file where the QPE result will be saved
        algorithm_name (str, optional): The name of the phase estimation algorithm to use, if overriding the default
            (options: "qdk_iterative", "qdk_standard")
        settings (Dict, optional): A dictionary of key, value pairs specifying which settings keys to replace
            with specific values (overrides defaults). Must include `num_bits` and `evolution_time`.
            Sub-algorithm overrides (``evolution_builder``, ``circuit_mapper``, ``circuit_executor``) can be
            specified as nested dicts — see the docstring above for the format.
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        str: Filename where QPE result was saved
        OR error message if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    state_prep_circuit_filename = _strip(state_prep_circuit_filename)
    qubit_hamiltonian_filename = _strip(qubit_hamiltonian_filename)
    out_qpe_result_filename = _strip(out_qpe_result_filename)
    out_qpe_result_filename, _err = _prepare_output(
        out_qpe_result_filename, "QpeResult", data.QpeResult, overwrite=overwrite
    )
    if _err:
        return _err

    state_prep_circuit, _err = _load_or_error(state_prep_circuit_filename, data.Circuit, "circuit")
    if _err:
        return _err

    try:
        qubit_hamiltonian = load_data_object(qubit_hamiltonian_filename, data.QubitHamiltonian)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load qubit hamiltonian from {qubit_hamiltonian_filename}: {e!s}"

    # Create the phase estimation algorithm
    phase_estimation = algorithms.create("phase_estimation", algorithm_name)

    settings = dict(settings or {})
    qpe_builder_settings: dict[str, Any] = {}
    for key in ("num_bits", "num_iteration", "phase_correction"):
        if key in settings:
            qpe_builder_settings[key] = settings.pop(key)

    evolution_builder_settings = settings.pop("evolution_builder", {}) or {}
    if "evolution_time" in settings:
        evolution_builder_settings = {**evolution_builder_settings, "time": settings.pop("evolution_time")}
    if evolution_builder_settings:
        return (
            "Full QPE with nested unitary-builder settings is not supported by the current AlgorithmRef API. "
            "Use run_time_evolution_builder and run_controlled_evolution_circuit_mapper for circuit construction."
        )

    circuit_mapper_settings = settings.pop("circuit_mapper", None)
    if circuit_mapper_settings:
        qpe_builder_settings["controlled_circuit_mapper"] = circuit_mapper_settings

    circuit_executor_settings = settings.pop("circuit_executor", None)
    if circuit_executor_settings:
        settings["circuit_executor"] = circuit_executor_settings

    if qpe_builder_settings:
        settings["qpe_circuit_builder"] = qpe_builder_settings

    _apply_settings(phase_estimation, settings)

    qpe_builder_ref = phase_estimation.settings()["qpe_circuit_builder"]
    qpe_builder = algorithms.create(qpe_builder_ref.algorithm_type, qpe_builder_ref.algorithm_name)
    _apply_settings(qpe_builder, qpe_builder_settings)

    # Check validity of settings
    if qpe_builder.settings()["num_bits"] == -1:
        return (
            "You need to set num_bits for QPE. A higher value will "
            "result in a better, but more expensive calculation. "
            "Consider the size of the problem setting, and what is "
            "feasible."
        )

    try:
        unitary_builder_settings = qpe_builder_settings.get("unitary_builder", {})
        unitary_builder_ref = qpe_builder.settings()["unitary_builder"]
        unitary_builder = algorithms.create(unitary_builder_ref.algorithm_type, unitary_builder_ref.algorithm_name)
        _apply_settings(unitary_builder, unitary_builder_settings)
        evolution_time = unitary_builder.settings()["time"]
    except (KeyError, RuntimeError):
        evolution_time = 0.0
    if evolution_time == 0.0:
        return (
            "You need to set evolution_time for QPE. The default value of 0.0 is invalid. "
            "Choose a value based on the eigenvalue range of your Hamiltonian. "
            "A good starting point is evolution_time = 2*pi / (E_max - E_min), "
            "where E_max and E_min are estimated energy bounds. "
            "Typical values range from 0.1 to 10.0 depending on the system."
        )

    # Run phase estimation
    try:
        qpe_result = _run_algorithm(
            phase_estimation,
            state_prep_circuit,
            qubit_hamiltonian,
            cache=cache,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as e:
        return f"Phase estimation failed: {e!s}"

    # Save to file
    try:
        save_data_object(qpe_result, out_qpe_result_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save QPE result to {out_qpe_result_filename}: {e!s}"

    return out_qpe_result_filename


@app.tool()
@_structured
@validate_project
def run_scf(
    project_name: str,
    structure_filename: str,
    out_wavefunction_filename: str,
    charge: int,
    spin_multiplicity: int,
    basis_set: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run a self-consistent field (SCF) calculation.

    The self-consistent field procedure is used to minimize the total energy of a system in a given basis set.
    This method is used for Hartree Fock (HF) and Density Functional Theory (DFT) calculations, where the
    wavefunction is defined using a single Slater determinant of molecular orbitals.

    Typical workflow context:

    This is typically the first step (or only step) in a quantum chemistry workflow starting from a structure.

    For single-reference calculations (closed-shell molecules at equilibrium):

    1. (THIS TOOL) Run `run_scf`
    2. Optionally add dynamical correlation with `run_dynamical_correlation_calculator`

    For multi-reference calculations (bond breaking, open-shell, transition metals, excited states):

    1. (THIS TOOL) Run `run_scf` with HF (default method) to get initial orbitals
    2. Run `run_active_space_selector` to define the active space
    3. Build Hamiltonian with `run_hamiltonian_constructor` from the active-space orbitals
    4. Run `run_multi_configuration_calculation` or `run_multi_configuration_scf`

    For quantum simulation (QPE):

    - Follow the multi-reference workflow above
    - Possible additional sparsification steps - please refer to docs of `run_phase_estimation` for example
    - Continue with `run_qubit_mapper`, `run_state_preparation`, and `run_phase_estimation`

    Usage guidelines:

    - The default SCF solver can be extracted using the function and MCP tool `get_algorithm_default_type`.
    - For closed-shell molecules at equilibrium geometries, Hartree-Fock (HF) is a reasonable starting point.
    - For multi-reference workflows, always start with HF (not DFT) to get clean canonical orbitals.
    - For better accuracy in single-reference energy predictions, consider using DFT with an appropriate functional.
    - The spin multiplicity should be set correctly: 1 for singlet (closed-shell), 2 for doublet, 3 for triplet, etc.
    - Common basis sets include: "sto-3g" (minimal, fast), "def2-svp" (balanced), "cc-pvdz" or "cc-pvtz" (accurate).

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - The most important setting is `method`: the default is HF, but
      DFT functionals can be specified (e.g., "b3lyp", "pbe",
      "m06-2x").
    - Other useful settings include `max_iterations` for convergence
      control and `convergence_threshold` for energy tolerance.
    - For difficult convergence cases, consider adjusting `damping` or `level_shift` settings if available.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        structure_filename (str): Name of the file containing the input structure in the current directory
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        charge (int): Total charge of the system
        spin_multiplicity (int): Spin multiplicity of the system
        basis_set (str): Basis set to use in the calculation
        algorithm_name (str, optional): The name of the scf solver method to use, if overriding the default
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults).
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[float, str]: The total energy returned by the SCF
            procedure and the filename where wavefunction was saved

        str: Containing error message, if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    structure_filename = _strip(structure_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    structure, _err = _load_or_error(structure_filename, data.Structure, "structure")
    if _err:
        return _err

    scf_solver = algorithms.create("scf_solver", algorithm_name)

    _apply_settings(scf_solver, settings)

    try:
        (total_energy, wavefunction) = _run_algorithm(
            scf_solver,
            structure,
            charge,
            spin_multiplicity,
            basis_set,
            cache=cache,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as e:
        return f"SCF calculation failed: {e!s}"

    # save to file
    try:
        save_data_object(wavefunction, out_wavefunction_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save wavefunction to {out_wavefunction_filename}: {e!s}"

    return (total_energy, out_wavefunction_filename)


@app.tool()
@_structured
@validate_project
def run_multi_configuration_calculation(
    project_name: str,
    hamiltonian_filename: str,
    out_wavefunction_filename: str,
    n_active_alpha_electrons: int,
    n_active_beta_electrons: int | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run multi-configurational calculation.

    Multi-configuration methods provide an improvement to the
    single-determinant Hartree-Fock (HF) method by defining the
    wavefunction as a linear combination of multiple determinants.
    They are also more expensive than single-determinant
    (single-reference) methods, but necessary for an accurate
    treatment of static correlation.

    Classic examples of systems with strong static correlation are:

    (1) Stretched molecules close to their dissociation limit or any systems involving bond-breaking
    (2) Open shell systems like diradicaloid organic species
        such as ethylene twisting, trimethylenemethane, or
        benzynes
    (3) Long conjugated π-systems: increasing π-conjugation
        lowers HOMO-LUMO gaps, making multiple determinants
        nearly degenerate.
    (4) Transition metal complexes or oxides: due to d-orbital near-degeneracies
    (5) Excited states: these often yield near-degenerate determinants

    Typical workflow context:

    1. Run `run_scf` with HF (default method) to get initial orbitals
    2. Run `run_active_space_selector` to define the active space
    3. Access orbitals from wavefunction or construct Hamiltonian object (depending on step 4)
    4. (THIS TOOL) Run `run_multi_configuration_calculation` or `run_multi_configuration_scf`
    5. ... possible next steps for quantum simulation (see docs for `run_phase_estimation` for example)

    The number of active electrons can be obtained from the wavefunction's `get_active_num_electrons()` method,
    which returns (n_alpha, n_beta). For the valence space selector, this is computed automatically.

    Usage guidelines:

    - The default multi-configuration calculator can be extracted
      using the function and MCP tool `get_algorithm_default_type`.
    - The Hamiltonian should be constructed from orbitals with a
      defined active space (use `run_active_space_selector` first).
    - For closed-shell systems, `n_active_beta_electrons` can be
      omitted and will default to the same value as
      `n_active_alpha_electrons`.
    - For open-shell systems (radicals, triplets), specify both alpha and beta electron counts explicitly.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - If you need access to reduced density matrices (RDMs), set
      `calculate_one_rdm` and/or `calculate_two_rdm` to `True`.
    - To tighten energy convergence, adjust `ci_residual_tolerance` to a smaller value.
    - The `davidson_iterations` setting controls the maximum number of Davidson solver iterations.
    - In most cases, the default settings should be sufficient for standard calculations.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        hamiltonian_filename (str): Name of the file containing the input Hamiltonian in the current directory
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        n_active_alpha_electrons (int): How many (alpha) electrons are in the active space
        n_active_beta_electrons (int, optional): For unrestricted/
            open-shell systems, we can separately specify how many
            beta electrons are in the opposite spin channel active
            space
        algorithm_name (str, optional): If we want to override the
            default algorithm for the multi configuration
            calculation, its name is passed here
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[float, str]: The resultant total energy and filename where wavefunction was saved
        str: Containing error message, if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    hamiltonian_filename = _strip(hamiltonian_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
    if _err:
        return _err

    mc_calculator = algorithms.create("multi_configuration_calculator", algorithm_name)

    _apply_settings(mc_calculator, settings)

    try:
        if n_active_beta_electrons is None:
            (total_energy, wavefunction) = _run_algorithm(
                mc_calculator,
                hamiltonian,
                n_active_alpha_electrons,
                n_active_alpha_electrons,
                cache=cache,
                overwrite=overwrite,
            )
        else:
            (total_energy, wavefunction) = _run_algorithm(
                mc_calculator,
                hamiltonian,
                n_active_alpha_electrons,
                n_active_beta_electrons,
                cache=cache,
                overwrite=overwrite,
            )
    except (RuntimeError, ValueError) as e:
        return f"Multi-configuration calculation failed: {e!s}"

    # save to file
    try:
        save_data_object(wavefunction, out_wavefunction_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save wavefunction to {out_wavefunction_filename}: {e!s}"

    return (total_energy, out_wavefunction_filename)


@app.tool()
@_structured
@validate_project
def run_multi_configuration_scf(
    project_name: str,
    orbitals_filename: str,
    out_wavefunction_filename: str,
    n_active_alpha_electrons: int,
    n_active_beta_electrons: int | None = None,
    ham_constructor_algorithm_name: str | None = None,
    ham_constructor_settings: dict | None = None,
    mc_calculator_algorithm_name: str | None = None,
    mc_calculator_settings: dict | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run multi-configuration self consistent field (MCSCF) calculation.

    MCSCF methods simultaneously optimize both the molecular orbitals and the CI coefficients,
    providing a more balanced description of static correlation than CASCI.
    This is especially important when the initial orbitals are not optimal for the active space problem.

    Typical workflow:

    1. Run `run_scf` with HF (default method) to get initial orbitals
    2. Run `run_active_space_selector` to define the active space
    3. Access orbitals from wavefunction or construct Hamiltonian object (depending on step 4)
    4. (THIS TOOL) Run `run_multi_configuration_calculation` or `run_multi_configuration_scf`
    5. ... possible next steps for quantum simulation (see docs for `run_phase_estimation` for example)

    Note: Unlike `run_multi_configuration_calculation`, this tool takes orbitals directly (not a Hamiltonian)
    and builds the Hamiltonian internally during the orbital optimization process.

    Usage guidelines:

    - The default MCSCF solver can be extracted using the function and MCP tool `get_algorithm_default_type`.
    - The input orbitals should have a defined active space (use `run_active_space_selector` on a wavefunction first).
    - For closed-shell systems, `n_active_beta_electrons` can be omitted.

    Guidelines on settings:

    - The current set of default settings can be obtained by using
      the function and MCP tool `get_algorithm_default_settings`.
    - For the `pyscf` solver (default), key settings include:
        - `max_cycle_macro`: Maximum number of macro cycles
        - `max_cycle_micro`: Maximum number of micro (CI) cycles per macro iteration
        - `conv_tol`: Energy convergence tolerance
        - `verbose`: Set to higher values (e.g., 4 or 5) for detailed output
    - The `ham_constructor_settings` and `mc_calculator_settings` allow fine-tuning of the
      Hamiltonian construction and CI solver components respectively.
    - In most cases, the default settings provide good convergence behavior.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        orbitals_filename (str): Name of the file containing the input orbitals in the current directory
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        n_active_alpha_electrons (int): How many (alpha) electrons are in the active space
        n_active_beta_electrons (int, optional): For unrestricted/
            open-shell systems, we can separately specify how many
            beta electrons are in the opposite spin channel active
            space
        ham_constructor_algorithm_name (str, optional): Override
            default Hamiltonian constructor algorithm
        ham_constructor_settings (Dict, optional): Settings for Hamiltonian constructor
        mc_calculator_algorithm_name (str, optional): Override default multi-configuration calculator algorithm
        mc_calculator_settings (Dict, optional): Settings for multi-configuration calculator
        settings (Dict, optional): A dictionary of key, value pairs
            specifying which settings keys to replace with specific
            values for MCSCF (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[float, str]: The total energy of the final system and filename where wavefunction was saved
        str: Containing error message, if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    orbitals_filename = _strip(orbitals_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    orbitals, _err = _load_or_error(orbitals_filename, data.Orbitals, "orbitals")
    if _err:
        return _err

    # Create and configure mcscf_calculator
    mcscf_calculator = algorithms.create("multi_configuration_scf", "pyscf")
    mcscf_settings = dict(settings or {})
    if mc_calculator_algorithm_name or mc_calculator_settings:
        nested_mc_settings = dict(mc_calculator_settings or {})
        if mc_calculator_algorithm_name:
            nested_mc_settings["algorithm_name"] = mc_calculator_algorithm_name
        mcscf_settings["multi_configuration_calculator"] = nested_mc_settings
    _apply_settings(mcscf_calculator, mcscf_settings)

    try:
        if n_active_beta_electrons is None:
            (total_energy, wavefunction) = _run_algorithm(
                mcscf_calculator,
                orbitals,
                n_active_alpha_electrons,
                n_active_alpha_electrons,
                cache=cache,
                overwrite=overwrite,
            )
        else:
            (total_energy, wavefunction) = _run_algorithm(
                mcscf_calculator,
                orbitals,
                n_active_alpha_electrons,
                n_active_beta_electrons,
                cache=cache,
                overwrite=overwrite,
            )
    except (RuntimeError, ValueError) as e:
        return f"MCSCF calculation failed: {e!s}"

    # save to file
    try:
        save_data_object(wavefunction, out_wavefunction_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save wavefunction to {out_wavefunction_filename}: {e!s}"

    return (total_energy, out_wavefunction_filename)


@app.tool()
@_structured
@validate_project
def run_projected_multi_configuration_calculation(
    project_name: str,
    hamiltonian_filename: str,
    configurations_json: str,
    out_wavefunction_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run a projected multi-configuration calculation on a specific set of determinants.

    This tool performs a CI calculation restricted to the determinants specified in the configurations
    input. It diagonalizes the Hamiltonian in the space spanned by those determinants to obtain
    optimized CI coefficients and the corresponding energy.

    Typical workflow context:

    This tool is commonly used in quantum simulation workflows to compute the energy of a sparse
    wavefunction:

    1. Run `run_multi_configuration_calculation` to get the full CASCI wavefunction and energy
    2. Run `get_top_configurations` to extract the most important configurations (determinants)
    3. (THIS TOOL) Run `run_projected_multi_configuration_calculation` with those configurations
       to get the sparse wavefunction with optimized CI coefficients
    4. Use the sparse wavefunction for `run_state_preparation` to get a shorter quantum circuit
    5. Continue with `run_qubit_mapper` and `run_phase_estimation` for quantum phase estimation

    Usage guidelines:

    - The configurations should be provided as a JSON array of configuration strings, where each
      string represents the occupation pattern of the active orbitals.
    - Use `get_top_configurations` to extract configurations from a reference wavefunction.
    - The Hamiltonian must be compatible with the configurations (same active space size).
    - For quantum simulation, selecting fewer determinants reduces circuit depth but may reduce
      accuracy. A typical approach is to start with the top 5-20 determinants by CI coefficient.

    Guidelines on settings:

    - The current set of default settings can be obtained by using the function
      and MCP tool `get_algorithm_default_settings`.
    - For the `macis_pmc` calculator (default), relevant settings include:

        - `iterative_solver_dimension_cutoff`: Matrix size cutoff for switching to iterative eigensolver
        - `H_thresh`: Threshold for Hamiltonian element screening
        - `h_el_tol`: Tolerance for Hamiltonian element evaluation
        - `davidson_res_tol`: Residual tolerance for Davidson solver convergence
        - `davidson_max_m`: Maximum subspace dimension for Davidson solver
    - In most cases, the default settings should be sufficient unless you encounter convergence issues.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        hamiltonian_filename (str): Name of the file containing the input Hamiltonian in the current directory
        configurations_json (str): A JSON string containing an array of configuration strings.
            Each configuration string represents the occupation pattern of active orbitals.
            Example: '["22000000", "20200000", "20020000"]'
            Use `get_top_configurations` to obtain these from a reference wavefunction.
        out_wavefunction_filename (str): Name of the file where the output wavefunction will be saved
        algorithm_name (str, optional): Algorithm name for the projected multi-configuration calculator,
            to override the default
        settings (Dict, optional): A dictionary of key, value pairs specifying which settings keys
            to replace with specific values (overrides defaults)
        cache (str, optional): Cache backend identifier for result caching
            to complete before returning a job handle. Default ``120``.
        overwrite (bool): If ``True``, overwrite existing output files
            without prompting. Default ``False``.

    Returns:
        Tuple[float, str]: Calculated total energy and filename where wavefunction was saved
        str: Containing error message, if there was a problem in the workflow

    """
    # Strip filenames in case full path is passed
    hamiltonian_filename = _strip(hamiltonian_filename)
    out_wavefunction_filename = _strip(out_wavefunction_filename)
    out_wavefunction_filename, _err = _prepare_output(
        out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
    )
    if _err:
        return _err

    hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
    if _err:
        return _err

    # Parse configurations from JSON
    try:
        config_strings = json.loads(configurations_json)
        if not isinstance(config_strings, list):
            return "configurations_json must be a JSON array of configuration strings"
        if not config_strings:
            return "configurations_json array is empty"
        configurations = [data.Configuration.from_spin_half_string(s) for s in config_strings]
    except json.JSONDecodeError as e:
        return f"Invalid JSON in configurations_json: {e!s}"
    except (RuntimeError, ValueError) as e:
        return f"Failed to parse configurations: {e!s}"

    pmc_calculator = algorithms.create("projected_multi_configuration_calculator", algorithm_name)

    _apply_settings(pmc_calculator, settings)

    try:
        (total_energy, out_wavefunction) = _run_algorithm(
            pmc_calculator,
            hamiltonian,
            configurations,
            cache=cache,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as e:
        return f"Projected multi-configuration calculation failed: {e!s}"

    # save to file
    try:
        save_data_object(out_wavefunction, out_wavefunction_filename)
    except (RuntimeError, ValueError) as e:
        return f"Failed to save wavefunction to {out_wavefunction_filename}: {e!s}"

    return (total_energy, out_wavefunction_filename)


@app.tool()
@_structured
@validate_project
def get_top_configurations(
    project_name: str,
    wavefunction_filename: str,
    max_determinants: int | None = None,
) -> str:
    """Get the top configurations (determinants) from a wavefunction ranked by CI coefficient magnitude.

    This tool extracts configuration strings from a multi-configurational wavefunction, ranked by
    the absolute value of their CI coefficients. This is useful for identifying the most important
    determinants for sparse CI calculations or quantum simulation.

    Typical workflow context:

    This tool is used to prepare inputs for ``run_projected_multi_configuration_calculation``:

    1. Run ``run_multi_configuration_calculation`` to get the full CASCI wavefunction
    2. (THIS TOOL) Run ``get_top_configurations`` to extract the most important configurations
    3. Pass the returned JSON directly to ``run_projected_multi_configuration_calculation``
    4. Use the resulting sparse wavefunction for ``run_state_preparation``

    Usage guidelines:

    - The returned JSON array contains configurations sorted by CI coefficient magnitude (largest first).
    - If `max_determinants` is not specified, all determinants in the wavefunction are returned.
    - The configuration strings represent the occupation pattern of the active orbitals.
    - For quantum simulation, selecting fewer determinants reduces circuit depth. A typical
      approach is to use the top 5-20 determinants, which often captures most of the wavefunction.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        wavefunction_filename (str): Name of the file containing the input wavefunction
        max_determinants (int, optional): Maximum number of configurations to return.
            If None, returns all configurations in the wavefunction.

    Returns:
        str: A JSON array of configuration strings, sorted by CI coefficient magnitude (largest first).
            This can be passed directly to `run_projected_multi_configuration_calculation`.
            Example: '["22000000", "20200000", "20020000"]'
            On error, returns an error message string (not valid JSON).

    """
    # Strip filename in case full path is passed
    wavefunction_filename = _strip(wavefunction_filename)

    wavefunction, _err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if _err:
        return _err

    # Get top determinants ranked by coefficient magnitude
    try:
        ranked_determinants = wavefunction.get_top_determinants(max_determinants=max_determinants)
        if not ranked_determinants:
            return f"No determinants found in wavefunction {wavefunction_filename}"

        # Extract configuration strings and return as JSON
        config_strings = [det.to_string() for det in ranked_determinants]
        return json.dumps(config_strings)
    except (RuntimeError, ValueError, AttributeError) as e:
        return f"Failed to extract configurations from wavefunction: {e!s}"


@app.tool()
@_structured
@validate_project
def get_circuit_stats(
    project_name: str,
    circuit_filename: str,
) -> dict | str:
    """Get resource statistics for a quantum circuit.

    Analyzes a saved Circuit file and returns gate counts, depth, qubit count,
    and Clifford/non-Clifford gate breakdowns. Use this after any step that
    produces a circuit (e.g., ``run_state_preparation``,
    ``run_controlled_evolution_circuit_mapper``) to understand the resource
    profile before proceeding.

    Typical workflow context:

    Call this tool after building a circuit to inspect its resource cost:

    1. ``run_state_preparation`` → produces state-prep circuit
    2. (THIS TOOL) ``get_circuit_stats`` → gate counts, depth, qubit count
    3. ``run_time_evolution_builder`` → produces time-evolution unitary
    4. ``run_controlled_evolution_circuit_mapper`` → produces controlled-U circuit
    5. (THIS TOOL) ``get_circuit_stats`` → controlled-U resource profile

    The returned statistics include:

    - ``num_qubits``: Number of qubits in the circuit
    - ``depth``: Circuit depth (longest path through the circuit)
    - ``total_gates``: Total number of gates
    - ``gate_counts``: Breakdown by gate type (e.g., ``{"cx": 12, "rz": 6, "h": 4}``)
    - ``single_qubit_clifford``: Count of single-qubit Clifford gates (H, S, X, Y, Z, etc.)
    - ``two_qubit_clifford``: Count of two-qubit Clifford gates (CNOT, CZ, etc.)
    - ``non_clifford``: Count of non-Clifford gates (T, Rz, etc.) — these dominate fault-tolerant cost

    All metrics are in terms of **logical qubits** — abstract, error-free qubits.

    Args:
        project_name (str): Name of the current qdk/chemistry project
        circuit_filename (str): Name of the file containing the circuit
            (e.g., ``"state_prep.circuit.json"``)

    Returns:
        Dict: Circuit statistics including qubit count, depth, gate breakdown
        str: Error message if there was a problem

    """
    circuit_filename = _strip(circuit_filename)

    circuit, _err = _load_or_error(circuit_filename, data.Circuit, "circuit")
    if _err:
        return _err

    try:
        from qdk_chemistry.plugins.qiskit._interop.circuit import CircuitInfo  # noqa: PLC0415

        qiskit_circuit = circuit.get_qiskit_circuit()
        info = CircuitInfo(circuit=qiskit_circuit)
        stats = info.summary()
        stats["gate_counts"] = dict(info.gate_counts)
        return stats
    except Exception as e:  # noqa: BLE001
        return f"Failed to analyze circuit {circuit_filename}: {e!s}"
