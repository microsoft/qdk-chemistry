"""MCP tools for the qdk_chemistry toolkit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# ruff: noqa: ARG001, PLR0911
# ARG001: All MCP tool functions accept ``project_name`` which is consumed by
# the ``@validate_project`` decorator before the function body runs.
# PLR0911: MCP tools use early-return error handling at each validation step,
# which legitimately requires many return statements.

import concurrent.futures
import functools
import hashlib
import inspect
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qdk_chemistry import algorithms, constants, data
from qdk_chemistry.data import AlgorithmRef as _AlgorithmRef

# Remote execution support — optional; the MCP server works without it.
try:
    from qdk_chemistry.remote.backends.base import available_backends, get_mcp_safe_config_options
    from qdk_chemistry.remote.cache import available_caches, resolve_cache
    from qdk_chemistry.remote.cache.folder import FolderCache
    from qdk_chemistry.remote.proxy import run as _remote_run

    _REMOTE_AVAILABLE = True
except Exception:  # noqa: BLE001  # ImportError, ModuleNotFoundError, or build issues
    _REMOTE_AVAILABLE = False
from qdk_chemistry.data.circuit_executor_data import CircuitExecutorData as _CircuitExecutorData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation as _UnitaryRepresentation
from qdk_chemistry.remote.job import Job
from qdk_chemistry.utils import (
    compute_valence_space_parameters,
)

from ._mcp import MCP_AVAILABLE, app
from ._mcp import MCPContext as _Context
from .config import config
from .io import (
    check_output_exists,
    check_output_path_exists,
    load_data_object,
    save_data_object,
)
from .validation import (
    FilenameFormatError,
    current_project_name,
    ensure_filename_format,
    resolve_project_file,
    resolve_project_path,
    strip_filename_path,
    validate_project,
)
from .workspace import bind_workspace as _bind_workspace
from .workspace import current_workspace_root, workspace_binding_middleware

_REMOTE_WAITER_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1024,
    thread_name_prefix="qdk-chemistry-remote",
)

if MCP_AVAILABLE:
    # Register MCP Apps visualization tools (interactive UI via ui:// resources).
    from .visualization import register_visualization_tools

    register_visualization_tools(app)

    app.middleware.append(workspace_binding_middleware)


@app.tool()
async def bind_workspace(ctx: _Context, workspace_root: str | None = None) -> dict[str, object]:
    """Bind this MCP process to a workspace root."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP support is unavailable")
    return await _bind_workspace(ctx, workspace_root)


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


class _JobSubmittedError(Exception):
    """Raised by ``_run_algorithm`` when a remote job is still running.

    Caught by ``_structured`` to produce a ``{"status": "submitted", ...}``
    envelope instead of letting the tool body crash on tuple unpacking.

    """

    def __init__(self, job: Job):
        self.job = job
        super().__init__(f"Job {job.job_id} submitted but not yet complete.")


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
        except _JobSubmittedError as sig:
            job = sig.job
            return {
                "status": "submitted",
                "message": "Job submitted but not yet complete. Use check_remote_job to monitor.",
                "job": {
                    "job_id": job.job_id,
                    "job_file": str(job.file_path) if job.file_path else None,
                    "backend": job.backend,
                    "status": job.status,
                    "run_hash": job.run_hash,
                    "submitted_at": job.submitted_at,
                },
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e), "error_type": type(e).__name__}
        return _wrap_result(result)

    # FastMCP reads the wrapper docstring when the tool is registered. Keep the
    # published description to the mechanical summary; detailed interface
    # information lives in the qdk-chemistry-mcp skill.
    wrapper.__doc__ = (func.__doc__ or "").strip().split("\n\n", maxsplit=1)[0]

    # Preserve the original function's parameter signature so that
    # MCPServer (which calls inspect.signature()) can build a correct
    # tool schema.  inspect.signature checks __signature__ first,
    # before following __wrapped__, so this survives the __wrapped__
    # deletion below.
    original_sig = inspect.signature(func)
    wrapper.__signature__ = original_sig.replace(return_annotation=dict[str, Any])  # type: ignore[attr-defined]

    # Override the return annotation so MCPServer/Pydantic validates against
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


def _validate_mcp_remote_config(name: str, remote_config: dict[str, Any]) -> None:
    """Reject backend constructor options that MCP clients may not control."""
    allowed = get_mcp_safe_config_options(name)
    unsupported = remote_config.keys() - allowed
    if unsupported:
        allowed_message = ", ".join(sorted(allowed)) or "none"
        raise ValueError(
            f"remote_config contains options that MCP clients cannot control for backend '{name}': "
            f"{', '.join(sorted(unsupported))}. Allowed options: {allowed_message}."
        )
    for option in {"poll_interval", "timeout"} & remote_config.keys():
        value = remote_config[option]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"remote_config '{option}' must be a finite, positive number for MCP clients.")


def _strip(filename: str) -> str:
    """Strip directory path from a filename, keeping only the base name."""
    return strip_filename_path(filename)


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


def _load_driven_hamiltonian(
    base_hamiltonian_filename: str,
    drive_hamiltonian_filename: str,
    drive_times: list[float],
    drive_values: list[float],
):
    """Build a driven Hamiltonian from two operators and a linear schedule."""
    base_hamiltonian, error = _load_or_error(base_hamiltonian_filename, data.QubitOperator, "base Hamiltonian")
    if error:
        return None, error
    drive_hamiltonian, error = _load_or_error(drive_hamiltonian_filename, data.QubitOperator, "drive Hamiltonian")
    if error:
        return None, error
    if len(drive_times) != len(drive_values) or not drive_times:
        return None, "drive_times and drive_values must be non-empty lists of equal length."
    if any(right <= left for left, right in itertools.pairwise(drive_times)):
        return None, "drive_times must be strictly increasing."

    times = np.asarray(drive_times, dtype=float)
    values = np.asarray(drive_values, dtype=float)

    def drive(time: float) -> float:
        return float(np.interp(time, times, values))

    try:
        return data.DrivenQubitHamiltonian(base_hamiltonian, drive_hamiltonian, drive=drive), None
    except (TypeError, ValueError) as error:
        return None, f"Failed to construct driven Hamiltonian: {error!s}"


def _dict_to_algorithm_ref(existing_ref, override_dict: dict):
    """Convert a plain dict into an ``AlgorithmRef``.

    The *existing_ref* supplies the immutable ``algorithm_type``. The dictionary
    may use either flattened setting overrides or the serialized form returned
    by :func:`get_algorithm_default_settings`, including its nested ``settings``
    mapping. An ``algorithm_name`` override selects a different implementation.

    Example dict::

        {"algorithm_name": "trotter", "order": 2, "target_accuracy": 1e-6}
    """
    d = dict(override_dict)
    d.pop("__type__", None)
    algorithm_type = d.pop("algorithm_type", existing_ref.algorithm_type)
    if algorithm_type != existing_ref.algorithm_type:
        raise ValueError(
            f"Nested algorithm type {algorithm_type!r} does not match expected type {existing_ref.algorithm_type!r}."
        )
    nested_settings = d.pop("settings", None)
    if nested_settings is not None:
        if not isinstance(nested_settings, dict):
            raise TypeError("Nested algorithm 'settings' must be a dictionary.")
        d = {**nested_settings, **d}
    algorithm_name = d.pop("algorithm_name", None) or existing_ref.algorithm_name
    ref = _AlgorithmRef(existing_ref.algorithm_type, algorithm_name)
    try:
        default_settings = algorithms.create(existing_ref.algorithm_type, algorithm_name).settings()
    except Exception:  # noqa: BLE001
        default_settings = None
    for k, override_value in d.items():
        value_to_set = override_value
        if isinstance(override_value, dict) and default_settings is not None:
            try:
                nested_ref = default_settings.get(k)
            except (KeyError, RuntimeError):
                nested_ref = None
            if isinstance(nested_ref, _AlgorithmRef):
                value_to_set = _dict_to_algorithm_ref(nested_ref, override_value)
        ref.settings.set(k, value_to_set)
    return ref


def _apply_settings(algorithm, settings: dict | None) -> None:
    """Apply a settings dict to an algorithm instance (no-op if None/empty).

    When a value is a ``dict`` and the corresponding setting currently
    holds an ``AlgorithmRef``, the dict is automatically converted into
    a new ``AlgorithmRef`` (see :func:`_dict_to_algorithm_ref`).  This
    allows callers to configure nested algorithms inline::

        settings={
            "qpe_circuit_builder": {
                "algorithm_name": "qdk_iterative",
                "num_bits": 10,
                "unitary_builder": {"algorithm_name": "trotter", "time": 1.0},
            },
        }
    """
    for key, value in (settings or {}).items():
        if isinstance(value, dict):
            existing = algorithm.settings().get(key)
            if isinstance(existing, _AlgorithmRef):
                algorithm.settings().set(key, _dict_to_algorithm_ref(existing, value))
                continue
        algorithm.settings().set(key, value)


def _jsonable_settings_value(value: Any) -> Any:
    """Convert settings values into MCP/JSON-friendly data."""
    if isinstance(value, _AlgorithmRef):
        serialized: dict[str, Any] = {
            "__type__": "algorithm_ref",
            "algorithm_type": value.algorithm_type,
            "algorithm_name": value.algorithm_name,
        }
        if value.settings is not None:
            serialized["settings"] = _jsonable_settings_dict(value.settings.to_dict())
        return serialized
    if isinstance(value, dict):
        return {str(key): _jsonable_settings_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_settings_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable_settings_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable_settings_dict(settings: dict[str, Any]) -> dict[str, Any]:
    """Convert a Settings.to_dict() mapping into MCP/JSON-friendly data."""
    return {key: _jsonable_settings_value(value) for key, value in settings.items()}


def _run_algorithm(
    algorithm, *args, cache=None, remote=None, remote_config=None, remote_timeout=120, overwrite=False, **kwargs
):
    """Execute an algorithm with automatic caching.

    Every run is cached by default so that identical computations are
    never repeated.  When no explicit *cache* is provided, a
    ``FolderCache(config.cache_dir)`` is used (typically
    ``<scratch>/cache``).  If a user supplies their own cache (path,
    backend name, or ``CacheBackend`` instance), it is used instead —
    this includes shared caches, ``TieredCache``, CosmosDB, etc.

    When *remote* is set and *remote_timeout* is not ``None``, the SDK
    call runs in a background thread.  If it completes within
    *remote_timeout* seconds the result is returned inline; otherwise a
    :class:`_JobSubmittedError` is raised so the ``_structured``
    decorator can produce a ``{"status": "submitted", ...}`` envelope
    while the job continues running in the cache.
    """
    if not _REMOTE_AVAILABLE and (cache is not None or remote is not None):
        return "Remote/cache execution is not available. The qdk_chemistry.remote module could not be imported."

    # Always use a cache so that identical local or remote runs are
    # never recomputed.  When the caller supplies an explicit cache it
    # takes precedence; otherwise a default FolderCache under
    # config.cache_dir is used.
    if cache is not None:
        resolved_cache = resolve_cache(cache)
    elif _REMOTE_AVAILABLE:
        resolved_cache = FolderCache(path=config.cache_dir)
    else:
        # Remote module unavailable and no explicit cache — run without.
        return algorithm.run(*args, **kwargs)

    # If remote is a name string and remote_config is provided,
    # create a pre-configured backend instance using only options that
    # are safe for untrusted MCP clients to control. This layer owns its
    # lifecycle because the remote proxy treats backend instances as caller-owned.
    resolved_remote = remote
    owns_resolved_remote = False
    if isinstance(remote, str) and remote_config:
        from qdk_chemistry.remote.backends import get_backend  # noqa: PLC0415

        _validate_mcp_remote_config(remote, remote_config)
        resolved_remote = get_backend(remote, **remote_config)
        owns_resolved_remote = True

    sdk_kwargs: dict[str, Any] = {
        "remote": resolved_remote,
        "force_rerun": overwrite,
    }
    project_name = current_project_name()
    owner = None
    if project_name is not None:
        workspace_root = current_workspace_root()
        owner = {
            "workspace_root": str(workspace_root) if workspace_root is not None else None,
            "project_name": project_name,
        }
        sdk_kwargs["_owner"] = owner
    sdk_kwargs["cache"] = resolved_cache

    def run_remote() -> Any:
        if owns_resolved_remote:
            resolved_remote.connect()
        try:
            return _remote_run(algorithm, *args, **sdk_kwargs, **kwargs)
        finally:
            if owns_resolved_remote:
                resolved_remote.disconnect()

    # For remote jobs with a timeout, run in a thread so we can return
    # early if the job is still running. The timeout starts only after
    # the SDK has persisted a durable job handle.
    if remote is not None and remote_timeout is not None:
        try:
            run_hash = algorithm.hash(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Cannot submit a timed remote job because algorithm.hash() failed. "
                "Timed remote execution requires a deterministic run hash so its job handle "
                f"can be persisted and monitored. Original error: {exc}"
            ) from exc
        if not isinstance(run_hash, str) or not run_hash:
            raise RuntimeError(
                "Cannot submit a timed remote job because algorithm.hash() did not return a non-empty string. "
                "Timed remote execution requires a deterministic run hash so its job handle can be persisted "
                "and monitored."
            )

        submitted: concurrent.futures.Future[Job] = concurrent.futures.Future()

        def on_job_submitted(job: Job) -> None:
            if not submitted.done():
                job.save(_job_record_path(job, run_hash))
                submitted.set_result(job)

        sdk_kwargs["_on_job_submitted"] = on_job_submitted
        future = _REMOTE_WAITER_POOL.submit(run_remote)
        try:
            completed, _ = concurrent.futures.wait(
                (future, submitted),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if future in completed:
                result = future.result()
            else:
                job = submitted.result()
                try:
                    result = future.result(timeout=remote_timeout)
                except concurrent.futures.TimeoutError:
                    if future.done():
                        # The remote backend raised TimeoutError; this was not
                        # the MCP return-window deadline.
                        result = future.result()
                    else:
                        raise _JobSubmittedError(job) from None
        except concurrent.futures.TimeoutError:
            # A backend TimeoutError before durable submission propagates.
            raise
        return result

    # No timeout (cache-only, or blocking remote) — call directly
    return run_remote()


def _resolve_seed_or_basis(seed_or_basis: str):
    """Return a basis-name string or load a saved seed object from the project."""
    seed_or_basis = _strip(seed_or_basis)
    if not seed_or_basis.endswith((".json", ".hdf5", ".h5")):
        return seed_or_basis, None

    for cls, label in (
        (data.Wavefunction, "wavefunction"),
        (data.Orbitals, "orbitals"),
        (data.BasisSet, "basis set"),
    ):
        obj, err = _load_or_error(seed_or_basis, cls, label)
        if err is None:
            return obj, None
    return None, f"Failed to load seed_or_basis from {seed_or_basis} as a wavefunction, orbitals, or basis set."


# ═══════════════════════════════════════════════════════════════════════════
# Discovery / introspection tools
# ═══════════════════════════════════════════════════════════════════════════


@app.tool()
@_structured
def list_algorithms(algorithm_type: str | None = None) -> dict:
    """List registered algorithm implementations and defaults."""
    registered = algorithms.available()
    defaults = algorithms.show_default()
    if algorithm_type is not None:
        algorithm_type = algorithm_type.strip()
        if algorithm_type not in registered:
            available_types = ", ".join(sorted(registered)) or "none"
            raise ValueError(f"Unknown algorithm type {algorithm_type!r}. Available types: {available_types}.")
        registered = {algorithm_type: registered[algorithm_type]}

    return {
        "algorithm_types": {
            name: {
                "default": defaults.get(name) or None,
                "implementations": sorted(implementations),
            }
            for name, implementations in sorted(registered.items())
        }
    }


@app.tool()
@_structured
def describe_algorithm(algorithm_type: str, algorithm_name: str | None = None) -> dict:
    """Describe a registered algorithm implementation and its settings."""
    available_names = algorithms.available(algorithm_type)
    if not available_names:
        if algorithm_name is not None:
            raise ValueError(f"Algorithm type {algorithm_type!r} has no registered implementations.")
        return {
            "algorithm_type": algorithm_type,
            "requested_name": None,
            "name": None,
            "aliases": [],
            "is_default": False,
            "interface_only": True,
            "default_settings": {},
            "settings": [],
        }

    default_name = algorithms.show_default(algorithm_type)
    selected_name = algorithm_name
    using_fallback = selected_name is None and default_name not in available_names
    if selected_name is None:
        selected_name = default_name if default_name in available_names else sorted(available_names)[0]

    instance = algorithms.create(algorithm_type, selected_name)
    canonical_name = instance.name()
    if using_fallback:
        selected_name = canonical_name
    aliases = instance.aliases()
    setting_schema = []
    for name, python_type, default, description, limits in algorithms.inspect_settings(algorithm_type, selected_name):
        setting_schema.append(
            {
                "name": name,
                "type": python_type,
                "default": _jsonable_settings_value(default),
                "description": description,
                "limits": _jsonable_settings_value(limits),
            }
        )

    return {
        "algorithm_type": algorithm_type,
        "requested_name": selected_name,
        "name": canonical_name,
        "aliases": sorted(aliases),
        "is_default": canonical_name == default_name,
        "interface_only": False,
        "default_settings": _jsonable_settings_dict(instance.settings().to_dict()),
        "settings": setting_schema,
    }


@app.tool()
@_structured
def list_cache_backends() -> dict:
    """List registered cache backend names."""
    if not _REMOTE_AVAILABLE:
        return {"backends": [], "note": "Remote/cache module not available."}
    return {"backends": available_caches()}


@app.tool()
@_structured
def list_remote_backends() -> dict:
    """List registered remote execution backend names."""
    if not _REMOTE_AVAILABLE:
        return {"backends": [], "note": "Remote/cache module not available."}
    return {"backends": available_backends()}


# ═══════════════════════════════════════════════════════════════════════════
# Project management tools
# ═══════════════════════════════════════════════════════════════════════════


@app.tool()
@_structured
def list_projects() -> dict:
    """List project directories in the workspace."""
    projects_dir = config.projects_dir
    if not projects_dir.exists():
        return {"projects": []}
    projects = sorted(d.name for d in projects_dir.iterdir() if d.is_dir() and not d.is_symlink())
    return {"projects": projects}


@app.tool()
@_structured
def create_project(project_name: str) -> dict | str:
    """Create a project directory and return its metadata."""
    if not project_name or not project_name.strip():
        return "ERROR: project_name must be a non-empty string."
    project_dir, error = resolve_project_path(project_name, config.projects_dir)
    if project_dir is None:
        return f"ERROR: {error}"
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        return f"ERROR: Cannot create project directory: {e}"
    return {"project_name": project_name, "path": str(project_dir)}


@app.tool()
@_structured
def list_project_files(project_name: str) -> dict | str:
    """List project files with sizes and inferred data types."""
    project_dir, error = resolve_project_path(project_name, config.projects_dir)
    if project_dir is None:
        return f"ERROR: {error}"
    if not project_dir.exists():
        return f"ERROR: Project '{project_name}' not found."

    files = []
    for f in sorted(project_dir.iterdir()):
        if f.is_file() and not f.is_symlink():
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

    Returns registered classes that provide ``from_json_file`` and can therefore
    be loaded from project files.

    """
    return [cls for cls in data.available_dataclasses().values() if callable(getattr(cls, "from_json_file", None))]


@app.tool()
@_structured
@validate_project
def get_summary(
    project_name: str,
    filename: str,
) -> dict | str:
    """Load a supported data file and return its summary."""
    filename = _strip(filename)

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
        "list_algorithms",
        "describe_algorithm",
        "get_algorithm_default_type",
        "get_algorithm_default_settings",
        "get_orbitals_from_input",
        "get_active_space_indices",
        "get_ansatz",
        "get_top_determinants",
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
        "run_population_analysis",
        "run_nuclear_derivative_calculator",
        "run_geometry_optimization",
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
        "create_majorana_mapping",
        "run_qubit_mapper",
        "run_state_preparation",
        "run_amplitude_amplification",
        "run_term_grouper",
        "run_qubit_hamiltonian_solver",
        "run_energy_estimator",
        "estimate_circuit",
    ],
    "qpe": [
        "run_time_evolution_builder",
        "run_evolution_circuit_builder",
        "run_hamiltonian_simulation",
        "run_controlled_evolution_circuit_mapper",
        "run_circuit_executor",
        "run_hadamard_test",
        "run_phase_estimation",
    ],
    "visualization": [
        "visualize_molecule",
        "visualize_orbitals",
        "visualize_orbital_entanglement",
        "visualize_circuit",
        "visualize_scatter_plot",
    ],
    "remote_execution": [
        "check_remote_job",
        "retrieve_remote_results",
        "list_remote_jobs",
        "cancel_remote_job",
        "list_cache_backends",
        "list_remote_backends",
        "describe_backend",
    ],
}


@app.tool()
@_structured
def list_tools(category: str | None = None) -> dict:
    """List MCP tool names by functional category."""
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
    """Convert Cartesian coordinates between Bohr and Angstrom."""
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
    """Convert an energy value between supported units."""
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
    """Describe accepted configuration fields for a cache or remote backend."""
    if not _REMOTE_AVAILABLE:
        return "Remote/cache module not available."

    import inspect as _inspect  # noqa: PLC0415

    if backend_type == "cache":
        from qdk_chemistry.remote.cache import _CACHES  # noqa: PLC0415

        registry = _CACHES
    elif backend_type == "remote":
        from qdk_chemistry.remote.backends.base import _BACKENDS  # noqa: PLC0415

        registry = _BACKENDS
    else:
        return f"backend_type must be 'cache' or 'remote', got '{backend_type}'"

    if name not in registry:
        return f"No {backend_type} backend registered with name '{name}'. Available: {', '.join(registry)}"

    cls = registry[name]
    sig = _inspect.signature(cls.__init__)
    safe_remote_options = get_mcp_safe_config_options(name) if backend_type == "remote" else frozenset()
    params = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if backend_type == "remote" and pname not in safe_remote_options:
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
    """Create and save a Structure from Bohr coordinates; use convert_coordinates for Angstrom input."""
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

    try:
        save_path = resolve_project_file(filename_to_save, allow_nested=True)
    except (OSError, RuntimeError, ValueError) as e:
        return f"ERROR: Cannot resolve output filename '{filename_to_save}': {e}"

    # Check if output file already exists
    if not overwrite:
        existing_check = check_output_path_exists(save_path, data.Structure)
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
    """Return the default implementation name for an algorithm type."""
    created_algorithm = algorithms.create(algorithm_type)
    return created_algorithm.name()


@app.tool()
@_structured
def get_algorithm_default_settings(algorithm_type: str, algorithm_name: str | None = None) -> dict:
    """Return default settings for an algorithm implementation."""
    created_algorithm = algorithms.create(algorithm_type, algorithm_name)

    return _jsonable_settings_dict(created_algorithm.settings().to_dict())


@app.tool()
@_structured
@validate_project
def get_orbitals_from_input(
    project_name: str, input_filename: str, out_orbitals_filename: str, overwrite: bool = False
) -> str:
    """Extract and save Orbitals from a supported electronic-structure object."""
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
    """Return active, inactive, and virtual orbital indices from a supported object."""
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
    """Combine a saved Hamiltonian and Wavefunction into a saved Ansatz."""
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Run an active-space selector on a Wavefunction and save the resulting Wavefunction."""
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

    active_space_selector = algorithms.create("active_space_selector", algorithm_name)
    _apply_settings(active_space_selector, settings)

    if charge is not None and algorithm_name == "qdk_valence":
        # grab valence electrons/orbitals count using helper function
        n_active_electrons, n_active_orbitals = compute_valence_space_parameters(wavefunction, charge)
        supplied_settings = settings or {}
        if "num_active_electrons" not in supplied_settings:
            active_space_selector.settings().set("num_active_electrons", n_active_electrons)
        if "num_active_orbitals" not in supplied_settings:
            active_space_selector.settings().set("num_active_orbitals", n_active_orbitals)

    # run active space selection
    out_wavefunction = _run_algorithm(
        active_space_selector,
        wavefunction,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run dynamical correlation for an Ansatz and save the resulting Wavefunction."""
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
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Build and save a fermionic Hamiltonian from molecular Orbitals."""
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
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
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
    """Construct and save a fermionic lattice-model Hamiltonian."""
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
    """Construct and save an Ising or Heisenberg QubitHamiltonian."""
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Localize selected Wavefunction orbitals and save the resulting Wavefunction."""
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
            overwrite=overwrite,
        )
    else:
        localized_wfn = _run_algorithm(
            localizer,
            wavefunction,
            loc_indices_alpha,
            loc_indices_beta,
            cache=cache,
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[bool, str]:
    """Evaluate orbital-rotation stability and save the StabilityResult."""
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
            overwrite=overwrite,
        )

    except RuntimeError as e:
        return f"The stability checker did not converge: {e}."

    # save to file
    save_data_object(stability_result, out_stability_result_filename)

    return (stability_bool, out_stability_result_filename)


@app.tool()
@_structured
@validate_project
def run_term_grouper(
    project_name: str,
    qubit_hamiltonian_filename: str,
    out_qubit_hamiltonian_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Partition a saved QubitOperator's Pauli terms and save the grouped operator."""
    out_qubit_hamiltonian_filename, error = _prepare_output(
        out_qubit_hamiltonian_filename,
        "QubitOperator",
        data.QubitOperator,
        overwrite=overwrite,
    )
    if error:
        return error
    qubit_hamiltonian, error = _load_or_error(qubit_hamiltonian_filename, data.QubitOperator, "qubit Hamiltonian")
    if error:
        return error

    term_grouper = algorithms.create("term_grouper", algorithm_name)
    _apply_settings(term_grouper, settings)
    grouped_hamiltonian = _run_algorithm(
        term_grouper,
        qubit_hamiltonian,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    save_data_object(grouped_hamiltonian, out_qubit_hamiltonian_filename)
    return out_qubit_hamiltonian_filename


@app.tool()
@_structured
@validate_project
def run_amplitude_amplification(
    project_name: str,
    state_prep_oracle_filename: str,
    good_state_oracle_filename: str,
    out_circuit_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Build and save an amplitude-amplified Circuit from state-preparation and good-state oracle Circuits."""
    out_circuit_filename, error = _prepare_output(out_circuit_filename, "Circuit", data.Circuit, overwrite=overwrite)
    if error:
        return error
    state_prep_oracle, error = _load_or_error(state_prep_oracle_filename, data.Circuit, "state-preparation oracle")
    if error:
        return error
    good_state_oracle, error = _load_or_error(good_state_oracle_filename, data.Circuit, "good-state oracle")
    if error:
        return error

    amplitude_amplification = algorithms.create("amplitude_amplification", algorithm_name)
    _apply_settings(amplitude_amplification, settings)
    circuit = _run_algorithm(
        amplitude_amplification,
        state_prep_oracle,
        good_state_oracle,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    save_data_object(circuit, out_circuit_filename)
    return out_circuit_filename


@app.tool()
@_structured
@validate_project
def run_hadamard_test(
    project_name: str,
    state_preparation_circuit_filename: str,
    unitary_filename: str,
    out_executor_data_filename: str,
    shots: int,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Run a Hadamard test for a saved state-preparation Circuit and UnitaryRepresentation."""
    out_executor_data_filename, error = _prepare_output(
        out_executor_data_filename,
        "CircuitExecutorData",
        data.CircuitExecutorData,
        overwrite=overwrite,
    )
    if error:
        return error
    state_preparation_circuit, error = _load_or_error(
        state_preparation_circuit_filename, data.Circuit, "state-preparation circuit"
    )
    if error:
        return error
    unitary, error = _load_or_error(unitary_filename, _UnitaryRepresentation, "unitary representation")
    if error:
        return error

    hadamard_test = algorithms.create("hadamard_test", algorithm_name)
    _apply_settings(hadamard_test, settings)
    executor_data = _run_algorithm(
        hadamard_test,
        state_preparation_circuit,
        unitary,
        shots,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    save_data_object(executor_data, out_executor_data_filename)
    return out_executor_data_filename


@app.tool()
@_structured
@validate_project
def run_evolution_circuit_builder(
    project_name: str,
    base_hamiltonian_filename: str,
    drive_hamiltonian_filename: str,
    drive_times: list[float],
    drive_values: list[float],
    state_preparation_circuit_filename: str,
    out_circuit_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Build a Circuit for H(t)=H0+f(t)H1 using a piecewise-linear drive schedule."""
    out_circuit_filename, error = _prepare_output(out_circuit_filename, "Circuit", data.Circuit, overwrite=overwrite)
    if error:
        return error
    hamiltonian, error = _load_driven_hamiltonian(
        base_hamiltonian_filename,
        drive_hamiltonian_filename,
        drive_times,
        drive_values,
    )
    if error:
        return error
    state_preparation_circuit, error = _load_or_error(
        state_preparation_circuit_filename, data.Circuit, "state-preparation circuit"
    )
    if error:
        return error

    circuit_builder = algorithms.create("evolution_circuit_builder", algorithm_name)
    _apply_settings(circuit_builder, settings)
    circuit = _run_algorithm(
        circuit_builder,
        hamiltonian,
        state_preparation_circuit,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    save_data_object(circuit, out_circuit_filename)
    return out_circuit_filename


@app.tool()
@_structured
@validate_project
def run_hamiltonian_simulation(
    project_name: str,
    base_hamiltonian_filename: str,
    drive_hamiltonian_filename: str,
    drive_times: list[float],
    drive_values: list[float],
    observable_filenames: list[str],
    state_preparation_circuit_filename: str,
    out_energy_result_filenames: list[str],
    out_measurement_data_filenames: list[str],
    shots: int = 1000,
    noise_model: Any | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> list[dict[str, str]] | str:
    """Evolve H(t)=H0+f(t)H1, measure observables, and save each result pair."""
    result_count = len(observable_filenames)
    if result_count == 0:
        return "observable_filenames must contain at least one observable."
    if len(out_energy_result_filenames) != result_count or len(out_measurement_data_filenames) != result_count:
        return "Each observable requires one energy-result filename and one measurement-data filename."

    energy_filenames: list[str] = []
    measurement_filenames: list[str] = []
    for requested_energy_filename, requested_measurement_filename in zip(
        out_energy_result_filenames, out_measurement_data_filenames, strict=True
    ):
        energy_filename, error = _prepare_output(
            requested_energy_filename,
            "EnergyExpectationResult",
            data.EnergyExpectationResult,
            overwrite=overwrite,
        )
        if error:
            return error
        measurement_filename, error = _prepare_output(
            requested_measurement_filename,
            "MeasurementData",
            data.MeasurementData,
            overwrite=overwrite,
        )
        if error:
            return error
        energy_filenames.append(energy_filename)
        measurement_filenames.append(measurement_filename)

    hamiltonian, error = _load_driven_hamiltonian(
        base_hamiltonian_filename,
        drive_hamiltonian_filename,
        drive_times,
        drive_values,
    )
    if error:
        return error
    observables = []
    for observable_filename in observable_filenames:
        observable, error = _load_or_error(observable_filename, data.QubitOperator, "observable")
        if error:
            return error
        observables.append(observable)
    state_preparation_circuit, error = _load_or_error(
        state_preparation_circuit_filename, data.Circuit, "state-preparation circuit"
    )
    if error:
        return error

    simulation = algorithms.create("hamiltonian_simulation", algorithm_name)
    _apply_settings(simulation, settings)
    results = _run_algorithm(
        simulation,
        hamiltonian,
        observables,
        state_preparation_circuit,
        shots,
        noise=noise_model,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    if len(results) != result_count:
        return f"Hamiltonian simulation returned {len(results)} results for {result_count} observables."

    outputs = []
    for (energy_result, measurement_data), energy_filename, measurement_filename in zip(
        results, energy_filenames, measurement_filenames, strict=True
    ):
        save_data_object(energy_result, energy_filename)
        save_data_object(measurement_data, measurement_filename)
        outputs.append({"energy_result": energy_filename, "measurement_data": measurement_filename})
    return outputs


@app.tool()
@_structured
@validate_project
def run_qubit_hamiltonian_solver(
    project_name: str,
    qubit_hamiltonian_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
) -> str | tuple[float, list]:
    """Diagonalize a QubitHamiltonian and return its mapped energy, excluding core energy, and eigenstate."""
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
        remote=remote,
        remote_config=remote_config,
    )

    return (energy, eigenstate.tolist())


@app.tool()
@_structured
@validate_project
def run_energy_estimator(
    project_name: str,
    circuit_filename: str,
    qubit_hamiltonian_filename: str,
    out_energy_result_filename: str,
    out_measurement_data_filename: str,
    total_shots: int,
    noise_model: Any | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[str, str]:
    """Estimate a mapped Hamiltonian expectation value, excluding core energy, and variance from a Circuit."""
    # Strip filenames in case full path is passed
    circuit_filename = _strip(circuit_filename)
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
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )

    # save to files
    save_data_object(energy_result, out_energy_result_filename)
    save_data_object(measurement_data, out_measurement_data_filename)

    return (out_energy_result_filename, out_measurement_data_filename)


@app.tool()
@_structured
@validate_project
def create_majorana_mapping(
    project_name: str,
    out_mapping_filename: str,
    encoding: str = "jordan-wigner",
    num_modes: int | None = None,
    hamiltonian_filename: str | None = None,
    overwrite: bool = False,
) -> str:
    """Create and save a MajoranaMapping for a mode count or Hamiltonian."""
    out_mapping_filename = _strip(out_mapping_filename)
    out_mapping_filename, _err = _prepare_output(
        out_mapping_filename, "MajoranaMapping", data.MajoranaMapping, overwrite=overwrite
    )
    if _err:
        return _err

    derived_num_modes = None
    if hamiltonian_filename is not None:
        hamiltonian_filename = _strip(hamiltonian_filename)
        hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
        if _err:
            return _err
        n_spatial_orbitals = hamiltonian.get_one_body_integrals()[0].shape[0]
        derived_num_modes = 2 * n_spatial_orbitals

    if num_modes is None:
        if derived_num_modes is None:
            return "ERROR: Provide either num_modes or hamiltonian_filename to create a MajoranaMapping."
        num_modes = derived_num_modes
    elif derived_num_modes is not None and num_modes != derived_num_modes:
        return (
            f"ERROR: num_modes ({num_modes}) does not match the Hamiltonian-derived spin-orbital mode count "
            f"({derived_num_modes})."
        )

    if num_modes <= 0:
        return f"ERROR: num_modes must be positive, got {num_modes}."

    mapping_factories = {
        "jordan-wigner": data.MajoranaMapping.jordan_wigner,
        "bravyi-kitaev": data.MajoranaMapping.bravyi_kitaev,
        "bravyi-kitaev-tree": data.MajoranaMapping.bravyi_kitaev_tree,
        "parity": data.MajoranaMapping.parity,
    }
    if encoding not in mapping_factories:
        return f"ERROR: Unsupported encoding '{encoding}'. Supported encodings: {', '.join(mapping_factories)}."

    mapping = mapping_factories[encoding](num_modes=num_modes)
    save_data_object(mapping, out_mapping_filename)

    return out_mapping_filename


@app.tool()
@_structured
@validate_project
def run_qubit_mapper(
    project_name: str,
    hamiltonian_filename: str,
    mapping_filename: str,
    out_qubit_hamiltonian_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | dict[str, str | float]:
    """Save a mapped QubitHamiltonian excluding core energy and return its filename with the companion offset."""
    # Strip filenames in case full path is passed
    hamiltonian_filename = _strip(hamiltonian_filename)
    mapping_filename = _strip(mapping_filename)
    out_qubit_hamiltonian_filename = _strip(out_qubit_hamiltonian_filename)
    out_qubit_hamiltonian_filename, _err = _prepare_output(
        out_qubit_hamiltonian_filename, "QubitHamiltonian", data.QubitHamiltonian, overwrite=overwrite
    )
    if _err:
        return _err

    hamiltonian, _err = _load_or_error(hamiltonian_filename, data.Hamiltonian, "hamiltonian")
    if _err:
        return _err

    mapping, _err = _load_or_error(mapping_filename, data.MajoranaMapping, "majorana mapping")
    if _err:
        return _err

    qubit_mapper = algorithms.create("qubit_mapper", algorithm_name)

    _apply_settings(qubit_mapper, settings)

    # run qubit mapping
    qubit_hamiltonian = _run_algorithm(
        qubit_mapper,
        hamiltonian,
        mapping,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(qubit_hamiltonian, out_qubit_hamiltonian_filename)

    return {
        "qubit_hamiltonian_filename": out_qubit_hamiltonian_filename,
        "core_energy": float(hamiltonian.get_core_energy()),
    }


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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Compile a Wavefunction into a saved Circuit."""
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
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )

    # save to file
    save_data_object(circuit, out_circuit_filename)

    return out_circuit_filename


@app.tool()
@_structured
@validate_project
def estimate_circuit(
    project_name: str,
    circuit_filename: str,
    params: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict | list | str:
    """Estimate a stored Circuit with its QDK estimator parameters and return the result inline."""
    circuit_filename = _strip(circuit_filename)
    circuit, error = _load_or_error(circuit_filename, data.Circuit, "circuit")
    if error:
        return error

    return circuit.estimate(params).data()


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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Build exp(-iHt) from a QubitHamiltonian and save the TimeEvolutionUnitary."""
    qubit_hamiltonian_filename = _strip(qubit_hamiltonian_filename)
    out_time_evolution_unitary_filename = _strip(out_time_evolution_unitary_filename)
    try:
        out_time_evolution_unitary_filename = ensure_filename_format(
            out_time_evolution_unitary_filename, "UnitaryRepresentation"
        )
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    if not overwrite:
        existing_check = check_output_exists(out_time_evolution_unitary_filename, _UnitaryRepresentation)
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Map a TimeEvolutionUnitary to a saved controlled Circuit."""
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
        time_evolution_unitary = load_data_object(time_evolution_unitary_filename, _UnitaryRepresentation)
    except (RuntimeError, ValueError) as e:
        return f"Failed to load time evolution unitary from {time_evolution_unitary_filename}: {e!s}"

    if power != 1:
        return (
            "Set controlled-U power on the time-evolution builder settings before mapping, "
            "for example run_time_evolution_builder(..., settings={'power': 4, 'power_strategy': 'repeat'})."
        )

    # Create the circuit mapper algorithm
    circuit_mapper = algorithms.create("controlled_circuit_mapper", algorithm_name)
    _apply_settings(circuit_mapper, settings)
    circuit_mapper.settings().set("control_indices", control_indices)

    try:
        circuit = _run_algorithm(
            circuit_mapper,
            time_evolution_unitary,
            cache=cache,
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
            overwrite=overwrite,
        )
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Execute a Circuit and save its CircuitExecutorData."""
    circuit_filename = _strip(circuit_filename)
    out_executor_data_filename = _strip(out_executor_data_filename)
    try:
        out_executor_data_filename = ensure_filename_format(out_executor_data_filename, "CircuitExecutorData")
    except FilenameFormatError as e:
        return f"Invalid output filename: {e!s}"

    if not overwrite:
        existing_check = check_output_exists(out_executor_data_filename, _CircuitExecutorData)
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str:
    """Run phase estimation and save a QpeResult whose mapped energies exclude core energy."""
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

    # Apply settings
    _apply_settings(phase_estimation, settings)

    # Check validity of settings
    qpe_circuit_builder = phase_estimation.settings().get("qpe_circuit_builder")
    qpe_circuit_builder_settings = getattr(qpe_circuit_builder, "settings", None)
    try:
        num_bits = qpe_circuit_builder_settings.get("num_bits") if qpe_circuit_builder_settings is not None else -1
    except (KeyError, RuntimeError):
        num_bits = -1
    if num_bits <= 0:
        return (
            "Invalid QPE setting: settings.qpe_circuit_builder.num_bits must be set to a positive integer; "
            f"received {num_bits!r}."
        )

    try:
        unitary_builder = (
            qpe_circuit_builder_settings.get("unitary_builder") if qpe_circuit_builder_settings is not None else None
        )
        unitary_builder_settings = getattr(unitary_builder, "settings", None)
        evolution_time = unitary_builder_settings.get("time") if unitary_builder_settings is not None else 0.0
    except (KeyError, RuntimeError):
        evolution_time = 0.0
    if evolution_time == 0.0:
        return (
            "Invalid QPE setting: settings.qpe_circuit_builder.unitary_builder.time must be set explicitly "
            f"to a nonzero value for the selected unitary builder; received {evolution_time!r}."
        )

    # Run phase estimation
    try:
        qpe_result = _run_algorithm(
            phase_estimation,
            state_prep_circuit,
            qubit_hamiltonian,
            cache=cache,
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run an HF or DFT self-consistent-field calculation and save its Wavefunction."""
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
def run_population_analysis(
    project_name: str,
    wavefunction_filename: str,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> dict | str:
    """Compute and return per-center populations from a Wavefunction."""
    wavefunction_filename = _strip(wavefunction_filename)
    wavefunction, error = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if error:
        return error

    analyzer = algorithms.create("population_analyzer", algorithm_name)
    _apply_settings(analyzer, settings)

    populations = _run_algorithm(
        analyzer,
        wavefunction,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    if isinstance(populations, str):
        return populations

    return {
        "wavefunction_filename": wavefunction_filename,
        "algorithm": analyzer.name(),
        "populations": [float(population) for population in populations],
        "population_sum": float(sum(populations)),
    }


@app.tool()
@_structured
@validate_project
def run_nuclear_derivative_calculator(
    project_name: str,
    structure_filename: str,
    out_gradients_filename: str,
    charge: int,
    spin_multiplicity: int,
    seed_or_basis: str,
    n_inactive_orbitals: int = 0,
    out_wavefunction_filename: str | None = None,
    out_hessian_filename: str | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> dict | str:
    """Compute nuclear derivatives for a Structure and save requested outputs."""
    structure_filename = _strip(structure_filename)
    out_gradients_filename, err = _prepare_output(
        out_gradients_filename, "NuclearGradients", data.NuclearGradients, overwrite=overwrite
    )
    if err:
        return err

    if out_wavefunction_filename is not None:
        out_wavefunction_filename, err = _prepare_output(
            out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
        )
        if err:
            return err

    if out_hessian_filename is not None:
        out_hessian_filename, err = _prepare_output(
            out_hessian_filename, "NuclearHessian", data.NuclearHessian, overwrite=overwrite
        )
        if err:
            return err

    structure, err = _load_or_error(structure_filename, data.Structure, "structure")
    if err:
        return err

    seed, err = _resolve_seed_or_basis(seed_or_basis)
    if err:
        return err

    calculator = algorithms.create("nuclear_derivative_calculator", algorithm_name)
    _apply_settings(calculator, settings)

    result = _run_algorithm(
        calculator,
        structure,
        charge,
        spin_multiplicity,
        seed,
        n_inactive_orbitals,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    if isinstance(result, str):
        return result

    energy, gradients, hessian, wavefunction = result
    save_data_object(gradients, out_gradients_filename)

    outputs: dict[str, Any] = {"energy": energy, "gradients_filename": out_gradients_filename}
    if wavefunction is not None and out_wavefunction_filename is not None:
        save_data_object(wavefunction, out_wavefunction_filename)
        outputs["wavefunction_filename"] = out_wavefunction_filename
    elif wavefunction is not None:
        outputs["wavefunction_available"] = True

    if hessian is not None and out_hessian_filename is not None:
        save_data_object(hessian, out_hessian_filename)
        outputs["hessian_filename"] = out_hessian_filename
    elif hessian is not None:
        outputs["hessian_available"] = True

    return outputs


@app.tool()
@_structured
@validate_project
def run_geometry_optimization(
    project_name: str,
    structure_filename: str,
    out_structure_filename: str,
    charge: int,
    spin_multiplicity: int,
    seed_or_basis: str,
    n_inactive_orbitals: int = 0,
    out_wavefunction_filename: str | None = None,
    out_hessian_filename: str | None = None,
    algorithm_name: str | None = None,
    settings: dict | None = None,
    cache: str | None = None,
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> dict | str:
    """Optimize molecular geometry and save the resulting Structure."""
    structure_filename = _strip(structure_filename)
    out_structure_filename, err = _prepare_output(
        out_structure_filename, "Structure", data.Structure, overwrite=overwrite
    )
    if err:
        return err

    if out_wavefunction_filename is not None:
        out_wavefunction_filename, err = _prepare_output(
            out_wavefunction_filename, "Wavefunction", data.Wavefunction, overwrite=overwrite
        )
        if err:
            return err

    if out_hessian_filename is not None:
        out_hessian_filename, err = _prepare_output(
            out_hessian_filename, "NuclearHessian", data.NuclearHessian, overwrite=overwrite
        )
        if err:
            return err

    structure, err = _load_or_error(structure_filename, data.Structure, "structure")
    if err:
        return err

    seed, err = _resolve_seed_or_basis(seed_or_basis)
    if err:
        return err

    try:
        optimizer = algorithms.create("geometry_optimizer", algorithm_name)
    except KeyError as exc:
        derivative_algorithms = algorithms.available("nuclear_derivative_calculator")
        if algorithm_name not in derivative_algorithms:
            raise
        raise ValueError(
            f"Algorithm {algorithm_name!r} is a nuclear derivative calculator, not a geometry optimizer. "
            "Leave algorithm_name unset (or use 'geometric') and select it through "
            f"settings['derivative_calculator'], for example "
            f"{{'algorithm_name': {algorithm_name!r}}}."
        ) from exc
    _apply_settings(optimizer, settings)

    result = _run_algorithm(
        optimizer,
        structure,
        charge,
        spin_multiplicity,
        seed,
        n_inactive_orbitals,
        cache=cache,
        remote=remote,
        remote_config=remote_config,
        remote_timeout=remote_timeout,
        overwrite=overwrite,
    )
    if isinstance(result, str):
        return result

    energy, optimized_structure, hessian, wavefunction = result
    save_data_object(optimized_structure, out_structure_filename)

    outputs: dict[str, Any] = {"energy": energy, "structure_filename": out_structure_filename}
    if wavefunction is not None and out_wavefunction_filename is not None:
        save_data_object(wavefunction, out_wavefunction_filename)
        outputs["wavefunction_filename"] = out_wavefunction_filename
    elif wavefunction is not None:
        outputs["wavefunction_available"] = True

    if hessian is not None and out_hessian_filename is not None:
        save_data_object(hessian, out_hessian_filename)
        outputs["hessian_filename"] = out_hessian_filename
    elif hessian is not None:
        outputs["hessian_available"] = True

    return outputs


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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run a multi-configuration calculation and save its Wavefunction."""
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
                remote=remote,
                remote_config=remote_config,
                remote_timeout=remote_timeout,
                overwrite=overwrite,
            )
        else:
            (total_energy, wavefunction) = _run_algorithm(
                mc_calculator,
                hamiltonian,
                n_active_alpha_electrons,
                n_active_beta_electrons,
                cache=cache,
                remote=remote,
                remote_config=remote_config,
                remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Run MCSCF from active-space Orbitals and save the resulting Wavefunction."""
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

    # Create and configure ham_constructor
    ham_constructor = algorithms.create("hamiltonian_constructor", ham_constructor_algorithm_name)
    ham_constructor_settings = ham_constructor_settings or {}
    for key, value in ham_constructor_settings.items():
        ham_constructor.settings().set(key, value)

    # Create and configure mc_calculator
    mc_calculator = algorithms.create("multi_configuration_calculator", mc_calculator_algorithm_name)
    mc_calculator_settings = mc_calculator_settings or {}
    for key, value in mc_calculator_settings.items():
        mc_calculator.settings().set(key, value)

    # Create and configure mcscf_calculator
    mcscf_calculator = algorithms.create("multi_configuration_scf", "pyscf")
    _apply_settings(mcscf_calculator, settings)

    try:
        if n_active_beta_electrons is None:
            (total_energy, wavefunction) = _run_algorithm(
                mcscf_calculator,
                orbitals,
                ham_constructor,
                mc_calculator,
                n_active_alpha_electrons,
                n_active_alpha_electrons,
                cache=cache,
                remote=remote,
                remote_config=remote_config,
                remote_timeout=remote_timeout,
                overwrite=overwrite,
            )
        else:
            (total_energy, wavefunction) = _run_algorithm(
                mcscf_calculator,
                orbitals,
                ham_constructor,
                mc_calculator,
                n_active_alpha_electrons,
                n_active_beta_electrons,
                cache=cache,
                remote=remote,
                remote_config=remote_config,
                remote_timeout=remote_timeout,
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
    remote: str | None = None,
    remote_config: dict | None = None,
    remote_timeout: int = 120,
    overwrite: bool = False,
) -> str | tuple[float, str]:
    """Solve a Hamiltonian in a supplied determinant subspace and save the Wavefunction."""
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
        configurations = [data.Configuration(s) for s in config_strings]
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
            remote=remote,
            remote_config=remote_config,
            remote_timeout=remote_timeout,
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
def get_top_determinants(
    project_name: str,
    wavefunction_filename: str,
    max_determinants: int | None = 10,
) -> dict | str:
    """Return ranked determinants and CI coefficient data from a Wavefunction."""
    wavefunction_filename = _strip(wavefunction_filename)
    if max_determinants is not None and max_determinants <= 0:
        return "max_determinants must be greater than zero or None"

    wavefunction, err = _load_or_error(wavefunction_filename, data.Wavefunction, "wavefunction")
    if err:
        return err

    try:
        ranked_determinants = wavefunction.get_top_determinants(max_determinants=max_determinants)
        norm = float(wavefunction.norm())
        total_weight = norm * norm
        cumulative_weight = 0.0
        determinants = []
        for rank, (determinant, coefficient) in enumerate(ranked_determinants.items(), start=1):
            complex_coefficient = complex(coefficient)
            magnitude = abs(complex_coefficient)
            weight = magnitude * magnitude / total_weight if total_weight > 0.0 else 0.0
            cumulative_weight += weight
            determinants.append(
                {
                    "rank": rank,
                    "configuration": determinant.to_string(),
                    "coefficient_real": float(complex_coefficient.real),
                    "coefficient_imaginary": float(complex_coefficient.imag),
                    "magnitude": float(magnitude),
                    "weight": float(weight),
                    "cumulative_weight": float(cumulative_weight),
                }
            )
        return {
            "wavefunction_filename": wavefunction_filename,
            "total_determinants": int(wavefunction.size()),
            "returned_determinants": len(determinants),
            "norm": norm,
            "determinants": determinants,
        }
    except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
        return f"Failed to inspect determinants in wavefunction: {exc!s}"


@app.tool()
@_structured
@validate_project
def get_top_configurations(
    project_name: str,
    wavefunction_filename: str,
    max_determinants: int | None = None,
) -> str:
    """Return configuration strings ranked by Wavefunction CI coefficient magnitude."""
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
    """Return logical-qubit, gate-count, and depth metrics for a saved Circuit."""
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


# =========================
# Remote / async job tools
# =========================


def _require_remote() -> str | None:
    """Return an error message if the remote module is not available."""
    if not _REMOTE_AVAILABLE:
        return (
            "Remote execution is not available. "
            "The qdk_chemistry.remote module could not be imported. "
            "Ensure the package is built with remote support."
        )
    return None


def _get_default_cache() -> FolderCache:
    """Return the default FolderCache backed by ``config.cache_dir``."""
    return FolderCache(path=config.cache_dir)


def _current_job_owner(project_name: str) -> dict[str, str | None]:
    """Return the workspace and project owner for a job-management request."""
    workspace_root = current_workspace_root()
    return {
        "workspace_root": str(workspace_root) if workspace_root is not None else None,
        "project_name": project_name,
    }


def _job_record_path(job: Job, run_hash: str) -> Path:
    """Return an owner-scoped path for a durable MCP job record."""
    if job.owner is None:
        return config.jobs_dir / f"{run_hash}.job.json"
    owner_json = json.dumps(job.owner, sort_keys=True, separators=(",", ":"))
    owner_digest = hashlib.sha256(owner_json.encode()).hexdigest()[:16]
    return config.jobs_dir / f"{owner_digest}.{run_hash}.job.json"


def _discover_cached_jobs(owner: dict[str, str | None]) -> list[Job]:
    """Discover only job files owned by the requesting workspace and project."""
    jobs = []
    seen: set[tuple[str, str | None]] = set()
    for jobs_dir in (config.jobs_dir, config.cache_dir):
        if not jobs_dir.exists():
            continue
        for p in sorted(jobs_dir.glob("*.job.json")):
            try:
                job = Job.load(p)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
            if job.owner != owner:
                continue
            identity = (job.job_id, job.run_hash)
            if identity in seen:
                continue
            seen.add(identity)
            jobs.append(job)
    return jobs


def _load_remote_job(job_id: str, owner: dict[str, str | None]):
    """Load a Job by its job_id from the configured job directories.

    Scans all discovered job files to find the one matching the given job_id.

    Returns:
        (Job, None) on success, or (None, error_string) on failure.

    """
    matches = [job for job in _discover_cached_jobs(owner) if job.job_id == job_id]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"Multiple remote jobs found with id '{job_id}' for this project."
    return None, f"No remote job found with id '{job_id}' in {config.cache_dir} or {config.jobs_dir}."


@app.tool()
@_structured
@validate_project
def check_remote_job(
    project_name: str,
    job_id: str,
) -> str | dict:
    """Query a remote job and update its persisted status record."""
    err = _require_remote()
    if err:
        return err

    job, load_err = _load_remote_job(job_id, _current_job_owner(project_name))
    if load_err:
        return load_err

    try:
        job_status = job.check()  # also updates & saves the job file
    except Exception as e:  # noqa: BLE001
        return f"Failed to query job status: {e}"

    # Calculate elapsed time
    elapsed = ""
    if job.submitted_at:
        try:
            submitted_dt = datetime.fromisoformat(job.submitted_at)
            delta = datetime.now(timezone.utc) - submitted_dt
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed = f"{hours}h {minutes}m {seconds}s"
        except (ValueError, TypeError):
            pass

    result: dict[str, Any] = {
        "job_id": job_id,
        "status": job_status.status,
        "elapsed": elapsed,
        "submitted_at": job.submitted_at,
        "logs": job_status.logs,
    }
    if job_status.error:
        result["error"] = job_status.error
    if job.run_hash:
        result["run_hash"] = job.run_hash
    if job.input_hashes:
        result["input_hashes"] = job.input_hashes
    if job.output_hashes:
        result["output_hashes"] = job.output_hashes
    return result


@app.tool()
@_structured
@validate_project
def retrieve_remote_results(
    project_name: str,
    job_id: str,
) -> str | dict:
    """Download a completed remote job's outputs into its project directory."""
    err = _require_remote()
    if err:
        return err

    job, load_err = _load_remote_job(job_id, _current_job_owner(project_name))
    if load_err:
        return load_err

    project_dir, project_error = resolve_project_path(project_name, config.projects_dir)
    if project_dir is None:
        return f"Failed to resolve project directory: {project_error}"
    try:
        job.fetch(local_dir=project_dir)  # updates status to "retrieved" & saves
    except Exception as e:  # noqa: BLE001
        return f"Failed to retrieve results for job '{job_id}': {e}"

    # Discover downloaded files from the manifest
    downloaded: list[str] = []
    primitives: dict[str, Any] = {}
    manifest_path = project_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        def _collect(entry_data: Any, name: str = "result") -> None:
            if isinstance(entry_data, dict):
                entry_type = entry_data.get("type")
                if entry_data.get("file"):
                    downloaded.append(entry_data["file"])
                elif "value" in entry_data and entry_type in ("float", "int", "str", "bool", "none"):
                    primitives[name] = entry_data["value"]
                elif entry_type in ("tuple", "list") and "items" in entry_data:
                    for i, item in enumerate(entry_data["items"]):
                        _collect(item, f"{name}_{i}")
                elif entry_type == "dict" and "entries" in entry_data:
                    for k, v in entry_data["entries"].items():
                        _collect(v, f"{name}_{k}")

        for i, result_entry in enumerate(manifest.get("results", [])):
            _collect(result_entry, f"result_{i}")

        manifest_path.unlink(missing_ok=True)

    return {
        "status": "retrieved",
        "job_id": job_id,
        "downloaded_files": downloaded,
        "values": primitives,
        "output_hashes": job.output_hashes,
    }


@app.tool()
@_structured
@validate_project
def list_remote_jobs(
    project_name: str,
    status_filter: str | None = None,
) -> str | dict:
    """List persisted remote jobs with an optional status filter."""
    err = _require_remote()
    if err:
        return err

    all_jobs = _discover_cached_jobs(_current_job_owner(project_name))

    jobs = []
    for j in all_jobs:
        if status_filter and j.status != status_filter:
            continue
        algo_info = j.algorithm_info or {}
        jobs.append(
            {
                "job_id": j.job_id,
                "algorithm": f"{algo_info.get('type', '?')}/{algo_info.get('name', '?')}",
                "backend": j.backend,
                "status": j.status,
                "submitted_at": j.submitted_at,
                "run_hash": j.run_hash,
                "input_hashes": j.input_hashes,
                "output_hashes": j.output_hashes,
            }
        )

    return {"jobs": jobs}


@app.tool()
@_structured
@validate_project
def cancel_remote_job(
    project_name: str,
    job_id: str,
) -> str | dict:
    """Cancel a running remote job and update its persisted record."""
    err = _require_remote()
    if err:
        return err

    job, load_err = _load_remote_job(job_id, _current_job_owner(project_name))
    if load_err:
        return load_err

    try:
        job.cancel()  # updates status to "canceled" & saves
    except Exception as e:  # noqa: BLE001
        return f"Failed to cancel job: {e}"

    return {"job_id": job_id, "status": "canceled"}
