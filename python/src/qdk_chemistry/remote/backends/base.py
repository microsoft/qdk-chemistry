"""Remote execution of QDK/Chemistry. Base classes for remote backends.

This module provides the abstract base class for remote execution backends.
Backends transfer serialized inputs, submit the worker process, poll its
status, and retrieve serialized outputs.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Type  # noqa: UP035

from qdk_chemistry._core import DuplicateRegistrationError as _DuplicateRegistrationError
from qdk_chemistry.utils.enum import CaseInsensitiveStrEnum

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.remote.job import Job

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 3600.0

__all__ = [
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_TIMEOUT",
    "JobState",
    "JobStatus",
    "RemoteBackend",
    "available_backends",
    "create_remote",
    "get_backend",
    "get_mcp_safe_config_options",
    "register_backend",
]


class JobState(CaseInsensitiveStrEnum):
    """Canonical states in the remote job lifecycle."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    CANCELLED = "cancelled"
    RETRIEVED = "retrieved"


@dataclass
class JobStatus:
    """Status of a remote job.

    Returned by :meth:`RemoteBackend.check` and related helpers.
    """

    TERMINAL_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELED,
            JobState.CANCELLED,
            JobState.RETRIEVED,
        }
    )

    job_id: str
    status: str  # Case-insensitive: submitted, running, succeeded, failed, canceled, retrieved.
    logs: str = ""
    error: str | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def normalize_status(status: str | None) -> str:
        """Return the canonical form of a status string.

        Args:
            status: Status string to normalize. If *None*, it is treated as an empty string.

        Returns:
            The case-folded status string.

        """
        return (status or "").casefold()

    @classmethod
    def is_terminal_status(cls, status: str | None) -> bool:
        """Return whether a status string represents a terminal state.

        Args:
            status: Status string to check.

        Returns:
            *True* if the status represents a terminal state; otherwise, *False*.

        """
        return cls.normalize_status(status) in cls.TERMINAL_STATUSES

    @classmethod
    def is_successful_status(cls, status: str | None) -> bool:
        """Return whether a status string represents successful execution.

        Args:
            status: Status string to check.

        Returns:
            *True* if the status represents successful execution; otherwise, *False*.

        """
        return cls.normalize_status(status) == JobState.SUCCEEDED

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal state.

        Returns:
            *True* if the job has reached a terminal state; otherwise, *False*.

        """
        return self.is_terminal_status(self.status)

    @property
    def is_successful(self) -> bool:
        """Whether the job completed successfully.

        Returns:
            *True* if the job completed successfully; otherwise, *False*.

        """
        return self.is_successful_status(self.status)


class RemoteBackend(ABC):
    """Abstract base class for remote execution backends.

    Backends must implement these core operations:

    - **connect** / **disconnect**: lifecycle management
    - **upload** / **download**: default file transfer to/from the remote system
    - **_submit**: launch a job asynchronously (returns job_id + state)
    - **check**: poll job status
    - **fetch**: download and deserialize results

    Backends may optionally implement:

    - **cancel**: cancel a running or queued job
    - **cleanup_job**: remove artifacts for a terminal job

    :meth:`upload` and :meth:`download` are the normal transport for serialized
    job files. A backend with access to a cache shared by the client and compute
    node may use it instead for cache-backed artifacts, avoiding redundant file
    transfers. Files unavailable through that cache still use the default
    transport.

    The remote node executes ``python -m qdk_chemistry.remote.worker`` which handles
    input deserialization, algorithm execution, caching, and output
    serialization.

    Backend artifacts are retained after a job reaches a terminal state. Callers
    are responsible for removing them with
    :meth:`~qdk_chemistry.remote.job.Job.cleanup` or
    :meth:`~qdk_chemistry.remote.job.Job.fetch` with ``cleanup=True``. Backend cleanup does not remove
    caller-owned local job records or result directories.

    To create a custom backend:

    1. Subclass RemoteBackend
    2. Implement the methods above
    3. Register it from a QdkChemistryPlugin
    4. Optionally declare ``mcp_safe_config_options`` for constructor options
       that MCP clients may control. The default is deny-all.

    Example:
        >>> from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin
        >>> class SlurmBackend(RemoteBackend):
        ...     name = "slurm"
        ...     mcp_safe_config_options = frozenset({"poll_interval", "timeout"})
        ...
        ...     def __init__(self, *, host, partition="default", poll_interval=5.0, timeout=3600.0):
        ...         super().__init__(
        ...             host=host,
        ...             partition=partition,
        ...             poll_interval=poll_interval,
        ...             timeout=timeout,
        ...         )
        ...         self.host = host
        ...         self.partition = partition
        ...
        ...     def connect(self):
        ...         self._client = SlurmClient(self.host)
        ...
        ...     def upload(self, local_path, remote_path):
        ...         self._client.sftp_put(local_path, remote_path)
        ...
        ...     def download(self, remote_path, local_path):
        ...         self._client.sftp_get(remote_path, local_path)
        ...
        ...     def disconnect(self):
        ...         self._client.close()
        ...
        >>> class SlurmPlugin(QdkChemistryPlugin):
        ...     def register(self, registrar: PluginRegistrar):
        ...         registrar.register_remote_backend("slurm", SlurmBackend)

    """

    name: str  # Backend name (e.g., "scheduler", "local")
    mcp_safe_config_options: ClassVar[frozenset[str]] = frozenset()
    """Constructor options that MCP clients may control.

    Concrete backends must declare this attribute directly on the class to
    expose any options through MCP. The default is deny-all.
    """

    def __init__(self, **backend_args: Any) -> None:
        """Store the arguments needed to recreate the concrete backend.

        Args:
            **backend_args: Constructor arguments supplied by the concrete backend.
                Persisted jobs normalize path-like values to strings and require
                every remaining value to be JSON-serializable.

        """
        self._backend_args = backend_args

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the remote system.

        This is called once before any upload/execute/download operations.
        Use this to set up network connections, authenticate with cloud services, etc.

        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the remote system.

        Called after connection-scoped operations are complete. This must not
        cancel submitted jobs or remove artifacts referenced by persisted jobs.

        """

    @abstractmethod
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file from local system to remote system.

        Backend implementations normally call this while staging the files
        produced by input serialization. Cache-backed files may be omitted
        when the compute node can read them from a shared cache.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path on the remote system.

        """

    @abstractmethod
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from remote system to local system.

        Backend implementations normally call this from :meth:`fetch` for
        serialized outputs that were not retrieved through a shared cache.

        Args:
            remote_path: Path to the file on the remote system.
            local_path: Destination path on the local system.

        """

    # ── Async job primitives ─────────────────────────────────────────────

    def submit(self, payload: dict, *, job_dir: str | Path | None = None) -> Job:
        """Submit a job and return immediately with a ``Job``.

        This method does **not** block.
        The returned ``Job`` is self-contained: it can be
        saved to disk, loaded in a different process, and used to
        ``Job.check()``, ``Job.cancel()``, or
        ``Job.fetch()`` results.

        Subclasses must override ``_submit()`` to provide the
        backend-specific implementation.

        Args:
            payload: Execution request containing algorithm metadata and inputs.
            job_dir: Optional directory where the job file is saved
                automatically (as ``<id>.job.json``).  If *None* the job
                is returned in-memory only.

        Returns:
            A ``Job`` that tracks this submission.

        """
        from qdk_chemistry.remote.job import Job, _prepare_persisted_value  # noqa: PLC0415

        backend_name = _prepare_persisted_value(self.name, "backend")
        backend_config = _prepare_persisted_value(dict(self._backend_args), "backend_config")
        algorithm_info = _prepare_persisted_value(
            {
                "type": payload.get("algorithm_type"),
                "name": payload.get("algorithm_name"),
                "settings": payload.get("settings"),
            },
            "algorithm_info",
        )
        run_hash = _prepare_persisted_value(payload.get("run_hash"), "run_hash")
        input_hashes = _prepare_persisted_value(payload.get("input_hashes"), "input_hashes")

        job_id, backend_state = self._submit(payload)
        try:
            persisted_job_id = _prepare_persisted_value(job_id, "job_id")
            persisted_backend_state = _prepare_persisted_value(backend_state, "backend_state")
        except TypeError:
            for operation, label in ((self.cancel, "cancel"), (self.cleanup_job, "clean up")):
                try:
                    operation(backend_state)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to %s remote job %s after metadata validation failed",
                        label,
                        job_id,
                        exc_info=True,
                    )
            raise

        job = Job(
            job_id=persisted_job_id,
            backend=backend_name,
            backend_config=backend_config,
            backend_state=persisted_backend_state,
            algorithm_info=algorithm_info,
            run_hash=run_hash,
            input_hashes=input_hashes,
        )
        job.attach_backend(self)

        if job_dir is not None:
            job_dir = Path(job_dir)
            job.save(job_dir / f"{job_id}.job.json")

        return job

    @abstractmethod
    def _submit(self, payload: dict) -> tuple[str, dict]:
        """Backend-specific async submission.

        Args:
            payload: Execution request.

        Returns:
            A ``(job_id, backend_state)`` tuple where *backend_state* is
            an opaque dict that will be passed back to :meth:`check`,
            :meth:`cancel`, and :meth:`fetch`. Its values must be
            JSON-serializable or path-like. Lifecycle methods must accept the
            persisted representation, in which path-like values are strings.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support async submission")

    @abstractmethod
    def check(self, backend_state: dict) -> JobStatus:
        """Query the current status of a previously submitted job.

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.

        Returns:
            A ``JobStatus`` describing the job's current state.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support status checks")

    def cancel(self, backend_state: dict) -> None:
        """Cancel a running or queued job.

        This operation is optional. The default implementation raises
        :class:`NotImplementedError`.

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support cancellation")

    @abstractmethod
    def fetch(
        self,
        backend_state: dict,
        local_dir: str | Path | None = None,
    ) -> Any:
        """Download and deserialize results for a completed job.

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.
            local_dir: Optional directory to download result files into.
                If *None*, a temporary directory is used and cleaned up
                after deserialization.

        Returns:
            The deserialized algorithm results (same format as the return
            value of the completed algorithm run).

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support result fetching")

    def cleanup_job(self, backend_state: dict) -> None:
        """Remove artifacts owned by a terminal job.

        This operation is optional. Implementations must make repeated calls
        safe and must not remove shared backend work directories. The default
        implementation raises :class:`NotImplementedError`.

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support job cleanup")


# ─────────────────────────────────────────────────────────────────────────────
# Backend Registry
# ─────────────────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[RemoteBackend]] = {}


def _declared_mcp_safe_config_options(cls: type[RemoteBackend]) -> frozenset[str]:
    """Return MCP-safe options declared directly by a backend class.

    Args:
        cls: Concrete backend class to inspect.

    Returns:
        The class's declared MCP-safe constructor options, or an empty set
        when the class does not declare any.

    """
    return cls.__dict__.get("mcp_safe_config_options", frozenset())


def get_mcp_safe_config_options(name: str) -> frozenset[str]:
    """Return MCP-safe constructor options for a registered backend.

    Unknown backends and those without a direct declaration expose no
    configurable options to MCP clients.

    Args:
        name: Registered backend name.

    Returns:
        The explicitly declared MCP-safe constructor options. Returns an empty
        set when the backend is unknown or has no direct declaration.

    """
    cls = _BACKENDS.get(name)
    if cls is None:
        return frozenset()
    return _declared_mcp_safe_config_options(cls)


def _register_backend(name: str, cls: Type[RemoteBackend]) -> Type[RemoteBackend]:  # noqa: UP006
    """Register one backend class after validating registry ownership.

    Args:
        name: Registry name for the backend.
        cls: Backend class to register.

    """
    if name in _BACKENDS:
        raise _DuplicateRegistrationError(f"Remote backend name '{name}' is already registered")
    for registered_name, registered_cls in _BACKENDS.items():
        if registered_cls is cls:
            raise _DuplicateRegistrationError(
                f"Remote backend class '{cls.__module__}.{cls.__qualname__}' is already registered "
                f"with name '{registered_name}'"
            )

    # Declarations are intentionally not inherited: a subclass may change the
    # meaning of constructor parameters and must make its own trust decision.
    safe_options = _declared_mcp_safe_config_options(cls)
    if not isinstance(safe_options, frozenset) or not all(
        isinstance(option, str) and option for option in safe_options
    ):
        raise TypeError(
            f"Remote backend '{cls.__module__}.{cls.__qualname__}' must declare "
            "mcp_safe_config_options as a frozenset of non-empty strings"
        )

    constructor_parameters = inspect.signature(cls.__init__).parameters
    keyword_parameters = {
        parameter_name
        for parameter_name, parameter in constructor_parameters.items()
        if parameter_name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown_options = safe_options - keyword_parameters
    if unknown_options:
        raise TypeError(
            f"Remote backend '{cls.__module__}.{cls.__qualname__}' declares MCP-safe options that are not "
            f"named constructor parameters: {', '.join(sorted(unknown_options))}"
        )

    cls.name = name
    _BACKENDS[name] = cls
    return cls


def register_backend(name: str) -> Callable[[Type[RemoteBackend]], Type[RemoteBackend]]:  # noqa: UP006
    """Decorator to register a backend class with a name.

    Example:
        >>> @register_backend("custom")
        ... class CustomBackend(RemoteBackend):
        ...     ...

    Args:
        name: The backend name (e.g., "custom" or "local").

    Returns:
        Decorator function that registers the backend class.

    Raises:
        DuplicateRegistrationError: If the remote backend name or class is already registered.

    """

    def decorator(cls: Type[RemoteBackend]) -> Type[RemoteBackend]:  # noqa: UP006
        return _register_backend(name, cls)

    return decorator


def get_backend(name: str, **config) -> RemoteBackend:
    """Create a backend instance by name.

    Args:
        name: Backend name (e.g., "custom" or "local")
        **config: Backend-specific configuration.

    Returns:
        Configured RemoteBackend instance

    Raises:
        ValueError: If no backend is registered with that name

    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys()) or "(none)"
        raise ValueError(f"No backend registered with name '{name}'. Available backends: {available}")

    return _BACKENDS[name](**config)


def create_remote(name: str, **config) -> RemoteBackend:
    """Create a configured remote backend instance.

    Args:
        name: Backend name (e.g., "custom" or "local")
        **config: Backend-specific configuration options.

    Returns:
        Configured RemoteBackend instance ready for use

    Examples:
        >>> from qdk_chemistry.remote import create_remote
        >>> from qdk_chemistry.algorithms import create
        >>>
        >>> remote = create_remote("local", timeout=7200, poll_interval=10.0)
        >>> scf = create("scf_solver")
        >>> energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
        ...                       cache="./cache", remote=remote)

    """
    backend = get_backend(name, **config)
    backend.connect()
    return backend


def available_backends() -> list[str]:
    """Return list of registered backend names."""
    return list(_BACKENDS.keys())
