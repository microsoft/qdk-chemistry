"""Remote execution of QDK/Chemistry. Base classes for remote backends.

This module provides the abstract base class for remote execution backends.
Backends define three user-configurable steps:
1. Upload: Transfer input HDF5 file to remote system
2. Execute: Run a generated Python script on the remote system
3. Download: Transfer output HDF5 file back to local system
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.remote.job import Job


@dataclass
class JobStatus:
    """Status of a remote job.

    Returned by :meth:`RemoteBackend.check` and related helpers.
    """

    job_id: str
    status: str  # "submitted", "running", "succeeded", "failed", "canceled"
    logs: str = ""
    error: str | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RemoteBackend(ABC):
    """Abstract base class for remote execution backends.

    Backends must implement:

    - **connect** / **disconnect**: lifecycle management
    - **upload** / **download**: file transfer to/from the remote system
    - **_submit**: launch a job asynchronously (returns job_id + state)
    - **check**: poll job status
    - **fetch**: download and deserialize results

    The remote node executes ``qdk_chem_cli remote-run`` which handles
    input deserialization, algorithm execution, caching, and output
    serialization.

    To create a custom backend:

    1. Subclass RemoteBackend
    2. Implement the methods above
    3. Register with @register_backend("name")

    Example:
        >>> @register_backend("slurm")
        ... class SlurmBackend(RemoteBackend):
        ...     name = "slurm"
        ...
        ...     def __init__(self, **config):
        ...         super().__init__(**config)
        ...         self.partition = config.get("partition", "default")
        ...
        ...     def connect(self):
        ...         self._client = SlurmClient(self.config.get("host"))
        ...
        ...     def upload(self, local_path, remote_path):
        ...         self._client.sftp_put(local_path, remote_path)
        ...
        ...     def download(self, remote_path, local_path):
        ...         self._client.sftp_get(remote_path, local_path)
        ...
        ...     def disconnect(self):
        ...         self._client.close()

    """

    name: str  # Backend name (e.g., "ssh", "local")

    def __init__(self, **config: Any):
        """Initialize the backend with configuration options.

        Args:
            **config: Backend-specific options such as:
                - host: Remote host to connect to
                - timeout: Maximum execution time in seconds
                - remote_workdir: Working directory on remote system
                - Any other backend-specific settings

        """
        self.config = config

        # Common defaults
        self.remote_workdir = config.get("remote_workdir", "/tmp/qdk_remote")
        self.timeout = config.get("timeout", 3600)

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the remote system.

        This is called once before any upload/execute/download operations.
        Use this to set up SSH connections, authenticate with cloud services, etc.

        """

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up connection to the remote system.

        Called after all operations are complete. Use this to close connections,
        clean up temporary files, etc.

        """

    @abstractmethod
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file from local system to remote system.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path on the remote system.

        """

    @abstractmethod
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from remote system to local system.

        Args:
            remote_path: Path to the file on the remote system.
            local_path: Destination path on the local system.

        """

    def submit_and_wait(self, payload: dict) -> dict:
        """Submit a job and block until results are available.

        Uses ``_submit()`` to launch the job asynchronously, polls via
        :meth:`check`, and then :meth:`fetch` es the result.

        Args:
            payload: Execution request containing algorithm_type,
                algorithm_name, settings, args, kwargs.

        Returns:
            The deserialized result from the algorithm.

        """
        import time  # noqa: PLC0415

        job_id, backend_state = self._submit(payload)
        poll_interval = getattr(self, "poll_interval", 5)

        while True:
            status = self.check(backend_state)
            if status.status in ("Succeeded", "Failed", "canceled"):
                break
            time.sleep(poll_interval)

        if status.status != "Succeeded":
            raise RuntimeError(
                f"Remote job {job_id} ended with status: {status.status}\n"
                f"Error: {status.error or 'unknown'}\nLogs:\n{status.logs}"
            )

        return self.fetch(backend_state)

    # ── Async job primitives ─────────────────────────────────────────────

    def submit(self, payload: dict, *, job_dir: str | Path | None = None) -> Job:
        """Submit a job and return immediately with a ``Job``.

        Unlike :meth:`submit_and_wait`, this method does **not** block.
        The returned ``Job`` is self-contained: it can be
        saved to disk, loaded in a different process, and used to
        ``Job.check()``, ``Job.cancel()``, or
        ``Job.fetch()`` results.

        Subclasses must override ``_submit()`` to provide the
        backend-specific implementation.

        Args:
            payload: Execution request (same format as *submit_and_wait*).
            job_dir: Optional directory where the job file is saved
                automatically (as ``job_<id>.json``).  If *None* the job
                is returned in-memory only.

        Returns:
            A ``Job`` that tracks this submission.

        """
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        job_id, backend_state = self._submit(payload)

        job = Job(
            job_id=job_id,
            backend=self.name,
            backend_config=self.config,
            backend_state=backend_state,
            algorithm_info={
                "type": payload.get("algorithm_type"),
                "name": payload.get("algorithm_name"),
                "settings": payload.get("settings"),
            },
            run_hash=payload.get("run_hash"),
            input_hashes=payload.get("input_hashes"),
        )

        if job_dir is not None:
            job_dir = Path(job_dir)
            job.save(job_dir / f"job_{job_id}.json")

        return job

    def _submit(self, payload: dict) -> tuple[str, dict]:
        """Backend-specific async submission (override in subclasses).

        Args:
            payload: Execution request.

        Returns:
            A ``(job_id, backend_state)`` tuple where *backend_state* is
            an opaque dict that will be passed back to :meth:`check`,
            :meth:`cancel`, and :meth:`fetch`.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support async submission")

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

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support cancellation")

    def fetch(self, backend_state: dict, local_dir: str | Path | None = None) -> dict:
        """Download and deserialize results for a completed job.

        Args:
            backend_state: The opaque state dict produced by ``_submit()``.
            local_dir: Optional directory to download result files into.
                If *None*, a temporary directory is used and cleaned up
                after deserialization.

        Returns:
            The deserialized algorithm results (same format as the return
            value of :meth:`submit_and_wait`).

        """
        raise NotImplementedError(f"Backend '{self.name}' does not support result fetching")


# ─────────────────────────────────────────────────────────────────────────────
# Backend Registry
# ─────────────────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[RemoteBackend]] = {}


def register_backend(name: str) -> Callable[[type[RemoteBackend]], type[RemoteBackend]]:
    """Decorator to register a backend class with a name.

    Example:
        >>> @register_backend("ssh")
        ... class SSHBackend(RemoteBackend):
        ...     ...

    Args:
        name: The backend name (e.g., "ssh" or "local").

    Returns:
        Decorator function that registers the backend class.

    """

    def decorator(cls: type[RemoteBackend]) -> type[RemoteBackend]:
        cls.name = name
        _BACKENDS[name] = cls
        return cls

    return decorator


def get_backend(name: str, **config) -> RemoteBackend:
    """Create a backend instance by name.

    Args:
        name: Backend name (e.g., "ssh" or "local")
        **config: Backend-specific configuration

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
        name: Backend name (e.g., "ssh" or "local")
        **config: Backend-specific configuration options

    Returns:
        Configured RemoteBackend instance ready for use

    Examples:
        >>> from qdk_chemistry.remote import create_remote
        >>> from qdk_chemistry.algorithms import create
        >>>
        >>> remote = create_remote("ssh", host="compute-server.example.com", timeout=7200)
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
