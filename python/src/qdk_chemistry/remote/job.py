"""Persistent job handle for QDK/Chemistry.

A ``Job`` records algorithm metadata, content hashes, and status for
cached computations.  Instances serialise to JSON so that results can
be recovered across sessions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
import pathlib
import tempfile
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdk_chemistry.remote.backends.base import JobStatus, RemoteBackend

__all__ = ["Job"]

logger = logging.getLogger(__name__)

_JOB_FILE_VERSION = 3


def _prepare_persisted_value(value: Any, field: str) -> Any:
    """Normalize supported values and verify that job metadata is JSON-safe.

    Args:
        value: Metadata value to normalize and validate.
        field: Metadata field name used in validation errors.

    """
    from qdk_chemistry.remote.serialization import _jsonable_settings_value  # noqa: PLC0415

    try:
        prepared = _jsonable_settings_value(value)
        json.dumps(prepared)
    except (TypeError, ValueError, RecursionError) as error:
        raise TypeError(f"Persisted job {field} must be JSON-serializable: {error}") from error
    return prepared


class Job:
    """Persistent handle for a cached computation.

    Instances serialise to a JSON file on disk, making them the canonical
    record of a cached algorithm run.

    Attributes:
        job_id:         Short unique identifier for this job.
        backend:        Registered backend name (e.g. ``"local"``).
        backend_config: Dict of configuration that was passed to the backend
                        constructor (pool, gpus, host, …).  Stored so the
                        backend can be re-created from scratch.
        backend_state:  Opaque dict written by the backend during *submit*.
                        Contains whatever the backend needs to poll / cancel /
                        fetch (operation IDs, remote paths, PIDs, …).
        algorithm_info: Dict with ``type``, ``name``, ``settings`` of the
                        algorithm that was submitted.
        status:         Last-known status string.
        submitted_at:   ISO-8601 timestamp of submission.
        file_path:      Path to the job file on disk (``None`` if not
                        persisted yet).
        run_hash:       Deterministic hash of the algorithm, settings, and
                        inputs.  Used for cache lookups.  ``None`` if not
                        computed.
        input_hashes:   Per-item content hashes of the submitted inputs,
                        keyed by namespaced argument name (e.g.
                        ``"args.arg_0"``, ``"kwargs.charge"``).  ``None``
                        if not recorded.
        output_hashes:  Per-item result descriptors.  Each entry is a dict
                        with ``"hash"`` and ``"type"`` keys.  Primitives
                        also carry a ``"value"`` key so they can be
                        reconstructed without a cache backend.  Populated
                        when results are fetched.  ``None`` until results
                        are retrieved.
        output_is_tuple: Whether the retrieved result is a tuple.  ``None``
                 until results are retrieved.
        owner:          Workspace and project permitted to manage the job
                through MCP. ``None`` for unowned SDK jobs.

    """

    def __init__(
        self,
        *,
        job_id: str,
        backend: str,
        backend_config: dict[str, Any],
        backend_state: dict[str, Any],
        algorithm_info: dict[str, Any] | None = None,
        status: str = "submitted",
        submitted_at: str | None = None,
        file_path: str | pathlib.Path | None = None,
        run_hash: str | None = None,
        input_hashes: dict[str, str] | None = None,
        output_hashes: list[dict[str, Any]] | None = None,
        output_is_tuple: bool | None = None,
        owner: dict[str, str | None] | None = None,
    ):
        """Initialise a Job from its constituent parts.

        Args:
            job_id: Unique identifier assigned by the backend.
            backend: Registered backend name.
            backend_config: Configuration used to reconstruct the backend.
            backend_state: Persisted backend-specific job state.
            algorithm_info: Submitted algorithm type, name, and settings.
            status: Initial job status.
            submitted_at: ISO-8601 submission timestamp.
            file_path: Optional path for the persisted job record.
            run_hash: Deterministic hash used for cache lookup.
            input_hashes: Content hashes for submitted inputs.
            output_hashes: Content-hash descriptors for retrieved outputs.
            output_is_tuple: Whether the retrieved result is a tuple.
            owner: Workspace and project permitted to manage this job through MCP.

        """
        self.job_id = job_id
        self.backend = backend
        self.backend_config = backend_config
        self.backend_state = backend_state
        self.algorithm_info = algorithm_info or {}
        self.status = status
        self.submitted_at = submitted_at or datetime.now(timezone.utc).isoformat()
        self.file_path: pathlib.Path | None = pathlib.Path(file_path) if file_path else None
        self.run_hash: str | None = run_hash
        self.input_hashes: dict[str, str] | None = input_hashes
        self.output_hashes: list[dict[str, Any]] | None = output_hashes
        self.output_is_tuple: bool | None = output_is_tuple
        self.owner: dict[str, str | None] | None = owner
        self._active_backend: RemoteBackend | None = None

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representing this job."""
        d: dict[str, Any] = {
            "version": _JOB_FILE_VERSION,
            "job_id": self.job_id,
            "backend": self.backend,
            "backend_config": self.backend_config,
            "backend_state": self.backend_state,
            "algorithm_info": self.algorithm_info,
            "status": self.status,
            "submitted_at": self.submitted_at,
        }
        if self.run_hash is not None:
            d["run_hash"] = self.run_hash
        if self.input_hashes is not None:
            d["input_hashes"] = self.input_hashes
        if self.output_hashes is not None:
            d["output_hashes"] = self.output_hashes
        if self.output_is_tuple is not None:
            d["output_is_tuple"] = self.output_is_tuple
        if self.owner is not None:
            d["owner"] = self.owner
        return _prepare_persisted_value(d, "metadata")

    def save(self, path: str | pathlib.Path | None = None) -> pathlib.Path:
        """Write the job file to disk atomically.

        Args:
            path: Explicit file path.  If *None*, uses :attr:`file_path`
                (which must have been set earlier, e.g. via *job_dir* at
                submit time).

        Returns:
            The path the file was written to.

        Raises:
            ValueError: If no path is available.

        """
        path = pathlib.Path(path) if path else self.file_path
        if path is None:
            raise ValueError("No file path specified.  Pass a path or set job.file_path.")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = pathlib.Path(file.name)
                json.dump(self.to_dict(), file, indent=2)
            os.replace(temporary_path, path)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        self.file_path = path
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Job:
        """Reconstruct a ``Job`` from a previously saved file.

        Args:
            path: Path to a ``*.job.json`` file.

        Returns:
            A fully re-hydrated ``Job``.

        """
        path = pathlib.Path(path)
        data = json.loads(path.read_text())
        version = data.get("version", 1)
        if version > _JOB_FILE_VERSION:
            raise ValueError(f"Unsupported job file version {version} (max supported {_JOB_FILE_VERSION})")
        if "status" not in data:
            raise ValueError("Job file is missing required field 'status'")
        return cls(
            job_id=data["job_id"],
            backend=data["backend"],
            backend_config=data.get("backend_config", {}),
            backend_state=data.get("backend_state", {}),
            algorithm_info=data.get("algorithm_info", {}),
            status=data["status"],
            submitted_at=data.get("submitted_at"),
            file_path=path,
            run_hash=data.get("run_hash"),
            input_hashes=data.get("input_hashes"),
            output_hashes=data.get("output_hashes"),
            output_is_tuple=data.get("output_is_tuple"),
            owner=data.get("owner"),
        )

    @classmethod
    def discover(cls, directory: str | pathlib.Path) -> list[Job]:
        """Find all job files in a directory.

        Args:
            directory: Folder to scan (non-recursively) for
                ``*.job.json`` files.

        Returns:
            List of ``Job`` instances, sorted by
            ``submitted_at`` (oldest first).

        """
        directory = pathlib.Path(directory)
        jobs: list[Job] = []
        for p in directory.glob("*.job.json"):
            try:
                jobs.append(cls.load(p))
            except (ValueError, KeyError, OSError):
                continue  # skip corrupt files
        jobs.sort(key=lambda j: j.submitted_at or "")
        return jobs

    # ── Backend interaction ──────────────────────────────────────────────

    def attach_backend(self, backend: RemoteBackend) -> None:
        """Associate this in-memory job with its submitting backend."""
        self._active_backend = backend

    def detach_backend(self) -> None:
        """Remove the non-persistent backend association."""
        self._active_backend = None

    def _get_backend(self) -> tuple[RemoteBackend, bool]:
        """Return an active backend and whether this job must disconnect it."""
        if self._active_backend is not None:
            return self._active_backend, False

        from qdk_chemistry.remote.backends import get_backend  # noqa: PLC0415

        backend = get_backend(self.backend, **self.backend_config)
        backend.connect()
        return backend, True

    def check(self) -> JobStatus:
        """Query the backend, persist the latest status, and return it."""
        from qdk_chemistry.remote.backends.base import JobState, JobStatus  # noqa: PLC0415

        backend, should_disconnect = self._get_backend()
        try:
            job_status = backend.check(self.backend_state)
        finally:
            if should_disconnect:
                backend.disconnect()

        if JobStatus.normalize_status(self.status) == JobState.RETRIEVED or self.output_hashes is not None:
            self.status = JobState.RETRIEVED
            job_status.status = JobState.RETRIEVED
        else:
            self.status = job_status.status
        if self.file_path is not None:
            self.save()
        return job_status

    def cancel(self) -> None:
        """Cancel the backend job and persist its canceled status."""
        backend, should_disconnect = self._get_backend()
        try:
            backend.cancel(self.backend_state)
        finally:
            if should_disconnect:
                backend.disconnect()

        self.status = "canceled"
        if self.file_path is not None:
            self.save()

    def fetch(
        self,
        local_dir: str | pathlib.Path | None = None,
        *,
        cleanup: bool = False,
    ) -> Any:
        """Download and persist results, then optionally remove backend artifacts.

        Args:
            local_dir: Optional directory to download result files into.
            cleanup: Whether to remove backend job artifacts after successful
                retrieval and persistence.

        Returns:
            The deserialized algorithm results.

        """
        backend, should_disconnect = self._get_backend()
        try:
            result = backend.fetch(self.backend_state, local_dir=local_dir)

            self.status = "retrieved"
            try:
                from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415

                self.output_hashes = collect_content_hashes(result)
                self.output_is_tuple = isinstance(result, tuple)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to collect output hashes for job %s; result will not be cached",
                    self.job_id,
                    exc_info=True,
                )
            if self.file_path is not None:
                self.save()
            if cleanup:
                backend.cleanup_job(self.backend_state)
            return result
        finally:
            if should_disconnect:
                backend.disconnect()

    def cleanup(self) -> None:
        """Remove backend artifacts for this terminal job.

        Repeated cleanup is safe when supported by the backend.

        Raises:
            RuntimeError: If the job has not reached a terminal state.

        """
        if not self.is_terminal:
            raise RuntimeError("Cannot clean up a job before it reaches a terminal state")

        backend, should_disconnect = self._get_backend()
        try:
            backend.cleanup_job(self.backend_state)
        finally:
            if should_disconnect:
                backend.disconnect()

    def wait(self) -> JobStatus:
        """Block until the job reaches a terminal state.

        Returns:
            The final status reported by the backend.

        Raises:
            TimeoutError: If the configured timeout expires before completion.

        """
        from qdk_chemistry.remote.backends.base import (  # noqa: PLC0415
            DEFAULT_POLL_INTERVAL,
            DEFAULT_TIMEOUT,
            JobStatus,
        )

        poll_interval = self.backend_config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        timeout = self.backend_config.get("timeout", DEFAULT_TIMEOUT)
        deadline = time.monotonic() + timeout
        status = JobStatus(job_id=self.job_id, status=self.status)
        backend, should_disconnect = self._get_backend()
        if should_disconnect:
            self.attach_backend(backend)
        try:
            while not self.is_terminal:
                status = self.check()
                if status.is_terminal:
                    return status
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Remote job {self.job_id} did not reach a terminal state within {timeout} seconds\n"
                        f"Last status: {status.status}\n"
                        f"Error: {status.error or 'unknown'}\nLogs:\n{status.logs}"
                    )
                time.sleep(min(poll_interval, remaining))
            return status
        finally:
            if should_disconnect:
                self.detach_backend()
                backend.disconnect()

    # ── Conveniences ─────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state."""
        from qdk_chemistry.remote.backends.base import JobStatus  # noqa: PLC0415

        return JobStatus.is_terminal_status(self.status)

    @property
    def is_successful(self) -> bool:
        """Whether the job completed successfully."""
        from qdk_chemistry.remote.backends.base import JobStatus  # noqa: PLC0415

        return JobStatus.is_successful_status(self.status)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"Job(id={self.job_id!r}, backend={self.backend!r}, status={self.status!r})"
