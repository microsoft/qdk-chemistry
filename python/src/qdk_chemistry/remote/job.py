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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_JOB_FILE_VERSION = 2


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
                        keyed by argument name (e.g. ``"arg_0"``,
                        ``"charge"``).  ``None`` if not recorded.
        output_hashes:  Per-item result descriptors.  Each entry is a dict
                        with ``"hash"`` and ``"type"`` keys.  Primitives
                        also carry a ``"value"`` key so they can be
                        reconstructed without a cache backend.  Populated
                        by :meth:`fetch`.  ``None`` until results are
                        retrieved.

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
        file_path: str | Path | None = None,
        run_hash: str | None = None,
        input_hashes: dict[str, str] | None = None,
        output_hashes: list[dict[str, Any]] | None = None,
    ):
        """Initialise a Job from its constituent parts."""
        self.job_id = job_id
        self.backend = backend
        self.backend_config = backend_config
        self.backend_state = backend_state
        self.algorithm_info = algorithm_info or {}
        self.status = status
        self.submitted_at = submitted_at or datetime.now(timezone.utc).isoformat()
        self.file_path: Path | None = Path(file_path) if file_path else None
        self.run_hash: str | None = run_hash
        self.input_hashes: dict[str, str] | None = input_hashes
        self.output_hashes: list[dict[str, Any]] | None = output_hashes

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
        return d

    def save(self, path: str | Path | None = None) -> Path:
        """Write the job file to disk.

        Args:
            path: Explicit file path.  If *None*, uses :attr:`file_path`
                (which must have been set earlier, e.g. via *job_dir* at
                submit time).

        Returns:
            The path the file was written to.

        Raises:
            ValueError: If no path is available.

        """
        path = Path(path) if path else self.file_path
        if path is None:
            raise ValueError("No file path specified.  Pass a path or set job.file_path.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        self.file_path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> Job:
        """Reconstruct a ``Job`` from a previously saved file.

        Args:
            path: Path to a ``*.job.json`` file.
        Returns:
            A fully re-hydrated ``Job`` ready for
            :meth:`check`, :meth:`cancel`, or :meth:`fetch`.

        """
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            job_id=data["job_id"],
            backend=data["backend"],
            backend_config=data.get("backend_config", {}),
            backend_state=data.get("backend_state", {}),
            algorithm_info=data.get("algorithm_info", {}),
            status=data.get("status", "unknown"),
            submitted_at=data.get("submitted_at"),
            file_path=path,
            run_hash=data.get("run_hash"),
            input_hashes=data.get("input_hashes"),
            output_hashes=data.get("output_hashes"),
        )

    @classmethod
    def discover(cls, directory: str | Path) -> list[Job]:
        """Find all job files in a directory.

        Args:
            directory: Folder to scan (non-recursively) for
                ``*.job.json`` files.

        Returns:
            List of ``Job`` instances, sorted by
            ``submitted_at`` (oldest first).

        """
        directory = Path(directory)
        jobs: list[Job] = []
        for p in directory.glob("*.job.json"):
            try:
                jobs.append(cls.load(p))
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # skip corrupt files
        jobs.sort(key=lambda j: j.submitted_at or "")
        return jobs

    # ── Conveniences ─────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state."""
        return (self.status or "").lower() in ("succeeded", "failed", "canceled", "cancelled", "retrieved")

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"Job(id={self.job_id!r}, backend={self.backend!r}, status={self.status!r})"
