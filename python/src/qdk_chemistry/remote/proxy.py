"""Remote algorithm proxy for QDK/Chemistry.

This module provides the RemoteAlgorithmProxy class that wraps algorithms
to redirect their execution to remote systems via configurable backends.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from qdk_chemistry.algorithms.base import Algorithm
    from qdk_chemistry.remote.backends import RemoteBackend
    from qdk_chemistry.remote.job import Job


class RemoteAlgorithmProxy:
    """Internal proxy used by the ``run()`` function for remote execution.

    Not intended for direct use. Use ``algorithm.run(..., remote=..., cache=...)``
    instead.

    """

    def __init__(
        self,
        algorithm: Algorithm,
        remote: str | RemoteBackend,
        **remote_config: Any,
    ):
        """Initialize the remote algorithm proxy.

        Args:
            algorithm: The algorithm instance to wrap.
            remote: Either a backend name (str) like "local" or a name
                registered by a plugin,
                or a pre-configured RemoteBackend instance.
            **remote_config: Backend-specific configuration options (only used
                if remote is a string). Options include:

                - host: Remote host to connect to
                - python_path: Remote Python interpreter path
                - timeout: Maximum execution time in seconds
                - remote_workdir: Working directory on remote system
                - Other backend-specific settings

        """
        self._algorithm = algorithm
        self._remote_config = remote_config

        # Store either the backend instance or the name for lazy creation
        if isinstance(remote, str):
            self._backend_name: str | None = remote
            self._backend: RemoteBackend | None = None
        else:
            self._backend_name = None
            self._backend = remote

    @cached_property
    def _resolved_backend(self) -> RemoteBackend:
        """Lazily resolve and connect to the backend."""
        if self._backend is not None:
            # Already have a backend instance
            return self._backend

        # Create backend from name
        from qdk_chemistry.remote.backends import get_backend  # noqa: PLC0415

        backend = get_backend(self._backend_name, **self._remote_config)
        backend.connect()
        return backend

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the algorithm on the remote system.

        This method:
        1. Serializes all arguments to HDF5
        2. Uploads to the remote system
        3. Generates a python script that deserializes the inputs, runs the algorithm, and serializes the outputs
        4. Executes the script on the remote system (possible polling)
        5. Downloads the results
        6. Deserializes all results

        Args:
            *args: Positional arguments for the algorithm's run method.
            **kwargs: Keyword arguments for the algorithm's run method.

        Returns:
            The result from the algorithm (e.g., (energy, wavefunction) tuple).

        Raises:
            RuntimeError: If remote execution fails.
            ConnectionError: If connection to remote system fails.

        """
        # Build the execution payload
        payload = _build_payload_for(self._algorithm, args, kwargs)

        backend = self._resolved_backend
        try:
            return backend.submit_and_wait(payload)
        finally:
            if self._backend_name is not None:
                backend.disconnect()
                self.__dict__.pop("_resolved_backend", None)

    def submit(self, *args: Any, job_dir: str | Path | None = None, **kwargs: Any) -> Job:
        """Submit the algorithm for remote execution without blocking.

        Returns a ``Job`` that persists to disk (if *job_dir*
        is given) and can be used to ``Job.check()``,
        ``Job.cancel()``, or ``Job.fetch()`` — even
        from a completely different script.

        Args:
            *args: Positional arguments for the algorithm's run method.
            job_dir: Optional directory where the job file is saved
                automatically (as ``job_<id>.json``).
            **kwargs: Keyword arguments for the algorithm's run method.

        Returns:
            A ``Job`` that tracks this submission.

        """
        payload = _build_payload_for(self._algorithm, args, kwargs)
        return self._resolved_backend.submit(payload, job_dir=job_dir)

    def settings(self):
        """Forward settings access to the wrapped algorithm.

        Returns:
            The algorithm's Settings object.

        """
        return self._algorithm.settings()

    def __getattr__(self, name: str) -> Any:
        """Forward all other attribute access to the wrapped algorithm."""
        return getattr(self._algorithm, name)

    def __repr__(self) -> str:
        """Return string representation of the proxy."""
        if self._backend_name:
            backend_name = self._backend_name
        elif self._backend is not None:
            backend_name = self._backend.name
        else:
            backend_name = "unknown"
        return f"<RemoteProxy({self._algorithm.name()}) -> {backend_name}>"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run — works with any algorithm (Python or C++)
# ─────────────────────────────────────────────────────────────────────────────


def _build_payload_for(algorithm: Any, args: tuple, kwargs: dict) -> dict:
    """Build an execution payload from any algorithm-like object."""
    import contextlib  # noqa: PLC0415

    from qdk_chemistry.data._hashing import _item_content_hash  # noqa: PLC0415

    payload: dict[str, Any] = {
        "algorithm_type": algorithm.type_name(),
        "algorithm_name": algorithm.name(),
        "settings": algorithm.settings().to_dict(),
        "args": args,
        "kwargs": kwargs,
    }

    with contextlib.suppress(Exception):
        payload["run_hash"] = algorithm.hash(*args, **kwargs)

    input_hashes: dict[str, str] = {}
    for i, arg in enumerate(args):
        input_hashes[f"args.arg_{i}"] = _item_content_hash(arg)
    for key, val in kwargs.items():
        input_hashes[f"kwargs.{key}"] = _item_content_hash(val)
    if input_hashes:
        payload["input_hashes"] = input_hashes

    return payload


def _store_result(cache: Any, run_hash: str, job: Any, result: Any) -> None:
    """Hash result items, persist DataClass blobs, update job in cache."""
    from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415

    job.output_hashes = collect_content_hashes(result)
    job.status = "retrieved"

    items = result if isinstance(result, tuple) else (result,)
    for entry, item in zip(job.output_hashes, items, strict=False):
        if "value" not in entry:
            cache.put_data(entry["hash"], item)

    cache.put_job(run_hash, job)


def _reconstruct_from_cache(cache: Any, job: Any) -> Any | None:
    """Reconstruct the full result from cached data, or None on partial miss."""
    items: list[Any] = []
    for entry in job.output_hashes:
        if "value" in entry:
            items.append(entry["value"])
        else:
            data = cache.get_data(entry["hash"])
            if data is None:
                return None
            items.append(data)
    return items[0] if len(items) == 1 else tuple(items)


def _poll_until_done(job: Any) -> None:
    """Block until the job reaches a terminal state."""
    from qdk_chemistry.remote.backends.base import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT  # noqa: PLC0415

    poll_interval = job.backend_config.get("poll_interval", DEFAULT_POLL_INTERVAL)
    timeout = job.backend_config.get("timeout", DEFAULT_TIMEOUT)
    deadline = time.monotonic() + timeout
    while not job.is_terminal:
        status = job.check()
        if status.is_terminal:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Remote job {job.job_id} did not reach a terminal state within {timeout} seconds\n"
                f"Last status: {status.status}\n"
                f"Error: {status.error or 'unknown'}\nLogs:\n{status.logs}"
            )
        time.sleep(min(poll_interval, remaining))


def _run_uncached(algorithm: Any, remote: Any, args: tuple, kwargs: dict) -> Any:
    """Execute without caching — locally or via a remote proxy."""
    if remote is not None:
        proxy = RemoteAlgorithmProxy(algorithm, remote)
        return proxy.run(*args, **kwargs)
    return algorithm.run(*args, **kwargs)


def run(
    algorithm: Any,
    *args: Any,
    cache: Any = None,
    remote: Any = None,
    force_rerun: bool = False,
    _on_job_submitted: Callable[[Job], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute any algorithm with optional caching and remote backend.

    Works with both Python and C++ algorithm implementations — anything
    with ``run()``, ``hash()``, ``type_name()``, ``name()``, and
    ``settings()`` methods.

    On a cache hit the result is returned immediately.  On a miss the
    algorithm is executed (locally or via *remote*) and the result is
    stored.  If a previous remote submission is still in-flight, polling
    resumes automatically — no duplicate submission.

    Args:
        algorithm: Any algorithm instance (from ``create(...)``).
        *args: Positional arguments for ``algorithm.run()``.
        cache: Cache backend — a ``CacheBackend``, a path (``str`` /
            ``Path`` → ``FolderCache``), or ``None``. For remote execution,
            shared backends are also used by the compute node. A
            ``TieredCache`` can combine local and shared backends.
        remote: Remote backend name or instance, or ``None`` for local.
        force_rerun: If ``True``, skip the cache lookup and re-execute,
            overwriting any previously cached result.
        _on_job_submitted: Internal callback invoked after a remote job handle
            is persisted to the local cache.
        **kwargs: Keyword arguments for ``algorithm.run()``.

    Returns:
        The algorithm result (e.g. ``(energy, wavefunction)``).

    Examples::

        # "scheduler" is provided by an installed plugin
        # Shared cache — both sides use the same backend
        shared = FolderCache("/mnt/shared/cache", is_shared=True)
        energy, wfn = run(scf, mol, 0, 1, "cc-pvdz",
                  cache=shared, remote="scheduler")

        # Local cache backed by a shared cache for remote execution
        cache = TieredCache([FolderCache("./cache"), shared])
        energy, wfn = run(scf, mol, 0, 1, "cc-pvdz",
                  cache=cache, remote="scheduler")

    """
    from qdk_chemistry.remote.cache import resolve_cache  # noqa: PLC0415

    resolved_cache = resolve_cache(cache)
    resolved_remote_cache = resolved_cache.for_remote() if remote is not None and resolved_cache is not None else None

    # No cache — just run
    if resolved_cache is None:
        return _run_uncached(algorithm, remote, args, kwargs)

    payload = _build_payload_for(algorithm, args, kwargs)
    run_hash = payload.get("run_hash")
    if run_hash is None:
        return _run_uncached(algorithm, remote, args, kwargs)

    # 1) Check the cache (skip on force_rerun)
    if not force_rerun:
        job = resolved_cache.get_job(run_hash)

        if job is not None:
            # 1a) Completed with outputs → reconstruct
            if job.is_terminal and job.output_hashes:
                result = _reconstruct_from_cache(resolved_cache, job)
                if result is not None:
                    return result

            # 1b) Still in-flight → resume polling
            if not job.is_terminal:
                if _on_job_submitted is not None:
                    _on_job_submitted(job)
                _poll_until_done(job)
                if job.is_successful:
                    result = job.fetch()
                    _store_result(resolved_cache, run_hash, job, result)
                    return result

            # 1c) Failed → fall through and re-submit

    # 2) Cache miss — execute
    if remote is not None:
        from qdk_chemistry.remote.backends import get_backend  # noqa: PLC0415

        owns_backend = isinstance(remote, str)
        if isinstance(remote, str):
            backend = get_backend(remote)
            backend.connect()
        else:
            backend = remote

        try:
            # If the caller provided a remote-reachable cache, serialize its
            # coordinates into the payload so the remote script can use it.
            if resolved_remote_cache is not None:
                payload["remote_cache"] = {
                    "name": resolved_remote_cache.name,
                    **resolved_remote_cache.to_config(),
                }
                # When the cache is shared (both sides see the same data),
                # pass the backend object so serialize_inputs can skip files
                # that already exist in the cache.
                if resolved_remote_cache.is_shared:
                    payload["remote_cache_backend"] = resolved_remote_cache

            if force_rerun:
                payload["force_rerun"] = True

            job = backend.submit(payload)
            job.run_hash = run_hash
            resolved_cache.put_job(run_hash, job)
            if _on_job_submitted is not None:
                _on_job_submitted(job)

            _poll_until_done(job)

            if not job.is_successful:
                resolved_cache.put_job(run_hash, job)
                raise RuntimeError(f"Remote job {job.job_id} ended with status: {job.status}")

            # If the remote wrote results to a shared cache, reconstruct
            # from there directly — avoiding an expensive fetch/download.
            result = None
            if resolved_remote_cache is not None and resolved_remote_cache.is_shared:
                remote_job = resolved_remote_cache.get_job(run_hash)
                if remote_job is not None and remote_job.output_hashes:
                    result = _reconstruct_from_cache(resolved_remote_cache, remote_job)
                    if result is not None:
                        job.output_hashes = remote_job.output_hashes
                        job.status = "retrieved"

            if result is None:
                result = job.fetch()
        finally:
            if owns_backend:
                backend.disconnect()
    else:
        result = algorithm.run(*args, **kwargs)

        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        job = Job(
            job_id=run_hash[:12],
            backend="local",
            backend_config={},
            backend_state={},
            algorithm_info={
                "type": payload.get("algorithm_type"),
                "name": payload.get("algorithm_name"),
                "settings": payload.get("settings"),
            },
            status="retrieved",
            run_hash=run_hash,
            input_hashes=payload.get("input_hashes"),
        )

    _store_result(resolved_cache, run_hash, job, result)
    return result
