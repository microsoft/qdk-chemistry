"""Remote execution and caching for QDK/Chemistry algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from qdk_chemistry.remote.job import Job

_CACHE_MISS = object()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run — works with any algorithm (Python or C++)
# ─────────────────────────────────────────────────────────────────────────────


def _build_payload_for(algorithm: Any, args: tuple, kwargs: dict) -> dict:
    """Build an execution payload from any algorithm-like object.

    Args:
        algorithm: Algorithm-like object providing execution metadata.
        args: Positional arguments for the algorithm.
        kwargs: Keyword arguments for the algorithm.

    """
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
    """Hash result items, persist DataClass blobs, update job in cache.

    Args:
        cache: Cache backend receiving result data and job metadata.
        run_hash: Deterministic cache key for the execution.
        job: Job record to update with output hashes.
        result: Algorithm result to persist.

    """
    from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415

    job.output_hashes = collect_content_hashes(result)
    job.output_is_tuple = isinstance(result, tuple)
    job.status = "retrieved"

    items = result if isinstance(result, tuple) else (result,)
    for entry, item in zip(job.output_hashes, items, strict=False):
        if "value" not in entry:
            cache.put_data(entry["hash"], item)

    cache.put_job(run_hash, job)


def _reconstruct_from_cache(cache: Any, job: Any) -> Any:
    """Reconstruct the full result from cached data, or return the cache-miss sentinel.

    Args:
        cache: Cache backend containing result data.
        job: Job record containing output-hash descriptors.

    """
    if job.output_hashes is None or job.output_is_tuple is None:
        return _CACHE_MISS

    items: list[Any] = []
    for entry in job.output_hashes:
        if "value" in entry:
            items.append(entry["value"])
        else:
            data = cache.get_data(entry["hash"])
            if data is None:
                return _CACHE_MISS
            items.append(data)
    if job.output_is_tuple:
        return tuple(items)
    return items[0] if len(items) == 1 else _CACHE_MISS


def submit(
    algorithm: Any,
    *args: Any,
    remote: Any,
    job_dir: str | Path | None = None,
    **kwargs: Any,
) -> Job:
    """Submit an algorithm for remote execution without blocking.

    Args:
        algorithm: Algorithm-like object to execute remotely.
        *args: Positional arguments for the algorithm.
        remote: Remote backend name or connected backend instance.
        job_dir: Optional directory where the job record is saved.
        **kwargs: Keyword arguments for the algorithm.

    Returns:
        A job handle that can be checked, canceled, fetched, or waited on.

    """
    from qdk_chemistry.remote.backends import get_backend  # noqa: PLC0415

    owns_backend = isinstance(remote, str)
    if owns_backend:
        backend = get_backend(remote)
        backend.connect()
    else:
        backend = remote

    try:
        job = backend.submit(_build_payload_for(algorithm, args, kwargs), job_dir=job_dir)
        if owns_backend:
            job.detach_backend()
        return job
    finally:
        if owns_backend:
            backend.disconnect()


def _run_uncached(algorithm: Any, remote: Any, args: tuple, kwargs: dict) -> Any:
    """Execute without caching, locally or through a remote job.

    Args:
        algorithm: Algorithm-like object to execute.
        remote: Remote backend name or instance, or ``None`` for local execution.
        args: Positional arguments for the algorithm.
        kwargs: Keyword arguments for the algorithm.

    """
    if remote is None:
        return algorithm.run(*args, **kwargs)

    job = submit(algorithm, *args, remote=remote, **kwargs)
    final_status = job.wait()
    if not job.is_successful:
        raise RuntimeError(
            f"Remote job {job.job_id} ended with status: {final_status.status}\n"
            f"Error: {final_status.error or 'unknown'}\nLogs:\n{final_status.logs}"
        )
    result = job.fetch()
    job.cleanup()
    return result


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
            complete caller-side records are cache hits whether or not the
            backend is shared. Shared backends are also used by the compute
            node as transport. A ``TieredCache`` can combine local and shared
            backends.
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
            if job.output_hashes is not None:
                result = _reconstruct_from_cache(resolved_cache, job)
                if result is not _CACHE_MISS:
                    return result

            # 1b) Still in-flight → resume polling
            if not job.is_terminal:
                if _on_job_submitted is not None:
                    _on_job_submitted(job)
                job.wait()

            # 1c) Execution finished but cached outputs are unavailable → fetch again
            if job.is_successful:
                result = job.fetch()
                _store_result(resolved_cache, run_hash, job, result)
                job.cleanup()
                return result

            # 1d) Failed → fall through and re-submit

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

            final_status = job.wait()

            if not job.is_successful:
                resolved_cache.put_job(run_hash, job)
                raise RuntimeError(
                    f"Remote job {job.job_id} ended with status: {final_status.status}\n"
                    f"Error: {final_status.error or 'unknown'}\nLogs:\n{final_status.logs}"
                )

            # If the remote wrote results to a shared cache, reconstruct
            # from there directly — avoiding an expensive fetch/download.
            result = _CACHE_MISS
            if resolved_remote_cache is not None and resolved_remote_cache.is_shared:
                remote_job = resolved_remote_cache.get_job(run_hash)
                if remote_job is not None and remote_job.output_hashes is not None:
                    result = _reconstruct_from_cache(resolved_remote_cache, remote_job)
                    if result is not _CACHE_MISS:
                        job.output_hashes = remote_job.output_hashes
                        job.output_is_tuple = remote_job.output_is_tuple
                        job.status = "retrieved"

            if result is _CACHE_MISS:
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
    if remote is not None:
        job.cleanup()
    return result
