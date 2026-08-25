"""Compute-node entrypoint for serialized QDK/Chemistry jobs."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_CACHE_MISS = object()


def _load_remote_cache(input_dir: Path) -> tuple[Any, str | None]:
    """Create the cache described by an input manifest, when available.

    Args:
        input_dir: Directory containing the serialized input manifest.

    """
    run_hash = None
    try:
        from qdk_chemistry.remote.serialization import _load_manifest  # noqa: PLC0415

        manifest = _load_manifest(input_dir / "manifest.json")
        run_hash = manifest.get("run_hash")
        cache_info = manifest.get("remote_cache")
        if not cache_info or not cache_info.get("name"):
            return None, run_hash

        from qdk_chemistry.remote.cache import get_cache  # noqa: PLC0415

        cache_name = cache_info["name"]
        cache_config = {key: value for key, value in cache_info.items() if key != "name"}
        return get_cache(cache_name, **cache_config), run_hash
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load remote cache", exc_info=True)
        return None, run_hash


def _get_cached_result(cache: Any, run_hash: str | None) -> Any:
    """Return a complete cached result or the cache-miss sentinel.

    Args:
        cache: Cache backend to query.
        run_hash: Deterministic execution hash to look up.

    """
    if cache is None or run_hash is None:
        return _CACHE_MISS

    try:
        from qdk_chemistry.remote.backends.base import JobState, JobStatus  # noqa: PLC0415

        job = cache.get_job(run_hash)
        status = JobStatus.normalize_status(job.status) if job is not None else None
        if (
            job is None
            or job.output_hashes is None
            or job.output_is_tuple is None
            or status not in (JobState.RETRIEVED, JobState.SUCCEEDED)
        ):
            return _CACHE_MISS

        items: list[Any] = []
        for entry in job.output_hashes:
            if "value" in entry:
                items.append(entry["value"])
                continue
            data = cache.get_data(entry["hash"])
            if data is None:
                return _CACHE_MISS
            items.append(data)
        if job.output_is_tuple:
            return tuple(items)
        return items[0] if len(items) == 1 else _CACHE_MISS
    except Exception:  # noqa: BLE001
        logger.warning("Failed to read cached result for run %s", run_hash, exc_info=True)
        return _CACHE_MISS


def _store_cached_result(cache: Any, run_hash: str | None, inputs: dict[str, Any], result: Any) -> None:
    """Persist a completed result to the compute node's cache when configured.

    Args:
        cache: Cache backend receiving the completed result.
        run_hash: Deterministic execution hash for the result.
        inputs: Deserialized algorithm metadata and arguments.
        result: Completed algorithm result to persist.

    """
    if cache is None or run_hash is None:
        return

    try:
        from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        output_hashes = collect_content_hashes(result)
        output_is_tuple = isinstance(result, tuple)
        result_items = result if isinstance(result, tuple) else (result,)
        for entry, item in zip(output_hashes, result_items, strict=False):
            if "value" not in entry:
                cache.put_data(entry["hash"], item)

        cache.put_job(
            run_hash,
            Job(
                job_id=run_hash[:12],
                backend="remote",
                backend_config={},
                backend_state={},
                algorithm_info={
                    "type": inputs["algorithm_type"],
                    "name": inputs["algorithm_name"],
                    "settings": inputs["settings"],
                },
                status="retrieved",
                run_hash=run_hash,
                input_hashes=inputs.get("input_hashes"),
                output_hashes=output_hashes,
                output_is_tuple=output_is_tuple,
            ),
        )
    except Exception:  # noqa: BLE001
        logger.warning("Failed to store cached result for run %s", run_hash, exc_info=True)


def execute_job(input_dir: str | Path, output_dir: str | Path) -> Any:
    """Execute one serialized algorithm job and write its serialized result.

    Args:
        input_dir: Directory containing serialized algorithm inputs.
        output_dir: Directory to receive serialized results.

    """
    from qdk_chemistry.algorithms import create as create_algorithm  # noqa: PLC0415
    from qdk_chemistry.remote.serialization import (  # noqa: PLC0415
        deserialize_inputs,
        serialize_outputs,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    cache, run_hash = _load_remote_cache(input_path)
    inputs = deserialize_inputs(input_path, cache=cache)
    result = _CACHE_MISS if inputs["force_rerun"] else _get_cached_result(cache, run_hash)

    if result is _CACHE_MISS:
        algorithm = create_algorithm(inputs["algorithm_type"], inputs["algorithm_name"])
        for key, value in inputs["settings"].items():
            algorithm.settings().set(key, value)
        result = algorithm.run(*inputs["args"], **inputs["kwargs"])
        _store_cached_result(cache, run_hash, inputs, result)

    serialize_outputs(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the compute-node worker command.

    Args:
        argv: Command-line arguments, or ``None`` to read process arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing serialized inputs")
    parser.add_argument("--output-dir", required=True, help="Directory for serialized outputs")
    args = parser.parse_args(argv)

    execute_job(args.input_dir, args.output_dir)
    print(json.dumps({"success": True, "output_dir": args.output_dir}))


if __name__ == "__main__":
    main()
