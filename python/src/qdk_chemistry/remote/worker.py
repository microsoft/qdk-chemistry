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


def _uses_remote_cache_transport(input_dir: Path) -> bool:
    """Return whether the job uses its shared cache for artifact transport."""
    try:
        manifest = json.loads((input_dir / "manifest.json").read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(manifest.get("remote_cache_transport"))


def _load_remote_cache(input_dir: Path) -> tuple[Any, str | None]:
    """Create the cache described by an input manifest, when available."""
    run_hash = None
    cache_transport = False
    try:
        manifest = json.loads((input_dir / "manifest.json").read_text())
        run_hash = manifest.get("run_hash")
        cache_transport = bool(manifest.get("remote_cache_transport"))
        cache_info = manifest.get("remote_cache")
        if not cache_info or not cache_info.get("name"):
            if cache_transport:
                raise RuntimeError("The remote cache transport was not configured")
            return None, run_hash

        from qdk_chemistry.remote.cache import get_cache  # noqa: PLC0415

        cache_name = cache_info["name"]
        cache_config = {key: value for key, value in cache_info.items() if key != "name"}
        return get_cache(cache_name, **cache_config), run_hash
    except Exception as exc:
        if cache_transport:
            raise RuntimeError("Failed to initialize the remote cache transport") from exc
        logger.warning("Failed to load remote cache", exc_info=True)
        return None, run_hash


def _get_cached_result(cache: Any, run_hash: str | None) -> Any:
    """Return a complete cached result or the cache-miss sentinel."""
    if cache is None or run_hash is None:
        return _CACHE_MISS

    try:
        job = cache.get_job(run_hash)
        if job is None or not job.output_hashes or (job.status or "").lower() not in ("retrieved", "succeeded"):
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
        return items[0] if len(items) == 1 else tuple(items)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to read cached result for run %s", run_hash, exc_info=True)
        return _CACHE_MISS


def _store_cached_result(
    cache: Any,
    run_hash: str | None,
    inputs: dict[str, Any],
    result: Any,
    *,
    required: bool = False,
) -> None:
    """Persist a completed result to the compute node's cache when configured."""
    if cache is None or run_hash is None:
        return

    try:
        from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        output_hashes = collect_content_hashes(result)
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
            ),
        )
    except Exception:
        if required:
            raise
        logger.warning("Failed to store cached result for run %s", run_hash, exc_info=True)


def execute_job(input_dir: str | Path, output_dir: str | Path) -> Any:
    """Execute one serialized algorithm job and write its serialized result."""
    from qdk_chemistry.algorithms import create as create_algorithm  # noqa: PLC0415
    from qdk_chemistry.remote.serialization import (  # noqa: PLC0415
        deserialize_inputs,
        serialize_outputs,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    cache_transport = _uses_remote_cache_transport(input_path)
    cache, run_hash = _load_remote_cache(input_path)
    result = _get_cached_result(cache, run_hash)

    if result is _CACHE_MISS:
        inputs = deserialize_inputs(input_path, cache=cache)
        algorithm = create_algorithm(inputs["algorithm_type"], inputs["algorithm_name"])
        for key, value in inputs["settings"].items():
            algorithm.settings().set(key, value)
        result = algorithm.run(*inputs["args"], **inputs["kwargs"])
        _store_cached_result(cache, run_hash, inputs, result, required=cache_transport)

    serialize_outputs(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the compute-node worker command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing serialized inputs")
    parser.add_argument("--output-dir", required=True, help="Directory for serialized outputs")
    args = parser.parse_args(argv)

    execute_job(args.input_dir, args.output_dir)
    print(json.dumps({"success": True, "output_dir": args.output_dir}))


if __name__ == "__main__":
    main()
