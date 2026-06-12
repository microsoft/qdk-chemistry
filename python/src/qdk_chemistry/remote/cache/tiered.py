"""Tiered (layered) cache backend for QDK/Chemistry.

Chains multiple cache backends in priority order so that reads hit the
fastest / closest tier first and writes propagate to every tier.

Typical setup pairs a fast local cache with a shared network cache::

    cache = TieredCache([
        FolderCache("./cache"),             # L1 — fast local
        FolderCache("/shared/team_cache"),   # L2 — shared
    ])

Read path
    Each tier is checked in order.  On a hit in a slower tier the result
    is *backfilled* into all faster tiers so subsequent reads are local.

Write path
    Data is written to **every** tier (write-through).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdk_chemistry.remote.cache.base import CacheBackend

if TYPE_CHECKING:
    from qdk_chemistry.data.base import DataClass
    from qdk_chemistry.remote.job import Job


class TieredCache(CacheBackend):
    """Composite cache that layers multiple backends.

    Args:
        tiers: Ordered list of cache backends, fastest first.

    Raises:
        ValueError: If *tiers* is empty.

    Example::

        from qdk_chemistry.remote.cache import FolderCache, TieredCache

        cache = TieredCache([
            FolderCache("./local_cache"),
            FolderCache("/shared/team_cache"),
        ])
        energy, wfn = scf.run(mol, 0, 1, "cc-pvdz", cache=cache)

    """

    name = "tiered"

    def __init__(self, tiers: list[CacheBackend], **_kwargs: Any):
        """Initialise with an ordered list of cache tiers."""
        super().__init__()
        if not tiers:
            raise ValueError("TieredCache requires at least one tier")
        self._tiers = list(tiers)

    @property
    def is_shared(self) -> bool:
        """A tiered cache is shared if any of its tiers is shared."""
        return any(tier.is_shared for tier in self._tiers)

    @property
    def tiers(self) -> list[CacheBackend]:
        """Return a copy of the tier list."""
        return list(self._tiers)

    # ── Job metadata ─────────────────────────────────────────────────────

    def get_job(self, run_hash: str) -> Job | None:
        """Check each tier in order; backfill faster tiers on a hit."""
        for i, tier in enumerate(self._tiers):
            job = tier.get_job(run_hash)
            if job is not None:
                # Backfill all faster tiers that missed
                for faster in self._tiers[:i]:
                    faster.put_job(run_hash, job)
                return job
        return None

    def put_job(self, run_hash: str, job: Job) -> None:
        """Write-through to every tier."""
        for tier in self._tiers:
            tier.put_job(run_hash, job)

    # ── DataClass blobs ──────────────────────────────────────────────────

    def get_data(self, content_hash: str) -> DataClass | list | None:
        """Check each tier in order; backfill faster tiers on a hit."""
        for i, tier in enumerate(self._tiers):
            data = tier.get_data(content_hash)
            if data is not None:
                for faster in self._tiers[:i]:
                    faster.put_data(content_hash, data)
                return data
        return None

    def put_data(self, content_hash: str, data: DataClass | list) -> None:
        """Write-through to every tier."""
        for tier in self._tiers:
            tier.put_data(content_hash, data)

    def has_data(self, content_hash: str) -> bool:
        """Return ``True`` if any tier contains the blob."""
        return any(tier.has_data(content_hash) for tier in self._tiers)

    # ── Deletion ─────────────────────────────────────────────────────────

    def delete_job(self, run_hash: str) -> bool:
        """Remove from every tier.  Returns ``True`` if any had it."""
        existed = False
        for tier in self._tiers:
            if tier.delete_job(run_hash):
                existed = True
        return existed

    def delete_data(self, content_hash: str) -> bool:
        """Remove from every tier.  Returns ``True`` if any had it."""
        existed = False
        for tier in self._tiers:
            if tier.delete_data(content_hash):
                existed = True
        return existed

    def clear(self) -> None:
        """Clear every tier."""
        for tier in self._tiers:
            tier.clear()
