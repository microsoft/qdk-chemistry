"""Abstract cache backend for QDK/Chemistry job results.

Cache backends provide content-addressed storage for algorithm results,
allowing repeated runs with identical inputs to skip execution entirely.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data._hashing import _numpy_scalar_to_python

if TYPE_CHECKING:
    from qdk_chemistry.remote.job import Job


def _is_cache_node(value: Any) -> bool:
    """Return whether a value can occur inside a cached list."""
    if isinstance(value, np.generic):
        value = _numpy_scalar_to_python(value)
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, CoreDataClass):
        return True
    if isinstance(value, np.ndarray):
        return not value.dtype.hasobject
    if isinstance(value, list | tuple):
        return all(_is_cache_node(item) for item in value)
    return False


def is_cacheable(value: Any) -> bool:
    """Return whether a value can be stored as a cache data entry."""
    if isinstance(value, CoreDataClass):
        return True
    if isinstance(value, np.ndarray):
        return not value.dtype.hasobject
    if isinstance(value, list):
        return all(_is_cache_node(item) for item in value)
    return False


class CacheBackend(ABC):
    """Abstract base class for result caches.

    Implementations must provide these operations:

    - ``get_job`` and ``put_job`` persist ``Job`` metadata by run hash.
    - ``get_data`` and ``put_data`` store supported values by content hash.
    - ``delete_job`` and ``delete_data`` remove cached metadata or blobs.
    - ``clear`` removes all cache entries.

    Args:
        is_shared: Set to ``True`` when the backing store is reachable
            from multiple machines (e.g. a network-mounted folder).
            Defaults to ``False``.

    """

    name: str  # Cache backend name (e.g. "folder", "sqlite")

    def __init__(self, *, is_shared: bool = False) -> None:
        """Initialise the cache backend."""
        self._is_shared = is_shared

    @abstractmethod
    def get_job(self, run_hash: str) -> Job | None:
        """Retrieve job metadata by *run_hash*, or ``None`` on miss."""

    @abstractmethod
    def put_job(self, run_hash: str, job: Job) -> None:
        """Store (or update) job metadata keyed by *run_hash*."""

    @abstractmethod
    def get_data(self, content_hash: str) -> Any | None:
        """Retrieve cached data by its content hash, or ``None``."""

    @abstractmethod
    def put_data(self, content_hash: str, data: Any, *, shared_only: bool = False) -> None:
        """Store data by content hash, optionally requiring shared storage."""

    @abstractmethod
    def delete_job(self, run_hash: str) -> bool:
        """Remove job metadata by *run_hash*.  Returns ``True`` if it existed."""

    @abstractmethod
    def delete_data(self, content_hash: str) -> bool:
        """Remove a DataClass blob by content hash.  Returns ``True`` if it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""

    # ── Optional helpers (have sensible defaults) ────────────────────────

    @property
    def is_shared(self) -> bool:
        """Whether this cache is reachable from both local and remote."""
        return self._is_shared

    def has_data(self, content_hash: str, *, shared_only: bool = False) -> bool:
        """Check whether a DataClass blob exists without deserializing it.

        The default implementation calls :meth:`get_data` and checks for
        ``None``.  Backends that can answer this more cheaply (e.g. a
        ``HEAD`` request or a ``glob``) should override. When *shared_only*
        is true, non-shared backends always return ``False``.
        """
        if shared_only and not self.is_shared:
            return False
        return self.get_data(content_hash) is not None

    def to_config(self) -> dict:
        """Return constructor kwargs sufficient to recreate this backend.

        Subclasses should override this if they accept configuration
        (paths, URLs, credentials, etc.).  The default returns an empty
        dict, which is only valid for backends that need no arguments.
        """
        return {}
