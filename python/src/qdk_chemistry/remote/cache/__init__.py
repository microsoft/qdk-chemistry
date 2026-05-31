"""Cache backends for QDK/Chemistry job results.

Built-in backends:
    - ``folder``: Plain-file content-addressed cache
    - ``tiered``: Layered cache that combines multiple backends

Custom backends can be registered with ``@register_cache`` or via
the ``qdk_chemistry.cache_backends`` entry-point group.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdk_chemistry.remote.cache.base import CacheBackend
from qdk_chemistry.remote.cache.folder import FolderCache
from qdk_chemistry.remote.cache.tiered import TieredCache

# ── Registry ─────────────────────────────────────────────────────────────────

_CACHES: dict[str, type[CacheBackend]] = {}


def register_cache(name: str):
    """Decorator to register a cache backend class.

    Args:
        name: The cache backend name (e.g. ``"folder"``).

    """

    def decorator(cls: type[CacheBackend]) -> type[CacheBackend]:
        cls.name = name
        _CACHES[name] = cls
        return cls

    return decorator


# Register built-in backends
register_cache("folder")(FolderCache)
register_cache("tiered")(TieredCache)


def get_cache(name: str, **config: Any) -> CacheBackend:
    """Create a cache backend by name.

    Args:
        name: Backend name (e.g. ``"folder"``).
        **config: Backend-specific configuration.

    Raises:
        ValueError: If no cache is registered with that name.

    """
    if name not in _CACHES:
        available = ", ".join(_CACHES) or "(none)"
        raise ValueError(f"No cache registered with name '{name}'. Available: {available}")
    return _CACHES[name](**config)


def resolve_cache(cache: str | Path | CacheBackend | None, **kwargs: Any) -> CacheBackend | None:
    """Normalise a user-supplied cache argument.

    Accepts any of the following:

    - ``None`` returns ``None``
    - A ``CacheBackend`` instance → returned as-is
    - A ``Path`` or path-like string → ``FolderCache(path=...)``
    - A registered name string → looked up in the registry; extra
      *kwargs* are forwarded to the backend constructor.

    """
    if cache is None:
        return None
    if isinstance(cache, CacheBackend):
        return cache
    if isinstance(cache, Path):
        return FolderCache(path=cache, **kwargs)
    # str — could be a registered name or a path
    if isinstance(cache, str) and cache in _CACHES and kwargs:
        return _CACHES[cache](**kwargs)
    # Treat as a filesystem path
    return FolderCache(path=cache, **kwargs)


def available_caches() -> list[str]:
    """Return list of registered cache backend names."""
    return list(_CACHES.keys())


def _load_plugin_caches() -> None:
    """Auto-discover cache backends from entry points."""
    try:
        from importlib.metadata import entry_points  # noqa: PLC0415

        eps = entry_points(group="qdk_chemistry.cache_backends")
        for ep in eps:
            cls = ep.load()
            register_cache(ep.name)(cls)
    except Exception:  # noqa: BLE001
        pass


_load_plugin_caches()

__all__ = [
    "CacheBackend",
    "FolderCache",
    "TieredCache",
    "available_caches",
    "get_cache",
    "register_cache",
    "resolve_cache",
]
