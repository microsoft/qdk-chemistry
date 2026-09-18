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

import logging
import pathlib
from typing import Any

from qdk_chemistry._core import DuplicateRegistrationError as _DuplicateRegistrationError
from qdk_chemistry.remote.cache.base import CacheBackend
from qdk_chemistry.remote.cache.folder import FolderCache
from qdk_chemistry.remote.cache.tiered import TieredCache

logger = logging.getLogger(__name__)

# ── Registry ─────────────────────────────────────────────────────────────────

_CACHES: dict[str, type[CacheBackend]] = {}


def _register_cache(name: str, cls: type[CacheBackend]) -> type[CacheBackend]:
    """Register one cache class after validating registry ownership.

    Args:
        name: Registry name for the cache backend.
        cls: Cache backend class to register.

    """
    if name in _CACHES:
        raise _DuplicateRegistrationError(f"Cache backend name '{name}' is already registered")
    for registered_name, registered_cls in _CACHES.items():
        if registered_cls is cls:
            raise _DuplicateRegistrationError(
                f"Cache backend class '{cls.__module__}.{cls.__qualname__}' is already registered "
                f"with name '{registered_name}'"
            )
    cls.name = name
    _CACHES[name] = cls
    return cls


def register_cache(name: str):
    """Decorator to register a cache backend class.

    Args:
        name: The cache backend name (e.g. ``"folder"``).

    Raises:
        DuplicateRegistrationError: If the cache backend name or class is already registered.

    """

    def decorator(cls: type[CacheBackend]) -> type[CacheBackend]:
        return _register_cache(name, cls)

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


def resolve_cache(cache: str | pathlib.Path | CacheBackend | None, **kwargs: Any) -> CacheBackend | None:
    """Normalise a user-supplied cache argument.

    Accepts any of the following:

    - ``None`` returns ``None``
    - A ``CacheBackend`` instance → returned as-is
    - A ``Path`` or path-like string → ``FolderCache(path=...)``
    - A registered name string → looked up in the registry; extra
      ``kwargs`` are forwarded to the backend constructor.

    Args:
        cache: Cache instance, registered backend name, filesystem path, or ``None``.
        **kwargs: Backend-specific configuration for a name or filesystem path.

    """
    if cache is None:
        return None
    if isinstance(cache, CacheBackend):
        return cache
    if isinstance(cache, pathlib.Path):
        return FolderCache(path=cache, **kwargs)
    # str — could be a registered name or a path
    if isinstance(cache, str) and cache in _CACHES:
        try:
            return _CACHES[cache](**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Cache name '{cache}' requires configuration. "
                f"Pass a filesystem path (e.g. './cache') or use get_cache('{cache}', ...)."
            ) from e
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
        logger.warning("Failed to load cache plugins", exc_info=True)


_load_plugin_caches()


__all__ = [
    "available_caches",
    "get_cache",
    "register_cache",
    "resolve_cache",
]
