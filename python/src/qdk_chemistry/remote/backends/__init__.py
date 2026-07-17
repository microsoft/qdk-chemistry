"""Remote execution backends for QDK/Chemistry.

This package provides pluggable backends for executing algorithms on remote systems.
Each backend implements the upload/execute/download pattern for transferring data
and running generated Python scripts.

Built-in backends:
    - ``ssh``: Execute via SSH on remote servers
    - ``local``: Local subprocess execution (for testing)

Custom backends can be registered using the ``@register_backend`` decorator or
via the ``qdk_chemistry.remote_backends`` entry-point group in ``pyproject.toml``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# Import built-in backends to register them
from qdk_chemistry.remote.backends import local, ssh
from qdk_chemistry.remote.backends.base import (
    available_backends,
    create_remote,
    get_backend,
    register_backend,
)

__all__ = [
    "JobStatus",
    "RemoteBackend",
    "available_backends",
    "create_remote",
    "get_backend",
    "register_backend",
]


def __getattr__(name: str):
    """Lazy import for re-exported types to avoid autodoc duplication."""
    if name == "JobStatus":
        from qdk_chemistry.remote.backends.base import JobStatus  # noqa: PLC0415

        return JobStatus
    if name == "RemoteBackend":
        from qdk_chemistry.remote.backends.base import RemoteBackend  # noqa: PLC0415

        return RemoteBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_plugin_backends() -> None:
    """Auto-discover and register backends from entry points."""
    try:
        from importlib.metadata import entry_points  # noqa: PLC0415

        eps = entry_points(group="qdk_chemistry.remote_backends")
        for ep in eps:
            backend_cls = ep.load()
            register_backend(ep.name)(backend_cls)
    except Exception:  # noqa: BLE001
        pass  # Fail silently if no plugins


_load_plugin_backends()
