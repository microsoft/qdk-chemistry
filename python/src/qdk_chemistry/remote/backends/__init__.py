"""Remote execution backends for QDK/Chemistry.

This package provides pluggable backends for executing algorithms on remote systems.
Each backend implements the upload/execute/download pattern for transferring data
and running generated Python scripts.

Built-in backend:
    - ``local``: Local subprocess execution and reference implementation

Plugin packages should register custom backends through ``QdkChemistryPlugin``
and the ``qdk_chemistry.plugins`` entry-point group. The ``@register_backend``
decorator remains available for direct registration.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# Import the built-in backend to register it
from qdk_chemistry.remote.backends import local
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
