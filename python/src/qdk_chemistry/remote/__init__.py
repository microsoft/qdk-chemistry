"""Remote execution and result caching for QDK/Chemistry algorithms.

Algorithms created through the registry accept ``remote`` and ``cache``
keyword arguments on ``run()``.

Usage:
    >>> from qdk_chemistry.algorithms import create
    >>> from qdk_chemistry.remote import create_remote
    >>>
    >>> scf = create("scf_solver")
    >>> backend = create_remote("local")
    >>> energy, wfn = scf.run(structure, 0, 1, "cc-pvdz", remote=backend)

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import TYPE_CHECKING

from qdk_chemistry.remote.backends import (
    available_backends,
    create_remote,
    get_backend,
)
from qdk_chemistry.remote.cache import resolve_cache
from qdk_chemistry.remote.proxy import run

if TYPE_CHECKING:
    from qdk_chemistry.remote.backends.base import RemoteBackend
    from qdk_chemistry.remote.job import Job

__all__ = [
    "Job",
    "RemoteBackend",
    "available_backends",
    "create_remote",
    "get_backend",
    "resolve_cache",
    "run",
]


def __getattr__(name: str):
    """Load re-exported remote types on first access.

    Args:
        name: Name of the re-exported remote type to load.

    """
    if name == "Job":
        from qdk_chemistry.remote.job import Job  # noqa: PLC0415

        return Job
    if name == "RemoteBackend":
        from qdk_chemistry.remote.backends.base import RemoteBackend  # noqa: PLC0415

        return RemoteBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
