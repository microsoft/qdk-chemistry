"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import qdk
from qdk import TargetProfile

__all__ = [
    "QSHARP_UTILS",
    "create_qsharp_context",
    "get_qsharp_context",
    "set_qsharp_context",
    "use_qsharp_context",
]

_PROJECT_ROOT = str(Path(__file__).parent)


# Utilities and any user op composed with them must share one qdk.Context, so we
# keep a single shared owning context callers can read or replace. Global access
# is lock-guarded; _thread_local holds an optional per-thread override.
class _SharedContext:
    """Lock-guarded holder for the process-wide shared Q# context."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.context: qdk.Context | None = None


_shared = _SharedContext()
_thread_local = threading.local()


def create_qsharp_context(target_profile: TargetProfile = TargetProfile.Base) -> qdk.Context:
    """Create a new, isolated QDK context for the vendored Q# utility project."""
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


def get_qsharp_context() -> qdk.Context:
    """Return the shared QDK context that owns the Q# chemistry utilities."""
    override = getattr(_thread_local, "context", None)
    if override is not None:
        return override

    with _shared.lock:
        if _shared.context is None:
            _shared.context = create_qsharp_context()
        return _shared.context


def set_qsharp_context(context: qdk.Context | None) -> None:
    """Set the process-wide QDK context, or pass None to reset to the default."""
    with _shared.lock:
        _shared.context = context


@contextmanager
def use_qsharp_context(context: qdk.Context) -> Iterator[qdk.Context]:
    """Temporarily use *context* for the Q# chemistry utilities on this thread."""
    previous = getattr(_thread_local, "context", None)
    _thread_local.context = context
    try:
        yield context
    finally:
        _thread_local.context = previous


class _QSharpUtilsProxy:
    """Resolve the chemistry Q# utilities against the active shared context."""

    def __getattr__(self, name: str):
        """Resolve *name* on the utilities namespace of the active context."""
        return getattr(get_qsharp_context().code.QDKChemistry.Utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
