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
    """Create a new, isolated ``qdk.Context`` preloaded with the Q# chemistry utilities.

    Every call returns a *fresh* context with its own Q# interpreter, so operations
    built against different contexts do not compose with one another. Most users never
    need this — the library maintains one shared context (see :func:`get_qsharp_context`).
    Reach for this only when you need a context with a non-default ``target_profile``;
    then register it with :func:`set_qsharp_context` if the chemistry builders should
    use it too.
    """
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


def get_qsharp_context() -> qdk.Context:
    """Return the shared ``qdk.Context`` that QDK/Chemistry uses for all Q# composition.

    This is the context that owns ``QSHARP_UTILS`` and every circuit the library builds
    internally. If you only use qdk-chemistry algorithms you never have to call this;
    the shared context is applied automatically and everything composes. Call it when you
    want to define your *own* Q# operation (e.g. a custom state preparation) and compose
    it with a chemistry builder: build the operation against the returned context so both
    sides share a single interpreter. The context is created lazily on first use and
    access is thread-safe.
    """
    override = getattr(_thread_local, "context", None)
    if override is not None:
        return override

    with _shared.lock:
        if _shared.context is None:
            _shared.context = create_qsharp_context()
        return _shared.context


def set_qsharp_context(context: qdk.Context | None) -> None:
    """Replace the process-wide shared Q# context (pass ``None`` to reset to the default).

    After this call both ``QSHARP_UTILS`` and :func:`get_qsharp_context` resolve against
    *context*, so a context you built with :func:`create_qsharp_context` is shared by the
    chemistry utilities and your own operations alike. The change is global and affects
    every thread; for a scoped, thread-local override prefer :func:`use_qsharp_context`.
    """
    with _shared.lock:
        _shared.context = context


@contextmanager
def use_qsharp_context(context: qdk.Context) -> Iterator[qdk.Context]:
    """Temporarily use *context* as the shared Q# context on the current thread.

    Unlike :func:`set_qsharp_context`, the override is thread-local and is automatically
    restored when the ``with`` block exits, so concurrent threads never clobber each
    other. Use it to run a block of chemistry work against a specific context without
    mutating global state.
    """
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
