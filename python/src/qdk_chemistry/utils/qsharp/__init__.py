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


class _SharedContext:
    """Lock-guarded holder for the process-wide shared Q# context."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.context: qdk.Context | None = None


_shared = _SharedContext()
_thread_local = threading.local()


def create_qsharp_context(target_profile: TargetProfile = TargetProfile.Base, **kwargs) -> qdk.Context:
    """Create a new, isolated ``qdk.Context`` preloaded with the Q# chemistry utilities.

    Every call returns a *fresh* context with its own Q# interpreter. Most users never
    need this — the library maintains one shared context (see :func:`get_qsharp_context`).
    Reach for this only when you need a context configured differently from the default
    (for example a non-default ``target_profile``); then register it with
    :func:`set_qsharp_context` if the chemistry builders should use it too.

    ``target_profile`` defaults to ``TargetProfile.Base``. Any additional keyword
    arguments are forwarded to :class:`qdk.Context` (``target_name``,
    ``language_features``, ``qdk_config``, ...). ``project_root`` is fixed to the vendored
    Q# utility project and cannot be overridden.
    """
    if "project_root" in kwargs:
        raise TypeError("project_root is fixed to the Q# chemistry utility project and cannot be overridden")
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile, **kwargs)


def get_qsharp_context() -> qdk.Context:
    """Return the shared ``qdk.Context`` that QDK/Chemistry uses for all Q# composition.

    Call it when you want to define your *own* Q# operation (e.g. a custom state
    preparation) and compose it with a chemistry builder. The context is created lazily
    on first use and access is thread-safe.
    """
    override = getattr(_thread_local, "context", None)
    if override is not None:
        return override

    with _shared.lock:
        if _shared.context is None:
            _shared.context = create_qsharp_context()
        return _shared.context


def set_qsharp_context(context: qdk.Context | None) -> None:
    """Replace the process-wide shared Q# context (pass ``None`` to reset to the default)."""
    with _shared.lock:
        _shared.context = context


@contextmanager
def use_qsharp_context(context: qdk.Context) -> Iterator[qdk.Context]:
    """Temporarily use *context* as the shared Q# context on the current thread."""
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
