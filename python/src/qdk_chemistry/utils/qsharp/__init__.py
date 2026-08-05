"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import atexit
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import qdk
from qdk import TargetProfile

__all__ = [
    "DEFAULT_TARGET_PROFILE",
    "QSHARP_UTILS",
    "create_qsharp_context",
    "get_qsharp_context",
    "set_qsharp_context",
    "use_qsharp_context",
]

_PROJECT_ROOT = str(Path(__file__).parent)
_SOURCE_ROOT = Path(__file__).parent / "src"
#: Profile the vendored Q# project is compiled for by default.
#:
#: ``Adaptive_RIF`` rather than ``Unrestricted``: unary iteration uncomputes by
#: measurement, so the profile has to permit mid-circuit measurement and feed-forward,
#: but ``Unrestricted`` is the one profile QIR code generation rejects outright. The two
#: lower measurement-based uncompute identically, so this buys QIR support for free.
DEFAULT_TARGET_PROFILE = TargetProfile.Adaptive_RIF

#: Q# sources that are supported by ``TargetProfile.Base``.
#:
#: ``Select`` and ``PrepSelPrep`` qualify: their only measurement-based uncompute is the
#: AND ladder in ``Reflect``, which the Q# standard library lowers to a measurement-free
#: decomposition when the target profile demands it. Unary iteration does not qualify --
#: see :func:`create_qsharp_context`.
_BASE_PROFILE_FILES = (
    "StatePreparation.qs",
    "CircuitComposition.qs",
    "IterativePhaseEstimation.qs",
    "StandardPhaseEstimation.qs",
    "ControlledPauliExp.qs",
    "HadamardTest.qs",
    "PauliExp.qs",
    "MeasurementBasis.qs",
    "Select.qs",
    "PrepSelPrep.qs",
)


@cache
def _base_project_root() -> str:
    """Stage the Base-supported sources as a standalone Q# project and return its root.

    A context can take sources either as a ``project_root`` or as a string passed to ``eval``,
    but only the former compiles them into a package that QIR generation can lower: circuits
    composed out of ``eval``-ed sources fail with "callable should exist in lowered package".
    The Base subset is therefore copied into a project of its own rather than concatenated and
    evaluated, which withholds the unary sources while keeping the rest QIR-capable.

    The directory is built once per process and removed at interpreter exit.
    """
    root = Path(tempfile.mkdtemp(prefix="qdk-chemistry-qsharp-base-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    shutil.copyfile(Path(_PROJECT_ROOT) / "qsharp.json", root / "qsharp.json")
    source_dir = root / _SOURCE_ROOT.name
    source_dir.mkdir()
    for name in _BASE_PROFILE_FILES:
        shutil.copyfile(_SOURCE_ROOT / name, source_dir / name)
    return str(root)


class _SharedContext:
    """Lock-guarded holder for the process-wide shared Q# context."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.context: qdk.Context | None = None


_shared = _SharedContext()
_thread_local = threading.local()


def create_qsharp_context(
    target_profile: TargetProfile = DEFAULT_TARGET_PROFILE,
    target_name: str | None = None,
    language_features: list[str] | None = None,
    qdk_config: dict[str, int | float | str | bool] | None = None,
) -> qdk.Context:
    """Create a new, isolated ``qdk.Context`` preloaded with the Q# chemistry utilities.

    Every call returns a *fresh* context with its own Q# interpreter. Most users never
    need this — the library maintains one shared context (see :func:`get_qsharp_context`).
    Reach for this only when you need a context configured differently from the default
    (for example a non-default ``target_profile``); then register it with
    :func:`set_qsharp_context` if the chemistry builders should use it too.

    :param target_profile: Target profile the Q# interpreter compiles for. Defaults to
        :data:`DEFAULT_TARGET_PROFILE`. A ``TargetProfile.Base`` context loads only the
        Base-*correct* subset of the vendored project, which is what callers that need
        measurement-free circuits should ask for. Sources that rely on
        measurement-based uncompute, such as unary iteration, are withheld from it: they
        would compile under Base and return silently wrong results, so they are made to
        fail as undefined symbols instead.
    :param target_name: Optional target machine name used to infer a compatible profile.
    :param language_features: Optional list of experimental Q# language feature flags.
    :param qdk_config: Optional configuration values exposed to Q# code via
        ``Std.Core.ConfigValue`` (values must be ``int``, ``float``, ``str``, or ``bool``).

    ``project_root`` is intentionally not exposed: it is fixed to the vendored Q# utility
    project so the chemistry utilities are always available on the returned context.
    """
    kwargs: dict = {}
    if target_name is not None:
        kwargs["target_name"] = target_name
    if language_features is not None:
        kwargs["language_features"] = language_features
    if qdk_config is not None:
        kwargs["qdk_config"] = qdk_config
    if target_profile == TargetProfile.Base:
        return qdk.Context(project_root=_base_project_root(), target_profile=target_profile, **kwargs)
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
