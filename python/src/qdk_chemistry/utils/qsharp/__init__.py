"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from contextlib import contextmanager
from pathlib import Path

import qdk
from qdk._native import TargetProfile

__all__ = [
    "QSHARP_PROJECT_ROOT",
    "QSHARP_UTILS",
    "get_qsharp_context",
    "get_qsharp_utils",
    "use_qsharp_profile",
]

_PROJECT_ROOT = str(Path(__file__).parent)
QSHARP_PROJECT_ROOT = _PROJECT_ROOT
_SOURCE_ROOT = Path(__file__).parent / "src"

# Q# sources that are legal under the Base profile. The Base context loads only this
# subset; the remaining sources use dynamic features that require an adaptive profile.
_BASE_PROFILE_FILES = [
    "StatePreparation.qs",
    "CircuitComposition.qs",
    "IterativePhaseEstimation.qs",
    "StandardPhaseEstimation.qs",
    "ControlledPauliExp.qs",
    "HadamardTest.qs",
    "PauliExp.qs",
    "MeasurementBasis.qs",
    "PrepSelPrep.qs",
    "Select.qs",
]

_DEFAULT_PROFILE = TargetProfile.Adaptive_RIF
_context_cache: dict[str, qdk.Context] = {}
_active_profile = _DEFAULT_PROFILE


def _build_qsharp_context(target_profile: TargetProfile) -> qdk.Context:
    """Build a fresh Q# context for *target_profile* (Base loads only the Base subset)."""
    if target_profile == TargetProfile.Base:
        context = qdk.Context(target_profile=target_profile)
        code = "\n".join((_SOURCE_ROOT / filename).read_text(encoding="utf-8") for filename in _BASE_PROFILE_FILES)
        context.eval(code)
        return context
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


def _cached_context(target_profile: TargetProfile) -> qdk.Context:
    """Return the shared, lazily built context for *target_profile* (used by QSHARP_UTILS)."""
    key = str(target_profile)
    context = _context_cache.get(key)
    if context is None:
        context = _build_qsharp_context(target_profile)
        _context_cache[key] = context
    return context


def get_qsharp_context(target_profile: TargetProfile = _DEFAULT_PROFILE) -> qdk.Context:
    """Return a fresh isolated ``qdk.Context`` with the vendored Q# utilities loaded.

    A new context is created on each call so callers (e.g. tests inspecting quantum state)
    get an isolated interpreter. The Base profile loads only the Base-compatible subset
    (:data:`_BASE_PROFILE_FILES`); other profiles load the full Q# project.

    Args:
        target_profile: The target profile for the context. Defaults to Adaptive_RIF.

    Returns:
        A fresh :class:`qdk.Context` with the Q# chemistry utilities loaded.

    """
    return _build_qsharp_context(target_profile)


def get_qsharp_utils(target_profile: TargetProfile = _DEFAULT_PROFILE):
    """Return the ``QDKChemistry.Utils`` namespace from a cached per-profile context.

    The context is built once per profile and reused, so resolved callables are stable and
    the (potentially expensive) project load happens only once.

    Args:
        target_profile: The profile whose context resolves the utilities. Defaults to Adaptive_RIF.

    Returns:
        The Q# ``QDKChemistry.Utils`` namespace; resolved callables carry their context.

    """
    return _cached_context(target_profile).code.QDKChemistry.Utils


@contextmanager
def use_qsharp_profile(target_profile: TargetProfile):
    """Resolve :data:`QSHARP_UTILS` from *target_profile*'s context within the block.

    Use around builders whose circuits are converted to QIR/Qiskit (``qir_to_qiskit``),
    which require the static Base profile.

    Args:
        target_profile: The profile to activate for :data:`QSHARP_UTILS` resolution.

    Yields:
        The :class:`qdk.Context` for *target_profile*.

    """
    global _active_profile  # noqa: PLW0603
    previous = _active_profile
    _active_profile = target_profile
    try:
        yield _cached_context(target_profile)
    finally:
        _active_profile = previous


class _QSharpUtilsProxy:
    """Lazily resolve the Q# utilities namespace from the active profile's context."""

    def __getattr__(self, name: str):
        """Resolve *name* on the active-profile utilities namespace.

        Args:
            name: The attribute being accessed on the Q# utilities namespace.

        """
        return getattr(get_qsharp_utils(_active_profile), name)


QSHARP_UTILS = _QSharpUtilsProxy()
