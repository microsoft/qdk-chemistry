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
    "ADAPTIVE_QSHARP_UTILS",
    "BASE_QSHARP_UTILS",
    "QSHARP_UTILS",
    "get_qsharp_context",
    "get_qsharp_utils",
    "use_qsharp_profile",
]

_PROJECT_ROOT = str(Path(__file__).parent)
_SOURCE_ROOT = Path(__file__).parent / "src"
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


def get_qsharp_context(target_profile: TargetProfile = TargetProfile.Adaptive_RIF) -> qdk.Context:
    """Create a QDK context for the vendored Q# utility project."""
    if target_profile == TargetProfile.Base:
        context = qdk.Context(target_profile=target_profile)
        code = "\n".join((_SOURCE_ROOT / filename).read_text(encoding="utf-8") for filename in _BASE_PROFILE_FILES)
        context.eval(code)
        return context
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


_BASE_QSHARP_CONTEXT = get_qsharp_context(TargetProfile.Base)
_ADAPTIVE_QSHARP_CONTEXT = get_qsharp_context(TargetProfile.Adaptive_RIF)
BASE_QSHARP_UTILS = _BASE_QSHARP_CONTEXT.code.QDKChemistry.Utils
ADAPTIVE_QSHARP_UTILS = _ADAPTIVE_QSHARP_CONTEXT.code.QDKChemistry.Utils


def get_qsharp_utils(target_profile: TargetProfile = TargetProfile.Base):
    """Return the Q# utilities namespace for a supported target profile.

    Args:
        target_profile: The target profile to use for the Q# interpreter. Defaults to Base.

    Returns:
        The Q# namespace for chemistry operations.

    """
    if target_profile == TargetProfile.Base:
        return BASE_QSHARP_UTILS
    if target_profile == TargetProfile.Adaptive_RIF:
        return ADAPTIVE_QSHARP_UTILS
    raise ValueError(f"Unsupported Q# utility target profile: {target_profile}")


class _ActiveQSharpUtils:
    """Proxy that forwards attribute access to the currently active Q# utilities namespace.

    A single shared global profile is used across the package. It defaults to the Base
    profile and can be temporarily switched (for example to Adaptive for MPS sparse state
    preparation) via :func:`use_qsharp_profile`.
    """

    def __init__(self, active: object) -> None:
        object.__setattr__(self, "_active", active)

    def swap_active(self, new_active: object) -> object:
        """Replace the active namespace and return the previous one."""
        previous = object.__getattribute__(self, "_active")
        object.__setattr__(self, "_active", new_active)
        return previous

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_active"), name)


QSHARP_UTILS = _ActiveQSharpUtils(BASE_QSHARP_UTILS)


@contextmanager
def use_qsharp_profile(target_profile: TargetProfile):
    """Temporarily switch the shared global :data:`QSHARP_UTILS` profile.

    The active namespace is switched to the one for ``target_profile`` on entry and
    restored to the previous namespace on exit.

    Args:
        target_profile: The target profile to activate for the duration of the context.

    Yields:
        The Q# utilities namespace for the requested target profile.

    """
    new_active = get_qsharp_utils(target_profile)
    previous = QSHARP_UTILS.swap_active(new_active)
    try:
        yield new_active
    finally:
        QSHARP_UTILS.swap_active(previous)
