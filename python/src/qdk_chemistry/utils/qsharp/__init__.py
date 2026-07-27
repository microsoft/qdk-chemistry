"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk._native import TargetProfile

__all__ = ["BASE_QSHARP_CONTEXT", "BASE_QSHARP_UTILS", "QSHARP_CONTEXT", "QSHARP_UTILS", "get_qsharp_utils"]

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


def _get_context(target_profile: TargetProfile) -> qdk.Context:
    """Create a QDK context for the vendored Q# utility project."""
    if target_profile == TargetProfile.Base:
        context = qdk.Context(target_profile=target_profile)
        code = "\n".join((_SOURCE_ROOT / filename).read_text(encoding="utf-8") for filename in _BASE_PROFILE_FILES)
        context.eval(code)
        return context
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


BASE_QSHARP_CONTEXT = _get_context(TargetProfile.Base)
QSHARP_CONTEXT = _get_context(TargetProfile.Adaptive_RIF)

BASE_QSHARP_UTILS = BASE_QSHARP_CONTEXT.code.QDKChemistry.Utils
QSHARP_UTILS = QSHARP_CONTEXT.code.QDKChemistry.Utils


def get_qsharp_utils(target_profile: TargetProfile = TargetProfile.Adaptive_RIF):
    """Return the Q# utilities namespace for a supported target profile.

    Args:
        target_profile: The target profile to use for the Q# interpreter. Defaults to Adaptive RIF.

    Returns:
        The Q# namespace for chemistry operations.

    """
    if target_profile == TargetProfile.Base:
        return BASE_QSHARP_UTILS
    if target_profile == TargetProfile.Adaptive_RIF:
        return QSHARP_UTILS
    raise ValueError(f"Unsupported Q# utility target profile: {target_profile}")
