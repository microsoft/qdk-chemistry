"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import qsharp
from qdk._native import TargetProfile

__all__ = ["QSHARP_UTILS", "get_qsharp_utils"]

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

_current_target_profile: TargetProfile | None = None


def get_qsharp_utils(target_profile: TargetProfile = TargetProfile.Adaptive_RIF):
    """Returns the Q# namespace for chemistry operations (lazy-loaded).

    Args:
        target_profile: The target profile to use for the Q# interpreter. Defaults to Adaptive RIF.

    Returns:
        The Q# namespace for chemistry operations.

    """
    global _current_target_profile  # noqa: PLW0603
    try:
        utils = qdk.code.QDKChemistry.Utils
        if _current_target_profile != target_profile:
            raise AttributeError("Profile changed, reinitialize")
        return utils
    except AttributeError:
        if target_profile == TargetProfile.Base:
            qsharp.init(target_profile=target_profile)
            code = "\n".join((_SOURCE_ROOT / filename).read_text(encoding="utf-8") for filename in _BASE_PROFILE_FILES)
            qsharp.eval(code)
        else:
            qsharp.init(project_root=_PROJECT_ROOT, target_profile=target_profile)
        _current_target_profile = target_profile
        return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        target = _current_target_profile if _current_target_profile is not None else TargetProfile.Adaptive_RIF
        utils = get_qsharp_utils(target_profile=target)
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
