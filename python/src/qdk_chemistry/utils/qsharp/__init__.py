"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS", "get_qsharp_utils"]

_PROJECT_ROOT = str(Path(__file__).parent)

_initialized = False


def _ensure_session():
    """Ensure the Q# interpreter is initialized with the unified project."""
    global _initialized
    if _initialized:
        try:
            _ = qdk.code.QDKChemistry.Utils.StatePreparation
            return
        except AttributeError:
            _initialized = False  # stale — interpreter was reset externally
    qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)
    _initialized = True


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded).

    Initializes the global Q# interpreter with the unified project on first call.
    All Q# code (SOSSA, MPS, state prep, etc.) shares a single context.
    """
    _ensure_session()
    return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        _ensure_session()
        if name in ("MPSSequential", "MPSSparse", "GivensDecomposition", "QroamStatePrep"):
            return getattr(qdk.code, name)
        return getattr(qdk.code.QDKChemistry.Utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
