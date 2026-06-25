"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import re
from pathlib import Path

import qdk
from qdk import TargetProfile, qsharp

__all__ = ["QSHARP_UTILS", "get_qsharp_utils"]

_PROJECT_ROOT = str(Path(__file__).parent)

_MPS_PROJECT_ROOT = str(Path(__file__).parent / "mps_sequential")

_state: dict[str, str | None] = {"mode": None}  # "base", "mps", or None


def _ensure_base_session():
    """Ensure interpreter is in Base mode with utility Q# files loaded."""
    if _state["mode"] == "base":
        try:
            _ = qdk.code.QDKChemistry.Utils.StatePreparation
            return
        except AttributeError:
            _state["mode"] = None  # stale — interpreter was reset externally
    if _state["mode"] == "mps":
        qsharp.init(target_profile=TargetProfile.Base)
    try:
        _ = qdk.code.QDKChemistry.Utils.StatePreparation
    except AttributeError:
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)
    _state["mode"] = "base"


def _ensure_mps_session():
    """Ensure interpreter has MPS project loaded (Unrestricted) plus utility files."""
    if _state["mode"] == "mps":
        try:
            _ = qdk.code.MPSSparse
            return
        except AttributeError:
            _state["mode"] = None  # stale — interpreter was reset externally
    qsharp.init(project_root=_MPS_PROJECT_ROOT)
    _state["mode"] = "mps"


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded).

    Initializes the global Q# interpreter with the MPS project on first call,
    then loads additional Q# utility files via eval. Use this when the MPS
    project must be available (e.g. for resource estimation of MPS circuits).
    """
    _ensure_mps_session()
    return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Falls through to MPS namespace for names not found in QDKChemistry.Utils.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        if name == "MPSSequential":
            _ensure_mps_session()
            return qdk.code.MPSSequential
        if name == "MPSSparse":
            _ensure_mps_session()
            return qdk.code.MPSSparse
        _ensure_base_session()
        return getattr(qdk.code.QDKChemistry.Utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
