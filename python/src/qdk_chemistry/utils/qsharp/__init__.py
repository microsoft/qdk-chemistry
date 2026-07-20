"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import re
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS", "get_qsharp_utils"]

_QS_FILES = [
    Path(__file__).parent / "StatePreparation.qs",
    Path(__file__).parent / "CircuitComposition.qs",
    Path(__file__).parent / "IterativePhaseEstimation.qs",
    Path(__file__).parent / "StandardPhaseEstimation.qs",
    Path(__file__).parent / "ControlledPauliExp.qs",
    Path(__file__).parent / "HadamardTest.qs",
    Path(__file__).parent / "PauliExp.qs",
    Path(__file__).parent / "MeasurementBasis.qs",
    Path(__file__).parent / "PrepSelPrep.qs",
    Path(__file__).parent / "Select.qs",
]

_MPS_PROJECT_ROOT = str(Path(__file__).parent / "mps_sequential")

_state: dict[str, str | None] = {"mode": None}


def _ensure_base_session():
    """Ensure the unified MPS project context and utility Q# files are loaded."""
    _ensure_mps_session()


def _ensure_mps_session():
    """Ensure interpreter has the MPS project and shared utility files loaded."""
    if _state["mode"] == "mps":
        try:
            _ = qdk.code.MPSSparse
            return
        except AttributeError:
            _state["mode"] = None  # stale — interpreter was reset externally
    qsharp.init(project_root=_MPS_PROJECT_ROOT)
    code = "\n".join(f.read_text() for f in _QS_FILES)
    qsharp.eval(code)
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
