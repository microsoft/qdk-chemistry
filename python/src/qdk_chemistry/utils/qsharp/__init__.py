"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import re
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS"]

_QS_FILES = [
    Path(__file__).parent / "StatePreparation.qs",
    Path(__file__).parent / "CircuitComposition.qs",
    Path(__file__).parent / "IterativePhaseEstimation.qs",
    Path(__file__).parent / "StandardPhaseEstimation.qs",
    Path(__file__).parent / "ControlledPauliExp.qs",
    Path(__file__).parent / "PauliExp.qs",
    Path(__file__).parent / "MeasurementBasis.qs",
]

_MPS_PROJECT_ROOT = str(Path(__file__).parent / "mps_sequential")

_mps_context = None


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded)."""
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        code = "\n".join(f.read_text() for f in _QS_FILES)
        qsharp.eval(code)
        return qdk.code.QDKChemistry.Utils


def _get_mps_context():
    """Returns a cached Q# Context with the MPS project loaded."""
    global _mps_context
    if _mps_context is None:
        _mps_context = qdk.Context(project_root=_MPS_PROJECT_ROOT)
    return _mps_context


def _get_mps_namespace():
    """Returns the MPS Q# namespace (lazy-loaded via project_root)."""
    return _get_mps_context().code.MPSSequential


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Falls through to MPS namespace for names not found in QDKChemistry.Utils.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        if name == "MPSSequential":
            return _get_mps_namespace()
        utils = get_qsharp_utils()
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
