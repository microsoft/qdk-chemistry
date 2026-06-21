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

_MPS_SEQUENTIAL_QS_FILES = [
    Path(__file__).parent / "mps_sequential" / "src" / "PhaseGradient.qs",
    Path(__file__).parent / "mps_sequential" / "src" / "GivensDecomposition.qs",
    Path(__file__).parent / "mps_sequential" / "src" / "QroamStatePrep.qs",
    Path(__file__).parent / "mps_sequential" / "src" / "MPSSequential.qs",
]

# Sibling-module imports to strip when loading MPS Sequential files via eval
# (file-as-namespace imports only work with qsharp.json projects)
_SIBLING_IMPORT_RE = re.compile(
    r"^import\s+(PhaseGradient|GivensDecomposition|QroamStatePrep|MPSSequential)\b.*$",
    re.MULTILINE,
)


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded)."""
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        code = "\n".join(f.read_text() for f in _QS_FILES)
        qsharp.eval(code)
        return qdk.code.QDKChemistry.Utils


def _get_mps_sequential_ns():
    """Returns the MPS Sequential Q# namespace (lazy-loaded via eval)."""
    try:
        return qdk.code.MPSSequential
    except AttributeError:
        code = "\n".join(f.read_text() for f in _MPS_SEQUENTIAL_QS_FILES)
        code = _SIBLING_IMPORT_RE.sub("", code)
        qsharp.eval(code)
        return qdk.code.MPSSequential


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Falls through to MPS Berry namespace for names not found in QDKChemistry.Utils.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        if name == "MPSSequential":
            return _get_mps_sequential_ns()
        utils = get_qsharp_utils()
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
