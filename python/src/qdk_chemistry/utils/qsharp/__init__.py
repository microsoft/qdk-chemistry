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


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded).

    Initializes the global Q# interpreter with the MPS project on first call,
    then loads additional Q# utility files via eval.
    """
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        qsharp.init(project_root=_MPS_PROJECT_ROOT)
        code = "\n".join(f.read_text() for f in _QS_FILES)
        qsharp.eval(code)
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
            get_qsharp_utils()  # ensure initialized
            return qdk.code.MPSSequential
        if name == "MPSSparse":
            get_qsharp_utils()  # ensure initialized
            return qdk.code.MPSSparse
        utils = get_qsharp_utils()
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
