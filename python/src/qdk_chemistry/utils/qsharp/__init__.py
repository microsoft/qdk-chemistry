"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS"]

_QS_FILES = [
    Path(__file__).parent / "StatePreparation.qs",
    Path(__file__).parent / "IterativePhaseEstimation.qs",
    Path(__file__).parent / "ControlledPauliExp.qs",
    Path(__file__).parent / "MeasurementBasisRotation.qs",
]


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded)."""
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        code = "\n".join(f.read_text() for f in _QS_FILES)
        qsharp.eval(code)
        return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        utils = get_qsharp_utils()
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
