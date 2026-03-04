"""QDK/Chemistry Q# Utilities Module."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import functools
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS"]

_QS_FILES = [
    Path(__file__).parent / "StatePreparation.qs",
    Path(__file__).parent / "IterativePhaseEstimation.qs",
    Path(__file__).parent / "ControlledPauliExp.qs",
]


@functools.lru_cache(maxsize=1)
def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded)."""
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        code = "\n".join(f.read_text() for f in _QS_FILES)
        qsharp.eval(code)
        return qdk.code.QDKChemistry.Utils


QSHARP_UTILS = get_qsharp_utils()
