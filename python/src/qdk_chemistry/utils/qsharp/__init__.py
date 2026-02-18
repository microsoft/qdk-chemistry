"""QDK/Chemistry Q# Utilities Module."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS"]


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations."""
    if not hasattr(qdk.code, "QDKChemistry.Utils"):
        state_preparation_code = (Path(__file__).parent / "StatePreparation.qs").read_text()
        iterative_phase_estimation_code = (Path(__file__).parent / "IterativePhaseEstimation.qs").read_text()
        controlled_pauli_exp_code = (Path(__file__).parent / "ControlledPauliExp.qs").read_text()
        code = state_preparation_code + "\n" + iterative_phase_estimation_code + "\n" + controlled_pauli_exp_code
        qsharp.eval(code)
    return qdk.code.QDKChemistry.Utils


QSHARP_UTILS = get_qsharp_utils()
