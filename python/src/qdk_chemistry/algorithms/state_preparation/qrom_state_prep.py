"""QDK/Chemistry QROM-based state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["QROMStatePreparation"]


class QROMStatePreparation(StatePreparation):
    r"""State preparation using QROM-based multiplexed rotations.

    Prepares an arbitrary n-qubit state using n layers of multiplexed Ry rotations,
    where each layer's angles are loaded from a QROM table.

    This approach uses only :math:`n = \lceil\log_2 L\rceil` qubits, but requires
    n QROM lookups.
    """

    def __init__(self, rotation_bit_precision: int = 10):
        """Initialize QROMStatePreparation.

        Args:
            rotation_bit_precision: Number of bits for Givens rotation angle
                precision. Higher values give more accurate rotations.
                Defaults to 10.

        """
        super().__init__()
        self._rotation_bit_precision = rotation_bit_precision

    def name(self) -> str:
        """Return the algorithm name."""
        return "qrom"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        r"""State preparation using QROM-based SBM decomposition from a Wavefunction.

        Extracts amplitudes from the wavefunction and builds a QROM state prep
        circuit using n layers of multiplexed Ry rotations.

        Args:
            wavefunction: The target wavefunction.

        Returns:
            Circuit: A Circuit wrapping the Q# QROM state prep callable and factory.

        """
        coeffs = np.asarray(wavefunction.get_coefficients())
        if np.iscomplexobj(coeffs):
            if not np.allclose(coeffs.imag, 0.0):
                raise ValueError("QROM state preparation requires real coefficients.")
            coeffs = coeffs.real

        amplitudes = coeffs.tolist()
        num_state_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1

        params = QSHARP_UTILS.QROMStatePrep.QROMStatePrepParams(
            amplitudes=amplitudes,
            rotationBitPrecision=self._rotation_bit_precision,
            numStateQubits=num_state_qubits,
        )

        qsharp_op = QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOp(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepCircuit,
            parameter={
                "amplitudes": amplitudes,
                "rotationBitPrecision": self._rotation_bit_precision,
                "numStateQubits": num_state_qubits,
            },
        )

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)

    @property
    def rotation_bit_precision(self) -> int:
        """Number of bits for rotation angle precision."""
        return self._rotation_bit_precision
