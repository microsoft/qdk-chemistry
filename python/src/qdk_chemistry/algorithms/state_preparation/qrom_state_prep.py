"""QDK/Chemistry QROM-based state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["QROMStatePreparation", "QROMStatePreparationSettings"]


class QROMStatePreparationSettings(Settings):
    """Settings for :class:`QROMStatePreparation`."""

    def __init__(self):
        """Initialize the QROMStatePreparationSettings."""
        super().__init__()
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Number of bits of precision used for the QROM-loaded Ry rotation angles. Higher "
            "values reduce the synthesis error of each multiplexed rotation at the cost of a "
            "wider QROM output register. The upper bound of 30 is a sanity limit as 2^-30 is already far "
            "below chemical accuracy.",
            (1, 30),
        )


class QROMStatePreparation(StatePreparation):
    r"""State preparation using Quantum Read-Only Memory (QROM) based multiplexed rotations.

    Prepares an arbitrary n-qubit state using n layers of multiplexed Ry rotations,
    where each layer's angles are loaded from a QROM table.
    """

    def __init__(self, rotation_bit_precision: int = 10):
        """Initialize QROMStatePreparation.

        Args:
            rotation_bit_precision: Number of bits for multiplexed :math:`R_y`
                angle precision. Higher values give more accurate rotations.
                Defaults to 10. Equivalent to setting the ``rotation_bit_precision``
                entry of ``settings()``.

        """
        super().__init__()
        self._settings = QROMStatePreparationSettings()
        self._settings.set("rotation_bit_precision", rotation_bit_precision)

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

        Raises:
            ValueError: If the wavefunction has no coefficients, has an imaginary part,
                contains a non-finite coefficient, or is all zeros.

        """
        coeffs = np.asarray(wavefunction.get_coefficients())
        if coeffs.size == 0:
            raise ValueError("QROM state preparation requires at least one coefficient.")
        if np.iscomplexobj(coeffs):
            if np.any(coeffs.imag != 0.0):
                raise ValueError("QROM state preparation requires real coefficients.")
            coeffs = coeffs.real
        coeffs = coeffs.astype(float, copy=False)

        if not np.all(np.isfinite(coeffs)):
            raise ValueError("QROM state preparation requires finite coefficients.")
        if not np.any(coeffs != 0.0):
            raise ValueError(
                "QROM state preparation requires at least one non-zero coefficient; an all-zero "
                "vector has no state to prepare."
            )

        amplitudes = coeffs.tolist()
        num_state_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
        rotation_bit_precision = int(self._settings.get("rotation_bit_precision"))

        params = QSHARP_UTILS.QROMStatePrep.QROMStatePrepParams(
            amplitudes=amplitudes,
            rotationBitPrecision=rotation_bit_precision,
            numStateQubits=num_state_qubits,
        )

        qsharp_op = QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOp(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepCircuit,
            parameter={
                "amplitudes": amplitudes,
                "rotationBitPrecision": rotation_bit_precision,
                "numStateQubits": num_state_qubits,
            },
        )

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)
