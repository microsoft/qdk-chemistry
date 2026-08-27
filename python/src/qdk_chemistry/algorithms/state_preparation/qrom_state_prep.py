"""QDK/Chemistry QROM-based state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
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
        params = self._build_params(wavefunction)

        qsharp_op = QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOpWithSharedGradient(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepCircuit,
            parameter={
                "amplitudes": params.amplitudes,
                "rotationBitPrecision": params.rotationBitPrecision,
                "numStateQubits": params.numStateQubits,
            },
        )

        return Circuit(
            qsharp_op=qsharp_op,
            qsharp_factory=qsharp_factory,
            num_qubits=params.numStateQubits + params.rotationBitPrecision,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=params.rotationBitPrecision),
        )

    def _build_params(self, wavefunction: Wavefunction):
        """Validate a wavefunction and build the Q# parameter record for it.

        Args:
            wavefunction: The target wavefunction.

        Returns:
            The Q# ``QROMStatePrepParams`` record.

        Raises:
            ValueError: If the wavefunction has no coefficients, has an imaginary part,
                contains a non-finite coefficient, or is all zeros.

        """
        coeffs, num_state_qubits = self._dense_state_vector(wavefunction, "QROM state preparation")
        if not np.all(np.isfinite(coeffs)) or not np.any(coeffs != 0.0):
            raise ValueError("QROM state preparation requires finite, non-zero coefficients.")

        return QSHARP_UTILS.QROMStatePrep.QROMStatePrepParams(
            amplitudes=coeffs.tolist(),
            rotationBitPrecision=int(self._settings.get("rotation_bit_precision")),
            numStateQubits=num_state_qubits,
        )
