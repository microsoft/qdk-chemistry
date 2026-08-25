"""QDK/Chemistry QROM-based state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from typing import Any

import numpy as np

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import PrepareLayout, StatePreparation

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
            ValueError: If the wavefunction has no coefficients, has an imaginary part, or
                contains a non-finite coefficient.

        """
        params = self._build_params(wavefunction)

        qsharp_op = QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOp(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepCircuit,
            parameter={
                "amplitudes": params.amplitudes,
                "rotationBitPrecision": params.rotationBitPrecision,
                "numStateQubits": params.numStateQubits,
            },
        )

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)

    def _build_params(self, wavefunction: Wavefunction):
        """Validate a wavefunction and build the Q# parameter record for it.

        Args:
            wavefunction: The target wavefunction.

        Returns:
            The Q# ``QROMStatePrepParams`` record.

        Raises:
            ValueError: If the wavefunction has no coefficients, has an imaginary part, or
                contains a non-finite coefficient.

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

        amplitudes = coeffs.tolist()
        return QSHARP_UTILS.QROMStatePrep.QROMStatePrepParams(
            amplitudes=amplitudes,
            rotationBitPrecision=int(self._settings.get("rotation_bit_precision")),
            numStateQubits=self._num_state_qubits(len(amplitudes)),
        )

    @staticmethod
    def _num_state_qubits(num_coefficients: int) -> int:
        """Width of the state register for a given coefficient count."""
        return math.ceil(math.log2(num_coefficients)) if num_coefficients > 1 else 1

    def prepare_layout(self, wavefunction: Wavefunction) -> PrepareLayout:
        """Return the register widths this oracle needs inside a block encoding.

        The prepared state is pure on the index register, so index and block ancilla
        widths coincide. The phase gradient is requested as shared ancilla: it is an
        eigenstate of every addition that consumes it, so one copy prepared for the whole
        circuit serves every rotation, and preparing it per call is where all of this
        oracle's arbitrary-angle rotations come from.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            PrepareLayout: An :math:`n`-qubit index with no garbage, plus a
            ``rotation_bit_precision``-wide shared phase gradient.

        """
        params = self._build_params(wavefunction)
        return PrepareLayout(
            num_select_qubits=params.numStateQubits,
            num_block_ancillas=params.numStateQubits,
            num_shared_ancillas=params.rotationBitPrecision,
        )

    def prepare_oracle(self, wavefunction: Wavefunction) -> tuple[Any, PrepareLayout]:
        """Return the shared-phase-gradient PREPARE callable and its layout.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            A Q# callable expecting ``[state | phaseGradient]``, and its layout.

        """
        params = self._build_params(wavefunction)
        return (
            QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOpShared(params),
            PrepareLayout(
                num_select_qubits=params.numStateQubits,
                num_block_ancillas=params.numStateQubits,
                num_shared_ancillas=params.rotationBitPrecision,
            ),
        )
