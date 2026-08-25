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
        self._set_default(
            "external_phase_gradient",
            "bool",
            True,
            "Whether the oracle reads a phase gradient register supplied by the caller instead of "
            "allocating and preparing its own."
        )


class QROMStatePreparation(StatePreparation):
    r"""State preparation using Quantum Read-Only Memory (QROM) based multiplexed rotations.

    Prepares an arbitrary n-qubit state using n layers of multiplexed Ry rotations,
    where each layer's angles are loaded from a QROM table.
    """

    def __init__(self, rotation_bit_precision: int = 10, external_phase_gradient: bool = True):
        """Initialize QROMStatePreparation.

        Args:
            rotation_bit_precision: Number of bits for multiplexed :math:`R_y`
                angle precision. Higher values give more accurate rotations.
                Defaults to 10. Equivalent to setting the ``rotation_bit_precision``
                entry of ``settings()``.
            external_phase_gradient: Whether :meth:`prepare_oracle` reads a caller-supplied
                phase gradient register rather than allocating its own. Defaults to True.
        """
        super().__init__()
        self._settings = QROMStatePreparationSettings()
        self._settings.set("rotation_bit_precision", rotation_bit_precision)
        self._settings.set("external_phase_gradient", external_phase_gradient)

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
        if not np.all(np.isfinite(coeffs)) or not np.any(coeffs != 0.0):
            raise ValueError("QROM state preparation requires finite, non-zero coefficients.")

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

    def num_system_qubits(self, wavefunction: Wavefunction) -> int:
        r"""Return the width of the index register SELECT controls on.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The state register width :math:`n = \lceil\log_2 L\rceil`.

        """
        return self._num_state_qubits(len(self._build_params(wavefunction).amplitudes))

    def num_phase_gradient_ancillas(self, wavefunction: Wavefunction) -> int:
        """Return the width of the phase gradient register the caller must supply.

        Zero unless ``external_phase_gradient`` is set, in which case every rotation this
        oracle applies reads the caller's gradient. Preparing that gradient is where all of
        this oracle's arbitrary-angle rotations come from, so hoisting it out of a repeated
        block encoding removes a cost that would otherwise scale with the repetition count.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            ``rotation_bit_precision`` qubits, or zero when the oracle allocates its own.

        """
        del wavefunction
        if not bool(self._settings.get("external_phase_gradient")):
            return 0
        return int(self._settings.get("rotation_bit_precision"))

    def prepare_oracle(self, wavefunction: Wavefunction) -> Any:
        """Return the PREPARE callable to embed in a block encoding.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            A Q# callable expecting ``[state | phaseGradient]`` when
            ``external_phase_gradient`` is set, and the state register alone otherwise.

        """
        return QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOracle(
            self._build_params(wavefunction),
            bool(self._settings.get("external_phase_gradient")),
        )
