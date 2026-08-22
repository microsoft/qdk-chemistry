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

# Relative bound for treating a complex coefficient vector as real. A bare absolute
# tolerance would reject legitimately real vectors that are merely large and accept tiny
# vectors whose imaginary part dominates, so the bound is scaled by the largest magnitude.
_IMAG_TOLERANCE = 1e-8


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
            "wider QROM output register.",
            (1, 30),
        )


class QROMStatePreparation(StatePreparation):
    r"""State preparation using QROM-based multiplexed rotations.

    Prepares an arbitrary n-qubit state using n layers of multiplexed Ry rotations,
    where each layer's angles are loaded from a QROM table.

    This approach uses only :math:`n = \lceil\log_2 L\rceil` state qubits, but requires
    n QROM lookups (plus scratch ancilla for each lookup).

    Index :math:`\ell` is the *position* of a coefficient in the wavefunction's coefficient
    vector, not a Jordan-Wigner determinant bit pattern, so the returned circuit carries no
    fermionic encoding.

    .. warning::

        **Negative coefficients are not supported yet.** Ry rotations only generate
        non-negative amplitudes, so signs are applied by a separate QROM-loaded ``Z`` phase
        kickback. That lookup is not correctly uncomputed: the sign ancilla is released
        while still entangled with the state register, so it is implicitly measured and the
        signs collapse at random. Magnitudes remain correct, but the sign pattern varies
        between simulator seeds. Passing a negative coefficient raises
        :class:`ValueError`; use a signed algorithm such as ``dense_pure_state`` until
        this is fixed.
    """

    def __init__(self, rotation_bit_precision: int = 10):
        """Initialize QROMStatePreparation.

        Args:
            rotation_bit_precision: Number of bits for Givens rotation angle
                precision. Higher values give more accurate rotations.
                Defaults to 10. Equivalent to setting the ``rotation_bit_precision``
                entry of :attr:`settings`.

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
            ValueError: If the wavefunction has no coefficients, has a non-negligible
                imaginary part, or contains a negative coefficient.

        """
        coeffs = np.asarray(wavefunction.get_coefficients())
        if coeffs.size == 0:
            raise ValueError("QROM state preparation requires at least one coefficient.")
        if np.iscomplexobj(coeffs):
            scale = max(1.0, float(np.abs(coeffs).max()))
            if not np.allclose(coeffs.imag, 0.0, rtol=0.0, atol=_IMAG_TOLERANCE * scale):
                raise ValueError("QROM state preparation requires real coefficients.")
            coeffs = coeffs.real
        coeffs = coeffs.astype(float, copy=False)

        if np.any(coeffs < 0.0):
            raise ValueError(
                "QROM state preparation does not support negative coefficients. Ry rotations only "
                "produce non-negative amplitudes, so the sign is applied by a separate QROM-loaded "
                "Z phase kickback; that lookup is not correctly uncomputed, so the sign ancilla "
                "stays entangled with the state register and is implicitly measured when released. "
                "The magnitudes would be right but the signs collapse at random, and the result "
                "varies between simulator seeds. Use an algorithm that supports signed amplitudes "
                "(for example 'dense_pure_state') until this is fixed."
            )

        amplitudes = coeffs.tolist()
        num_state_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
        rotation_bit_precision = self.rotation_bit_precision

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

    @property
    def rotation_bit_precision(self) -> int:
        """Number of bits for rotation angle precision."""
        return int(self._settings.get("rotation_bit_precision"))

    @rotation_bit_precision.setter
    def rotation_bit_precision(self, value: int) -> None:
        self._settings.set("rotation_bit_precision", value)
