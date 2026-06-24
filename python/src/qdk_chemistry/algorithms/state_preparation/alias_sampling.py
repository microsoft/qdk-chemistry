"""QDK/Chemistry alias sampling state preparation algorithm."""

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

__all__: list[str] = ["AliasSamplingStatePreparation"]


class AliasSamplingStatePreparation(StatePreparation):
    r"""State preparation using the alias sampling method.

    The algorithm implements section III.D. of :cite:`Babbush2018`, to
    prepares an arbitrary probability distribution over L terms using:
      - :math:`\lceil\log_2 L\rceil` index qubits
      - :math:`\mu` comparison (uniform) qubits
      - 1 flag qubit
      - :math:`\mu + \lceil\log_2 L\rceil` QROM output qubits

    Total ancilla: :math:`2\lceil\log_2 L\rceil + 2\mu + 1` qubits.

    The circuit proceeds:
      1. PrepareUniformSuperposition over L terms
      2. H⊗μ on comparison register
      3. QROM load of (keep_l, alt_l) alias table
      4. Comparison: flag = (sigma >= keep_l)
      5. Conditional swap: if flag, index <-> alt_l

    The Toffoli cost scales as O(L) for the QROM, independent of precision μ.

    """

    def __init__(self, bits_precision: int = 10):
        """Initialize AliasSamplingStatePreparation.

        Args:
            bits_precision: Number of bits μ for keep-coefficient precision.
                Higher values give more accurate state preparation at the cost
                of more ancilla qubits. Defaults to 10.

        """
        super().__init__()
        self._bits_precision = bits_precision

    def name(self) -> str:
        """Return the algorithm name."""
        return "alias_sampling"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        r"""State preparation using the alias sampling method.

        Extracts coefficients from the wavefunction and builds an alias sampling
        circuit that prepares the state.

        Args:
            wavefunction: The target wavefunction.

        Returns:
            Circuit: A Circuit wrapping the Q# alias sampling callable and factory.

        """
        coeffs = np.asarray(wavefunction.get_coefficients())
        if np.iscomplexobj(coeffs):
            if not np.allclose(coeffs.imag, 0.0):
                raise ValueError("Alias sampling state preparation requires real coefficients.")
            coeffs = coeffs.real

        coefficients = coeffs.tolist()
        num_index_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1
        padded_len = 1 << num_index_qubits
        if len(coefficients) < padded_len:
            coefficients = coefficients + [0.0] * (padded_len - len(coefficients))
        total_qubits = 2 * num_index_qubits + 2 * self._bits_precision + 1

        params = QSHARP_UTILS.AliasSampling.AliasSamplingParams(
            coefficients=coefficients,
            bitsPrecision=self._bits_precision,
            numIndexQubits=num_index_qubits,
            numQubits=total_qubits,
        )

        qsharp_op = QSHARP_UTILS.AliasSampling.MakeAliasSamplingOp(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.AliasSampling.MakeAliasSamplingCircuit,
            parameter={
                "coefficients": coefficients,
                "bitsPrecision": self._bits_precision,
                "numIndexQubits": num_index_qubits,
                "numQubits": total_qubits,
            },
        )

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)

    @property
    def bits_precision(self) -> int:
        """Number of bits for keep-coefficient precision."""
        return self._bits_precision
