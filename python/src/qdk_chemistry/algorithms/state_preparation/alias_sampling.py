"""QDK/Chemistry alias sampling state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["AliasSamplingStatePreparation"]


class AliasSamplingStatePreparation(StatePreparation):
    r"""State preparation using the Walker alias sampling method.

    Prepares an arbitrary probability distribution over L terms using:
      - :math:`\lceil\log_2 L\rceil` index qubits
      - :math:`\mu` comparison (uniform) qubits
      - 1 flag qubit
      - :math:`\mu + \lceil\log_2 L\rceil` QROM output qubits

    Total ancilla: :math:`2\lceil\log_2 L\rceil + 2\mu + 1` qubits.

    The circuit implements (Babbush et al. arXiv:1805.03662):
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
        """Not directly applicable -- use prepare_from_statevector instead.

        Raises:
            NotImplementedError: Use prepare_from_statevector for block encoding PREPARE.

        """
        raise NotImplementedError(
            "AliasSamplingStatePreparation does not support direct Wavefunction input. "
            "Use prepare_from_statevector() for block encoding PREPARE oracles."
        )

    def prepare_from_statevector(
        self,
        statevector: np.ndarray,
        num_qubits: int,
        qubit_indices: list[int],
    ) -> Circuit:
        r"""Create a PREPARE circuit using alias sampling.

        The input statevector contains amplitudes :math:`\sqrt{p_\ell}` for each term.
        The alias sampling circuit prepares this state using O(L) Toffoli gates.

        Args:
            statevector: A 1-D array of real amplitudes to load (length L).
                The squared magnitudes are used as the probability distribution.
            num_qubits: Number of qubits in the index (prepare) register.
            qubit_indices: Qubit indices for the prepare register (used for layout).

        Returns:
            Circuit: A Circuit wrapping the Q# alias sampling callable and factory.

        """
        coefficients = statevector.tolist()
        num_index_qubits = num_qubits
        # Total qubits: index + uniform + flag + qrom_output
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
