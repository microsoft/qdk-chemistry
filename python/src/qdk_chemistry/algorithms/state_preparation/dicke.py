"""QDK/Chemistry Dicke state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import comb

import numpy as np

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["DickeStatePreparation"]


class DickeStatePreparation(StatePreparation):
    r"""State preparation for a uniform Dicke state :math:`|D^n_k\rangle`.

    A Dicke state is the equal-amplitude superposition of all :math:`n`-qubit
    computational-basis states of Hamming weight :math:`k`:

    .. math::

        |D^n_k\rangle = \binom{n}{k}^{-1/2} \sum_{|x| = k} |x\rangle .

    Unlike a dense amplitude load, this algorithm exploits the permutation
    symmetry that defines a Dicke state to emit a compact, structure-preserving
    preparation circuit.  The input :class:`~qdk_chemistry.data.Wavefunction`
    must describe such a uniform Dicke state: every determinant shares the same
    Hamming weight :math:`k`, the support covers all :math:`\binom{n}{k}` weight-
    :math:`k` basis states, and all coefficients have equal magnitude.

    .. note::
        Only the weight-1 case (the uniform one-hot superposition) is currently
        supported.  Higher weights raise :class:`NotImplementedError`.

    """

    def __init__(self):
        """Initialize the DickeStatePreparation."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"dicke"``.

        """
        return "dicke"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        r"""Prepare a circuit for the uniform Dicke state described by *wavefunction*.

        Args:
            wavefunction: The target wavefunction, which must describe a uniform Dicke state :math:`|D^n_k\rangle`.

        Returns:
            Circuit: A Circuit object implementing the Dicke state preparation.

        Raises:
            ValueError: If the wavefunction does not describe a uniform Dicke state.
            NotImplementedError: If the Dicke weight is greater than 1.

        """
        config_set = wavefunction.get_configuration_set()
        dets = wavefunction.get_active_determinants()
        coeffs = np.asarray(wavefunction.get_coefficients())

        if len(dets) == 0:
            raise ValueError("Dicke state preparation requires a non-empty wavefunction.")

        n_qubits = config_set.num_modes() * dets[0].bits_per_mode()
        weight = self._validate_uniform_dicke(dets, coeffs, n_qubits)

        if weight != 1:
            raise NotImplementedError("Dicke state preparation currently supports only weight-1 states.")

        params = QSHARP_UTILS.Dicke.DickeParams(numQubits=n_qubits, weight=weight)
        qsharp_op = QSHARP_UTILS.Dicke.MakeDickeOp(params)
        qsharp_factory = QsharpFactoryData(program=QSHARP_UTILS.Dicke.MakeDickeCircuit, parameter=vars(params))
        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory, encoding="jordan-wigner")

    @staticmethod
    def _validate_uniform_dicke(dets: list, coeffs: np.ndarray, n_qubits: int) -> int:
        r"""Validate that the determinants and coefficients describe a uniform Dicke state.

        Args:
            dets: The determinants of the wavefunction.
            coeffs: The coefficients of the wavefunction.
            n_qubits: The register size :math:`n`.

        Returns:
            int: The common Hamming weight :math:`k` of the Dicke state.

        Raises:
            ValueError: If the determinants and coefficients do not describe a uniform Dicke state.

        """
        weights = {sum(det.to_bits(n_qubits)) for det in dets}
        if len(weights) != 1:
            raise ValueError("Dicke state preparation requires all determinants to share the same Hamming weight.")
        weight = weights.pop()

        if len(dets) != comb(n_qubits, weight):
            raise ValueError(
                f"Dicke state preparation requires the full weight-{weight} support "
                f"({comb(n_qubits, weight)} determinants), got {len(dets)}."
            )

        magnitudes = np.abs(coeffs)
        if not np.allclose(magnitudes, magnitudes[0]):
            raise ValueError("Dicke state preparation requires all coefficients to have equal magnitude.")

        return weight
