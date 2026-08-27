"""QDK/Chemistry alias sampling state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["AliasSamplingStatePreparation", "AliasSamplingStatePreparationSettings"]


class AliasSamplingStatePreparationSettings(Settings):
    """Settings for :class:`AliasSamplingStatePreparation`."""

    def __init__(self):
        """Initialize the AliasSamplingStatePreparationSettings."""
        super().__init__()
        self._set_default(
            "bits_precision",
            "int",
            10,
            "Number of bits mu of precision for the alias table's keep probabilities. Each "
            "prepared probability is within 1/(L 2^mu) of the target for L coefficients. "
            "The upper bound of 30 is a sanity limit as 2^-30 is far below chemical accuracy.",
            (1, 30),
        )


class AliasSamplingStatePreparation(StatePreparation):
    r"""LCU PREPARE oracle built with coherent alias sampling.

    Implements section III.D of :cite:`Babbush2018`. Given :math:`L` real, non-negative
    coefficients :math:`c_\ell`, this prepares

    .. math::

        \sum_{\ell} \sqrt{\tilde{p}_\ell}\,|\ell\rangle\,|\text{garbage}_\ell\rangle,
        \qquad \tilde{p}_\ell \approx p_\ell = \frac{c_\ell^2}{\sum_k c_k^2},

    where :math:`\tilde{p}` is :math:`p` discretized to :math:`\mu` bits. The index
    amplitudes are therefore :math:`c_\ell / \lVert c \rVert_2`, matching
    ``dense_pure_state``, so the same coefficient vector means the same thing to either.

    .. warning::

        **The index register stays entangled with ancilla.** This is a block-encoding
        subroutine, not a general state preparation for algorithms like QPE. It is only
        meaningful as the PREPARE subroutine of an LCU or qubitization circuit, where
        PREPARE\ :sup:`†` later uncomputes the garbage and projects onto the correct
        subspace.

        Negative coefficients are not supported. Index :math:`\ell` is the determinant's
        bit pattern, matching ``dense_pure_state``.

    The circuit proceeds:

    1. ``PrepareUniformSuperposition`` over L terms
    2. :math:`H^{\otimes\mu}` on the comparison register
    3. QROM load of the ``(keep_l, alt_l)`` alias table
    4. Comparison: ``flag = (sigma >= keep_l)``
    5. Conditional swap: if ``flag`` is set, ``index <- alt_l``

    Named registers total :math:`2\lceil\log_2 L\rceil + 2\mu + 1` qubits:
    :math:`\lceil\log_2 L\rceil` index, :math:`\mu` uniform, 1 flag, and
    :math:`\mu + \lceil\log_2 L\rceil` QROM output. The QROM and conditional-swap
    implementations allocate further scratch ancilla on top of that.

    The Toffoli count is dominated by the :math:`O(L)` QROM lookup; the comparator adds a
    further :math:`O(\mu)`, so the total grows slowly with :math:`\mu` rather than being
    independent of it.

    """

    def __init__(self, bits_precision: int = 10):
        """Initialize AliasSamplingStatePreparation.

        Args:
            bits_precision: Number of bits μ for keep-coefficient precision.
                Higher values give more accurate state preparation at the cost
                of more ancilla qubits. Defaults to 10. Equivalent to setting the
                ``bits_precision`` entry of ``settings()``.

        """
        super().__init__()
        self._settings = AliasSamplingStatePreparationSettings()
        self._settings.set("bits_precision", bits_precision)

    def name(self) -> str:
        """Return the algorithm name."""
        return "alias_sampling"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        r"""Build the alias sampling PREPARE circuit for a wavefunction.

        Args:
            wavefunction: The target wavefunction. Its coefficients must be real and
                non-negative; see the class docstring for why.

        Returns:
            Circuit: A Circuit wrapping the Q# alias sampling callable and factory.

        Raises:
            ValueError: If the wavefunction has no coefficients, has an imaginary part,
                contains a non-finite or negative coefficient, or is all zeros.

        """
        coefficients, num_index_qubits = self._sampling_weights(wavefunction)
        bits_precision = int(self._settings.get("bits_precision"))
        total_qubits = 2 * num_index_qubits + 2 * bits_precision + 1

        params = QSHARP_UTILS.AliasSampling.AliasSamplingParams(
            coefficients=coefficients,
            bitsPrecision=bits_precision,
            numIndexQubits=num_index_qubits,
            numQubits=total_qubits,
        )

        qsharp_op = QSHARP_UTILS.AliasSampling.MakeAliasSamplingOp(params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.AliasSampling.MakeAliasSamplingCircuit,
            parameter={
                "coefficients": coefficients,
                "bitsPrecision": bits_precision,
                "numIndexQubits": num_index_qubits,
                "numQubits": total_qubits,
            },
        )

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory, num_qubits=total_qubits)

    @classmethod
    def _sampling_weights(cls, wavefunction: Wavefunction) -> tuple[list[float], int]:
        """Return the sampling weights ``|c|^2`` for a wavefunction's amplitudes.

        The Q# layer normalizes these itself, so they are returned unnormalized.

        Args:
            wavefunction: The target wavefunction.

        Returns:
            The squared amplitudes indexed by determinant, and the index register width.

        Raises:
            ValueError: If the coefficients are empty, complex, non-finite, negative, or
                all zero.

        """
        coeffs, num_index_qubits = cls._dense_state_vector(wavefunction, "Alias sampling state preparation")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Alias sampling state preparation requires finite coefficients.")
        if np.any(coeffs < 0.0):
            raise ValueError("Alias sampling state preparation requires non-negative coefficients.")
        weights = coeffs**2
        if not np.all(np.isfinite(weights)):
            raise ValueError("Alias sampling state preparation overflows to infinity when squaring; rescale first.")
        if not np.any(weights != 0.0):
            raise ValueError("Alias sampling state preparation requires at least one non-zero coefficient.")
        return weights.tolist(), num_index_qubits
