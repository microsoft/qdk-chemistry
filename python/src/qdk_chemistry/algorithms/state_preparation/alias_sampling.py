"""QDK/Chemistry alias sampling state preparation algorithm."""

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
            "prepared probability is within 2^-mu of the target, at the cost of one extra "
            "uniform qubit and one extra Quantum Read-Only Memory (QROM) output qubit per bit. "
            "The upper bound of 30 is a sanity limit rather than an algorithmic one: 2^-30 is "
            "already far below chemical accuracy.",
            (1, 30),
        )


class AliasSamplingStatePreparation(StatePreparation):
    r"""LCU PREPARE oracle built with coherent alias sampling.

    Implements section III.D of :cite:`Babbush2018`. Given :math:`L` real coefficients
    :math:`c_\ell`, this prepares

    .. math::

        \sum_{\ell} \sqrt{\tilde{p}_\ell}\,|\ell\rangle\,|\text{garbage}_\ell\rangle,
        \qquad \tilde{p}_\ell \approx p_\ell = \frac{|c_\ell|}{\sum_k |c_k|},

    where :math:`\tilde{p}` is :math:`p` discretized to :math:`\mu` bits.

    .. warning::

        **This is not a drop-in replacement for** ``dense_pure_state`` **or**
        ``sparse_isometry``. It differs from them in two ways that matter:

        1. **Amplitudes are square roots of normalized magnitudes**, not the coefficients
           themselves. It realizes :math:`\sqrt{|c_\ell| / \sum_k |c_k|}`, not
           :math:`c_\ell / \lVert c \rVert_2`. Coefficient signs are discarded, which is why
           negative coefficients are rejected outright.
        2. **The index register stays entangled with ancilla.** The output is not a pure
           state on the index register alone. This circuit is only meaningful as the PREPARE
           subroutine of a block encoding (LCU or qubitization), where PREPARE\ :sup:`†`
           later uncomputes the garbage and projects onto the correct subspace.

        Index :math:`\ell` is the *position* of a coefficient in the wavefunction's
        coefficient vector, not a Jordan-Wigner determinant bit pattern, so the returned
        circuit carries no fermionic encoding.

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
        coeffs = np.asarray(wavefunction.get_coefficients())
        if coeffs.size == 0:
            raise ValueError("Alias sampling state preparation requires at least one coefficient.")
        if np.iscomplexobj(coeffs):
            if np.any(coeffs.imag != 0.0):
                raise ValueError("Alias sampling state preparation requires real coefficients.")
            coeffs = coeffs.real
        coeffs = coeffs.astype(float, copy=False)

        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Alias sampling state preparation requires finite coefficients.")
        if np.any(coeffs < 0.0):
            raise ValueError(
                "Alias sampling state preparation requires non-negative coefficients. It is an LCU "
                "PREPARE oracle: it prepares amplitudes sqrt(|c_l| / sum_k |c_k|) and has no way to "
                "represent a coefficient's sign, so accepting a negative value would silently "
                "prepare the wrong state. Pass the absolute values and carry the signs in the "
                "accompanying SELECT oracle, or use a signed algorithm such as 'dense_pure_state'."
            )
        if not np.any(coeffs > 0.0):
            raise ValueError(
                "Alias sampling state preparation requires at least one non-zero coefficient; "
                "an all-zero vector has no probability distribution to prepare."
            )

        coefficients = coeffs.tolist()
        num_index_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1
        padded_len = 1 << num_index_qubits
        if len(coefficients) < padded_len:
            coefficients = coefficients + [0.0] * (padded_len - len(coefficients))
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

        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)
