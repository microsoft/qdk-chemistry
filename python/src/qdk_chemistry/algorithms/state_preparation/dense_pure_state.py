"""QDK/Chemistry dense pure-state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import StateVectorContainer, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["DensePureStatePreparation"]


class DensePureStatePreparation(StatePreparation):
    r"""State preparation using the Q# ``PreparePureStateD`` operation.

    This is the simplest dense amplitude-loading strategy: given an arbitrary
    real-valued amplitude vector, it uses ``PreparePureStateD`` to prepare the
    corresponding state on a qubit register.

    """

    def __init__(self):
        """Initialize the DensePureStatePreparation."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"dense_pure_state"``.

        """
        return "dense_pure_state"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit from a Wavefunction using PreparePureStateD.

        Builds a dense statevector directly from the wavefunction's
        determinants and coefficients, normalizes it, and wraps it in a
        Q# ``StatePreparation`` circuit.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            Circuit: A Circuit object implementing the state preparation.

        """
        container = wavefunction.get_container()
        if isinstance(container, StateVectorContainer):
            config_set = container.get_configuration_set()
        else:
            raise ValueError("Dense state preparation requires a state vector container.")
        dets = wavefunction.get_active_determinants()
        coeffs = np.asarray(wavefunction.get_coefficients())
        if np.iscomplexobj(coeffs):
            if not np.allclose(coeffs.imag, 0.0):
                raise ValueError("Dense state preparation requires real coefficients (imaginary part must be zero).")
            coeffs = coeffs.real
        n_bits = config_set.num_modes() * dets[0].bits_per_mode()
        n_qubits = n_bits
        if n_qubits > 32:
            raise ValueError("Dense state preparation is only supported for up to 32 qubits.")
        statevector = np.zeros(2**n_qubits, dtype=float)
        for coeff, det in zip(coeffs, dets, strict=True):
            bits = det.to_bits(n_bits)
            idx = 0
            for i, b in enumerate(bits):
                idx |= b << i
            statevector[idx] += coeff

        row_map = list(range(n_qubits - 1, -1, -1))
        state_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=row_map,
            stateVector=statevector.tolist(),
            expansionOps=[],
            numQubits=n_qubits,
        )

        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=vars(state_prep_params),
        )
        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory, encoding="jordan-wigner")
