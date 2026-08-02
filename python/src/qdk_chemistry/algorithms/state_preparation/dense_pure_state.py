"""QDK/Chemistry dense pure-state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np

from qdk_chemistry.data import Wavefunction
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

    def _run_impl(self, wavefunction: Wavefunction | Sequence[float] | np.ndarray) -> Circuit:
        """Prepare a quantum circuit using PreparePureStateD.

        Accepts either a :class:`~qdk_chemistry.data.Wavefunction`, in which case
        a dense statevector is built from its determinants and coefficients, or a
        dense amplitude vector of length ``2**n_qubits`` given directly as a
        sequence or array.

        The amplitude-vector form exists because ``PreparePureStateD`` itself has
        no notion of orbitals or determinants: once a statevector is in hand, the
        remainder of this routine is independent of where it came from. Requiring
        a ``Wavefunction`` forces callers who already hold amplitudes -- model
        Hamiltonians, spin systems, or externally computed guiding states -- to
        construct orbitals that carry no meaning for their problem.

        Args:
            wavefunction: The target wavefunction, or a real amplitude vector
                whose length is a power of two.

        Returns:
            Circuit: A Circuit object implementing the state preparation.

        Raises:
            ValueError: If amplitudes are complex, the vector length is not a
                power of two, the vector has zero norm, or more than 32 qubits
                would be required.

        """
        if not isinstance(wavefunction, Wavefunction):
            return self._circuit_from_statevector(np.asarray(wavefunction))

        config_set = wavefunction.get_configuration_set()
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

        return self._circuit_from_statevector(statevector, n_qubits=n_qubits)

    @staticmethod
    def _circuit_from_statevector(statevector: np.ndarray, n_qubits: int | None = None) -> Circuit:
        """Wrap a dense real statevector in a Q# ``PreparePureStateD`` circuit."""
        statevector = np.asarray(statevector)
        if np.iscomplexobj(statevector):
            if not np.allclose(statevector.imag, 0.0):
                raise ValueError("Dense state preparation requires real amplitudes (imaginary part must be zero).")
            statevector = statevector.real
        statevector = statevector.astype(float, copy=True)

        if n_qubits is None:
            size = statevector.size
            if size < 2 or size & (size - 1):
                raise ValueError(f"Amplitude vector length must be a power of two greater than one. Got {size}.")
            n_qubits = size.bit_length() - 1
            if n_qubits > 32:
                raise ValueError("Dense state preparation is only supported for up to 32 qubits.")
            norm = float(np.linalg.norm(statevector))
            if norm == 0.0:
                raise ValueError("Cannot prepare a state from an all-zero amplitude vector.")
            statevector /= norm

        state_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=list(range(n_qubits - 1, -1, -1)),
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
