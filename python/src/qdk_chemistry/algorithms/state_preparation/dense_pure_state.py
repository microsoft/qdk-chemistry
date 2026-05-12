"""QDK/Chemistry dense pure-state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit from a Wavefunction using PreparePureStateD.

        Extracts coefficients and determinants from the wavefunction, converts
        them to a full statevector in the computational basis, normalizes it,
        and wraps it in a Q# ``StatePreparation`` circuit.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            Circuit: A Circuit object implementing the state preparation.

        """
        alpha_indices, beta_indices = wavefunction.get_orbitals().get_active_space_indices()
        if alpha_indices != beta_indices:
            raise ValueError(
                f"Active space contains {len(alpha_indices)} alpha orbitals and "
                f"{len(beta_indices)} beta orbitals. Asymmetric active spaces for "
                "alpha and beta orbitals are not supported for state preparation."
            )
        coeffs = wavefunction.get_coefficients()
        dets = wavefunction.get_active_determinants()
        num_orbitals = max(len(alpha_indices), len(beta_indices))

        # Convert determinants to bitstrings (little-endian convention)
        bitstrings = []
        for det in dets:
            alpha_str, beta_str = det.to_binary_strings(num_orbitals)
            bitstring = beta_str[::-1] + alpha_str[::-1]
            bitstrings.append(bitstring)

        n_qubits = len(bitstrings[0])

        # Build the full statevector: place each coefficient at the index
        # corresponding to its bitstring
        statevector = np.zeros(2**n_qubits, dtype=float)
        for coeff, bitstring in zip(coeffs, bitstrings, strict=True):
            index = int(bitstring, 2)
            statevector[index] += coeff

        # Normalize
        norm = np.linalg.norm(statevector)
        if norm > 0:
            statevector /= norm

        # All qubits participate in the dense preparation, no expansion ops
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

    def prepare_from_statevector(
        self,
        statevector: np.ndarray,
        num_qubits: int,
        qubit_indices: list[int],
    ) -> Circuit:
        """Create a PREPARE circuit from a statevector and qubit layout.

        Uses ``PreparePureStateD`` to load the given amplitudes into a qubit
        register via the ``StatePreparation`` Q# operation with the specified
        qubit indices and no expansion ops.

        Args:
            statevector: A 1-D array of real amplitudes to load.
            num_qubits: Number of qubits in the prepare register.
            qubit_indices: Qubit indices for the prepare register.

        Returns:
            Circuit: A Circuit wrapping the Q# PREPARE callable and factory.

        TODO: Refactor Wavefunction to adpot PREPARE oracle for block encoding

        """
        amplitudes = statevector.tolist()
        prepare_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=qubit_indices,
            stateVector=amplitudes,
            expansionOps=[],
            numQubits=num_qubits,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(prepare_params)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter={
                "rowMap": qubit_indices,
                "stateVector": amplitudes,
                "expansionOps": [],
                "numQubits": num_qubits,
            },
        )
        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)
