"""QDK/Chemistry dense pure-state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["DensePureStatePreparation"]


class DensePureStatePreparation(StatePreparation):
    r"""State preparation using the Q# ``PreparePureStateD`` operation.

    This is the simplest dense amplitude-loading strategy: given an arbitrary
    real-valued amplitude vector, it uses ``PreparePureStateD`` to prepare the
    corresponding state on a qubit register.

    This algorithm is used both for:

    * **Electronic-structure state preparation** (via :meth:`run` with a
      :class:`~qdk_chemistry.data.Wavefunction`), and
    * **Block-encoding PREPARE oracles** (via :meth:`prepare_from_statevector`
      with a raw amplitude array).

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

    def _run_impl(self, wavefunction):
        """Prepare a quantum circuit from a Wavefunction using PreparePureStateD.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            Circuit: A Circuit object implementing the state preparation.

        Raises:
            NotImplementedError: Full wavefunction preparation is not yet implemented.
                Use :meth:`prepare_from_statevector` for amplitude-vector preparation.

        """
        raise NotImplementedError(
            "Full wavefunction preparation via DensePureStatePreparation is not yet implemented. "
            "Use prepare_from_statevector() for direct amplitude-vector preparation."
        )

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
