"""Regular isometry module for quantum state preparation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging

from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager

from qdk_chemistry.algorithms import register
from qdk_chemistry.algorithms.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import Wavefunction
from qdk_chemistry.plugins.qiskit._interop.transpiler import (
    MergeZBasisRotations,
    RemoveZBasisOnZeroState,
    SubstituteCliffordRz,
)

_LOGGER = logging.getLogger(__name__)


class RegularIsometryStatePreparation(StatePreparation):
    """State preparation using a regular isometry approach.

    This class implements the isometry-based state preparation proposed by
    Matthias Christandl in `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_.
    """

    def __init__(self):
        """Initialize the RegularIsometryStatePreparation."""
        super().__init__()
        self._settings = StatePreparationSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> str:
        """Create a quantum circuit that prepares the state using regular isometry.

        Args:
            wavefunction: Wavefunction to prepare state from

        Returns:
            A QASM string representation of the quantum circuit.

        """
        # Active Space Consistency Check
        alpha_indices, beta_indices = wavefunction.get_orbitals().get_active_space_indices()
        if alpha_indices != beta_indices:
            raise ValueError(
                f"Active space contains {len(alpha_indices)} alpha orbitals and "
                f"{len(beta_indices)} beta orbitals. Asymmetric active spaces for "
                "alpha and beta orbitals are not supported for state preparation."
            )

        num_orbitals = len(alpha_indices)
        n_qubits = num_orbitals * 2
        num_dets = wavefunction.size()
        _LOGGER.debug(f"Using {num_dets} determinants for state preparation")

        # Create statevector using efficient C++ implementation
        statevector_data = wavefunction.to_statevector(normalize=True)

        # Create the circuit
        circuit = QuantumCircuit(n_qubits, name=f"regular_isometry_{num_dets}_det")

        # Use the StatePreparation class which implements efficient decomposition
        state_prep = QiskitStatePreparation(Statevector(statevector_data), normalize=True)
        circuit.append(state_prep, range(n_qubits))

        # Transpile the circuit if needed
        basis_gates = self._settings.get("basis_gates")
        do_transpile = self._settings.get("transpile")
        if do_transpile and basis_gates:
            opt_level = self._settings.get("transpile_optimization_level")
            circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=opt_level)
            pass_manager = PassManager(
                [
                    MergeZBasisRotations(),
                    RemoveZBasisOnZeroState(),
                    SubstituteCliffordRz(),
                ]
            )
            circuit = pass_manager.run(circuit)

        return qasm3.dumps(circuit)

    def name(self) -> str:
        """Return the name of the state preparation method."""
        return "regular_isometry"


register(lambda: RegularIsometryStatePreparation())
