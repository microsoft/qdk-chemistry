"""Regular isometry module for quantum state preparation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager

from qdk_chemistry.algorithms.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.plugins.qiskit._interop.transpiler import (
    MergeZBasisRotations,
    RemoveZBasisOnZeroState,
    SubstituteCliffordRz,
)
from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction
from qdk_chemistry.utils import Logger

__all__ = ["RegularIsometryStatePreparation"]


class RegularIsometryStatePreparation(StatePreparation):
    """State preparation using a regular isometry approach.

    This class implements the isometry-based state preparation proposed by
    Matthias Christandl in arXiv:1501.06911 :cite:`Christandl2016`.
    """

    def __init__(self):
        """Initialize the RegularIsometryStatePreparation."""
        Logger.trace_entering()
        super().__init__()
        self._settings = StatePreparationSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Create a quantum circuit that prepares the state using regular isometry.

        Args:
            wavefunction: Wavefunction to prepare state from

        Returns:
            A Circuit object containing a QASM string representation of the quantum circuit.

        """
        Logger.trace_entering()
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
        Logger.debug(f"Using {num_dets} determinants for state preparation")

        # Create statevector using Python conversion function
        statevector_data = create_statevector_from_wavefunction(wavefunction, normalize=True)

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

        return Circuit(qasm=qasm3.dumps(circuit), encoding="jordan-wigner")

    def name(self) -> str:
        """Return the name of the state preparation method."""
        Logger.trace_entering()
        return "qiskit_regular_isometry"
