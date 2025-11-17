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
from qdk_chemistry.algorithms.state_preparation.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import Wavefunction
from qdk_chemistry.plugins.qiskit._interop.transpiler import (
    MergeZBasisRotations,
    RemoveZBasisOnZeroState,
    SubstituteCliffordRz,
)
from qdk_chemistry.utils.bitstring import separate_alpha_beta_to_binary_string
from qdk_chemistry.utils.statevector import _create_statevector_from_coeffs_and_dets_string

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

        coeffs = wavefunction.get_coefficients()
        dets = wavefunction.get_active_determinants()
        num_orbitals = len(wavefunction.get_orbitals().get_active_space_indices()[0])
        bitstrings = []
        for det in dets:
            alpha_str, beta_str = separate_alpha_beta_to_binary_string(det.to_string()[:num_orbitals])
            bitstring = beta_str[::-1] + alpha_str[::-1]  # Qiskit uses little-endian convention
            bitstrings.append(bitstring)

        if not bitstrings:
            raise ValueError("No valid bitstrings found. The determinants list might be empty.")
        n_qubits = len(bitstrings[0])
        _LOGGER.debug(f"Using {len(bitstrings)} determinants for state preparation")

        # Create a statevector from the filtered terms
        statevector_data = _create_statevector_from_coeffs_and_dets_string(coeffs, bitstrings, n_qubits)

        # Create the circuit
        circuit = QuantumCircuit(n_qubits, name=f"regular_isometry_{len(bitstrings)}_det")

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
