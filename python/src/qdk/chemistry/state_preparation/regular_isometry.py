"""Regular isometry module for quantum state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from qdk.chemistry.data import Wavefunction
from qdk.chemistry.utils.statevector import create_statevector_from_coeffs_and_dets_string

from .base import StatePrep, StatePrepAlgorithm


class RegularIsometryStatePrep(StatePrep):
    """State preparation using a regular isometry approach.

    This class implements the isometry-based state preparation proposed by
    Matthias Christandl in `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_.
    """

    algorithm = StatePrepAlgorithm.REGULAR_ISOMETRY

    def __init__(self, wavefunction: Wavefunction, max_dets: int | None = None, amplitude_threshold: float = 0.0):
        """Initialize the regular isometry state preparation object.

        Args:
            wavefunction: Wavefunction to prepare state from
            max_dets: Maximum number of determinants to include in the isometry (default to include all)
            amplitude_threshold: Amplitude threshold for including determinants (default: 0.0)

        """
        super().__init__(wavefunction, max_dets, amplitude_threshold)

    def create_circuit_qasm(self) -> str:
        """Create a quantum circuit that prepares the state using regular isometry.

        Returns:
            A QASM string representation of the quantum circuit.

        """
        circuit = self._create_regular_isometry_circuit()

        return qasm3.dumps(circuit)

    def _create_regular_isometry_circuit(self) -> QuantumCircuit:
        """Create a quantum circuit implementing regular isometry.

        Reference: `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_

        Returns:
            A quantum circuit implementing the regular isometry

        """
        # Calculate the number of qubits (2 * number of orbitals)
        num_qubits = 2 * self.num_orbitals

        # Filter coefficients and bitstrings based on threshold and max_dets
        filtered_coeffs, filtered_bitstrings = self._filter_terms()

        # Create a statevector from the filtered terms
        statevector_data = create_statevector_from_coeffs_and_dets_string(
            filtered_coeffs, filtered_bitstrings, num_qubits
        )

        # Create the circuit
        circuit = QuantumCircuit(num_qubits, name=f"circuit_{self.algorithm!s}")

        # Use the StatePreparation class which implements efficient decomposition
        state_prep = StatePreparation(Statevector(statevector_data), normalize=True)
        circuit.append(state_prep, range(num_qubits))

        return circuit
