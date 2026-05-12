"""Integration tests for QPE using the LCU block encoding unitary builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
)


class TestIQPEWithLCU:
    """Integration tests for IQPE using LCU unitary builder."""

    def test_iterative_qpe_with_lcu_two_qubit(self):
        """Verify IQPE with LCU quantum walk recovers the expected energy for a 2-qubit Hamiltonian.

        Uses H = (pi/4)*ZI + (pi/4)*IZ, a 2-term diagonal Hamiltonian.
        Eigenstate |00> has eigenvalue (pi/4) + (pi/4) = pi/2.
        lambda = pi/2, cos(2*pi*phi) = (pi/2)/(pi/2) = 1, phi = 0,
        exactly representable with 4 bits.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(
            pauli_strings=["ZI", "IZ"],
            coefficients=np.array([coeff, coeff]),
        )

        # |00> is eigenstate with eigenvalue (pi/4) + (pi/4) = pi/2
        state_vector = [1.0, 0.0, 0.0, 0.0]
        state_prep_params = {
            "rowMap": [1, 0],
            "stateVector": state_vector,
            "expansionOps": [],
            "numQubits": 2,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        # Configure IQPE with LCU
        iqpe = IterativePhaseEstimation(
            num_bits=4,
            shots_per_bit=3,
        )
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        )

        result = iqpe.run(
            qubit_hamiltonian=hamiltonian,
            state_preparation=state_prep,
        )

        expected_energy = np.pi / 2.0  # pi/4 + pi/4
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_lcu_h2_quantum_walk(self):
        """Verify QPE with block encoding quantum walk recovers H2 ground-state energy.

        Uses the full H2/STO-3G qubit Hamiltonian (15 Pauli terms, 4 qubits)
        with the exact ground state as the initial state. The quantum walk
        operator W = R*B[H] is used with IQPE to extract the eigenphase, and
        the energy is recovered via E = lambda * cos(2*pi*phi).

        Reference ground-state energy: -2.2472 Ha (from exact diagonalization).
        With 4 phase bits, the discretization error is ~0.02 Ha.
        """
        # H2 / STO-3G qubit Hamiltonian (Jordan-Wigner, 4 qubits, 15 terms)
        h2_pauli_strings = [
            "ZIZI",
            "YYYY",
            "XXYY",
            "IIII",
            "XXXX",
            "IIIZ",
            "IZII",
            "IIZI",
            "ZIII",
            "ZIIZ",
            "IIZZ",
            "IZZI",
            "ZZII",
            "IZIZ",
            "YYXX",
        ]
        h2_coefficients = np.array(
            [
                0.19176479,
                0.04104867,
                0.04104867,
                -0.5734373,
                0.04104867,
                0.23708567,
                0.23708567,
                -0.46083546,
                -0.46083546,
                0.18168163,
                0.14063296,
                0.18168163,
                0.14063296,
                0.18454294,
                0.04104867,
            ]
        )
        hamiltonian = QubitHamiltonian(
            pauli_strings=h2_pauli_strings,
            coefficients=h2_coefficients,
        )

        # Exact ground state of the qubit Hamiltonian (from diagonalization)
        ground_state_vector = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.99828664,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05851314,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        # Reference energy from exact diagonalization
        reference_energy = -2.2472250318780063

        num_qubits = 4
        state_prep_params = {
            "rowMap": list(range(num_qubits)),  # [0, 1, 2, 3]
            "stateVector": ground_state_vector,
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        # Configure IQPE with LCU quantum walk
        iqpe = IterativePhaseEstimation(
            num_bits=4,
            shots_per_bit=3,
        )
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        )

        result = iqpe.run(
            qubit_hamiltonian=hamiltonian,
            state_preparation=state_prep,
        )

        # With 4 phase bits the phase 0.376 rounds to 6/16 = 0.375,
        # giving E = lambda * cos(2*pi * 0.375) ~ -2.231.
        # Allow 0.02 Ha tolerance for discretization error.
        assert np.isclose(
            result.raw_energy,
            reference_energy,
            atol=0.02,
        )

    def test_iterative_qpe_with_lcu_off_diagonal(self):
        """Verify IQPE with LCU quantum walk handles off-diagonal Pauli terms.

        Uses H = (pi/4)*XX + (pi/4)*ZZ, a Hamiltonian with both off-diagonal (XX)
        and diagonal (ZZ) terms.
        Eigenstate (|00> + |11>)/sqrt(2) has eigenvalue pi/2.
        lambda = pi/2, cos(2*pi*phi) = 1, phi = 0.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([coeff, coeff]),
        )

        # (|00> + |11>)/sqrt(2) is eigenstate with eigenvalue pi/2
        s = 1.0 / np.sqrt(2.0)
        state_vector = [s, 0.0, 0.0, s]
        state_prep_params = {
            "rowMap": [1, 0],
            "stateVector": state_vector,
            "expansionOps": [],
            "numQubits": 2,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        iqpe = IterativePhaseEstimation(num_bits=4, shots_per_bit=3)
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = np.pi / 2.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_lcu_negative_coefficient(self):
        """Verify IQPE with LCU quantum walk handles negative Hamiltonian coefficients.

        Uses H = -(pi/4)*XX + (pi/4)*ZZ, mixing negative coefficients with
        off-diagonal terms.
        Eigenstate (|01> + |10>)/sqrt(2) has eigenvalue -pi/2.
        lambda = pi/2, cos(2*pi*phi) = -1, phi = 0.5.
        Tests sign encoding in the PREPARE oracle.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([-coeff, coeff]),
        )

        # (|01> + |10>)/sqrt(2) is eigenstate with eigenvalue -pi/2
        s = 1.0 / np.sqrt(2.0)
        state_vector = [0.0, s, s, 0.0]
        state_prep_params = {
            "rowMap": [1, 0],
            "stateVector": state_vector,
            "expansionOps": [],
            "numQubits": 2,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        iqpe = IterativePhaseEstimation(num_bits=4, shots_per_bit=3)
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = -np.pi / 2.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_lcu_three_qubit(self):
        """Verify IQPE with LCU quantum walk on a 3-qubit, 3-term Hamiltonian.

        Uses H = (pi/4)*ZII + (pi/4)*IZI + (pi/4)*IIZ (sum of single-qubit Z
        operators). With 3 terms, ceil(log2(3)) = 2 ancilla qubits are needed.
        Eigenstate |111> has eigenvalue -(pi/4)*3 = -3*pi/4.
        lambda = 3*pi/4, cos(2*pi*phi) = -1, phi = 0.5.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(
            pauli_strings=["ZII", "IZI", "IIZ"],
            coefficients=np.array([coeff, coeff, coeff]),
        )

        # |111> is eigenstate: each Z gives -1, so E = 3 * (-pi/4) = -3*pi/4
        state_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        state_prep_params = {
            "rowMap": [2, 1, 0],
            "stateVector": state_vector,
            "expansionOps": [],
            "numQubits": 3,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        iqpe = IterativePhaseEstimation(num_bits=4, shots_per_bit=3)
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = -3.0 * np.pi / 4.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )
