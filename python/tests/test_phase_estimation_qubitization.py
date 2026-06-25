"""Integration tests for iterative QPE with qubitization (LCU quantum walk)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
)

_builder_params = [
    pytest.param("qdk_iterative", id="qdk_iterative"),
    pytest.param(
        "qiskit_iterative",
        id="qiskit_iterative",
        marks=[
            pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
            pytest.mark.xfail(
                reason="QIR-to-Qiskit converter does not support Adaptive_RIFLA profile",
                raises=Exception,
            ),
        ],
    ),
]


def _qubitization_circuit_builder_ref(num_bits: int = 4, builder: str = "qdk_iterative") -> AlgorithmRef:
    """Return an AlgorithmRef for the iterative QPE circuit builder with qubitization."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        builder,
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
    )


@pytest.fixture
def h2_hamiltonian() -> QubitHamiltonian:
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
    return QubitHamiltonian(
        pauli_strings=h2_pauli_strings,
        coefficients=h2_coefficients,
    )


class TestIQPEWithQubitization:
    """Integration tests for iterative QPE with qubitization (LCU quantum walk)."""

    def test_iterative_qpe_with_qubitization_two_qubit(self):
        """Verify IQPE with qubitization recovers the expected energy for a 2-qubit Hamiltonian.

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

        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set("qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=4))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
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

    @pytest.mark.parametrize("builder_name", _builder_params)
    def test_iterative_qpe_with_qubitization_h2(self, builder_name, h2_hamiltonian):
        """Verify QPE with qubitization recovers H2 ground-state energy.

        Uses the full H2/STO-3G qubit Hamiltonian (15 Pauli terms, 4 qubits)
        with the exact ground state as the initial state. The quantum walk
        operator W = R*B[H] is used with IQPE to extract the eigenphase, and
        the energy is recovered via E = lambda * cos(2*pi*phi).

        Reference ground-state energy: -2.2472 Ha (from exact diagonalization).
        With 4 phase bits, the discretization error is ~0.02 Ha.
        """
        # Exact ground state from qubit Hamiltonian solver (dense diagonalization)
        solver = create("qubit_hamiltonian_solver", "qdk_dense_matrix_solver")
        reference_energy, ground_state_vector = solver.run(h2_hamiltonian)
        ground_state_vector = ground_state_vector.real.tolist()

        num_qubits = h2_hamiltonian.num_qubits
        state_prep_params = {
            "rowMap": [3, 2, 1, 0],
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

        num_bits = 4
        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set(
            "qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=num_bits, builder=builder_name)
        )
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(
            qubit_hamiltonian=h2_hamiltonian,
            state_preparation=state_prep,
        )

        # With 4 phase bits the phase 0.376 rounds to 6/16 = 0.375,
        # giving E = lambda * cos(2*pi * 0.375) ~ -2.231.
        # Allow 0.02 Ha tolerance for discretization error.
        lambda_norm = np.sum(np.abs(h2_hamiltonian.coefficients))
        phi_exact = np.arccos(reference_energy / lambda_norm) / (2 * np.pi)
        # Discretize to num_bits
        num_levels = 2**num_bits  # e.g. 16 for 4 bits
        phi_disc = round(phi_exact * num_levels) / num_levels
        # Conjugate branch (QPE may measure either)
        phi_disc_alt = 1.0 - phi_disc
        # Discretized energy (same for both branches)
        energy_disc = lambda_norm * np.cos(2 * np.pi * phi_disc)
        assert np.isclose(
            result.phase_fraction,
            phi_disc,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ) or np.isclose(
            result.phase_fraction,
            phi_disc_alt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            result.raw_energy,
            energy_disc,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(result.raw_energy, reference_energy, atol=0.02)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.xfail(
        reason="QIR-to-Qiskit converter does not support Adaptive_RIFLA profile",
        raises=Exception,
    )
    def test_standard_qpe_with_qubitization_h2(self, h2_hamiltonian):
        """Verify standard QPE with qubitization recovers H2 ground-state energy."""
        # Exact ground state from qubit Hamiltonian solver (dense diagonalization)
        solver = create("qubit_hamiltonian_solver", "qdk_dense_matrix_solver")
        reference_energy, ground_state_vector = solver.run(h2_hamiltonian)
        ground_state_vector = ground_state_vector.real.tolist()

        num_qubits = h2_hamiltonian.num_qubits
        state_prep_params = {
            "rowMap": [3, 2, 1, 0],
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

        num_bits = 4
        qpe = StandardPhaseEstimation(shots=3)
        qpe.settings().set(
            "qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=num_bits, builder="qiskit_standard")
        )
        qpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qiskit_aer_simulator"),
        )

        result = qpe.run(
            qubit_hamiltonian=h2_hamiltonian,
            state_preparation=state_prep,
        )

        # With 4 phase bits the phase 0.376 rounds to 6/16 = 0.375,
        # giving E = lambda * cos(2*pi * 0.375) ~ -2.231.
        # Allow 0.02 Ha tolerance for discretization error.
        lambda_norm = np.sum(np.abs(h2_hamiltonian.coefficients))
        phi_exact = np.arccos(reference_energy / lambda_norm) / (2 * np.pi)
        # Discretize to num_bits
        num_levels = 2**num_bits  # e.g. 16 for 4 bits
        phi_disc = round(phi_exact * num_levels) / num_levels
        # Conjugate branch (QPE may measure either)
        phi_disc_alt = 1.0 - phi_disc
        # Discretized energy (same for both branches)
        energy_disc = lambda_norm * np.cos(2 * np.pi * phi_disc)
        assert np.isclose(
            result.phase_fraction,
            phi_disc,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ) or np.isclose(
            result.phase_fraction,
            phi_disc_alt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            result.raw_energy,
            energy_disc,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(result.raw_energy, reference_energy, atol=0.02)

    def test_iterative_qpe_with_qubitization_off_diagonal(self):
        """Verify IQPE with qubitization handles off-diagonal Pauli terms.

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

        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set("qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=4))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = np.pi / 2.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_qubitization_negative_coefficient(self):
        """Verify IQPE with qubitization handles negative Hamiltonian coefficients.

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

        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set("qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=4))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = -np.pi / 2.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_qubitization_three_qubit(self):
        """Verify IQPE with qubitization on a 3-qubit, 3-term Hamiltonian.

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

        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set("qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=4))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = -3.0 * np.pi / 4.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )

    def test_iterative_qpe_with_qubitization_asymmetric_pauli(self):
        """Verify IQPE with qubitization on an asymmetric Hamiltonian.

        Uses H = (pi/4)*XI + (pi/4)*IZ, where the Pauli terms are NOT symmetric
        under qubit swap (XI != IX). This tests that the SELECT oracle applies
        each Pauli to the correct qubit.

        Little-endian label convention (rightmost char = qubit 0):
        "XI" applies X to qubit 1, I to qubit 0.
        "IZ" applies I to qubit 1, Z to qubit 0.
        Eigenstate |+>_1 |0>_0 has eigenvalue (pi/4)(+1) + (pi/4)(+1) = pi/2.
        lambda = pi/2, cos(2*pi*phi) = 1, phi = 0.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IZ"],
            coefficients=np.array([coeff, coeff]),
        )

        # |+>_q1 x |0>_q0:  X on q1 gives +1, Z on q0 gives +1 -> E = pi/2.
        # In Q# big-endian register: targets[0] = q1 (MSB), targets[1] = q0 (LSB).
        #   index 0 = |q1=0,q0=0>, index 1 = |q1=0,q0=1>,
        #   index 2 = |q1=1,q0=0>, index 3 = |q1=1,q0=1>
        # |+>_q1 |0>_q0 = (|q1=0,q0=0> + |q1=1,q0=0>)/sqrt(2) = (idx 0 + idx 2)/sqrt(2)
        s = 1.0 / np.sqrt(2.0)
        state_vector = [s, 0.0, s, 0.0]
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

        iqpe = IterativePhaseEstimation(shots_per_bit=3)
        iqpe.settings().set("qpe_circuit_builder", _qubitization_circuit_builder_ref(num_bits=4))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_prep)

        expected_energy = np.pi / 2.0
        assert np.isclose(
            result.raw_energy,
            expected_energy,
            rtol=float_comparison_relative_tolerance,
            atol=qpe_energy_tolerance,
        )
