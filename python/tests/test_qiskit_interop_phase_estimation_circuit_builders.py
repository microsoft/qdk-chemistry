"""Tests for Qiskit-specific phase estimation circuit builders."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit, qasm3

    from qdk_chemistry.plugins.qiskit.phase_estimation_circuit_builder import (
        QiskitIterativeQpeCircuitBuilder,
        QiskitStandardQpeCircuitBuilder,
    )

pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")


@dataclass(frozen=True)
class CircuitBuilderProblem:
    """Container describing a reproducible circuit builder test scenario."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: Circuit
    evolution_time: float
    num_bits: int
    num_system_qubits: int


@pytest.fixture
def two_qubit_circuit_problem() -> CircuitBuilderProblem:
    """Return the two-qubit circuit builder test scenario with QASM state prep."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    # Prepare state |psi> ~ 0.6|00> + 0.8|11> using Ry + CNOT
    qc = QuantumCircuit(2)
    qc.ry(1.2870, 0)  # arccos(0.6)*2
    qc.cx(0, 1)
    state_prep = Circuit(qasm=qasm3.dumps(qc))

    return CircuitBuilderProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        num_system_qubits=2,
    )


@pytest.fixture
def four_qubit_circuit_problem() -> CircuitBuilderProblem:
    """Return the four-qubit circuit builder test scenario with QASM state prep."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    # Prepare a simple 4-qubit state
    qc = QuantumCircuit(4)
    qc.x(3)
    qc.h(0)
    qc.cx(0, 1)
    state_prep = Circuit(qasm=qasm3.dumps(qc))

    return CircuitBuilderProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        num_system_qubits=4,
    )


class TestQiskitStandardQpeCircuitBuilder:
    """Tests for the Qiskit standard (QFT-based) phase estimation circuit builder."""

    def test_builder_run_returns_circuits(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QiskitStandardQpeCircuitBuilder.run produces a single QPE circuit."""
        builder = QiskitStandardQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        circuits = builder.run(
            state_preparation=two_qubit_circuit_problem.state_prep,
            qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
        )

        assert isinstance(circuits, list)
        assert len(circuits) == 1
        circuit = circuits[0]
        assert isinstance(circuit, Circuit)
        # Standard QPE: num_bits ancilla + num_system qubits
        qc = circuit.get_qiskit_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == two_qubit_circuit_problem.num_bits + two_qubit_circuit_problem.num_system_qubits
        assert qc.num_clbits == two_qubit_circuit_problem.num_bits

    def test_builder_run_four_qubit(self, four_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate standard QPE with a four-qubit Hamiltonian."""
        builder = QiskitStandardQpeCircuitBuilder(num_bits=four_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=four_qubit_circuit_problem.evolution_time),
        )

        circuits = builder.run(
            state_preparation=four_qubit_circuit_problem.state_prep,
            qubit_hamiltonian=four_qubit_circuit_problem.hamiltonian,
        )

        assert len(circuits) == 1
        qc = circuits[0].get_qiskit_circuit()
        assert qc.num_qubits == four_qubit_circuit_problem.num_bits + four_qubit_circuit_problem.num_system_qubits
        assert qc.num_clbits == four_qubit_circuit_problem.num_bits

    def test_builder_raises_invalid_num_bits_error(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QiskitStandardQpeCircuitBuilder raises ValueError for invalid num_bits."""
        builder = QiskitStandardQpeCircuitBuilder(num_bits=0)  # Invalid number of bits
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        with pytest.raises(ValueError, match="num_bits must be a positive integer"):
            builder.run(
                state_preparation=two_qubit_circuit_problem.state_prep,
                qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            )


class TestQiskitIterativeQpeCircuitBuilder:
    """Tests for the Qiskit iterative phase estimation circuit builder."""

    def test_power_calculation(self) -> None:
        """Test that the power calculation is correct for different iterations."""
        hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
        state_prep = QuantumCircuit(1)
        state_prep.h(0)
        state_prep_circuit = Circuit(qasm=qasm3.dumps(state_prep))
        iqpe = QiskitIterativeQpeCircuitBuilder(num_bits=5, num_iteration=0)
        iqpe.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=float(np.pi)),
        )
        iter_0_circuits = iqpe.run(
            state_preparation=state_prep_circuit,
            qubit_hamiltonian=hamiltonian,
        )
        iter_0_circuit = iter_0_circuits[0]
        # For the first iteration, powers should be 16
        assert iter_0_circuit.qasm.count("rz(pi)") == 2 ** (iqpe._settings.get("num_bits") - 0 - 1)

    def test_builder_initialization(self) -> None:
        """Test QiskitIterativeQpeCircuitBuilder initialization with parameters."""
        num_bits = 4
        phase_correction = 0.5
        num_iteration = 1

        builder = QiskitIterativeQpeCircuitBuilder(
            num_bits=num_bits,
            phase_correction=phase_correction,
            num_iteration=num_iteration,
        )

        assert builder.settings().get("num_bits") == num_bits
        assert builder.settings().get("phase_correction") == phase_correction
        assert builder.settings().get("num_iteration") == num_iteration

    def test_builder_run_returns_circuits(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QiskitIterativeQpeCircuitBuilder.run produces iteration circuits."""
        builder = QiskitIterativeQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        circuits = builder.run(
            state_preparation=two_qubit_circuit_problem.state_prep,
            qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
        )

        assert isinstance(circuits, list)
        assert len(circuits) == two_qubit_circuit_problem.num_bits
        for circuit in circuits:
            assert isinstance(circuit, Circuit)
            # Iterative QPE: 1 ancilla + num_system qubits, 1 classical bit
            qc = circuit.get_qiskit_circuit()
            assert isinstance(qc, QuantumCircuit)
            assert qc.num_qubits == two_qubit_circuit_problem.num_system_qubits + 1
            assert qc.num_clbits == 1

    def test_builder_run_four_qubit(self, four_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate iterative QPE with a four-qubit Hamiltonian."""
        builder = QiskitIterativeQpeCircuitBuilder(num_bits=four_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=four_qubit_circuit_problem.evolution_time),
        )

        circuits = builder.run(
            state_preparation=four_qubit_circuit_problem.state_prep,
            qubit_hamiltonian=four_qubit_circuit_problem.hamiltonian,
        )

        assert len(circuits) == four_qubit_circuit_problem.num_bits
        assert len(circuits) == 6
        for circuit in circuits:
            qc = circuit.get_qiskit_circuit()
            assert qc.num_qubits == four_qubit_circuit_problem.num_system_qubits + 1
            assert qc.num_clbits == 1

    @pytest.mark.parametrize("iteration", [0, 2, 3])
    def test_run_specific_iteration(self, two_qubit_circuit_problem: CircuitBuilderProblem, iteration: int) -> None:
        """Validate that specifying num_iteration returns only that single circuit."""
        builder = QiskitIterativeQpeCircuitBuilder(
            num_bits=two_qubit_circuit_problem.num_bits,
            num_iteration=iteration,
        )
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        circuits = builder.run(
            state_preparation=two_qubit_circuit_problem.state_prep,
            qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
        )

        assert len(circuits) == 1
        qc = circuits[0].get_qiskit_circuit()
        assert qc.num_qubits == two_qubit_circuit_problem.num_system_qubits + 1
        assert qc.num_clbits == 1

    def test_builder_raises_invalid_num_bits_error(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QiskitIterativeQpeCircuitBuilder raises ValueError for invalid num_bits."""
        builder = QiskitIterativeQpeCircuitBuilder(num_bits=0)  # Invalid number of bits
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        with pytest.raises(ValueError, match="num_bits must be a positive integer"):
            builder.run(
                state_preparation=two_qubit_circuit_problem.state_prep,
                qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            )

    def test_builder_invalid_iteration_raises_error(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QiskitIterativeQpeCircuitBuilder raises ValueError for invalid num_iteration."""
        builder = QiskitIterativeQpeCircuitBuilder(
            num_bits=two_qubit_circuit_problem.num_bits,
            num_iteration=two_qubit_circuit_problem.num_bits,  # num_iteration >= num_bits
        )
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_circuit_problem.evolution_time),
        )

        with pytest.raises(ValueError, match="must be less than num_bits"):
            builder.run(
                state_preparation=two_qubit_circuit_problem.state_prep,
                qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            )

    def test_builder_raises_error_for_qsharp_op_input(self) -> None:
        """Validate that QiskitIterativeQpeCircuitBuilder raises when given a Q# op state prep."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
        state_prep_params = {
            "rowMap": [1, 0],
            "stateVector": [0.6, 0.0, 0.0, 0.8],
            "expansionOps": [],
            "numQubits": 2,
        }
        factories = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=factories, qsharp_op=qsharp_op)

        builder = QiskitIterativeQpeCircuitBuilder(num_bits=4)
        builder.settings().set(
            "circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=float(np.pi / 2)),
        )

        with pytest.raises(AttributeError):
            builder.run(
                state_preparation=state_prep,
                qubit_hamiltonian=hamiltonian,
            )
