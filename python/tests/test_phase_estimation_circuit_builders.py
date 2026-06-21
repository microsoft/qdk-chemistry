"""Tests for phase estimation circuit builders (core)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.iterative_builder import (
    QdkIterativeQpeCircuitBuilder,
    _validate_iteration_inputs,
)
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import (
    QdkStandardQpeCircuitBuilder,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


@dataclass(frozen=True)
class CircuitBuilderProblem:
    """Container describing a reproducible circuit builder test scenario."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: Circuit
    evolution_time: float
    num_bits: int
    num_qubits: int


@pytest.fixture
def two_qubit_circuit_problem() -> CircuitBuilderProblem:
    """Return the two-qubit circuit builder test scenario."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_vector = [0.6, 0.0, 0.0, 0.8]
    state_prep_params = {"rowMap": [1, 0], "stateVector": state_vector, "expansionOps": [], "numQubits": 2}
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)

    return CircuitBuilderProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        num_qubits=3,  # 2 system + 1 ancilla
    )


@pytest.fixture
def four_qubit_circuit_problem() -> CircuitBuilderProblem:
    """Return the four-qubit circuit builder test scenario."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    state_vector = np.zeros(2**4, dtype=float)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep_params = {
        "rowMap": [3, 2, 1, 0],
        "stateVector": state_vector.tolist(),
        "expansionOps": [],
        "numQubits": 4,
    }
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)

    return CircuitBuilderProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        num_qubits=5,  # 4 system + 1 ancilla
    )


class TestIterativeQpeCircuitBuilder:
    """Tests for the iterative phase estimation circuit builder."""

    def test_generates_correct_number_of_circuits(
        self,
        two_qubit_circuit_problem: CircuitBuilderProblem,
    ) -> None:
        """Test that the builder generates the correct number of iteration circuits."""
        iqpe_circuit_builder = QdkIterativeQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)

        circuits = iqpe_circuit_builder.run(
            qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            state_preparation=two_qubit_circuit_problem.state_prep,
        )

        assert len(circuits) == two_qubit_circuit_problem.num_bits

    def test_invalid_iteration_negative(self) -> None:
        """Test that _validate_iteration_inputs raises ValueError for negative iteration."""
        with pytest.raises(ValueError, match="iteration index -1 is outside the valid range"):
            _validate_iteration_inputs(iteration=-1, total_iterations=4)

        with pytest.raises(ValueError, match="iteration index 4 is outside the valid range"):
            _validate_iteration_inputs(iteration=4, total_iterations=4)

    def test_invalid_total_iterations_zero(self) -> None:
        """Test that _validate_iteration_inputs raises ValueError for total_iterations <= 0."""
        with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
            _validate_iteration_inputs(iteration=0, total_iterations=0)

        with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
            _validate_iteration_inputs(iteration=0, total_iterations=-1)

    def test_run_returns_circuits(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that IterativeQpeCircuitBuilder.run produces one circuit per phase bit."""
        builder = QdkIterativeQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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
            qsc_json = json.loads(circuit.get_qsharp_circuit().json())
            assert len(qsc_json["qubits"]) == two_qubit_circuit_problem.num_qubits
            # Ancilla qubit (q_0) has a measurement result
            assert qsc_json["qubits"][0]["numResults"] == 1

    def test_run_returns_circuits_four_qubit(self, four_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate circuit builder with a four-qubit Hamiltonian produces more circuits."""
        builder = QdkIterativeQpeCircuitBuilder(num_bits=four_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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

        # Four-qubit problem has num_bits=6 vs two-qubit's num_bits=4
        assert len(circuits) == four_qubit_circuit_problem.num_bits
        assert len(circuits) == 6
        for circuit in circuits:
            qsc_json = json.loads(circuit.get_qsharp_circuit().json())
            assert len(qsc_json["qubits"]) == four_qubit_circuit_problem.num_qubits
            assert qsc_json["qubits"][0]["numResults"] == 1

    @pytest.mark.parametrize("iteration", [0, 1, 3])
    def test_run_specific_iteration(self, two_qubit_circuit_problem: CircuitBuilderProblem, iteration: int) -> None:
        """Validate that specifying num_iteration returns only that single circuit."""
        builder = QdkIterativeQpeCircuitBuilder(
            num_bits=two_qubit_circuit_problem.num_bits,
            num_iteration=iteration,
        )
        builder.settings().set(
            "controlled_circuit_mapper",
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
        qsc_json = json.loads(circuits[0].get_qsharp_circuit().json())
        assert len(qsc_json["qubits"]) == two_qubit_circuit_problem.num_qubits
        assert qsc_json["qubits"][0]["numResults"] == 1

    def test_run_specific_iteration_out_of_range(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that num_iteration >= num_bits raises ValueError."""
        builder = QdkIterativeQpeCircuitBuilder(
            num_bits=two_qubit_circuit_problem.num_bits,
            num_iteration=two_qubit_circuit_problem.num_bits,  # equal to num_bits, out of range
        )

        with pytest.raises(ValueError, match="num_iteration"):
            builder.run(
                state_preparation=two_qubit_circuit_problem.state_prep,
                qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            )

    def test_raises_error_for_qasm_only_state_prep(self) -> None:
        """Validate that passing a QASM-only state prep (no Q# op) raises RuntimeError."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
        qasm_str = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nry(1.2870) q[0];\ncx q[0], q[1];\n'
        state_prep = Circuit(qasm=qasm_str)

        builder = QdkIterativeQpeCircuitBuilder(num_bits=4)
        builder.settings().set(
            "controlled_circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=float(np.pi / 2)),
        )

        with pytest.raises(RuntimeError, match="Q# operations are not available"):
            builder.run(
                state_preparation=state_prep,
                qubit_hamiltonian=hamiltonian,
            )


class TestStandardQpeCircuitBuilder:
    """Tests for the QDK standard (QFT-based) phase estimation circuit builder."""

    def test_run_returns_single_circuit(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that QdkStandardQpeCircuitBuilder.run produces a single QPE circuit."""
        builder = QdkStandardQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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
        assert isinstance(circuits[0], Circuit)

    def test_circuit_has_correct_qubit_count(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that the QPE circuit has ancilla + system qubits."""
        builder = QdkStandardQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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

        qsc_json = json.loads(circuits[0].get_qsharp_circuit().json())
        num_system_qubits = two_qubit_circuit_problem.hamiltonian.num_qubits
        expected_total = two_qubit_circuit_problem.num_bits + num_system_qubits
        assert len(qsc_json["qubits"]) == expected_total

    def test_circuit_has_measurements_on_ancillas(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that ancilla qubits have measurement results."""
        builder = QdkStandardQpeCircuitBuilder(num_bits=two_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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

        qsc_json = json.loads(circuits[0].get_qsharp_circuit().json())
        # Each ancilla qubit should have a measurement
        for i in range(two_qubit_circuit_problem.num_bits):
            assert qsc_json["qubits"][i]["numResults"] == 1

    def test_run_four_qubit(self, four_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate circuit builder works with a four-qubit Hamiltonian."""
        builder = QdkStandardQpeCircuitBuilder(num_bits=four_qubit_circuit_problem.num_bits)
        builder.settings().set(
            "controlled_circuit_mapper",
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
        qsc_json = json.loads(circuits[0].get_qsharp_circuit().json())
        num_system_qubits = four_qubit_circuit_problem.hamiltonian.num_qubits
        expected_total = four_qubit_circuit_problem.num_bits + num_system_qubits
        assert len(qsc_json["qubits"]) == expected_total

    def test_invalid_num_bits_raises_error(self, two_qubit_circuit_problem: CircuitBuilderProblem) -> None:
        """Validate that num_bits <= 0 raises ValueError."""
        builder = QdkStandardQpeCircuitBuilder(num_bits=0)

        with pytest.raises(ValueError, match="num_bits must be a positive integer"):
            builder.run(
                state_preparation=two_qubit_circuit_problem.state_prep,
                qubit_hamiltonian=two_qubit_circuit_problem.hamiltonian,
            )

    def test_raises_error_for_qasm_only_state_prep(self) -> None:
        """Validate that passing a QASM-only state prep (no Q# op) raises RuntimeError."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
        qasm_str = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nry(1.2870) q[0];\ncx q[0], q[1];\n'
        state_prep = Circuit(qasm=qasm_str)

        builder = QdkStandardQpeCircuitBuilder(num_bits=4)
        builder.settings().set(
            "controlled_circuit_mapper",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        builder.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=float(np.pi / 2)),
        )

        with pytest.raises(RuntimeError, match="Q# operations are not available"):
            builder.run(
                state_preparation=state_prep,
                qubit_hamiltonian=hamiltonian,
            )

    def test_name(self) -> None:
        """Validate that the builder returns the correct name."""
        builder = QdkStandardQpeCircuitBuilder()
        assert builder.name() == "qdk_standard"
        assert builder.type_name() == "qpe_circuit_builder"
