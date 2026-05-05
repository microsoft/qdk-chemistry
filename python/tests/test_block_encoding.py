"""Tests for QPE with block encoding unitary builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.controlled_circuit_mapper import BlockEncodingMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import BlockEncodingBuilder
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
)


class TestLCUBuilder:
    """Tests for the block encoding builder algorithm."""

    def test_basic_construction(self):
        """Test that the block encoding builder produces a BlockEncodingContainer from a simple Hamiltonian."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = BlockEncodingBuilder()
        result = builder.run(hamiltonian)

        container = result.get_container()
        assert isinstance(container, BlockEncodingContainer)
        assert container.type == "block_encoding"
        assert container.num_system_qubits == 2
        assert container.num_qubits == 3  # 2 system + 1 select
        assert len(container.select.controlled_operations) == 2
        assert container.num_select_qubits == 1
        assert all(s in (1, -1) for s in container.select.signs)

    def test_num_select_qubits(self):
        """Test correct computation of select qubit count."""
        # 2 terms -> 1 select qubit
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = BlockEncodingBuilder()
        result = builder.run(hamiltonian)
        assert result.get_container().num_select_qubits == 1

        # 3 terms -> 2 select qubits (ceil(log2(3)) = 2)
        hamiltonian3 = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, 0.5, 0.1]),
        )
        builder3 = BlockEncodingBuilder()
        result3 = builder3.run(hamiltonian3)
        assert result3.get_container().num_select_qubits == 2

    def test_lcu_builder_registered_in_registry(self):
        """Verify block encoding builder and mapper are accessible via the registry."""
        # Block encoding builder should be registered
        builder = registry.create("hamiltonian_unitary_builder", "block_encoding")
        assert isinstance(builder, BlockEncodingBuilder)
        assert builder.name() == "block_encoding"

        # Block encoding circuit mapper should be registered
        mapper = registry.create("controlled_circuit_mapper", "block_encoding")
        assert isinstance(mapper, BlockEncodingMapper)
        assert mapper.name() == "block_encoding"


class TestBlockEncodingMapper:
    """Tests for the block encoding circuit mapper algorithm."""

    def test_basic_mapping(self):
        """Test basic mapping of ControlledUnitary with BlockEncodingContainer to Circuit."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = BlockEncodingBuilder()
        unitary_rep = builder.run(hamiltonian)

        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        mapper = BlockEncodingMapper()
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None


class TestBlockEncodingContainer:
    """Test for the BlockEncodingContainer data class."""

    def test_lcu_container_serialization_roundtrip(self):
        """Test JSON serialization round-trip for BlockEncodingContainer."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = BlockEncodingBuilder()
        result = builder.run(hamiltonian)
        container = result.get_container()

        # Round-trip through JSON
        json_data = container.to_json()
        restored = BlockEncodingContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.num_select_qubits == container.num_select_qubits
        assert np.array_equal(restored.select.signs, container.select.signs)
        assert [op.operation for op in restored.select.controlled_operations] == [
            op.operation for op in container.select.controlled_operations
        ]

    def test_unitary_representation_serialization_roundtrip(self):
        """Test JSON serialization round-trip via UnitaryRepresentation dispatch."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = BlockEncodingBuilder()
        unitary_rep = builder.run(hamiltonian)

        # Serialize through UnitaryRepresentation
        json_data = unitary_rep.to_json()
        assert json_data["container_type"] == "block_encoding"

        # Deserialize through UnitaryRepresentation.from_json
        restored_rep = UnitaryRepresentation.from_json(json_data)
        restored_container = restored_rep.get_container()

        assert isinstance(restored_container, BlockEncodingContainer)
        assert restored_container.type == "block_encoding"
        assert restored_container.num_qubits == unitary_rep.get_container().num_qubits
        assert restored_container.num_select_qubits == unitary_rep.get_container().num_select_qubits


class TestQPEWithLCU:
    """Integration tests for QPE using LCU unitary builder."""

    def test_iterative_qpe_with_lcu_two_qubit(self):
        """Verify QPE with LCU recovers the expected energy for a 2-qubit Hamiltonian.

        Uses H = (pi/4)*ZZ which has eigenvalue pi/4 on |00>.
        With evolution_time=1 (block encoding), phase_fraction = (pi/4)/(2*pi) = 1/8,
        exactly representable with 4 bits.
        """
        coeff = np.pi / 4.0
        hamiltonian = QubitHamiltonian(pauli_strings=["ZZ"], coefficients=np.array([coeff]))

        # State |00> is an eigenstate of ZZ with eigenvalue +1,
        # so it's an eigenstate of H with eigenvalue pi/4.
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

        # Configure IQPE with LCU (evolution_time=1.0 implicitly)
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
            AlgorithmRef("controlled_circuit_mapper", "block_encoding"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "block_encoding"),
        )

        result = iqpe.run(
            qubit_hamiltonian=hamiltonian,
            state_preparation=state_prep,
        )

        expected_energy = coeff  # pi/4
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
            AlgorithmRef("controlled_circuit_mapper", "block_encoding"),
        )
        iqpe.settings().set(
            "unitary_builder",
            AlgorithmRef("hamiltonian_unitary_builder", "block_encoding", quantum_walk=True),
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
