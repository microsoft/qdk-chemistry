"""Tests for the PauliSequenceMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import numpy as np
import pytest
import qsharp
import scipy

from qdk_chemistry.algorithms.controlled_circuit_mapper.pauli_sequence_mapper import (
    PauliSequenceMapper,
)
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.controlled_unitary import (
    ControlledUnitary,
)
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator


@pytest.fixture
def simple_ppf_container():
    """Create a simple PauliProductFormulaContainer for testing."""
    terms = [
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.25),
    ]

    return PauliProductFormulaContainer(
        step_terms=terms,
        step_reps=1,
        num_qubits=2,
    )


@pytest.fixture
def controlled_unitary(simple_ppf_container):
    """Create a ControlledUnitary for testing."""
    teu = UnitaryRepresentation(container=simple_ppf_container)
    return ControlledUnitary(
        unitary=teu,
        control_indices=[2],
    )


class TestPauliSequenceMapper:
    """Tests for the PauliSequenceMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = PauliSequenceMapper()
        assert mapper.name() == "pauli_sequence"

    def test_basic_mapping(self, controlled_unitary):
        """Test basic mapping of ControlledUnitary to Circuit."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), qsharp._native.Circuit)

    def test_default_target_indices(self, controlled_unitary):
        """Test that default target indices are used when none are provided."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(controlled_unitary)
        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])  # 2 system qubits + 1 control qubit
        assert num_qubits == 3

        def _find_control_qubits(node):
            """Recursively collect control-qubit indices for X gates."""
            control_qubits = []
            if isinstance(node, dict):
                if node.get("gate") == "X" and "controls" in node:
                    for ctrl in node["controls"]:
                        qubit_idx = ctrl.get("qubit")
                        if qubit_idx is not None:
                            control_qubits.append(qubit_idx)
                for value in node.values():
                    control_qubits.extend(_find_control_qubits(value))
            elif isinstance(node, list):
                for item in node:
                    control_qubits.extend(_find_control_qubits(item))
            return control_qubits

        # Check that there is at least one X gate controlled by qubit 2
        control_qubits = _find_control_qubits(qsc_json.get("componentGrid", []))

        assert set(control_qubits) == {2}

    def test_target_indices_validation(self, simple_ppf_container):
        """Test that invalid target indices raise ValueError."""
        teu = UnitaryRepresentation(container=simple_ppf_container)

        # Overlapping target and control indices
        with pytest.raises(ValueError, match="must not overlap"):
            ControlledUnitary(unitary=teu, control_indices=[2], target_indices=[1, 2])

        # Non-overlapping indices in a larger circuit are valid
        cu = ControlledUnitary(unitary=teu, control_indices=[2], target_indices=[0, 1])
        assert cu.target_indices == [0, 1]

        cu = ControlledUnitary(unitary=teu, control_indices=[2], target_indices=[3, 4])
        assert cu.target_indices == [3, 4]

    def test_invalid_container_type_raises(self):
        """Test that an invalid container type raises a ValueError."""

        # Create a new UnitaryRepresentation with invalid container type
        class MockContainer:
            """Mock container class."""

            @property
            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = UnitaryRepresentation(container=MockContainer())
        invalid_controlled = ControlledUnitary(
            unitary=invalid_teu,
            control_indices=[2],
        )

        mapper = PauliSequenceMapper()

        with pytest.raises(ValueError, match="not supported"):
            mapper.run(invalid_controlled)

    def test_rotation_parameters(self, controlled_unitary):
        """Test that rotation parameters are correctly set in the mapped circuit."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(controlled_unitary)

        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])  # 2 system qubits + 1 control qubit
        assert num_qubits == 3
        operations = qsc_json["componentGrid"][0]["components"][0]["children"][0]["components"][0]["children"]
        # Check that "X0" on qubit 0 and "Z1" on qubit 1 are present in the circuit with correct parameters
        for op in operations:
            for component in op["components"]:
                if component["gate"] == "Rz":
                    params = float(component["args"][0])
                    target_qubit = component["targets"][0]["qubit"]
                    if target_qubit == 0:
                        assert np.isclose(
                            abs(params),
                            0.5,
                            rtol=float_comparison_relative_tolerance,
                            atol=float_comparison_absolute_tolerance,
                        )  # X on qubit 0
                    elif target_qubit == 1:
                        assert np.isclose(
                            abs(params),
                            0.25,
                            rtol=float_comparison_relative_tolerance,
                            atol=float_comparison_absolute_tolerance,
                        )  # Z on qubit 1

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_controlled_u_circuit_matrix(self, controlled_unitary):
        """Test that the constructed controlled-U circuit has the expected matrix."""
        mapper = PauliSequenceMapper()
        circuit = mapper.run(controlled_unitary)

        # Extract angles from the container
        container = controlled_unitary.unitary.get_container()
        angle_x = container.step_terms[0].angle
        angle_z = container.step_terms[1].angle

        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        x_0 = np.kron(identity, pauli_x)
        z_1 = np.kron(pauli_z, identity)
        u_1 = scipy.linalg.expm(-1j * angle_x * x_0)
        u_2 = scipy.linalg.expm(-1j * angle_z * z_1)
        u = u_2 @ u_1

        # CU = (|0><0| ⊗ I₄) + (|1><1| ⊗ U)
        p_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        i_4 = np.eye(4, dtype=complex)
        expected_matrix = np.kron(p_0, i_4) + np.kron(p_1, u)

        qc = circuit.get_qiskit_circuit()
        actual_matrix = Operator(qc).data

        assert np.allclose(
            actual_matrix,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
