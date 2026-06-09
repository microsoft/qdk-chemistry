"""Tests for the LCU block encoding builder and LCU container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer, LCUContainer

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestLCUBuilder:
    """Tests for the LCU block encoding builder algorithm."""

    def test_basic_construction(self):
        """Test that the block encoding builder produces an LCUContainer from a simple Hamiltonian."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        result = builder.run(hamiltonian)

        container = result.get_container()
        assert isinstance(container, LCUContainer)
        assert isinstance(container, BlockEncodingContainer)
        assert container.type == "lcu"
        assert container.num_qubits == 3  # 2 system + 1 select
        assert len(container.select.controlled_operations) == 2
        assert all(s in (1, -1) for s in container.select.phases)

    def test_num_select_qubits(self):
        """Test correct computation of select qubit count."""
        # 2 terms -> 1 select qubit
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        result = builder.run(hamiltonian)
        assert result.get_container().prepare.num_prepare_qubits == 1

        # 3 terms -> 2 select qubits (ceil(log2(3)) = 2)
        hamiltonian3 = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, 0.5, 0.1]),
        )
        builder3 = LCUBuilder()
        result3 = builder3.run(hamiltonian3)
        assert result3.get_container().prepare.num_prepare_qubits == 2

    def test_lcu_builder_registered_in_registry(self):
        """Verify block encoding builder is accessible via the registry."""
        builder = registry.create("hamiltonian_unitary_builder", "lcu")
        assert isinstance(builder, LCUBuilder)
        assert builder.name() == "lcu"

    def test_prepare_statevector_encodes_normalized_coefficients(self):
        """Verify PREPARE statevector is sqrt(|alpha_j|/lambda) for each term.

        For H = 0.25*XX + 0.5*ZZ, lambda = 0.75,
        statevector = [sqrt(0.25/0.75), sqrt(0.5/0.75)] padded to length 2.
        """
        coefficients = np.array([0.25, 0.5])
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=coefficients)
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        lam = np.sum(np.abs(coefficients))
        expected_sv = np.sqrt(np.abs(coefficients) / lam)
        # Pad to 2^num_prepare_qubits
        padded = np.zeros(2**container.prepare.num_prepare_qubits)
        padded[: len(expected_sv)] = expected_sv

        assert np.allclose(
            container.prepare.statevector,
            padded,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_prepare_statevector_three_terms(self):
        """Verify PREPARE statevector for 3 terms."""
        coefficients = np.array([0.25, -0.5, 0.3])
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ", "XZ"], coefficients=coefficients)
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        lam = np.sum(np.abs(coefficients))
        expected_sv = np.sqrt(np.abs(coefficients) / lam)

        assert np.allclose(
            container.prepare.statevector,
            expected_sv,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_select_operations_match_pauli_strings(self):
        """Verify SELECT controlled operations match the input Pauli strings."""
        pauli_strings = ["XI", "IZ", "XZ"]
        hamiltonian = QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array([0.3, 0.5, 0.2]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        for i, op in enumerate(container.select.controlled_operations):
            assert op.operation == pauli_strings[i]
            assert op.ctrl_state == i

    def test_select_phases_match_coefficient_signs(self):
        """Verify SELECT phases encode coefficient signs correctly."""
        coefficients = np.array([0.3, -0.5, 0.2, -0.1])
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IZ", "XZ", "ZX"],
            coefficients=coefficients,
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        expected_phases = np.sign(coefficients).astype(int)
        assert np.array_equal(container.select.phases, expected_phases)

    def test_quantum_walk_flag(self):
        """Verify quantum_walk setting produces container with quantum_walk=True."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()
        assert container.quantum_walk is True

        builder_no_walk = LCUBuilder()
        container_no_walk = builder_no_walk.run(hamiltonian).get_container()
        assert container_no_walk.quantum_walk is False

    def test_rejects_zero_l1_norm(self):
        """Verify LCUBuilder raises ValueError when all coefficients are zero."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.0, 0.0]),
        )
        builder = LCUBuilder()
        with pytest.raises(ValueError, match="L1 norm is too small"):
            builder.run(hamiltonian)


class TestLCUContainer:
    """Tests for the LCUContainer data class."""

    def test_lcu_container_serialization_roundtrip(self):
        """Test JSON serialization round-trip for LCUContainer."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder()
        result = builder.run(hamiltonian)
        container = result.get_container()

        # Round-trip through JSON
        json_data = container.to_json()
        restored = LCUContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.prepare.num_prepare_qubits == container.prepare.num_prepare_qubits
        assert np.array_equal(restored.select.phases, container.select.phases)
        assert [op.operation for op in restored.select.controlled_operations] == [
            op.operation for op in container.select.controlled_operations
        ]

    def test_unitary_representation_serialization_roundtrip(self):
        """Test JSON serialization round-trip via UnitaryRepresentation dispatch."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder()
        unitary_rep = builder.run(hamiltonian)

        # Serialize through UnitaryRepresentation
        json_data = unitary_rep.to_json()
        assert json_data["container_type"] == "lcu"

        # Deserialize through UnitaryRepresentation.from_json
        restored_rep = UnitaryRepresentation.from_json(json_data)
        restored_container = restored_rep.get_container()

        assert isinstance(restored_container, LCUContainer)
        assert isinstance(restored_container, BlockEncodingContainer)
        assert restored_container.type == "lcu"
        assert restored_container.num_qubits == unitary_rep.get_container().num_qubits
        assert restored_container.prepare.num_prepare_qubits == unitary_rep.get_container().prepare.num_prepare_qubits

    def test_serialization_preserves_statevector(self):
        """Verify that serialization preserves the PREPARE statevector exactly."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        json_data = container.to_json()
        restored = LCUContainer.from_json(json_data)

        assert np.allclose(
            restored.prepare.statevector,
            container.prepare.statevector,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_hdf5_serialization_roundtrip(self):
        """Test HDF5 serialization round-trip for LCUContainer."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            filename = tmp.name

        try:
            with h5py.File(filename, "w") as f:
                container.to_hdf5(f)

            with h5py.File(filename, "r") as f:
                restored = LCUContainer.from_hdf5(f)

            assert restored.type == container.type
            assert restored.power == container.power
            assert restored.quantum_walk == container.quantum_walk
            assert restored.prepare.num_prepare_qubits == container.prepare.num_prepare_qubits
            assert np.allclose(restored.prepare.statevector, container.prepare.statevector)
            assert np.array_equal(restored.select.phases, container.select.phases)
            assert len(restored.select.controlled_operations) == len(container.select.controlled_operations)
            for r_op, c_op in zip(
                restored.select.controlled_operations, container.select.controlled_operations, strict=False
            ):
                assert r_op.ctrl_state == c_op.ctrl_state
                assert r_op.operation == c_op.operation
        finally:
            Path(filename).unlink()

    def test_get_summary(self):
        """Test that get_summary returns a descriptive string."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        summary = container.get_summary()
        assert "LCU Container" in summary
        assert "Power" in summary
        assert "Prepare" in summary
        assert "Select" in summary
        assert "Quantum Walk" in summary
