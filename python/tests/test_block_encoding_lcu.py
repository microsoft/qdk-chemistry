"""Tests for the LCU block encoding builder and LCU container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import AlgorithmRef, QubitOperator
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer, LCUContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _reverse_bits(value: int, num_bits: int) -> int:
    """Reverse the bit order of *value* within a *num_bits* field."""
    reversed_value = 0
    for bit in range(num_bits):
        reversed_value |= ((value >> bit) & 1) << (num_bits - 1 - bit)
    return reversed_value


def _block_encoding_action(circuit, num_system_qubits: int, system_amplitudes: np.ndarray) -> np.ndarray:
    r"""Apply ``circuit`` to :math:`|\psi\rangle|0\rangle_\mathrm{anc}` and project back on the ancillas.

    Returns :math:`(\langle 0|_\mathrm{anc} \otimes I) B[H] (|0\rangle_\mathrm{anc} \otimes I) |\psi\rangle`,
    which the block encoding identity makes :math:`H |\psi\rangle / \lambda`.

    ``PSPMapper`` lays the register out as ``[system | ancilla]`` while
    :func:`dump_operation_on_state` numbers basis states big-endian, so the system register takes
    the high bits and its own index runs the other way -- hence the bit reversal on both ends.
    """
    stride = 2 ** (circuit.num_qubits - num_system_qubits)
    dimension = 2**num_system_qubits

    initial_state = [0.0] * ((dimension - 1) * stride + 1)
    for index, amplitude in enumerate(system_amplitudes):
        initial_state[_reverse_bits(index, num_system_qubits) * stride] = amplitude

    statevector = dump_operation_on_state(
        circuit._qsharp_op, circuit.num_qubits, initial_state, context=get_qsharp_context()
    )
    return np.array([statevector[_reverse_bits(index, num_system_qubits) * stride] for index in range(dimension)])


class TestLCUBuilder:
    """Tests for the LCU block encoding builder algorithm."""

    def test_basic_construction(self):
        """Test that the block encoding builder produces an LCUContainer from a simple Hamiltonian."""
        hamiltonian = QubitOperator(
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
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        result = builder.run(hamiltonian)
        assert result.get_container().num_prepare_ancillas == 1

        # 3 terms -> 2 select qubits (ceil(log2(3)) = 2)
        hamiltonian3 = QubitOperator(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, 0.5, 0.1]),
        )
        builder3 = LCUBuilder()
        result3 = builder3.run(hamiltonian3)
        assert result3.get_container().num_prepare_ancillas == 2

    def test_lcu_builder_registered_in_registry(self):
        """Verify block encoding builder is accessible via the registry."""
        builder = registry.create("hamiltonian_unitary_builder", "lcu")
        assert isinstance(builder, LCUBuilder)
        assert builder.name() == "lcu"

    def test_prepare_statevector_encodes_normalized_coefficients(self):
        """Verify PREPARE wavefunction amplitudes are sqrt(|alpha_j|/lambda) for each term.

        For H = 0.25*XX + 0.5*ZZ, lambda = 0.75,
        amplitudes = [sqrt(0.25/0.75), sqrt(0.5/0.75)].
        """
        coefficients = np.array([0.25, 0.5])
        hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ"], coefficients=coefficients)
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        lam = np.sum(np.abs(coefficients))
        expected_amplitudes = np.sqrt(np.abs(coefficients) / lam)

        actual_amplitudes = np.array(container.prepare.get_coefficients())
        assert np.allclose(
            actual_amplitudes,
            expected_amplitudes,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_prepare_statevector_three_terms(self):
        """Verify PREPARE wavefunction amplitudes for 3 terms."""
        coefficients = np.array([0.25, -0.5, 0.3])
        hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ", "XZ"], coefficients=coefficients)
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        lam = np.sum(np.abs(coefficients))
        expected_amplitudes = np.sqrt(np.abs(coefficients) / lam)

        actual_amplitudes = np.array(container.prepare.get_coefficients())
        assert np.allclose(
            actual_amplitudes,
            expected_amplitudes,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_select_operations_match_pauli_strings(self):
        """Verify SELECT controlled operations match the input Pauli strings."""
        pauli_strings = ["XI", "IZ", "XZ"]
        hamiltonian = QubitOperator(
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
        hamiltonian = QubitOperator(
            pauli_strings=["XI", "IZ", "XZ", "ZX"],
            coefficients=coefficients,
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        expected_phases = np.sign(coefficients).astype(int)
        assert np.array_equal(container.select.phases, expected_phases)

    def test_quantum_walk_flag(self):
        """Verify quantum_walk setting produces LCUWalkContainer."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()
        assert isinstance(container, LCUWalkContainer)

        builder_no_walk = LCUBuilder()
        container_no_walk = builder_no_walk.run(hamiltonian).get_container()
        assert isinstance(container_no_walk, LCUContainer)
        assert not isinstance(container_no_walk, LCUWalkContainer)

    def test_rejects_zero_l1_norm(self):
        """Verify LCUBuilder raises ValueError when all coefficients are zero."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.0, 0.0]),
        )
        builder = LCUBuilder()
        with pytest.raises(ValueError, match="L1 norm is too small"):
            builder.run(hamiltonian)

    def test_prepare_select_prepare_with_alias_sampling(self):
        """Verify alias sampling supplies its entangled scratch register to PREPARE."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, 0.5, 0.1]),
        )
        unitary = LCUBuilder().run(hamiltonian)
        mapper = registry.create(
            "circuit_mapper",
            "prepare_select_prepare",
            prepare=AlgorithmRef("state_prep", "alias_sampling", bits_precision=4),
        )

        circuit = mapper.run(unitary)

        assert circuit.num_qubits == 15
        assert circuit._qsharp_factory.parameter["numSelectQubits"] == 2
        assert circuit._qsharp_factory.parameter["numBlockAncillaQubits"] == 13

    def test_alias_sampling_block_encodes_the_hamiltonian(self):
        r"""Verify :math:`\langle 0|_\mathrm{anc} B[H] |0\rangle_\mathrm{anc} = H/\lambda`."""
        coefficients = np.array([0.25, 0.5, 0.1])
        bits_precision = 4
        hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ", "XZ"], coefficients=coefficients)
        circuit = registry.create(
            "circuit_mapper",
            "prepare_select_prepare",
            prepare=AlgorithmRef("state_prep", "alias_sampling", bits_precision=bits_precision),
        ).run(LCUBuilder().run(hamiltonian))

        num_system_qubits = hamiltonian.num_qubits
        dimension = 2**num_system_qubits
        expected_block = hamiltonian.to_matrix() / np.sum(np.abs(coefficients))
        alias_tolerance = 2.0**-bits_precision

        inputs = [*np.eye(dimension), np.full(dimension, 1.0 / math.sqrt(dimension))]
        for system_state in inputs:
            actual = _block_encoding_action(circuit, num_system_qubits, system_state)
            expected = expected_block @ system_state

            assert np.abs(actual.imag).max() < alias_tolerance, f"unexpected phase in {actual}"
            sign = 1.0 if np.vdot(actual, expected).real >= 0.0 else -1.0
            assert np.allclose(sign * actual, expected, rtol=0.0, atol=alias_tolerance), (
                f"block encoding failed on {system_state}: got {sign * actual}, expected {expected}"
            )


class TestPSPMapperPrepareGuards:
    """Tests for PSPMapper's checks on the PREPARE circuit it is handed."""

    @staticmethod
    def _unitary():
        """Build a three-term LCU, whose PREPARE indexes two qubits."""
        hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ", "XZ"], coefficients=np.array([0.25, 0.5, 0.1]))
        return LCUBuilder().run(hamiltonian)

    def test_rejects_a_prepare_that_wants_a_phase_gradient(self):
        """QROM state prep declares shared ancilla this mapper never allocates.

        Only when it is asked to share one: left to its default the callable allocates and
        prepares its own gradient internally, declares none, and embeds fine.
        """
        mapper = registry.create(
            "circuit_mapper",
            "prepare_select_prepare",
            prepare=AlgorithmRef("state_prep", "qrom", rotation_bit_precision=4, allocate_phase_gradient=False),
        )
        with pytest.raises(ValueError, match="phase gradient ancilla"):
            mapper.run(self._unitary())


class TestLCUContainer:
    """Tests for the LCUContainer data class."""

    def test_lcu_container_serialization_roundtrip(self):
        """Test JSON serialization round-trip for LCUContainer."""
        hamiltonian = QubitOperator(
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
        assert restored.num_prepare_ancillas == container.num_prepare_ancillas
        assert np.array_equal(restored.select.phases, container.select.phases)
        assert [op.operation for op in restored.select.controlled_operations] == [
            op.operation for op in container.select.controlled_operations
        ]

    def test_unitary_representation_serialization_roundtrip(self):
        """Test JSON serialization round-trip via UnitaryRepresentation dispatch."""
        hamiltonian = QubitOperator(
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
        assert restored_container.num_prepare_ancillas == unitary_rep.get_container().num_prepare_ancillas

    def test_serialization_preserves_statevector(self):
        """Verify that serialization preserves the PREPARE wavefunction coefficients."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        json_data = container.to_json()
        restored = LCUContainer.from_json(json_data)

        original_coeffs = np.array(container.prepare.get_coefficients())
        restored_coeffs = np.array(restored.prepare.get_coefficients())
        assert np.allclose(
            restored_coeffs,
            original_coeffs,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_hdf5_serialization_roundtrip(self):
        """Test HDF5 serialization round-trip for LCUContainer."""
        hamiltonian = QubitOperator(
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
            assert restored.num_prepare_ancillas == container.num_prepare_ancillas
            original_coeffs = np.array(container.prepare.get_coefficients())
            restored_coeffs = np.array(restored.prepare.get_coefficients())
            assert np.allclose(restored_coeffs, original_coeffs)
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
        hamiltonian = QubitOperator(
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

    def test_lcu_walk_container_json_roundtrip(self):
        """Test JSON serialization round-trip for LCUWalkContainer."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, -0.5, 0.3]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()
        assert isinstance(container, LCUWalkContainer)

        json_data = container.to_json()
        restored = LCUWalkContainer.from_json(json_data)

        assert isinstance(restored, LCUWalkContainer)
        assert restored.power == container.power
        assert restored.block_encoding.power == container.block_encoding.power
        original_coeffs = np.array(container.block_encoding.prepare.get_coefficients())
        restored_coeffs = np.array(restored.block_encoding.prepare.get_coefficients())
        assert np.allclose(
            restored_coeffs,
            original_coeffs,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_lcu_walk_container_hdf5_roundtrip(self):
        """Test HDF5 serialization round-trip for LCUWalkContainer."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            filename = tmp.name

        try:
            with h5py.File(filename, "w") as f:
                container.to_hdf5(f)

            with h5py.File(filename, "r") as f:
                restored = LCUWalkContainer.from_hdf5(f)

            assert isinstance(restored, LCUWalkContainer)
            assert restored.power == container.power
            original_coeffs = np.array(container.block_encoding.prepare.get_coefficients())
            restored_coeffs = np.array(restored.block_encoding.prepare.get_coefficients())
            assert np.allclose(
                restored_coeffs,
                original_coeffs,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )
        finally:
            Path(filename).unlink()

    def test_lcu_walk_container_get_summary(self):
        """Test that LCUWalkContainer.get_summary returns a descriptive string."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()

        summary = container.get_summary()
        assert "LCU Walk Operator Container" in summary
        assert "Power" in summary
        assert "Block Encoding" in summary

    def test_walk_container_scale_equals_schatten_norm(self):
        """LCUWalkContainer scale should equal the Hamiltonian's Schatten (L1) norm."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ", "XZ"],
            coefficients=np.array([0.25, 0.5, 0.1]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()
        assert isinstance(container, LCUWalkContainer)
        assert np.isclose(
            container.scale,
            hamiltonian.schatten_norm,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_walk_container_eigenvalue_from_phase(self):
        """LCUWalkContainer eigenvalue_from_phase recovers E = λ·cos(2πφ)."""
        hamiltonian = QubitOperator(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder(quantum_walk=True)
        container = builder.run(hamiltonian).get_container()
        lam = hamiltonian.schatten_norm

        # φ=0 → E=λ
        assert np.isclose(
            container.eigenvalue_from_phase(0.0),
            lam,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        # φ=0.25 → E=0
        assert np.isclose(
            container.eigenvalue_from_phase(0.25),
            0.0,
            atol=float_comparison_absolute_tolerance,
        )
