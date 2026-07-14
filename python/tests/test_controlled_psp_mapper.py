"""Tests for the PREPARE-SELECT controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms.controlled_circuit_mapper import ControlledPSPMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import Circuit, QubitOperator
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _build_unitary_rep(pauli_strings, coefficients, *, quantum_walk=False):
    """Helper: build UnitaryRepresentation from Pauli strings and coefficients."""
    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
    builder = LCUBuilder(quantum_walk=quantum_walk)
    return builder.run(hamiltonian)


def _extract_block_encoding_submatrix(full_unitary, num_target, num_ancilla):
    r"""Extract the H/\lambda submatrix from a controlled block-encoding unitary.

    Projects onto control=1 and ancilla=|0> to verify the block encoding identity:

    .. math::
        (\langle 0|_\mathrm{anc} \otimes I_\mathrm{sys}) B[H] (|0\rangle_\mathrm{anc} \otimes I_\mathrm{sys})
        = H / \lambda

    """
    n_total = num_target + num_ancilla + 1  # +1 for control
    dim = 2**n_total
    assert full_unitary.shape == (dim, dim)

    # Find indices where control=1 (bit 0 set) AND all ancilla bits=0
    indices = []
    for i in range(dim):
        ctrl_bit = (i >> 0) & 1
        anc_bits = i >> (1 + num_target)  # ancilla occupies highest bits
        if ctrl_bit == 1 and anc_bits == 0:
            indices.append(i)
    return full_unitary[np.ix_(indices, indices)]


class TestPrepareSelectMapper:
    """Tests for the PREPARE-SELECT circuit mapper algorithm."""

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        mapper = ControlledPSPMapper()
        assert mapper.name() == "prepare_select_prepare"
        assert mapper.type_name() == "controlled_circuit_mapper"

    def test_basic_mapping_produces_circuit_with_factory(self):
        """Test that mapping produces a Circuit with both qsharp_op and qsharp_factory."""
        unitary_rep = _build_unitary_rep(["XX", "ZZ"], np.array([0.25, 0.5]))
        mapper = ControlledPSPMapper()
        circuit = mapper.run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_rejects_non_block_encoding_container(self):
        """Verify ControlledPSPMapper raises ValueError for non-BlockEncoding containers."""

        class MockContainer:
            """Mock container that is not a BlockEncodingContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        mapper = ControlledPSPMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(unitary_rep)

    def test_rejects_multiple_control_qubits(self):
        """Verify ControlledPSPMapper raises ValueError for multiple control qubits."""
        unitary_rep = _build_unitary_rep(["XX", "ZZ"], np.array([0.25, 0.5]))

        mapper = ControlledPSPMapper()
        mapper.settings().set("control_indices", [0, 1])
        with pytest.raises(ValueError, match="single control qubit"):
            mapper.run(unitary_rep)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    @pytest.mark.parametrize(
        ("pauli_strings", "coefficients", "description"),
        [
            # 1-qubit, mixed-sign coefficients
            (["X", "Z"], [0.5, -0.3], "1q mixed sign"),
            # 2-qubit, molecular-inspired (from Qiskit IQPE benchmark)
            (
                ["XI", "IX", "ZI", "IZ", "XX", "ZZ"],
                [-0.0289, -0.0289, 0.0541, 0.0541, 0.015, 0.059],
                "2q molecular H2-like",
            ),
            # 4-qubit, transverse-field Ising chain: H = J*sum(ZZ) + h*sum(X)
            (
                ["ZZII", "IZZI", "IIZZ", "XIII", "IXII", "IIXI", "IIIX"],
                [0.5, 0.5, 0.5, -0.3, -0.3, -0.3, -0.3],
                "4q transverse-field Ising",
            ),
        ],
        ids=["1q_mixed_sign", "2q_molecular", "4q_ising"],
    )
    def test_block_encoding_identity(self, pauli_strings, coefficients, description, initialize_qsharp_base_profile):
        r"""Verify the block encoding identity <0|_anc B[H] |0>_anc = H/\lambda.

        Parametrized over several Hamiltonians of increasing complexity, from
        textbook 1-qubit examples to a molecular-inspired 2-qubit benchmark.
        """
        coefficients = np.array(coefficients)
        hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
        num_target = hamiltonian.num_qubits

        unitary_rep = _build_unitary_rep(pauli_strings, coefficients)
        mapper = ControlledPSPMapper()
        circuit = mapper.run(unitary_rep)
        qc = circuit.get_qiskit_circuit()
        full_u = Operator(qc).data

        # num_ancilla = total circuit qubits - 1 (control) - num_target
        num_ancilla = qc.num_qubits - 1 - num_target
        h_over_lam = _extract_block_encoding_submatrix(full_u, num_target=num_target, num_ancilla=num_ancilla)

        lam = np.sum(np.abs(coefficients))
        expected = hamiltonian.to_matrix() / lam

        assert np.allclose(
            h_over_lam, expected, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        ), f"Block encoding identity failed for: {description}"

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_quantum_walk_eigenvalues(self, initialize_qsharp_base_profile):
        r"""Verify quantum walk operator eigenvalues satisfy the arccos relation.

        For H = \alpha_1*X + \alpha_2*Z on 1 qubit with \lambda = |\alpha_1| + |\alpha_2|,
        the walk operator W = R*B[H] has eigenvalues :math:`e^{\pm i \arccos(E_k / \lambda)}`
        where E_k are eigenvalues of H.

        H = 0.5*X + 0.3*Z has eigenvalues +-sqrt(0.5^2 + 0.3^2) = +-sqrt(0.34).
        \lambda = 0.8. The phases are arccos(+-sqrt(0.34)/0.8).
        """
        coeffs = np.array([0.5, 0.3])
        unitary_rep = _build_unitary_rep(["X", "Z"], coeffs, quantum_walk=True)

        mapper = ControlledPSPMapper()
        circuit = mapper.run(unitary_rep)
        full_u = Operator(circuit.get_qiskit_circuit()).data

        # Extract the ctrl=1 block (the full walk operator W acting on system+ancilla)
        dim = full_u.shape[0]
        ctrl1_indices = [i for i in range(dim) if i & 1]
        walk_u = full_u[np.ix_(ctrl1_indices, ctrl1_indices)]

        eigenvalues = np.linalg.eigvals(walk_u)
        phases = np.sort(np.angle(eigenvalues))

        # H eigenvalues: ±sqrt(0.5² + 0.3²) = ±sqrt(0.34)
        lam = np.sum(np.abs(coeffs))
        e_plus = np.sqrt(0.34)
        theta_plus = np.arccos(e_plus / lam)
        theta_minus = np.arccos(-e_plus / lam)
        # Walk eigenvalues: e^{±i·theta_plus} and e^{±i·theta_minus}
        expected_phases = np.sort([-theta_minus, -theta_plus, theta_plus, theta_minus])

        assert np.allclose(
            phases, expected_phases, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_reflect_via_dump_machine(self):
        r"""Verify the Reflect oracle via Q# dump_machine.

        The Reflect operation implements :math:`R = 2|0\rangle\langle 0| - I` so that:

        - :math:`R|0\rangle = +|0\rangle`
        - :math:`R|1\rangle = -|1\rangle`

        Uses ``qsharp.eval`` + ``qsharp.dump_machine()`` to inspect the quantum
        state after applying Reflect.
        """
        # Ensure Q# sources are loaded
        _ = QSHARP_UTILS.PrepSelPrep.Reflect

        # Allocate a qubit (persists across eval calls in the interpreter session)
        qsharp.eval("use q = Qubit();")

        # R|0> = +|0>
        qsharp.eval("QDKChemistry.Utils.PrepSelPrep.Reflect([q]);")
        state = qsharp.dump_machine()
        assert state.check_eq([1.0, 0.0]), f"R|0> should be |0>, got {state.as_dense_state()}"

        # State is |0> after Reflect(|0>). Apply X to get |1>, then Reflect.
        # R|1> = -|1>
        qsharp.eval("X(q); QDKChemistry.Utils.PrepSelPrep.Reflect([q]);")
        state = qsharp.dump_machine()
        assert state.check_eq([0.0, -1.0]), f"R|1> should be -|1>, got {state.as_dense_state()}"

        # Clean up qubit
        qsharp.eval("Reset(q)")
