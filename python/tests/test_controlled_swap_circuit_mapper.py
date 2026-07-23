"""Tests for the ControlledSwapPauliSequenceMapper in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

try:
    from qdk._native import Circuit as QdkCircuitType
except ImportError:
    from qsharp._native import Circuit as QdkCircuitType

from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_swap_pauli_sequence_mapper import (
    ControlledSwapPauliSequenceMapper,
)
from qdk_chemistry.data.circuit import Circuit
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
def diagonal_ppf_container():
    """Create a diagonal (Z/I-only) PauliProductFormulaContainer for testing.

    Using only Z/I terms guarantees that the all-zero vacuum register is an
    eigenstate of the evolution, so the CSWAP sandwich does not leak amplitude
    out of the vacuum codespace and the effective operator is unitary.
    """
    terms = [
        ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.7),
    ]

    return PauliProductFormulaContainer(
        step_terms=terms,
        step_reps=1,
        num_qubits=2,
    )


@pytest.fixture
def unitary_rep(diagonal_ppf_container):
    """Create a UnitaryRepresentation for testing."""
    return UnitaryRepresentation(container=diagonal_ppf_container)


class TestControlledSwapPauliSequenceMapper:
    """Tests for the ControlledSwapPauliSequenceMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = ControlledSwapPauliSequenceMapper()
        assert mapper.name() == "cswap_pauli_sequence"

    def test_basic_mapping(self, unitary_rep):
        """Test basic mapping of unitary to Circuit."""
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        circuit = mapper.run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)

    def test_multiple_control_indices_raises(self, unitary_rep):
        """Test that supplying more than one control qubit raises a ValueError."""
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2, 3])

        with pytest.raises(ValueError, match="single control qubit"):
            mapper.run(unitary_rep)

    def test_invalid_container_type_raises(self):
        """Test that an invalid container type raises a ValueError."""

        class MockContainer:
            """Mock container class."""

            @property
            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = UnitaryRepresentation(container=MockContainer())

        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        with pytest.raises(ValueError, match="not supported"):
            mapper.run(invalid_teu)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_cswap_sandwich_controlled_u_matrix(self, unitary_rep, diagonal_ppf_container):
        r"""Validate the CSWAP-sandwich construction as a controlled-U matrix check.

        The CSWAP sandwich does not equal :math:`C\text{-}U` on the full
        ancilla+system+vacuum Hilbert space. It equals a controlled-U only on the
        codespace where the vacuum register stays in :math:`|0\ldots0\rangle`.
        Because the *uncontrolled* evolution is applied to the vacuum register, the
        target eigenphase lands on the :math:`|1\rangle` control branch (the
        standard controlled-U convention). The effective operator on the codespace
        (with control as the most-significant qubit) is

        .. math::
            M_{\mathrm{eff}} = e^{i\phi_0}\,|0\rangle\langle0| \otimes I
                             + |1\rangle\langle1| \otimes U,

        where :math:`e^{i\phi_0}` is the phase the evolution imprints on the vacuum
        state :math:`U|0\ldots0\rangle = e^{i\phi_0}|0\ldots0\rangle`. It reduces to
        the textbook :math:`C\text{-}U` exactly when :math:`\phi_0 = 0`.

        This test builds the full circuit unitary, extracts the vacuum
        :math:`|0\rangle \to |0\rangle` block, confirms it is unitary (i.e. no
        amplitude leaks out of the codespace), and compares it to
        :math:`M_{\mathrm{eff}}`.
        """
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        circuit = mapper.run(unitary_rep)

        # Qubit layout of the generated circuit: q0, q1 = system; q2 = control;
        # q3, q4 = internally allocated vacuum register.
        qc = circuit.get_qiskit_circuit()
        assert qc.num_qubits == 5

        full = Operator(qc).data  # 32x32, qiskit little-endian: index bits = q4 q3 q2 q1 q0
        # Vacuum qubits q3, q4 are the two most-significant bits; the vacuum = |0>
        # codespace is therefore the leading 8x8 block (indices 0..7).
        block = full[0:8, 0:8]

        # Reconstruct the target time-evolution unitary U = exp(-i H t) from the container.
        angle_z0 = diagonal_ppf_container.step_terms[0].angle
        angle_z1 = diagonal_ppf_container.step_terms[1].angle
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        z_0 = np.kron(identity, pauli_z)
        z_1 = np.kron(pauli_z, identity)
        u = scipy.linalg.expm(-1j * angle_z1 * z_1) @ scipy.linalg.expm(-1j * angle_z0 * z_0)

        # Vacuum phase e^{i phi0} = <0|U|0>.
        vacuum_phase = u[0, 0]

        # Control-one effective operator: e^{i phi0} I on the |0> branch, U on the |1> branch.
        p_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        i_4 = np.eye(4, dtype=complex)
        expected_matrix = vacuum_phase * np.kron(p_0, i_4) + np.kron(p_1, u)

        # No leakage: the codespace block must itself be unitary.
        assert np.allclose(
            block @ block.conj().T,
            np.eye(8, dtype=complex),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

        # The codespace block reproduces the expected controlled-U (control-one) operator.
        assert np.allclose(
            block,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
