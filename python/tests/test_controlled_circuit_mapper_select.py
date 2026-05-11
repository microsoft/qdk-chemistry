"""Tests for the LCU SELECT oracle mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.controlled_circuit_mapper import LCUSelectMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import Circuit, QubitHamiltonian
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestLCUSelectMapper:
    """Tests for the LCU SELECT oracle mapper algorithm."""

    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        mapper = LCUSelectMapper()
        assert mapper.name() == "lcu_select"
        assert mapper.type_name() == "select_mapper"

    def test_registered_in_registry(self):
        """Verify LCU select mapper is accessible via the registry."""
        mapper = registry.create("select_mapper", "lcu_select")
        assert isinstance(mapper, LCUSelectMapper)

    def test_select_circuit_has_factory(self):
        """Test that the SELECT circuit has both qsharp_op and qsharp_factory."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = LCUSelectMapper()
        circuit = mapper.run(container.select)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_select_unitary_single_qubit_two_terms(self):
        r"""Verify SELECT unitary for H = \alpha_1 X + \alpha_2 Z (1 target, 1 select qubit).

        The textbook SELECT oracle is:

        .. math::
            \mathrm{SELECT} = |0\rangle\langle 0| \otimes P_0 + |1\rangle\langle 1| \otimes P_1

        With positive coefficients, signs are +1, so SELECT = |0><0| \otimes X + |1><1| \otimes Z.
        Qiskit uses little-endian: q0=select (LSB), q1=target (MSB),
        so the expected matrix is kron(X, |0><0|) + kron(Z, |1><1|).
        """
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=np.array([0.5, 0.3]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = LCUSelectMapper()
        circuit = mapper.run(container.select)
        actual = Operator(circuit.get_qiskit_circuit()).data

        # Construct expected SELECT in Qiskit LE convention
        expected = np.kron(self.pauli_x, self.proj_0) + np.kron(self.pauli_z, self.proj_1)

        assert np.allclose(
            actual, expected, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_select_unitary_negative_coefficient(self):
        r"""Verify SELECT with a negative coefficient applies a sign flip.

        For H = 0.5X - 0.3Z:
          SELECT = |0><0| \otimes (+1)*X + |1><1| \otimes (-1)*Z
        The -1 sign is implemented as a global phase of pi on the target.
        """
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=np.array([0.5, -0.3]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = LCUSelectMapper()
        circuit = mapper.run(container.select)
        actual = Operator(circuit.get_qiskit_circuit()).data

        # sign(0.5)=+1, sign(-0.3)=-1
        expected = np.kron(self.pauli_x, self.proj_0) + np.kron(-self.pauli_z, self.proj_1)

        assert np.allclose(
            actual, expected, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_select_unitary_two_qubit_two_terms(self):
        r"""Verify SELECT for H = 0.25*XX + 0.5*ZZ (2 target qubits, 1 select qubit).

        SELECT = |0><0| \otimes (X \otimes X) + |1><1| \otimes (Z \otimes Z).
        Qiskit LE layout: q0=select, q1=target0, q2=target1.
        """
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = LCUSelectMapper()
        circuit = mapper.run(container.select)
        actual = Operator(circuit.get_qiskit_circuit()).data

        xx = np.kron(self.pauli_x, self.pauli_x)
        zz = np.kron(self.pauli_z, self.pauli_z)
        expected = np.kron(xx, self.proj_0) + np.kron(zz, self.proj_1)

        assert np.allclose(
            actual, expected, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
