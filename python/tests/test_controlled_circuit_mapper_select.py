"""Tests for the multi-control SELECT oracle mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.controlled_circuit_mapper import MultiControlledSelectMapper, SelectMapperFactory
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import Circuit, QubitHamiltonian
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestMultiControlledSelectMapper:
    """Tests for the multi-control SELECT oracle mapper algorithm."""

    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        mapper = MultiControlledSelectMapper()
        assert mapper.name() == "multi_controlled_select"
        assert mapper.type_name() == "select_mapper"

    def test_registered_in_registry(self):
        """Verify multi-control select mapper is accessible via the registry."""
        mapper = registry.create("select_mapper", "multi_controlled_select")
        assert isinstance(mapper, MultiControlledSelectMapper)

    def test_factory_default_algorithm_name(self):
        """Verify SelectMapperFactory default is multi_controlled_select."""
        factory = SelectMapperFactory()
        assert factory.default_algorithm_name() == "multi_controlled_select"

    def test_select_circuit_has_factory(self):
        """Test that the SELECT circuit has both qsharp_op and qsharp_factory."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=np.array([0.25, 0.5]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = MultiControlledSelectMapper()
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

        mapper = MultiControlledSelectMapper()
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

        mapper = MultiControlledSelectMapper()
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

        mapper = MultiControlledSelectMapper()
        circuit = mapper.run(container.select)
        actual = Operator(circuit.get_qiskit_circuit()).data

        xx = np.kron(self.pauli_x, self.pauli_x)
        zz = np.kron(self.pauli_z, self.pauli_z)
        expected = np.kron(xx, self.proj_0) + np.kron(zz, self.proj_1)

        assert np.allclose(
            actual, expected, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_select_single_term_negative_sign(self):
        r"""Verify SELECT for H = -1.0*Z applies the -1 global phase.

        With one term, ceil(log2(1)) = 0, so the select register is empty.
        The sign correction must still produce -Z, not +Z.

        Uses ``qsharp.eval`` + ``qsharp.dump_machine()`` to inspect the quantum
        state including global phase, which Qiskit Operator extraction drops.

        - SELECT(-Z)|0> should give -Z|0> = -|0>  (amplitude -1)
        - SELECT(-Z)|1> should give -Z|1> = +|1>  (amplitude +1)
        """
        hamiltonian = QubitHamiltonian(
            pauli_strings=["Z"],
            coefficients=np.array([-1.0]),
        )
        builder = LCUBuilder()
        container = builder.run(hamiltonian).get_container()

        mapper = MultiControlledSelectMapper()
        circuit = mapper.run(container.select)

        # Build Q# expression from the mapper's factory parameters
        factory = circuit._qsharp_factory
        params = factory.parameter["params"]
        num_select = factory.parameter["numSelectQubits"]
        num_target = factory.parameter["numTargetQubits"]

        def _pauli_to_qs(p):
            return "Pauli" + str(p).split(".")[1]

        pauli_terms_str = (
            "[" + ", ".join("[" + ", ".join(_pauli_to_qs(p) for p in term) + "]" for term in params.pauliTerms) + "]"
        )
        params_expr = (
            f"new QDKChemistry.Utils.Select.PauliSelectParams {{"
            f" pauliTerms = {pauli_terms_str},"
            f" signs = {list(params.signs)},"
            f" controlStates = {list(params.controlStates)} }}"
        )
        select_call = f"QDKChemistry.Utils.Select.PauliSelect({params_expr}, selectReg, targets);"

        qsharp.eval(f"use selectReg = Qubit[{num_select}];")
        qsharp.eval(f"use targets = Qubit[{num_target}];")

        # Apply SELECT(-Z) to |0>: expect -Z|0> = -|0>
        qsharp.eval(select_call)
        state = qsharp.dump_machine()
        assert state.check_eq([-1.0, 0.0]), f"SELECT(-Z)|0> should be -|0>, got {state.as_dense_state()}"

        # Reset, prepare |1>, apply SELECT(-Z): expect -Z|1> = +|1>
        qsharp.eval("ResetAll(targets);")
        qsharp.eval(f"X(targets[0]); {select_call}")
        state = qsharp.dump_machine()
        assert state.check_eq([0.0, 1.0]), f"SELECT(-Z)|1> should be +|1>, got {state.as_dense_state()}"

        qsharp.eval("ResetAll(targets);")
