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

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_swap_pauli_sequence_mapper import (
    ControlledSwapPauliSequenceMapper,
    vacuum_preserving_blocks,
)
from qdk_chemistry.data import QubitOperator
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.pauli_qubit_flip import (
    pauli_label_zero_state_action,
    pauli_map_zero_state_action,
)

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


class TestVacuumPreservationValidation:
    """Tests for the vacuum-preservation validation of the input product formula."""

    @staticmethod
    def _make_rep(terms):
        """Wrap the given step terms in a two-qubit UnitaryRepresentation."""
        return UnitaryRepresentation(container=PauliProductFormulaContainer(terms, step_reps=1, num_qubits=2))

    @staticmethod
    def _make_mapper():
        """Return a mapper configured with a single control qubit."""
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        return mapper

    def test_grouped_cancellation_partners_are_accepted(self):
        """``XX, YY, Z0`` keeps the partners adjacent and preserves the vacuum."""
        # H = 0.5 (XX + YY) - 0.5 Z0, the JW image of a0^dag a1 + a1^dag a0 + a0^dag a0 up to a constant.
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X", 1: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={0: "Y", 1: "Y"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=-0.5),
        ]

        circuit = self._make_mapper().run(self._make_rep(terms))
        assert isinstance(circuit, Circuit)

    def test_interleaved_cancellation_partners_raise(self):
        """``XX, Z0, YY`` splits the partners and leaks the vacuum."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X", 1: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=-0.5),
            ExponentiatedPauliTerm(pauli_term={0: "Y", 1: "Y"}, angle=0.5),
        ]

        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            self._make_mapper().run(self._make_rep(terms))

    def test_uncancelled_off_diagonal_term_raises(self):
        """A lone off-diagonal rotation rotates the vacuum out of the codespace."""
        terms = [ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3)]

        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            self._make_mapper().run(self._make_rep(terms))

    def test_diagonal_evolution_is_accepted(self, unitary_rep):
        """Pure I/Z evolutions never move the vacuum and are always valid."""
        circuit = self._make_mapper().run(unitary_rep)
        assert isinstance(circuit, Circuit)

    def test_qubit_flip_grouper_produces_an_accepted_product_formula(self):
        """Trotterising a qubit_flip-grouped Hamiltonian yields a valid ordering.

        The commuting grouper is free to interleave the ``XX``/``YY`` cancellation
        partners with the diagonal ``Z`` terms (they all commute here), which leaks
        the vacuum; the qubit_flip grouper keeps the partners adjacent.
        """
        # 0.5 (XX + YY) - 0.5 Z0 + 0.5 Z1: number conserving, so |00> is an eigenstate.
        hamiltonian = QubitOperator(["XX", "YY", "IZ", "ZI"], np.array([0.5, 0.5, -0.5, 0.5]))
        grouped = registry.create("term_grouper", "qubit_flip").run(hamiltonian)

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 1, "num_divisions": 1, "time": 0.5})
        unitary = trotter.run(grouped)

        circuit = self._make_mapper().run(unitary)
        assert isinstance(circuit, Circuit)


class TestVacuumPreservingBlocks:
    """Tests for the ``vacuum_preserving_blocks`` helper."""

    def test_blocks_are_cut_as_early_as_possible(self):
        """The helper returns the finest valid split."""
        terms = [
            ({0: "X", 1: "X"}, 0.5),
            ({0: "Y", 1: "Y"}, 0.5),
            ({0: "Z"}, -0.5),
            ({}, 0.5),
        ]
        assert vacuum_preserving_blocks(terms) == [(0, 1), (2,), (3,)]

    def test_non_commuting_block_is_rejected(self):
        """A block whose factors anticommute is not equal to the exponential of its sum."""
        terms = [
            ({0: "X", 1: "X"}, 0.5),
            ({0: "Z"}, -0.5),
            ({0: "Y", 1: "Y"}, 0.5),
        ]
        assert vacuum_preserving_blocks(terms) is None

    def test_trailing_residual_is_rejected(self):
        """Amplitude left outside the vacuum at the end invalidates the sequence."""
        assert vacuum_preserving_blocks([({0: "X"}, 0.3)]) is None

    def test_empty_sequence(self):
        """An empty product formula is trivially vacuum preserving."""
        assert vacuum_preserving_blocks([]) == []


class TestPauliVacuumAction:
    """Tests for the Pauli-on-vacuum helpers."""

    @pytest.mark.parametrize(
        ("label", "support", "amplitude"),
        [
            ("II", frozenset(), 1 + 0j),
            ("ZZ", frozenset(), 1 + 0j),
            ("IX", frozenset({0}), 1 + 0j),
            ("IY", frozenset({0}), 1j),
            ("YY", frozenset({0, 1}), -1 + 0j),
            ("XZ", frozenset({1}), 1 + 0j),
        ],
    )
    def test_pauli_label_zero_state_action(self, label, support, amplitude):
        """Labels map the vacuum to a single basis state with an i^{n_Y} prefactor."""
        result_support, result_amplitude = pauli_label_zero_state_action(label)
        assert result_support == support
        assert np.isclose(result_amplitude, amplitude)

    def test_label_and_map_agree(self):
        """Label- and map-based helpers describe the same action."""
        assert pauli_label_zero_state_action("YXZ") == pauli_map_zero_state_action({2: "Y", 1: "X", 0: "Z"})

    def test_invalid_label_raises(self):
        """Unknown Pauli characters are rejected."""
        with pytest.raises(ValueError, match="Invalid character"):
            pauli_label_zero_state_action("XQ")

    def test_invalid_axis_raises(self):
        """Unknown Pauli axes are rejected."""
        with pytest.raises(ValueError, match="Invalid Pauli axis"):
            pauli_map_zero_state_action({0: "Q"})
