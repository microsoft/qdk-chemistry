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


#: Qubit index used as the control in every mapper built by these tests.
CONTROL_INDEX = 2

#: Single-qubit Pauli matrices, keyed by axis.
PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

#: Sparse Pauli terms of the two-mode worked example.
XX = {0: "X", 1: "X"}
YY = {0: "Y", 1: "Y"}
Z0 = {0: "Z"}
IDENTITY: dict[int, str] = {}

#: Time step at which the interleaved ordering leaks exactly half the vacuum amplitude.
LEAKING_TIME_STEP = np.pi / 2

#: H = 0.5 (XX + YY) + 0.5 (I - Z0) as ``(pauli_term, coefficient)``, in two orderings.
GROUPED_ORDERING = [(XX, 0.5), (YY, 0.5), (Z0, -0.5), (IDENTITY, 0.5)]
INTERLEAVED_ORDERING = [(XX, 0.5), (Z0, -0.5), (YY, 0.5), (IDENTITY, 0.5)]

#: 0.5 (XX + YY) - 0.5 Z0 + 0.5 Z1: number conserving, so H|00> = 0.
END_TO_END_PAULI_STRINGS = ["XX", "YY", "IZ", "ZI"]
END_TO_END_COEFFICIENTS = [0.5, 0.5, -0.5, 0.5]
END_TO_END_EVOLUTION_TIME = 0.5


def build_pauli_matrix(pauli_term, num_qubits):
    """Build the dense matrix of a sparse Pauli term (qubit 0 = least significant)."""
    matrix = np.array([[1]], dtype=complex)
    for qubit in reversed(range(num_qubits)):
        matrix = np.kron(matrix, PAULI_MATRICES[pauli_term.get(qubit, "I")])
    return matrix


def build_product_formula_matrix(terms, num_qubits):
    r"""Multiply out :math:`\prod_j e^{-i\theta_j P_j}` in the order the terms are applied.

    Each factor is evaluated in closed form via :math:`e^{-i\theta P} = \cos\theta\,I - i\sin\theta\,P`,
    which holds because every Pauli string squares to the identity.
    """
    unitary = np.eye(2**num_qubits, dtype=complex)
    for pauli_term, angle in terms:
        pauli = build_pauli_matrix(pauli_term, num_qubits)
        factor = np.cos(angle) * np.eye(2**num_qubits, dtype=complex) - 1j * np.sin(angle) * pauli
        unitary = factor @ unitary
    return unitary


def evolve_vacuum(ordering, time_step):
    """Apply one Trotter step of ``(pauli_term, coefficient)`` pairs to |00>."""
    terms = [(pauli_term, coefficient * time_step) for pauli_term, coefficient in ordering]
    vacuum = np.zeros(4, dtype=complex)
    vacuum[0] = 1.0
    return build_product_formula_matrix(terms, num_qubits=2) @ vacuum


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


@pytest.fixture
def cswap_mapper():
    """Create a CSWAP mapper configured with a single control qubit."""
    mapper = ControlledSwapPauliSequenceMapper()
    mapper.settings().set("control_indices", [CONTROL_INDEX])
    return mapper


@pytest.fixture
def make_two_qubit_rep():
    """Return a factory turning ``(pauli_term, angle)`` pairs into a UnitaryRepresentation."""

    def _make(terms):
        step_terms = [ExponentiatedPauliTerm(pauli_term=pauli_term, angle=angle) for pauli_term, angle in terms]
        return UnitaryRepresentation(container=PauliProductFormulaContainer(step_terms, step_reps=1, num_qubits=2))

    return _make


@pytest.fixture
def make_ordering_rep(make_two_qubit_rep):
    """Return a factory turning a ``(pauli_term, coefficient)`` ordering into a UnitaryRepresentation."""

    def _make(ordering, time_step=LEAKING_TIME_STEP):
        return make_two_qubit_rep([(pauli_term, coefficient * time_step) for pauli_term, coefficient in ordering])

    return _make


@pytest.fixture
def qubit_flip_unitary():
    """Group the end-to-end Hamiltonian with ``qubit_flip`` and Trotterise it."""
    hamiltonian = QubitOperator(END_TO_END_PAULI_STRINGS, np.array(END_TO_END_COEFFICIENTS))
    grouped = registry.create("term_grouper", "qubit_flip").run(hamiltonian)

    trotter = registry.create("hamiltonian_unitary_builder", "trotter")
    trotter.settings().update({"order": 1, "num_divisions": 1, "time": END_TO_END_EVOLUTION_TIME})
    return trotter.run(grouped)


class TestControlledSwapPauliSequenceMapper:
    """Tests for the ControlledSwapPauliSequenceMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = ControlledSwapPauliSequenceMapper()
        assert mapper.name() == "cswap_pauli_sequence"

    def test_basic_mapping(self, cswap_mapper, unitary_rep):
        """Test basic mapping of unitary to Circuit."""
        circuit = cswap_mapper.run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)

    def test_multiple_control_indices_raises(self, cswap_mapper, unitary_rep):
        """Test that supplying more than one control qubit raises a ValueError."""
        cswap_mapper.settings().set("control_indices", [2, 3])

        with pytest.raises(ValueError, match="single control qubit"):
            cswap_mapper.run(unitary_rep)

    def test_invalid_container_type_raises(self, cswap_mapper):
        """Test that an invalid container type raises a ValueError."""

        class MockContainer:
            """Mock container class."""

            @property
            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = UnitaryRepresentation(container=MockContainer())

        with pytest.raises(ValueError, match="not supported"):
            cswap_mapper.run(invalid_teu)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_cswap_sandwich_controlled_u_matrix(self, cswap_mapper, unitary_rep, diagonal_ppf_container):
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
        circuit = cswap_mapper.run(unitary_rep)

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

    def test_grouped_cancellation_partners_are_accepted(self, cswap_mapper, make_two_qubit_rep):
        """``XX, YY, Z0`` keeps the partners adjacent and preserves the vacuum."""
        # H = 0.5 (XX + YY) - 0.5 Z0, the JW image of a0^dag a1 + a1^dag a0 + a0^dag a0 up to a constant.
        terms = [(XX, 0.5), (YY, 0.5), (Z0, -0.5)]

        circuit = cswap_mapper.run(make_two_qubit_rep(terms))
        assert isinstance(circuit, Circuit)

    def test_interleaved_cancellation_partners_raise(self, cswap_mapper, make_two_qubit_rep):
        """``XX, Z0, YY`` splits the partners and leaks the vacuum."""
        terms = [(XX, 0.5), (Z0, -0.5), (YY, 0.5)]

        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            cswap_mapper.run(make_two_qubit_rep(terms))

    def test_uncancelled_off_diagonal_term_raises(self, cswap_mapper, make_two_qubit_rep):
        """A lone off-diagonal rotation rotates the vacuum out of the codespace."""
        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            cswap_mapper.run(make_two_qubit_rep([({0: "X"}, 0.3)]))

    def test_diagonal_evolution_is_accepted(self, cswap_mapper, unitary_rep):
        """Pure I/Z evolutions never move the vacuum and are always valid."""
        circuit = cswap_mapper.run(unitary_rep)
        assert isinstance(circuit, Circuit)


class TestVacuumPreservingBlocks:
    """Tests for the ``vacuum_preserving_blocks`` helper."""

    def test_blocks_are_cut_as_early_as_possible(self):
        """The helper returns the finest valid split."""
        terms = [(XX, 0.5), (YY, 0.5), (Z0, -0.5), (IDENTITY, 0.5)]
        assert vacuum_preserving_blocks(terms) == [(0, 1), (2,), (3,)]

    def test_non_commuting_block_is_rejected(self):
        """A block whose factors anticommute is not equal to the exponential of its sum."""
        terms = [(XX, 0.5), (Z0, -0.5), (YY, 0.5)]
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


class TestVacuumLeakageWithoutGrouping:
    r"""Reproduce the worked example motivating the qubit-flip grouper.

    For the Jordan-Wigner image of
    :math:`H = a_0^\dagger a_1 + a_1^\dagger a_0 + a_0^\dagger a_0
    = \tfrac12 (XX + YY) + \tfrac12 (I - Z_0)`
    we have :math:`H|00\rangle = 0`, so the exact :math:`e^{-iHt}` fixes the vacuum.
    A single Trotter step at :math:`\Delta t = \pi/2` only reproduces that when the
    ``XX``/``YY`` cancellation partners stay adjacent.
    """

    def test_interleaved_ordering_leaks_half_the_vacuum(self):
        r"""``XX, Z0, YY, I`` sends :math:`|00\rangle` to a superposition with :math:`|11\rangle`.

        The step yields :math:`\tfrac{1-i}{2}|00\rangle + \tfrac{-1+i}{2}|11\rangle`, so
        :math:`|\langle 11|U|00\rangle|^2 = 1/2`. Inside the CSWAP sandwich that entangles
        the vacuum register with the control, and the final ``ResetAll`` destroys the
        control coherence.
        """
        state = evolve_vacuum(INTERLEAVED_ORDERING, LEAKING_TIME_STEP)

        assert np.isclose(state[0], (1 - 1j) / 2)
        assert np.isclose(state[3], (-1 + 1j) / 2)
        assert np.isclose(abs(state[3]) ** 2, 0.5)

    def test_grouped_ordering_preserves_the_vacuum(self):
        """``XX, YY, Z0, I`` keeps the partners adjacent and returns |00> exactly."""
        state = evolve_vacuum(GROUPED_ORDERING, LEAKING_TIME_STEP)

        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(state, expected, atol=float_comparison_absolute_tolerance)

    def test_mapper_rejects_the_leaking_ordering(self, cswap_mapper, make_ordering_rep):
        """The mapper's validation matches the numerics above."""
        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            cswap_mapper.run(make_ordering_rep(INTERLEAVED_ORDERING))

    def test_mapper_accepts_the_grouped_ordering(self, cswap_mapper, make_ordering_rep):
        """The ordering that preserves the vacuum passes validation."""
        assert isinstance(cswap_mapper.run(make_ordering_rep(GROUPED_ORDERING)), Circuit)


class TestQubitFlipGroupingEndToEnd:
    """End-to-end: group a qubit Hamiltonian, Trotterise it, and control it with CSWAP."""

    def test_grouped_trotter_step_preserves_the_vacuum(self, qubit_flip_unitary):
        """The Trotterised product formula fixes |00>, which is what the mapper requires."""
        container = qubit_flip_unitary.get_container()
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]

        vacuum = np.zeros(2**container.num_qubits, dtype=complex)
        vacuum[0] = 1.0
        evolved = build_product_formula_matrix(terms, container.num_qubits) @ vacuum

        # Only a global phase is allowed; no amplitude may leave the vacuum.
        assert np.isclose(abs(evolved[0]), 1.0, atol=float_comparison_absolute_tolerance)
        assert np.allclose(evolved[1:], 0.0, atol=float_comparison_absolute_tolerance)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_cswap_circuit_from_grouped_hamiltonian_is_a_controlled_unitary(self, cswap_mapper, qubit_flip_unitary):
        r"""The CSWAP circuit built from the grouped Hamiltonian is a genuine controlled-:math:`U`.

        This is the pipeline a caller actually writes: ``qubit_flip`` grouper ->
        ``trotter`` unitary builder -> ``cswap_pauli_sequence`` mapper. The circuit's
        vacuum codespace block must be unitary (no leakage) and reproduce
        :math:`e^{i\phi_0}|0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes U`.
        """
        container = qubit_flip_unitary.get_container()
        circuit = cswap_mapper.run(qubit_flip_unitary)

        # q0, q1 = system; q2 = control; q3, q4 = internally allocated vacuum register.
        qc = circuit.get_qiskit_circuit()
        assert qc.num_qubits == 5
        block = Operator(qc).data[0:8, 0:8]

        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        step = build_product_formula_matrix(terms, container.num_qubits)
        u = np.linalg.matrix_power(step, container.step_reps)

        p_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        expected_matrix = u[0, 0] * np.kron(p_0, np.eye(4, dtype=complex)) + np.kron(p_1, u)

        assert np.allclose(
            block @ block.conj().T,
            np.eye(8, dtype=complex),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.allclose(
            block,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
