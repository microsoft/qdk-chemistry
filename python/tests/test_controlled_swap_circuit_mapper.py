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
    _vacuum_eigenphase,
)
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    LatticeGraph,
    MajoranaMapping,
    QubitOperator,
    Symmetries,
)
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_nontrivial_test_hamiltonian, create_test_orbitals

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector


#: Sparse Pauli terms of the two-mode worked example.
XX = {0: "X", 1: "X"}
YY = {0: "Y", 1: "Y"}
Z0 = {0: "Z"}
IDENTITY: dict[int, str] = {}

#: Fermion-to-qubit mappings the vacuum-annihilating reconstruction is expected to hold for.
FERMION_TO_QUBIT_MAPPINGS = {
    "jordan-wigner": MajoranaMapping.jordan_wigner,
    "bravyi-kitaev": MajoranaMapping.bravyi_kitaev,
    "bravyi-kitaev-tree": MajoranaMapping.bravyi_kitaev_tree,
    "parity": MajoranaMapping.parity,
}


def build_pauli_matrix(pauli_term, num_qubits):
    """Build the dense matrix of a sparse Pauli term (qubit 0 = least significant)."""
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    matrix = np.array([[1]], dtype=complex)
    for qubit in reversed(range(num_qubits)):
        matrix = np.kron(matrix, paulis[pauli_term.get(qubit, "I")])
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


def controlled_unitary(u):
    """Build the two-qubit control-one controlled-U, control being the most significant qubit."""
    p_0 = np.array([[1, 0], [0, 0]], dtype=complex)
    p_1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(p_0, np.eye(4, dtype=complex)) + np.kron(p_1, u)


def codespace_block(circuit):
    r"""Return the vacuum :math:`|0\rangle \to |0\rangle` block of a generated CSWAP circuit.

    Qubit layout: q0, q1 = system; q2 = control; q3, q4 = internally allocated vacuum register.
    Qiskit is little-endian, so the vacuum qubits are the two most significant bits and the
    codespace is the leading 8x8 block.  The final ``ResetAll`` has no matrix representation and
    acts as the identity on that codespace, so it is dropped before building the operator.
    """
    qc = circuit.get_qiskit_circuit()
    assert qc.num_qubits == 5

    unitary_part = qc.copy_empty_like()
    for instruction in qc.data:
        if instruction.operation.name != "reset":
            unitary_part.append(instruction)
    return Operator(unitary_part).data[0:8, 0:8]


@pytest.fixture
def diagonal_ppf_container():
    """Diagonal (Z/I-only) product formula: the vacuum stays an eigenstate and picks up a phase."""
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
    mapper.settings().set("control_indices", [2])
    return mapper


@pytest.fixture
def make_two_qubit_rep():
    """Return a factory turning ``(pauli_term, angle)`` pairs into a UnitaryRepresentation."""

    def _make(terms):
        step_terms = [ExponentiatedPauliTerm(pauli_term=pauli_term, angle=angle) for pauli_term, angle in terms]
        return UnitaryRepresentation(container=PauliProductFormulaContainer(step_terms, step_reps=1, num_qubits=2))

    return _make


@pytest.fixture
def vacuum_annihilating_unitary():
    """Group the end-to-end Hamiltonian with ``vacuum_annihilating`` and Trotterise it."""
    # 0.5 (XX + YY) - 0.5 Z0 + 0.5 Z1: number conserving, so H|00> = 0.
    hamiltonian = QubitOperator(["XX", "YY", "IZ", "ZI"], np.array([0.5, 0.5, -0.5, 0.5]))
    grouped = registry.create("term_grouper", "vacuum_annihilating").run(hamiltonian)

    trotter = registry.create("hamiltonian_unitary_builder", "trotter")
    trotter.settings().update({"order": 1, "num_divisions": 1, "time": 0.5})
    return trotter.run(grouped)


@pytest.fixture
def make_mapped_unitary():
    """Return a factory running molecular Hamiltonian -> qubit mapping -> ``vacuum_annihilating`` -> Trotter."""

    def _make(mapping_name):
        hamiltonian = create_nontrivial_test_hamiltonian()
        num_spin_orbitals = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapping = FERMION_TO_QUBIT_MAPPINGS[mapping_name](num_spin_orbitals)

        qubit_hamiltonian = registry.create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        grouped = registry.create("term_grouper", "vacuum_annihilating").run(qubit_hamiltonian)

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 1, "num_divisions": 1, "time": 0.5})
        return trotter.run(grouped)

    return _make


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
        r"""The sandwich equals :math:`C\text{-}U` on the codespace where the vacuum register stays :math:`|0\rangle`.

        The evolution runs *uncontrolled* on the vacuum register, so the eigenphase lands on the
        :math:`|1\rangle` control branch, and the vacuum's own phase
        :math:`U|0\ldots0\rangle = e^{i\phi_0}|0\ldots0\rangle` is cancelled by the mapper's
        :math:`R_1(\phi_0)`. The codespace block must therefore be unitary (no leakage) and equal
        the textbook :math:`C\text{-}U` up to a global phase.
        """
        circuit = cswap_mapper.run(unitary_rep)
        block = codespace_block(circuit)

        # Reconstruct the target time-evolution unitary U = exp(-i H t) from the container.
        angle_z0 = diagonal_ppf_container.step_terms[0].angle
        angle_z1 = diagonal_ppf_container.step_terms[1].angle
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        z_0 = np.kron(identity, pauli_z)
        z_1 = np.kron(pauli_z, identity)
        u = scipy.linalg.expm(-1j * angle_z1 * z_1) @ scipy.linalg.expm(-1j * angle_z0 * z_0)

        # Vacuum phase e^{i phi0} = <0|U|0> is cancelled by the mapper's R1 on the control, so the
        # codespace block is a textbook controlled-U up to a global phase.
        expected_matrix = controlled_unitary(u)

        # No leakage: the codespace block must itself be unitary.
        assert np.allclose(
            block @ block.conj().T,
            np.eye(8, dtype=complex),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

        # The codespace block reproduces the expected controlled-U (control-one) operator.
        assert np.allclose(
            block / block[0, 0],
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_vacuum_phase_correction_preserves_the_eigenphase(self, cswap_mapper, make_two_qubit_rep):
        r"""The kickback is the eigenphase of the system state, not the vacuum-shifted one.

        For :math:`U = e^{-i 0.3 Z_0} e^{-i 0.7 Z_1}` the vacuum picks up :math:`\phi_0 = -1.0`
        while :math:`|01\rangle` is an eigenstate with phase :math:`-0.4`. Without the correction
        the control would register :math:`-0.4 - \phi_0 = +0.6`.
        """
        circuit = cswap_mapper.run(make_two_qubit_rep([({0: "Z"}, 0.3), ({1: "Z"}, 0.7)]))

        prepared = QuantumCircuit(5)
        prepared.x(0)  # system in |01>, an eigenstate of U
        prepared.h(2)
        prepared.compose(circuit.get_qiskit_circuit(), inplace=True)

        # Little-endian index bits q4 q3 q2 q1 q0: |01> with the control at 0 -> 1, at 1 -> 5.
        amplitudes = Statevector(prepared).data
        assert abs(amplitudes[1]) == pytest.approx(2**-0.5, abs=float_comparison_absolute_tolerance)
        assert abs(amplitudes[5]) == pytest.approx(2**-0.5, abs=float_comparison_absolute_tolerance)
        assert amplitudes[5] / amplitudes[1] == pytest.approx(np.exp(-0.4j), abs=float_comparison_absolute_tolerance)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_vacuum_phase_covers_every_repetition(self, cswap_mapper, diagonal_ppf_container):
        """``step_reps`` copies of the step imprint ``step_reps`` times the phase."""
        repetitions = 3
        container = PauliProductFormulaContainer(
            step_terms=diagonal_ppf_container.step_terms,
            step_reps=repetitions,
            num_qubits=2,
        )
        circuit = cswap_mapper.run(UnitaryRepresentation(container=container))

        block = codespace_block(circuit)
        step = build_product_formula_matrix([(t.pauli_term, t.angle) for t in container.step_terms], 2)
        expected_matrix = controlled_unitary(np.linalg.matrix_power(step, repetitions))

        assert np.allclose(
            block / block[0, 0],
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

    def test_tolerance_setting_controls_acceptance(self, cswap_mapper, make_two_qubit_rep):
        """``vacuum_preservation_tolerance`` decides how exactly the partners have to cancel."""
        rep = make_two_qubit_rep([(XX, 0.5), (YY, 0.5 + 1e-7)])

        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            cswap_mapper.run(rep)

        # Settings lock once an algorithm has run, so the loose tolerance needs a fresh mapper.
        tolerant = ControlledSwapPauliSequenceMapper()
        tolerant.settings().set("control_indices", [2])
        tolerant.settings().set("vacuum_preservation_tolerance", 1e-6)
        assert isinstance(tolerant.run(rep), Circuit)

    def test_sub_tolerance_prefix_does_not_hide_exact_cancellation(self, cswap_mapper, make_two_qubit_rep):
        """An approximately-zero prefix must not close before its exact cancelling partner."""
        x0 = {0: "X"}
        terms = [(x0, 0.75e-9), (x0, 0.75e-9), (x0, -1.5e-9)]

        assert isinstance(cswap_mapper.run(make_two_qubit_rep(terms)), Circuit)

    def test_tolerance_is_budgeted_over_the_repetitions(self, cswap_mapper, make_two_qubit_rep):
        """A residual that is harmless once leaks ``step_reps`` times over, so the budget scales."""
        terms = [(XX, 0.5), (YY, 0.5 + 1e-10)]
        assert isinstance(cswap_mapper.run(make_two_qubit_rep(terms)), Circuit)

        step_terms = [ExponentiatedPauliTerm(pauli_term=pauli_term, angle=angle) for pauli_term, angle in terms]
        repeated = UnitaryRepresentation(
            container=PauliProductFormulaContainer(step_terms, step_reps=1000, num_qubits=2)
        )
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        with pytest.raises(ValueError, match="vacuum-preserving product formula"):
            mapper.run(repeated)


class TestVacuumPreservingBlocks:
    """Tests for the ``_vacuum_eigenphase`` helper."""

    def test_cancelling_partners_are_accepted(self):
        """Partners that cancel on the vacuum split into commuting blocks."""
        terms = [(XX, 0.5), (YY, 0.5), (Z0, -0.5), (IDENTITY, 0.5)]
        assert _vacuum_eigenphase(terms, 1e-9) == pytest.approx(0.0)

    def test_non_commuting_block_is_rejected(self):
        """A block whose factors anticommute is not equal to the exponential of its sum."""
        terms = [(XX, 0.5), (Z0, -0.5), (YY, 0.5)]
        assert _vacuum_eigenphase(terms, 1e-9) is None

    def test_trailing_residual_is_rejected(self):
        """Amplitude left outside the vacuum at the end invalidates the sequence."""
        assert _vacuum_eigenphase([({0: "X"}, 0.3)], 1e-9) is None

    def test_diagonal_terms_set_the_eigenphase(self):
        """Diagonal terms leave the vacuum in place and contribute -sum(angles) of phase."""
        assert _vacuum_eigenphase([(Z0, 0.3), (IDENTITY, 0.2)], 1e-9) == pytest.approx(-0.5)

    def test_tolerance_controls_amplitude_cancellation(self):
        """Residual amplitude below ``atol`` counts as cancelled."""
        terms = [(XX, 0.5), (YY, 0.5 + 1e-7)]
        assert _vacuum_eigenphase(terms, 1e-9) is None
        assert _vacuum_eigenphase(terms, 1e-6) == pytest.approx(0.0)

    def test_tolerance_is_a_budget_over_all_supports(self):
        """Two supports leaking 0.75e-9 each exceed a 1e-9 budget together."""
        x0, x1 = {0: "X"}, {1: "X"}
        assert _vacuum_eigenphase([(x0, 0.75e-9)], 1e-9) == pytest.approx(0.0)
        assert _vacuum_eigenphase([(x0, 0.75e-9), (x1, 0.75e-9)], 1e-9) is None

    def test_empty_sequence(self):
        """An empty product formula is trivially vacuum preserving."""
        assert _vacuum_eigenphase([], 1e-9) == pytest.approx(0.0)


class TestVacuumLeakageWithoutGrouping:
    r"""The worked example motivating the vacuum-annihilating grouper.

    For the Jordan-Wigner image of
    :math:`H = a_0^\dagger a_1 + a_1^\dagger a_0 + a_0^\dagger a_0
    = \tfrac12 (XX + YY) + \tfrac12 (I - Z_0)`
    we have :math:`H|00\rangle = 0`, and a single Trotter step at :math:`\Delta t = \pi/2`
    reproduces that only when the ``XX``/``YY`` cancellation partners stay adjacent.
    """

    def test_interleaved_ordering_leaks_half_the_vacuum(self):
        r"""``XX, Z0, YY, I`` sends :math:`|00\rangle` to a superposition with :math:`|11\rangle`.

        Inside the sandwich that entangles the vacuum register with the control, and the final
        ``ResetAll`` destroys the control coherence.
        """
        state = evolve_vacuum([(XX, 0.5), (Z0, -0.5), (YY, 0.5), (IDENTITY, 0.5)], np.pi / 2)
        assert np.isclose(state[0], (1 - 1j) / 2)
        assert np.isclose(state[3], (-1 + 1j) / 2)
        assert np.isclose(abs(state[3]) ** 2, 0.5)

    def test_grouped_ordering_preserves_the_vacuum(self):
        """``XX, YY, Z0, I`` keeps the partners adjacent and returns |00> exactly."""
        state = evolve_vacuum([(XX, 0.5), (YY, 0.5), (Z0, -0.5), (IDENTITY, 0.5)], np.pi / 2)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(state, expected, atol=float_comparison_absolute_tolerance)


class TestVacuumAnnihilatingGroupingEndToEnd:
    """End-to-end: group a qubit Hamiltonian, Trotterise it, and control it with CSWAP."""

    def test_grouped_trotter_step_preserves_the_vacuum(self, vacuum_annihilating_unitary):
        """The Trotterised product formula fixes |00>, which is what the mapper requires."""
        container = vacuum_annihilating_unitary.get_container()
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]

        vacuum = np.zeros(2**container.num_qubits, dtype=complex)
        vacuum[0] = 1.0
        evolved = build_product_formula_matrix(terms, container.num_qubits) @ vacuum

        # Only a global phase is allowed; no amplitude may leave the vacuum.
        assert np.isclose(abs(evolved[0]), 1.0, atol=float_comparison_absolute_tolerance)
        assert np.allclose(evolved[1:], 0.0, atol=float_comparison_absolute_tolerance)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_cswap_circuit_from_grouped_hamiltonian_is_a_controlled_unitary(
        self, cswap_mapper, vacuum_annihilating_unitary
    ):
        r"""The pipeline yields a controlled-:math:`U`.

        The vacuum codespace block must be unitary (no leakage) and reproduce
        :math:`|0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes U` up to a global phase.
        """
        container = vacuum_annihilating_unitary.get_container()
        circuit = cswap_mapper.run(vacuum_annihilating_unitary)
        block = codespace_block(circuit)

        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        step = build_product_formula_matrix(terms, container.num_qubits)
        u = np.linalg.matrix_power(step, container.step_reps)

        expected_matrix = controlled_unitary(u)

        assert np.allclose(
            block @ block.conj().T,
            np.eye(8, dtype=complex),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.allclose(
            block / block[0, 0],
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_transverse_field_ising_is_rejected(self):
        """A TFIM does not conserve particle number, so its X terms have nothing to cancel against."""
        hamiltonian = create_ising_hamiltonian(LatticeGraph.chain(4, periodic=False), j=1.0, h=0.5)

        with pytest.raises(ValueError, match="uncancelled vacuum amplitude"):
            registry.create("term_grouper", "vacuum_annihilating").run(hamiltonian)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_nonzero_vacuum_energy_still_yields_a_controlled_unitary(self, cswap_mapper):
        r"""An operator with :math:`E_0 \ne 0` phases the :math:`|0\rangle` branch, which the mapper cancels."""
        hamiltonian = QubitOperator(["ZI", "IZ", "II"], np.array([0.5, 0.5, 3.0]))
        grouped = registry.create("term_grouper", "vacuum_annihilating").run(hamiltonian)

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 1, "num_divisions": 1, "time": 0.5})
        unitary = trotter.run(grouped)
        container = unitary.get_container()

        block = codespace_block(cswap_mapper.run(unitary))
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        u = np.linalg.matrix_power(build_product_formula_matrix(terms, container.num_qubits), container.step_reps)

        # <0|U|0> != 1, so this only matches because the vacuum phase is cancelled on the control.
        assert not np.isclose(u[0, 0], 1.0, atol=float_comparison_absolute_tolerance)
        assert np.allclose(
            block / block[0, 0],
            controlled_unitary(u),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_symmetry_conserving_bravyi_kitaev_is_rejected(self):
        """Qubit tapering can move the reference off |0...0>, and the grouper catches that.

        SCBK fixes the symmetry sectors and drops the corresponding qubits, so the all-zero state
        of the tapered register is an occupation-number state inside the retained sector.  For a
        Hamiltonian with hopping terms it is not an eigenstate.
        """
        hamiltonian = create_nontrivial_test_hamiltonian()
        num_spin_orbitals = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(num_spin_orbitals, Symmetries(1, 1))
        qubit_hamiltonian = registry.create("qubit_mapper", "qdk").run(hamiltonian, mapping)

        with pytest.raises(ValueError, match="uncancelled vacuum amplitude"):
            registry.create("term_grouper", "vacuum_annihilating").run(qubit_hamiltonian)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_cswap_circuit_from_scbk_hamiltonian_is_a_controlled_unitary(self, cswap_mapper):
        r"""A tapered image that does annihilate |0...0> goes through the pipeline unchanged.

        Diagonal one-body integrals give an SCBK image with :math:`Z`/:math:`I` strings only, so
        the tapered reference stays an eigenstate and only picks up :math:`\varphi_0 = -E_0 t`.
        """
        num_orbitals = 2
        hamiltonian = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                np.diag([1.0, -0.5]),
                np.zeros(num_orbitals**4),
                create_test_orbitals(num_orbitals),
                0.5,
                np.eye(0),
            )
        )
        mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(2 * num_orbitals, Symmetries(1, 1))
        qubit_hamiltonian = registry.create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        assert qubit_hamiltonian.tapering is not None

        grouped = registry.create("term_grouper", "vacuum_annihilating").run(qubit_hamiltonian)
        assert grouped.tapering == qubit_hamiltonian.tapering

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 1, "num_divisions": 1, "time": 0.5})
        unitary = trotter.run(grouped)
        container = unitary.get_container()

        block = codespace_block(cswap_mapper.run(unitary))
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        u = np.linalg.matrix_power(build_product_formula_matrix(terms, container.num_qubits), container.step_reps)

        # <0|U|0> != 1, so this only matches because the vacuum phase is cancelled on the control.
        assert not np.isclose(u[0, 0], 1.0, atol=float_comparison_absolute_tolerance)
        assert np.allclose(
            block / block[0, 0],
            controlled_unitary(u),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )


class TestVacuumAnnihilatingAcrossFermionToQubitMappings:
    """The vacuum-annihilating reconstruction must not be specific to Jordan-Wigner.

    The fermionic provenance of the Pauli strings is discarded by the mapping, so ``vacuum_annihilating``
    recovers the vacuum-annihilating groups from Pauli structure alone.  Whether that
    reconstruction is faithful depends on the encoding placing every string of a fermionic term
    on a common flipped-qubit set, which these tests pin down for the supported mappings.
    """

    @pytest.mark.parametrize("mapping_name", list(FERMION_TO_QUBIT_MAPPINGS))
    def test_grouped_product_formula_is_vacuum_preserving(self, make_mapped_unitary, mapping_name):
        """A particle-conserving molecular Hamiltonian survives Trotterisation under every mapping."""
        container = make_mapped_unitary(mapping_name).get_container()
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]

        assert _vacuum_eigenphase(terms, 1e-9) is not None

    @pytest.mark.parametrize("mapping_name", list(FERMION_TO_QUBIT_MAPPINGS))
    def test_predicted_phase_matches_the_evolved_vacuum(self, make_mapped_unitary, mapping_name):
        """The classically predicted phase equals the one the product formula actually imprints."""
        container = make_mapped_unitary(mapping_name).get_container()
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]

        vacuum = np.zeros(2**container.num_qubits, dtype=complex)
        vacuum[0] = 1.0
        evolved = build_product_formula_matrix(terms, container.num_qubits) @ vacuum

        phase = _vacuum_eigenphase(terms, 1e-9)
        assert np.allclose(evolved[1:], 0.0, atol=float_comparison_absolute_tolerance)
        assert np.isclose(evolved[0], np.exp(1j * phase), atol=float_comparison_absolute_tolerance)

    @pytest.mark.parametrize("mapping_name", list(FERMION_TO_QUBIT_MAPPINGS))
    def test_mapper_accepts_the_grouped_unitary(self, make_mapped_unitary, mapping_name):
        """The full pipeline builds a circuit instead of rejecting the ordering."""
        unitary = make_mapped_unitary(mapping_name)
        mapper = ControlledSwapPauliSequenceMapper()
        mapper.settings().set("control_indices", [unitary.get_num_qubits()])

        assert isinstance(mapper.run(unitary), Circuit)
