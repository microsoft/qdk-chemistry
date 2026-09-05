"""Tests for the plaquette Trotter builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from typing import ClassVar

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms import create, register
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter import (
    PlaquetteTrotter,
    _plaquette_terms,
    plaquette_sections,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, LatticeGraph, MajoranaMapping, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

#: Name under which the single-plaquette builder below is registered.
_SINGLE_PLAQUETTE_ALGORITHM = "single_plaquette_for_tests"


class _SinglePlaquetteBuilder(Trotter):
    """Evolve one four-cycle by the plaquette decomposition, for the IQPE test.

    The real builder needs a lattice of at least 4x4, which is 32 qubits and far past
    state-vector simulation. A lone plaquette is small enough to simulate and is the
    same decomposition, so it exercises the emitted terms, the container's
    control-exemption check, the controlled mapper, and the Q# end to end.
    """

    def name(self) -> str:
        """Return the registered algorithm name."""
        return _SINGLE_PLAQUETTE_ALGORITHM

    def _decompose_trotter_step(self, qubit_hamiltonian, time, atol=1e-12):  # noqa: ARG002
        """Return the plaquette factors for the cycle 0-1-2-3-0."""
        if qubit_hamiltonian.num_qubits != 4:
            raise ValueError(
                f"This test builder is pinned to a single four-mode plaquette, "
                f"got {qubit_hamiltonian.num_qubits} qubits."
            )
        return _plaquette_terms((0, 1, 2, 3), hopping=1.0, time=time)


register(lambda: _SinglePlaquetteBuilder())


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron_all(matrices):
    out = np.array([[1.0 + 0j]])
    for matrix in matrices:
        out = np.kron(out, matrix)
    return out


def _annihilation(mode: int, num_modes: int) -> np.ndarray:
    """Jordan-Wigner annihilation operator with mode 0 as the leading factor."""
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    return _kron_all([_PAULI["Z"]] * mode + [lower] + [_PAULI["I"]] * (num_modes - mode - 1))


def _cycle_hamiltonian(sites, num_modes: int) -> np.ndarray:
    """Second-quantized hopping Hamiltonian of one four-cycle, with t = 1."""
    ops = [_annihilation(m, num_modes) for m in range(num_modes)]
    out = np.zeros((2**num_modes, 2**num_modes), dtype=complex)
    for index in range(4):
        p, q = sites[index], sites[(index + 1) % 4]
        out += -1.0 * (ops[p].conj().T @ ops[q] + ops[q].conj().T @ ops[p])
    return out


def _unitary_from_terms(terms, num_modes: int) -> np.ndarray:
    """Compose terms under the container convention: angle ``a`` means ``exp(-i a P)``."""
    out = np.eye(2**num_modes, dtype=complex)
    for term in terms:
        if not term.pauli_term:
            out = np.exp(-1j * term.angle) * out
            continue
        operator = _kron_all([_PAULI[term.pauli_term.get(m, "I")] for m in range(num_modes)])
        out = scipy.linalg.expm(-1j * term.angle * operator) @ out
    return out


class TestPlaquetteSections:
    """Tests for the lattice tiling."""

    @pytest.mark.parametrize("side", [4, 6, 8])
    def test_covers_every_bond_exactly_once(self, side):
        """The two sections together tile the lattice without overlap."""
        section_a, section_b = plaquette_sections(side, side)
        bonds = [frozenset((cycle[k], cycle[(k + 1) % 4])) for cycle in section_a + section_b for k in range(4)]
        assert len(bonds) == len(set(bonds)), "a bond is covered twice"
        assert len(set(bonds)) == 2 * side * side, "not every bond is covered"

    @pytest.mark.parametrize("side", [4, 6, 8])
    def test_cycles_within_a_section_are_vertex_disjoint(self, side):
        """Vertex-disjoint cycles commute, so a section carries no Trotter error."""
        for section in plaquette_sections(side, side):
            seen: set[int] = set()
            for cycle in section:
                assert not (seen & set(cycle)), "cycles in a section share a site"
                seen |= set(cycle)

    def test_rectangular_lattice_is_supported(self):
        """Tiling does not require a square lattice."""
        section_a, section_b = plaquette_sections(4, 6)
        assert len(section_a) == len(section_b) == 6

    @pytest.mark.parametrize("shape", [(3, 4), (4, 5)])
    def test_odd_sides_rejected(self, shape):
        """An odd side cannot be tiled by period-two plaquettes."""
        with pytest.raises(ValueError, match="even side lengths"):
            plaquette_sections(*shape)

    @pytest.mark.parametrize("shape", [(2, 2), (2, 4), (4, 2)])
    def test_sides_below_four_rejected(self, shape):
        """Periodic wrap folds both sections onto the same cycles below side four."""
        with pytest.raises(ValueError, match="at least four"):
            plaquette_sections(*shape)


class TestPlaquetteTerms:
    """Tests for the single-plaquette decomposition."""

    @pytest.mark.parametrize("sites", [(0, 1, 2, 3), (0, 1, 4, 3), (1, 2, 4, 5)])
    @pytest.mark.parametrize("time", [0.05, 0.4, 1.3])
    def test_reproduces_exact_evolution(self, sites, time):
        """The emitted terms equal ``exp(-i t H)`` even for non-adjacent modes."""
        num_modes = 6
        terms = _plaquette_terms(sites, hopping=1.0, time=time)
        expected = scipy.linalg.expm(-1j * time * _cycle_hamiltonian(sites, num_modes))
        assert np.allclose(_unitary_from_terms(terms, num_modes), expected, atol=1e-10)

    def test_only_two_terms_need_rotation_synthesis(self):
        """The Givens network is fixed-angle; only the eigenvalue phases are arbitrary."""
        terms = _plaquette_terms((0, 1, 4, 3), hopping=1.0, time=0.05)
        eighth = math.pi / 8.0
        arbitrary = [term for term in terms if not np.isclose(term.angle / eighth, round(term.angle / eighth))]
        assert len(terms) == 14
        assert len(arbitrary) == 2, "a plaquette must cost exactly two synthesized rotations"

    def test_emission_uses_cycle_order_not_sorted_order(self):
        """Sorting the modes would permute the cycle and lose the fixed angles.

        The regression this guards is subtle: a sorted-order decomposition still
        reproduces the evolution, so correctness tests pass, but its angles become
        arbitrary and the scheme costs more than the term-by-term path it replaces.
        """
        # (0, 1, 4, 3) is a cycle whose sorted order is a different cycle.
        terms = _plaquette_terms((0, 1, 4, 3), hopping=1.0, time=0.05)
        eighth = math.pi / 8.0
        fixed = [term for term in terms if np.isclose(term.angle / eighth, round(term.angle / eighth))]
        assert len(fixed) == 12, "the Givens network lost its fixed angles"


class TestPlaquetteTrotter:
    """Tests for the builder."""

    @staticmethod
    def _hubbard(side, interaction=8.0):
        lattice = LatticeGraph.square(side, side, periodic_x=True, periodic_y=True)
        return create("qubit_mapper").run(
            create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=interaction),
            mapping=MajoranaMapping.jordan_wigner(2 * side * side),
        )

    def test_requires_a_lattice_shape(self):
        """Without a lattice the builder cannot know the tiling."""
        with pytest.raises(ValueError, match="lattice_width"):
            PlaquetteTrotter(order=2, time=0.05, num_divisions=1).run(self._hubbard(4))

    def test_rejects_a_lattice_that_does_not_match_the_operator(self):
        """A shape mismatch is an error rather than a silently wrong circuit."""
        with pytest.raises(ValueError, match="needs 72 qubits"):
            PlaquetteTrotter(lattice_width=6, lattice_height=6, order=2, time=0.05, num_divisions=1).run(
                self._hubbard(4)
            )

    def test_uses_four_times_fewer_rotations_on_the_hopping(self):
        """The whole point: four bonds cost two synthesized rotations, not eight."""
        side = 4
        operator = self._hubbard(side)
        container = (
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1)
            .run(operator)
            .get_container()
        )
        eighth = math.pi / 8.0
        fixed = sum(
            1
            for term in container.step_terms
            if term.pauli_term and np.isclose(term.angle / eighth, round(term.angle / eighth))
        )
        # Four section applications, two spins, four cycles each, twelve fixed terms.
        assert fixed == 4 * 2 * 4 * 12

    def test_rejects_non_uniform_hopping(self):
        """The fixed-angle Fourier network only exists for a uniform cycle."""
        side = 4
        operator = self._hubbard(side)
        # Detune a single bond so the hopping is no longer uniform.
        labels = list(operator.pauli_strings)
        coefficients = operator.coefficients.copy()
        for index, label in enumerate(labels):
            if sum(1 for axis in label if axis in "XY") == 2:
                coefficients[index] *= 1.5
                break
        detuned = QubitOperator(
            pauli_strings=labels,
            coefficients=coefficients,
            encoding=operator.encoding,
            fermion_mode_order=operator.fermion_mode_order,
        )
        with pytest.raises(ValueError, match="uniform hopping"):
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1).run(detuned)

    def test_rejects_interleaved_mode_ordering(self):
        """The tiling reads the register as spin-blocked; interleaved would mis-address sites."""
        side = 4
        operator = self._hubbard(side)
        interleaved = QubitOperator(
            pauli_strings=list(operator.pauli_strings),
            coefficients=operator.coefficients.copy(),
            encoding=operator.encoding,
            fermion_mode_order="interleaved",
        )
        with pytest.raises(ValueError, match="spin-blocked"):
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1).run(
                interleaved
            )

    def test_rejects_a_hopping_graph_that_is_not_the_declared_lattice(self):
        """A bond graph mismatch must raise rather than emit a circuit for another Hamiltonian."""
        side = 4
        operator = self._hubbard(side)
        # Drop one bond's two Pauli terms, leaving the lattice with a hole.
        labels = list(operator.pauli_strings)

        def support(label):
            return frozenset(i for i, axis in enumerate(reversed(label)) if axis in "XY")

        target = support(next(label for label in labels if len(support(label)) == 2))
        keep = [index for index, label in enumerate(labels) if support(label) != target]
        punctured = QubitOperator(
            pauli_strings=[labels[i] for i in keep],
            coefficients=operator.coefficients[keep],
            encoding=operator.encoding,
            fermion_mode_order=operator.fermion_mode_order,
        )
        with pytest.raises(ValueError, match="does not match a periodic|different hopping graphs"):
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1).run(
                punctured
            )

    def test_rejects_spin_flip_hopping(self):
        """Each spin sector is tiled separately, so cross-block hopping cannot be expressed."""
        side = 4
        num_qubits = 2 * side * side
        label = ["I"] * num_qubits
        label[0] = "X"
        label[side * side] = "Y"
        operator = QubitOperator(pauli_strings=["".join(reversed(label))], coefficients=np.array([0.5]))
        with pytest.raises(ValueError, match="spin-up and spin-down"):
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1).run(operator)


class TestControlExemption:
    """Tests for the factors the plaquette scheme exempts from control."""

    def test_conjugating_factors_are_exempt_and_phases_are_not(self):
        """Only the two eigenvalue phases should need controlling."""
        terms = _plaquette_terms((0, 1, 4, 3), hopping=1.0, time=0.05)
        exempt = [t for t in terms if not t.needs_control]
        controlled = [t for t in terms if t.needs_control]
        assert len(exempt) == 12, "the Givens network should be exempt from control"
        assert len(controlled) == 2, "only the eigenvalue phases should be controlled"

    def test_control_off_branch_is_the_identity(self):
        """With the control off only the exempt factors run, and they must cancel.

        This is the property that makes the exemption sound. If it fails the circuit
        is wrong on the control-off branch, which is invisible to any check that only
        inspects the control-on evolution.
        """
        num_modes = 6
        terms = _plaquette_terms((0, 1, 4, 3), hopping=1.0, time=0.37)
        control_off = [t for t in terms if not t.needs_control]
        assert np.allclose(_unitary_from_terms(control_off, num_modes), np.eye(2**num_modes), atol=1e-10)

    def test_control_on_branch_still_reproduces_the_evolution(self):
        """Exempting factors must not change the control-on evolution."""
        num_modes, time = 6, 0.37
        sites = (0, 1, 4, 3)
        terms = _plaquette_terms(sites, hopping=1.0, time=time)
        expected = scipy.linalg.expm(-1j * time * _cycle_hamiltonian(sites, num_modes))
        assert np.allclose(_unitary_from_terms(terms, num_modes), expected, atol=1e-10)

    def test_container_rejects_exempt_factors_that_do_not_cancel(self):
        """A factor exempted from control without a matching inverse must be caught."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3, needs_control=False),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.1),
        ]
        with pytest.raises(ValueError, match="do not cancel"):
            PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=2, scale=1.0)

    def test_container_rejects_a_near_miss_inverse(self):
        """A factor undone by the wrong angle is not a cancellation."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3, needs_control=False),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.4, needs_control=False),
        ]
        with pytest.raises(ValueError, match="do not cancel"):
            PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=2, scale=1.0)

    def test_container_accepts_a_nested_sandwich(self):
        """Conjugations nest, so cancellation is last-in-first-out rather than pairwise."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3, needs_control=False),
            ExponentiatedPauliTerm(pauli_term={1: "Y"}, angle=0.2, needs_control=False),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.9),
            ExponentiatedPauliTerm(pauli_term={1: "Y"}, angle=-0.2, needs_control=False),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.3, needs_control=False),
        ]
        PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=2, scale=1.0)

    def test_a_whole_step_passes_the_container_check(self):
        """Consecutive plaquettes interleave their sandwiches; the check must still hold."""
        side = 4
        operator = TestPlaquetteTrotter._hubbard(side)
        container = (
            PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=0.05, num_divisions=1)
            .run(operator)
            .get_container()
        )
        exempt = sum(1 for t in container.step_terms if not t.needs_control)
        assert exempt == 4 * 2 * 4 * 12


class TestPlaquetteIterativePhaseEstimation:
    """End-to-end: simulate the plaquette circuit under IQPE and read back the energy.

    A single four-cycle is the entire Hamiltonian, so the plaquette decomposition is
    exact there and phase estimation must land on an exact eigenvalue. That makes this
    the sharpest available check on the whole chain -- the emitted terms, the
    container's control-exemption bookkeeping, the controlled mapper, and the Q#
    -- because a fault anywhere in it shifts the recovered phase.

    In particular it exercises the control-OFF branch, which no test of the evolution
    alone can see: if the conjugating factors failed to cancel, the ancilla would
    entangle with the system and the phase would be wrong.
    """

    # Jordan-Wigner image of -t sum_<ab> (a_a^dag a_b + h.c.) on the cycle 0-1-2-3-0,
    # verified against the dense second-quantized operator.
    _LABELS: ClassVar[list[str]] = ["IIXX", "IIYY", "IXXI", "IYYI", "XXII", "YYII", "XZZX", "YZZY"]
    _COEFFS: ClassVar[list[float]] = [-0.5] * 8

    # The eigenvalue -2 with t = pi/8 puts the phase at exactly 1/8, which four bits
    # represent exactly, so the expected bitstring is unambiguous.
    _ENERGY = -2.0
    _TIME = math.pi / 8.0
    _PHASE = 0.125
    _BITS: ClassVar[list[int]] = [0, 0, 1, 0]

    @staticmethod
    def _eigenstate():
        """Return an exact eigenvector of the cycle Hamiltonian for energy -2."""
        dense = pauli_to_dense_matrix(
            TestPlaquetteIterativePhaseEstimation._LABELS, TestPlaquetteIterativePhaseEstimation._COEFFS
        )
        values, vectors = np.linalg.eigh(dense)
        index = int(np.argmin(np.abs(values - TestPlaquetteIterativePhaseEstimation._ENERGY)))
        assert np.isclose(values[index], TestPlaquetteIterativePhaseEstimation._ENERGY)
        return np.real(vectors[:, index])

    def _run(self, unitary_builder_name: str, num_bits: int = 4):
        state = self._eigenstate()
        num_qubits = int(np.log2(len(state)))
        params = {
            "rowMap": list(range(num_qubits - 1, -1, -1)),
            "stateVector": state.tolist(),
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        state_prep = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params),
        )

        iqpe = IterativePhaseEstimation(shots_per_bit=25)
        iqpe.settings().set(
            "qpe_circuit_builder",
            AlgorithmRef(
                "qpe_circuit_builder",
                "qdk_iterative",
                num_bits=num_bits,
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
                unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", unitary_builder_name, time=self._TIME),
            ),
        )
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))
        return iqpe.run(
            qubit_hamiltonian=QubitOperator(pauli_strings=self._LABELS, coefficients=self._COEFFS),
            state_preparation=state_prep,
        )

    def test_recovers_the_exact_eigenvalue(self):
        """The plaquette circuit must reproduce the eigenvalue to simulator precision."""
        result = self._run(_SINGLE_PLAQUETTE_ALGORITHM)
        assert list(result.bits_msb_first or []) == self._BITS
        assert np.isclose(result.phase_fraction, self._PHASE, atol=1e-9)
        assert np.isclose(result.raw_energy, self._ENERGY, atol=1e-9)

    def test_is_at_least_as_accurate_as_the_term_by_term_path(self):
        """The plaquette decomposition is exact here; term-by-term Trotter is not."""
        plaquette = self._run(_SINGLE_PLAQUETTE_ALGORITHM)
        assert abs(plaquette.raw_energy - self._ENERGY) <= 1e-9
