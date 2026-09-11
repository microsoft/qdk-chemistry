"""Tests for the plaquette Trotter builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import os
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
import scipy
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter import (
    PlaquetteTrotter,
)
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    LatticeGraph,
    MajoranaMapping,
    QubitOperator,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

from .test_helpers import dense_matrix

batch_equal_angles = PlaquetteTrotter._batch_equal_angles
plaquette_sections = PlaquetteTrotter._plaquette_sections

_HWP_BREAK_EVEN = 8


def _expand_groups(
    groups: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm],
) -> list[ExponentiatedPauliTerm]:
    """Expand structural equal-angle groups into plain Pauli exponentials."""
    expanded: list[ExponentiatedPauliTerm] = []
    for group in groups:
        if isinstance(group, ExponentiatedPauliTerm):
            expanded.append(group)
        else:
            expanded.extend(ExponentiatedPauliTerm(pauli_term=term, angle=group.angle) for term in group.pauli_terms)
    return expanded


def _expand_terms(
    terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm],
) -> list[ExponentiatedPauliTerm]:
    """Expand batches and implicit conjugate tails into execution-order leaves."""
    expanded: list[ExponentiatedPauliTerm] = []
    for term in terms:
        if isinstance(term, ConjugatedExponentiatedPauliTerm):
            within = _expand_groups(term.within_terms)
            expanded.extend(within)
            expanded.extend(_expand_groups(term.apply_terms))
            expanded.extend(
                ExponentiatedPauliTerm(pauli_term=leaf.pauli_term, angle=-leaf.angle) for leaf in reversed(within)
            )
        else:
            expanded.extend(_expand_groups([term]))
    return expanded


def _structured_batches(
    terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm],
) -> list[BatchedExponentiatedPauliTerm]:
    """Collect batches from direct terms and conjugated apply blocks."""
    batches: list[BatchedExponentiatedPauliTerm] = []
    for term in terms:
        if isinstance(term, BatchedExponentiatedPauliTerm):
            batches.append(term)
        elif isinstance(term, ConjugatedExponentiatedPauliTerm):
            batches.extend(group for group in term.apply_terms if isinstance(group, BatchedExponentiatedPauliTerm))
    return batches


def _unbatch_container(container: PauliProductFormulaContainer) -> PauliProductFormulaContainer:
    """Return an equivalent structure with every HWP group expanded into leaves."""
    step_terms: list[ExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm] = []
    for term in container.step_terms:
        if isinstance(term, ConjugatedExponentiatedPauliTerm):
            step_terms.append(
                ConjugatedExponentiatedPauliTerm(
                    within_terms=_expand_groups(term.within_terms),
                    apply_terms=_expand_groups(term.apply_terms),
                )
            )
        else:
            step_terms.extend(_expand_groups([term]))
    return PauliProductFormulaContainer(
        step_terms=step_terms,
        step_reps=container.step_reps,
        num_qubits=container.num_qubits,
        scale=container.scale,
        conjugating_terms=_expand_groups(container.conjugating_terms),
    )


def _term_signature(terms: Sequence[ExponentiatedPauliTerm]) -> list[tuple[tuple[tuple[int, str], ...], float]]:
    """Return a sortable signature for an ordered leaf sequence."""
    return [(tuple(sorted(term.pauli_term.items())), round(term.angle, 12)) for term in terms]


def _single_spin_hop_layer(
    section: list[tuple[int, ...]], num_sites: int, hopping: float, time: float
) -> list[ExponentiatedPauliTerm]:
    """Return the first spin block from a hopping layer built for both spins."""
    layer = PlaquetteTrotter()._hop_layer(
        section,
        num_sites=num_sites,
        hopping=hopping,
        time=time,
    )
    assert layer is not None
    terms = _expand_terms([layer])
    return [term for term in terms if all(qubit < num_sites for qubit in term.pauli_term)]


def _plaquette_terms(sites: tuple[int, ...], hopping: float, time: float) -> list[ExponentiatedPauliTerm]:
    """Return one plaquette's factors through the hopping-layer implementation."""
    return _single_spin_hop_layer([sites], max(sites) + 1, hopping, time)


#: Name under which the plaquette builder is registered.
_PLAQUETTE_ALGORITHM = "plaquette"


def _hubbard_operator(width: int, height: int, interaction: float, *, hopping: float = 1.0):
    """Jordan-Wigner image of the periodic Hubbard model on a ``width`` by ``height`` lattice.

    Both directions wrap. At 2x2 the two wrap directions land on the same pair of
    sites, so each bond is traversed twice and its weight doubles: the effective
    hopping is 2t rather than t. That is uniform across the lattice, so the builder
    accepts it, but it does move the spectrum, so energies here are derived from this
    operator rather than quoted from the textbook chain.
    """
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    return create("qubit_mapper").run(
        create_hubbard_hamiltonian(lattice, epsilon=0.0, t=hopping, U=interaction),
        mapping=MajoranaMapping.jordan_wigner(2 * width * height),
    )


def _shifted_hubbard_operator(side: int, interaction: float = 4.0) -> QubitOperator:
    r"""Return Campbell's particle-hole shifted Hamiltonian without its known scalar offset.

    Setting the one-body site energy to ``-U/2`` turns
    :math:`U n_{i\uparrow}n_{i\downarrow}` into :math:`(U/4)Z_{i\uparrow}Z_{i\downarrow}-U/4`.
    The scalar is removed because it is restored classically when converting the
    phase-estimation result back to the unshifted fixed-particle-number energy.
    """
    lattice = LatticeGraph.square(side, side, periodic_x=True, periodic_y=True)
    mapped = create("qubit_mapper").run(
        create_hubbard_hamiltonian(lattice, epsilon=-interaction / 2.0, t=1.0, U=interaction),
        mapping=MajoranaMapping.jordan_wigner(2 * side * side),
    )
    keep = [index for index, label in enumerate(mapped.pauli_strings) if set(label) != {"I"}]
    return QubitOperator(
        pauli_strings=[mapped.pauli_strings[index] for index in keep],
        coefficients=mapped.coefficients[keep],
        encoding=mapped.encoding,
        fermion_mode_order=mapped.fermion_mode_order,
    )


def _plaquette_builder(side: int, *, time: float = 0.05) -> PlaquetteTrotter:
    """A second-order plaquette builder for a square ``side`` x ``side`` lattice."""
    return PlaquetteTrotter(lattice_width=side, lattice_height=side, order=2, time=time, num_divisions=1)


def _plaquette_container(side: int, *, time: float = 0.05, interaction: float = 8.0):
    """One second-order plaquette step of the periodic ``side`` x ``side`` Hubbard model."""
    operator = _hubbard_operator(side, side, interaction)
    return _plaquette_builder(side, time=time).run(operator).get_container()


def _pauli_label(terms: dict[int, str], num_qubits: int) -> str:
    """Encode a sparse Pauli map as the operator's big-endian label."""
    axes = ["I"] * num_qubits
    for qubit, axis in terms.items():
        axes[-qubit - 1] = axis
    return "".join(axes)


def _bound_parameter_operator(num_sites: int, hopping: float, interaction: float) -> QubitOperator:
    """Minimal operator carrying the Hubbard parameters read by the bound."""
    num_qubits = 2 * num_sites

    return QubitOperator(
        pauli_strings=[
            _pauli_label({0: "X", 1: "X"}, num_qubits),
            _pauli_label({0: "Y", 1: "Y"}, num_qubits),
            _pauli_label({num_sites: "X", num_sites + 1: "X"}, num_qubits),
            _pauli_label({num_sites: "Y", num_sites + 1: "Y"}, num_qubits),
            _pauli_label({0: "Z", num_sites: "Z"}, num_qubits),
        ],
        coefficients=np.array([-hopping / 2.0, -hopping / 2.0, -hopping / 2.0, -hopping / 2.0, interaction / 4.0]),
    )


def _resolved_plaquette_steps(
    time: float,
    target_accuracy: float,
    num_divisions: int = 1,
    *,
    side: int = 8,
    hopping: float = 1.0,
    interaction: float = 4.0,
) -> int:
    """Resolve the step count from representative uniform Hubbard parameters."""
    builder = PlaquetteTrotter(
        lattice_width=side,
        lattice_height=side,
        time=time,
        target_accuracy=target_accuracy,
        num_divisions=num_divisions,
    )
    operator = _bound_parameter_operator(side * side, hopping, interaction)
    return builder._resolve_num_divisions(operator, time)


def _inferred_plaquette_error_constant(side: int, hopping: float = 1.0, interaction: float = 4.0) -> float:
    """Recover W from the resolver with negligible integer-rounding error."""
    probe_time = 1e9
    divisions = _resolved_plaquette_steps(
        time=probe_time,
        target_accuracy=1.0,
        side=side,
        hopping=hopping,
        interaction=interaction,
    )
    return (divisions / probe_time) ** 2


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
    for term in _expand_terms(terms):
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

    @pytest.mark.parametrize("shape", [(2, 4), (4, 2), (2, 6), (6, 2)])
    def test_sides_below_four_rejected(self, shape):
        """A lone short side leaves bonds covered twice, which no choice of sections repairs."""
        with pytest.raises(ValueError, match="at least four"):
            plaquette_sections(*shape)

    def test_two_by_two_is_a_single_plaquette(self):
        """The 2x2 torus is itself one four-cycle, so the second section is empty.

        Both sides degenerate together here, unlike a 2xL lattice: the lone cycle
        already covers all four bonds exactly once, so emitting a separate gold tiling as
        well would evolve every bond twice over.
        """
        section_a, section_b = plaquette_sections(2, 2)

        assert section_b == [], "the 2x2 torus needs no second section"
        assert len(section_a) == 1

        bonds = [frozenset((section_a[0][k], section_a[0][(k + 1) % 4])) for k in range(4)]
        assert len(bonds) == len(set(bonds)), "a bond is covered twice"
        assert set(bonds) == {frozenset(pair) for pair in [(0, 1), (1, 3), (3, 2), (2, 0)]}


class TestPlaquetteTerms:
    """Tests for the single-plaquette decomposition."""

    @pytest.mark.parametrize("sites", [(0, 1, 2, 3), (0, 1, 4, 3), (1, 2, 4, 5), (4, 3, 0, 1)])
    @pytest.mark.parametrize("time", [0.05, 0.4, 1.3])
    @pytest.mark.parametrize("hopping", [1.0, -1.0])
    def test_reproduces_exact_evolution(self, sites, time, hopping):
        """The emitted terms equal ``exp(-i t H)`` regardless of the cycle's orientation.

        The sign conventions here are subtle -- four of eight plausible ones are wrong --
        so the cases span the axes that flip a sign. ``(0, 1, 4, 3)`` and ``(1, 2, 4, 5)``
        interleave non-adjacent modes, whose Jordan-Wigner strings must thread through the
        butterfly correctly. ``(4, 3, 0, 1)`` is the same four-cycle as ``(0, 1, 4, 3)``
        walked in the opposite orientation, so both its butterfly pairs descend and its
        first bond descends: exactly the gold cycles the tiling really emits (e.g.
        ``(15, 12, 0, 3)`` at 4x4), which exercise the ``-pi/8`` butterfly branch that an
        ascending-only cycle never reaches.
        """
        num_modes = 6
        terms = _plaquette_terms(sites, hopping=hopping, time=time)
        expected = scipy.linalg.expm(-1j * time * hopping * _cycle_hamiltonian(sites, num_modes))
        assert np.allclose(_unitary_from_terms(terms, num_modes), expected, atol=1e-10)

    def test_only_two_terms_need_rotation_synthesis(self):
        """The Givens network is fixed-angle; only the eigenvalue phases are arbitrary."""
        terms = _plaquette_terms((0, 1, 4, 3), hopping=1.0, time=0.05)
        eighth = math.pi / 8.0
        arbitrary = [term for term in terms if not np.isclose(term.angle / eighth, round(term.angle / eighth))]
        # Four network factors, the two fused rotations, four adjoint factors. Campbell's
        # fusion absorbs the innermost butterfly into the phases, so the network is two
        # factors shorter at each end than the six-Givens form it replaces.
        assert len(terms) == 10
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
        # Eight rather than twelve: the fused butterfly no longer appears as fixed
        # factors, having been absorbed into the two arbitrary-angle rotations.
        assert len(fixed) == 8, "the Givens network lost its fixed angles"


class TestPlaquetteTrotter:
    """Tests for the builder."""

    @staticmethod
    def _xy_support(label: str) -> frozenset[int]:
        """Return the X/Y endpoint positions in a little-endian Pauli label."""
        return frozenset(index for index, axis in enumerate(reversed(label)) if axis in "XY")

    def test_requires_a_lattice_shape(self):
        """Without a lattice the builder cannot know the tiling."""
        with pytest.raises(ValueError, match="lattice_width"):
            PlaquetteTrotter(order=2, time=0.05, num_divisions=1).run(_hubbard_operator(4, 4, 8.0))

    def test_rejects_a_lattice_that_does_not_match_the_operator(self):
        """A shape mismatch is an error rather than a silently wrong circuit."""
        with pytest.raises(ValueError, match="needs 72 qubits"):
            _plaquette_builder(6).run(_hubbard_operator(4, 4, 8.0))

    def test_rejects_an_order_other_than_two(self):
        """The error constant is second order, so no other order may be sized by it.

        A first-order product has error O(t^2/r) rather than O(t^3/r^2), so sizing it
        with Campbell's W_PLAQ (arXiv:2012.09238v4, Eq. (20) and App. D) would
        understate its error rather than fail loudly.
        """
        for order in (1, 3, 4):
            with pytest.raises(ValueError, match="order 2 only"):
                PlaquetteTrotter(lattice_width=4, lattice_height=4, order=order, time=0.05, num_divisions=1)

    def test_rejects_an_order_set_after_construction(self):
        """Settings are mutable, so the constructor check alone would be bypassable."""
        builder = _plaquette_builder(4)
        builder.settings().set("order", 1)
        with pytest.raises(ValueError, match="order 2 only"):
            builder.run(_hubbard_operator(4, 4, 8.0))

    def test_decomposes_the_repeated_body_into_one_interaction_and_three_tile_applications(self):
        """The decomposition itself must emit Campbell's four-factor Eq. (E2) body."""
        side = 4
        time = 0.05
        operator = _shifted_hubbard_operator(side)
        builder = _plaquette_builder(side, time=time)
        body = builder._decompose_trotter_step(operator, time=time)
        container = builder.run(operator).get_container()

        assert body == container.step_terms
        hopping = body[:3]
        assert all(isinstance(term, ConjugatedExponentiatedPauliTerm) for term in hopping)
        hopping = [term for term in hopping if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        interaction = _expand_terms(body[3:])
        assert len(interaction) == side * side
        assert all(term.pauli_term and set(term.pauli_term.values()) <= {"Z"} for term in interaction)
        assert sum(2 * len(term.within_terms) for term in hopping) == 3 * (side * side // 2) * 8
        assert sum(len(_expand_groups(term.apply_terms)) for term in hopping) == 3 * (side * side // 2) * 2

    def test_uses_four_times_fewer_rotations_on_the_hopping(self):
        """The whole point: four bonds cost two synthesized rotations, not eight."""
        container = _plaquette_container(4)
        eighth = math.pi / 8.0
        hopping = [term for term in container.step_terms if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        fixed = sum(
            2
            for layer in hopping
            for term in layer.within_terms
            if np.isclose(term.angle / eighth, round(term.angle / eighth))
        )
        # Campbell's Eq. (D2) applies the pink tiling at half time twice and the gold
        # tiling at full time once, so three tiling applications, two
        # spins, four cycles each, eight fixed terms -- eight rather than twelve because
        # the innermost butterfly is fused into the eigenvalue phases (App. E Eq. (E13)).
        assert fixed == 3 * 2 * 4 * 8

    def test_rejects_non_uniform_hopping(self):
        """The fixed-angle Fourier network only exists for a uniform cycle."""
        side = 4
        operator = _hubbard_operator(side, side, 8.0)
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
            _plaquette_builder(side).run(detuned)

    def test_preserves_the_hopping_sign(self):
        """The plaquette phases reverse when the Hamiltonian's hopping reverses."""
        side = 2
        time = 0.13
        positive = (
            _plaquette_builder(side, time=time)
            .run(_hubbard_operator(side, side, interaction=0.0, hopping=1.0))
            .get_container()
        )
        negative = (
            _plaquette_builder(side, time=time)
            .run(_hubbard_operator(side, side, interaction=0.0, hopping=-1.0))
            .get_container()
        )

        positive_phases = [
            term.angle
            for layer in positive.step_terms
            if isinstance(layer, ConjugatedExponentiatedPauliTerm)
            for term in _expand_groups(layer.apply_terms)
        ]
        negative_phases = [
            term.angle
            for layer in negative.step_terms
            if isinstance(layer, ConjugatedExponentiatedPauliTerm)
            for term in _expand_groups(layer.apply_terms)
        ]
        assert negative_phases == pytest.approx([-angle for angle in positive_phases])

    def test_rejects_interleaved_mode_ordering(self):
        """The tiling reads the register as spin-blocked; interleaved would mis-address sites."""
        side = 4
        operator = _hubbard_operator(side, side, 8.0)
        interleaved = QubitOperator(
            pauli_strings=list(operator.pauli_strings),
            coefficients=operator.coefficients.copy(),
            encoding=operator.encoding,
            fermion_mode_order="interleaved",
        )
        with pytest.raises(ValueError, match="spin-blocked"):
            _plaquette_builder(side).run(interleaved)

    def test_rejects_a_hopping_graph_that_is_not_the_declared_lattice(self):
        """A bond graph mismatch must raise rather than emit a circuit for another Hamiltonian."""
        side = 4
        operator = _hubbard_operator(side, side, 8.0)
        # Drop one bond's two Pauli terms, leaving the lattice with a hole.
        labels = list(operator.pauli_strings)
        target = self._xy_support(next(label for label in labels if len(self._xy_support(label)) == 2))
        keep = [index for index, label in enumerate(labels) if self._xy_support(label) != target]
        punctured = QubitOperator(
            pauli_strings=[labels[i] for i in keep],
            coefficients=operator.coefficients[keep],
            encoding=operator.encoding,
            fermion_mode_order=operator.fermion_mode_order,
        )
        with pytest.raises(ValueError, match=r"does not match a periodic|different hopping graphs"):
            _plaquette_builder(side).run(punctured)

    def test_rejects_spin_flip_hopping(self):
        """Each spin sector is tiled separately, so cross-block hopping cannot be expressed."""
        side = 4
        num_qubits = 2 * side * side
        label = ["I"] * num_qubits
        label[0] = "X"
        label[side * side] = "Y"
        operator = QubitOperator(pauli_strings=["".join(reversed(label))], coefficients=np.array([0.5]))
        with pytest.raises(ValueError, match="spin-up and spin-down"):
            _plaquette_builder(side).run(operator)

    def test_repeated_formula_has_one_interaction_and_three_hopping_sections_per_body(self):
        """Campbell E2 hoists the interaction half-layers outside the repeated bulk body."""
        side = 4
        operator = _hubbard_operator(side, side, interaction=8.0)
        container = (
            PlaquetteTrotter(
                lattice_width=side,
                lattice_height=side,
                time=0.15,
                num_divisions=3,
            )
            .run(operator)
            .get_container()
        )

        assert container.step_reps == 3
        assert container.conjugating_terms
        interaction_half = _expand_groups(container.conjugating_terms)
        full_interaction = [term for term in _expand_terms(container.step_terms[3:]) if term.pauli_term]
        assert len(full_interaction) == len(interaction_half)
        for half, full in zip(interaction_half, full_interaction, strict=True):
            assert full.pauli_term == half.pauli_term
            assert full.angle == pytest.approx(2.0 * half.angle)

        hopping = container.step_terms[:3]
        assert all(isinstance(term, ConjugatedExponentiatedPauliTerm) for term in hopping)
        hopping = [term for term in hopping if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        assert sum(2 * len(term.within_terms) for term in hopping) == 3 * (side * side // 2) * 8
        assert sum(len(_expand_groups(term.apply_terms)) for term in hopping) == 3 * (side * side // 2) * 2

    def test_hopping_only_repetitions_keep_the_whole_step(self):
        """The E2 rewrite is unnecessary when the interaction layer is empty."""
        side = 2
        operator = _hubbard_operator(side, side, interaction=0.0)
        container = (
            PlaquetteTrotter(
                lattice_width=side,
                lattice_height=side,
                time=0.15,
                num_divisions=2,
            )
            .run(operator)
            .get_container()
        )

        assert container.step_reps == 2
        assert container.step_terms
        assert container.conjugating_terms == []


class TestConjugationStructure:
    """Tests for the plaquette basis changes represented by Q# ``within`` blocks."""

    def test_basis_change_and_phases_are_separate_blocks(self):
        """Each spin-resolved plaquette has four basis factors and two phases."""
        layer = PlaquetteTrotter()._hop_layer([(0, 1, 4, 3)], num_sites=5, hopping=1.0, time=0.05)
        assert layer is not None
        assert len(layer.within_terms) == 8
        assert len(_expand_groups(layer.apply_terms)) == 4

    def test_control_off_branch_is_the_identity(self):
        """The implicit adjoint of a ``within`` block exactly cancels its basis change."""
        num_modes = 5
        layer = PlaquetteTrotter()._hop_layer([(0, 1, 4, 3)], num_sites=5, hopping=1.0, time=0.37)
        assert layer is not None
        within = [
            term for term in _expand_groups(layer.within_terms) if all(qubit < num_modes for qubit in term.pauli_term)
        ]
        control_off = within + [
            ExponentiatedPauliTerm(pauli_term=term.pauli_term, angle=-term.angle) for term in reversed(within)
        ]
        assert np.allclose(_unitary_from_terms(control_off, num_modes), np.eye(2**num_modes), atol=1e-10)

    def test_control_on_branch_still_reproduces_the_evolution(self):
        """The structured basis change must not alter the control-on evolution."""
        num_modes, time = 6, 0.37
        sites = (0, 1, 4, 3)
        terms = _plaquette_terms(sites, hopping=1.0, time=time)
        expected = scipy.linalg.expm(-1j * time * _cycle_hamiltonian(sites, num_modes))
        assert np.allclose(_unitary_from_terms(terms, num_modes), expected, atol=1e-10)

    def test_the_emitted_step_uses_three_conjugated_hopping_blocks(self):
        """The pink, gold, and pink hopping layers each own their implicit adjoint."""
        side = 4
        container = _plaquette_container(side)
        assert all(isinstance(term, ConjugatedExponentiatedPauliTerm) for term in container.step_terms[:3])

    def test_a_whole_step_has_the_expected_basis_change_count(self):
        """The three hopping tilings have Campbell's fixed-factor count."""
        container = _plaquette_container(4)
        hopping = [term for term in container.step_terms[:3] if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        fixed = sum(2 * len(term.within_terms) for term in hopping)
        # Three tiling applications, two spins, four cycles, eight conjugating factors
        # per cycle once the innermost butterfly is fused into the phases.
        assert fixed == 3 * 2 * 4 * 8


class TestPlaquetteIterativePhaseEstimation:
    """End-to-end: simulate the real plaquette circuit under IQPE and read back the energy.

    A 2x2 lattice is a single plaquette, so with the interaction switched off the
    decomposition is exact -- the emitted step reproduces exp(-iHt) to machine
    precision -- and phase estimation must land on an exact eigenvalue. That makes
    this the sharpest available check on the whole chain: the builder's lattice
    validation, the tiling, the emitted conjugations, the controlled mapper, and Q#.
    A fault anywhere in that chain shifts the
    recovered phase.

    In particular it exercises the control-OFF branch, which no test of the evolution
    alone can see: if the conjugating factors failed to cancel, the ancilla would
    entangle with the system and the phase would be wrong.

    Trotter error is deliberately absent here so the expected phase is exact; the
    class at the end of this file turns the interaction on and measures the error
    instead.
    """

    _WIDTH = 2
    _HEIGHT = 2

    # Both wrap directions coincide at 2x2, so the effective hopping is 2t = 2 and the
    # single-particle levels are -4, 0, 0, +4. Filling the lowest level for each spin
    # gives -8. The zero levels can be filled freely, so the eigenspace is degenerate,
    # but every vector in it carries the same phase and any one of them will do.
    _ENERGY = -8.0
    # -E * time / (2 pi) is then exactly 1/8, which four bits represent exactly, so the
    # expected bitstring is unambiguous.
    _TIME = math.pi / 32.0
    _PHASE = 0.125
    _BITS: ClassVar[list[int]] = [0, 0, 1, 0]

    @classmethod
    def _operator(cls):
        """Hopping-only Hubbard operator on the single-plaquette lattice."""
        return _hubbard_operator(cls._WIDTH, cls._HEIGHT, interaction=0.0)

    @classmethod
    def _eigenstate(cls, operator):
        """Return an exact ground-state eigenvector, and check its energy is the quoted one."""
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        values, vectors = np.linalg.eigh(pauli_to_dense_matrix(list(labels), list(coefficients)))
        assert np.isclose(values[0], cls._ENERGY), f"expected a ground energy of {cls._ENERGY}, got {values[0]}"
        state = np.real(vectors[:, 0])
        return state / np.linalg.norm(state)

    def _run(self, num_bits: int = 4):
        operator = self._operator()
        state = self._eigenstate(operator)
        num_qubits = operator.num_qubits
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
                unitary_builder=AlgorithmRef(
                    "hamiltonian_unitary_builder",
                    _PLAQUETTE_ALGORITHM,
                    time=self._TIME,
                    num_divisions=1,
                    order=2,
                    lattice_width=self._WIDTH,
                    lattice_height=self._HEIGHT,
                ),
            ),
        )
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))
        return iqpe.run(qubit_hamiltonian=operator, state_preparation=state_prep)

    def _emitted_unitary(self, operator: QubitOperator, name: str, **settings) -> np.ndarray:
        """Build one step with *name* and return its dense unitary."""
        container = (
            create("hamiltonian_unitary_builder", name, time=self._TIME, num_divisions=1, order=2, **settings)
            .run(operator)
            .get_container()
        )
        return _unitary_from_terms(container.step_terms, operator.num_qubits)

    def test_recovers_the_exact_eigenvalue(self):
        """The plaquette circuit must reproduce the eigenvalue to simulator precision."""
        result = self._run()
        assert list(result.bits_msb_first or []) == self._BITS
        assert np.isclose(result.phase_fraction, self._PHASE, atol=1e-9)
        assert np.isclose(result.raw_energy, self._ENERGY, atol=1e-9)

    def test_the_step_is_exact_where_term_by_term_trotter_is_not(self):
        """Justifies the exactness asserted above, and shows it is not free.

        Without this, the test above would pass for any decomposition accurate enough
        to round to the same four bits. Comparing both paths against the true
        exponential shows the plaquette step matches it to machine precision on a
        lattice where the ordinary term-by-term product does not.
        """
        operator = self._operator()
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        dense = pauli_to_dense_matrix(list(labels), list(coefficients))
        exact = scipy.linalg.expm(-1j * self._TIME * dense)

        plaquette = self._emitted_unitary(
            operator, _PLAQUETTE_ALGORITHM, lattice_width=self._WIDTH, lattice_height=self._HEIGHT
        )
        term_by_term = self._emitted_unitary(operator, "trotter")
        assert np.max(np.abs(plaquette - exact)) < 1e-12
        assert np.max(np.abs(term_by_term - exact)) > 1e-6


class TestPlaquetteErrorConstant:
    """Tests for Campbell's arXiv:2012.09238v4 plaquette-specific second-order error constant."""

    # Table I of arXiv:2012.09238v4, the W_PLAQ row, at u/tau = 4. Quoted to two
    # significant figures, which sets the tolerance below.
    _TABLE_I: ClassVar[dict[int, float]] = {4: 1.3e2, 6: 3.0e2, 8: 5.3e2, 12: 1.2e3, 16: 2.1e3}

    @pytest.mark.parametrize("side", [4, 6, 8, 12, 16])
    def test_reproduces_campbells_published_constant(self, side):
        """The constant must match the published table, not merely scale like it."""
        ours = _inferred_plaquette_error_constant(side)
        assert ours == pytest.approx(self._TABLE_I[side], rel=0.05)

    def test_vanishes_for_the_lattice_whose_sections_commute(self):
        """At 4x4 the pink and gold hopping terms commute, so only :math:`H_I` survives.

        Campbell's Table III (arXiv:2012.09238v4) records the commutator norm as
        exactly 0 there, which is a sharp check that the tiling matches his.
        """
        with_interaction = _inferred_plaquette_error_constant(4)
        # W_SO2 alone, i.e. the constant with the commutator contribution removed
        w_so2 = 4.0 / 6.0 * 16 * (math.sqrt(5.0) + 8.0) + 16.0 / 24.0 * 24.0
        assert with_interaction == pytest.approx(w_so2, rel=1e-9)

    def test_exact_and_extensive_branches_agree_at_the_cutoff(self):
        """The switch to the thermodynamic limit must not jump."""
        below = _inferred_plaquette_error_constant(40) / 1600
        above = _inferred_plaquette_error_constant(44) / 1936
        assert below == pytest.approx(above, rel=1e-3)

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() not in {"1", "true", "yes"},
        reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    )
    def test_commutator_limit_matches_exact_10000_site_norm(self):
        """The fitted per-site limit agrees with the exact 100x100 lattice norm."""
        num_sites = 10_000
        fitted = _inferred_plaquette_error_constant(100, interaction=0.0) * (24.0 / 3.0) / num_sites

        section_matrices = []
        for cycles in plaquette_sections(100, 100):
            matrix = np.zeros((num_sites, num_sites))
            for cycle in cycles:
                for index in range(4):
                    site_a, site_b = cycle[index], cycle[(index + 1) % 4]
                    matrix[site_a, site_b] = matrix[site_b, site_a] = -1.0
            section_matrices.append(matrix)
        matrix_p, matrix_g = section_matrices
        inner = matrix_p @ matrix_g - matrix_g @ matrix_p
        outer = inner @ matrix_g - matrix_g @ inner
        exact = float(np.linalg.svd(outer, compute_uv=False).sum()) / num_sites

        assert fitted == pytest.approx(exact, rel=2e-3)

    def test_step_count_falls_as_accuracy_loosens(self):
        """R scales as sqrt(1/epsilon); a looser target cannot need more steps."""
        tight = _resolved_plaquette_steps(time=1.0, target_accuracy=0.01)
        loose = _resolved_plaquette_steps(time=1.0, target_accuracy=1.0)
        assert tight > loose
        assert loose >= 1

    def test_step_count_grows_linearly_with_time(self):
        """The energy bound gives r proportional to t, not to t^{3/2}.

        A step of duration s biases the energy by W s^2 and that bias does not
        accumulate over the repetitions (arXiv:2012.09238v4, Eq. (F2) and Sec. V), so
        r = t sqrt(W/epsilon). The t^{3/2} form belongs to the per-step *unitary* error
        W s^3, which is a dimensionless quantity and not the one phase estimation reads.
        """
        base = _resolved_plaquette_steps(time=1.0, target_accuracy=1e-3)
        quadrupled = _resolved_plaquette_steps(time=4.0, target_accuracy=1e-3)
        assert quadrupled == pytest.approx(base * 4, rel=0.02)

    def test_zero_accuracy_uses_manual_divisions(self):
        """Zero target accuracy disables automatic step sizing."""
        assert _resolved_plaquette_steps(time=1.0, target_accuracy=0.0, num_divisions=7) == 7


class TestCampbellTableIResources:
    """Pre-HWP non-Clifford counts quoted for one bulk plaquette step in Table I."""

    @pytest.mark.parametrize("side", [4, 6, 8, 12, 16])
    def test_shifted_bulk_matches_the_published_gate_formulas(self, side):
        """The particle-hole shift leaves one interaction rotation per lattice site."""
        container = (
            PlaquetteTrotter(
                lattice_width=side,
                lattice_height=side,
                time=0.2,
                num_divisions=2,
            )
            .run(_shifted_hubbard_operator(side))
            .get_container()
        )

        hopping = [term for term in container.step_terms[:3] if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        fixed_t = sum(2 * len(term.within_terms) for term in hopping)
        hopping_rotations = sum(len(_expand_groups(term.apply_terms)) for term in hopping)
        interaction_rotations = len([term for term in _expand_terms(container.step_terms[3:]) if term.pauli_term])
        arbitrary_rotations = hopping_rotations + interaction_rotations
        assert fixed_t == 12 * side * side
        assert arbitrary_rotations == 4 * side * side


class TestRecoversTheClassicalGroundStateEnergy:
    """The circuit must reproduce an energy that classical diagonalization already knows.

    Every other test here checks a piece: the tiling, the emitted factors, the control
    exemption, the gate counts. This checks the thing those pieces exist to deliver.

    A 2x2 lattice is the smallest one the builder accepts and the smallest on which
    convergence can actually be observed: it holds a single plaquette, so with the
    on-site interaction switched on the step carries genuine Trotter error, because
    the hopping and the interaction do not commute. Larger lattices are past
    state-vector simulation, and a 4x4 one would not show error anyway, since its two
    hopping sections commute.
    """

    _INTERACTION = 8.0
    _WIDTH = 2
    _HEIGHT = 2

    @classmethod
    def _operator(cls):
        """Jordan-Wigner image of the periodic Hubbard model on the single-plaquette lattice."""
        return _hubbard_operator(cls._WIDTH, cls._HEIGHT, interaction=cls._INTERACTION)

    @classmethod
    def _reference(cls):
        """Exact ground-state energy and eigenvector, by dense diagonalization."""
        operator = cls._operator()
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        values, vectors = np.linalg.eigh(pauli_to_dense_matrix(list(labels), list(coefficients)))
        state = np.real(vectors[:, 0])
        return operator, float(values[0]), state / np.linalg.norm(state)

    @classmethod
    def _evolved_energy(cls, operator, state, time, divisions):
        """Energy carried by the emitted circuit, read through the mapper and Q#.

        The state is an eigenvector of the step up to the Trotter error, so one
        application suffices: the inner product is the eigenvalue. Phase estimation
        would need 2**bits applications to read the same number, and its readout is
        already pinned exactly by the phase-estimation test above.
        """
        container = (
            create(
                "hamiltonian_unitary_builder",
                _PLAQUETTE_ALGORITHM,
                time=time,
                num_divisions=divisions,
                order=2,
                lattice_width=cls._WIDTH,
                lattice_height=cls._HEIGHT,
            )
            .run(operator)
            .get_container()
        )
        circuit = create("circuit_mapper", "pauli_sequence").run(UnitaryRepresentation(container))
        evolved = np.array(
            dump_operation_on_state(
                circuit._qsharp_op, operator.num_qubits, state.tolist(), context=get_qsharp_context()
            ),
            dtype=complex,
        )
        eigenvalue = complex(np.vdot(state.astype(complex), evolved))
        return float(-np.angle(eigenvalue) / time), abs(eigenvalue)

    def test_the_hopping_and_interaction_do_not_commute(self):
        """Guards every assertion below: without this the step would be exact.

        A 4x4 lattice has commuting hopping sections, so a convergence test there
        passes whatever the decomposition does. This instance cannot.
        """
        operator = self._operator()
        hopping: list[tuple[str, float]] = []
        interaction: list[tuple[str, float]] = []
        for label, coeff in operator.get_real_coefficients(tolerance=1e-14):
            (hopping if any(axis in "XY" for axis in label) else interaction).append((label, coeff))
        left = pauli_to_dense_matrix(*(list(part) for part in zip(*hopping, strict=True)))
        right = pauli_to_dense_matrix(*(list(part) for part in zip(*interaction, strict=True)))
        assert np.max(np.abs(left @ right - right @ left)) > 1.0

    def test_converges_to_the_classical_energy(self):
        """The recovered energy must approach the exact one as the step count grows."""
        operator, exact, state = self._reference()
        time = 0.9 * np.pi / abs(exact)

        errors = []
        for divisions in (2, 4, 8):
            energy, overlap = self._evolved_energy(operator, state, time, divisions)
            errors.append(abs(energy - exact))
            assert overlap > 0.7, f"r={divisions} lost the eigenvector, overlap {overlap}"

        assert errors[0] > errors[1] > errors[2], f"error must fall with the step count, got {errors}"
        # Second order: quartering the error per doubling. Allowing a factor of two of
        # slack keeps this from failing on the higher-order terms still present here.
        assert errors[1] / errors[2] > 2.0, f"convergence too slow to be second order: {errors}"
        assert errors[2] < 0.05 * abs(exact), f"r=8 should be within 5% of {exact}, got {errors[2]}"


#: dump_operation rounds to about six decimals, so exact agreement lands near 1e-6.
_TOL = 1e-5


class TestBatchEqualAngles:
    """The structural grouping the plaquette builder applies to equal-angle terms.

    These exercise ``batch_equal_angles`` directly -- no builder, no Q# -- so they are
    the fast, unit-level checks on the grouping itself.
    """

    @pytest.mark.parametrize(("count", "is_batched"), [(2, True), (1, False)])
    def test_groups_every_nontrivial_family(self, count, is_batched):
        """Python records equal-angle structure; Q# owns the HWP cost threshold."""
        terms = [ExponentiatedPauliTerm({i: "Z"}, 0.25) for i in range(count)]
        grouped = batch_equal_angles(terms)
        assert isinstance(grouped[0], BatchedExponentiatedPauliTerm) is is_batched

    def test_separates_families_with_different_angles(self):
        """Two angles cannot share a register, so they become separate objects."""
        terms = [ExponentiatedPauliTerm({i: "Z"}, 0.25) for i in range(_HWP_BREAK_EVEN)]
        terms += [ExponentiatedPauliTerm({i: "Z", i + 1: "Z"}, -0.25) for i in range(100, 100 + 2 * _HWP_BREAK_EVEN, 2)]
        grouped = batch_equal_angles(terms)
        batches = [term for term in grouped if isinstance(term, BatchedExponentiatedPauliTerm)]
        assert len(batches) == 2
        assert {batch.angle for batch in batches} == {-0.25, 0.25}

    def test_splits_an_overlapping_family(self):
        """A family with no disjoint pair stays as plain exponentials."""
        terms = [ExponentiatedPauliTerm({0: "Z", i: "Z"}, 0.25) for i in range(1, 5)]
        assert all(isinstance(term, ExponentiatedPauliTerm) for term in batch_equal_angles(terms))

    def test_max_batch_chunks_a_large_family(self):
        """A batch needs about one ancilla per member, so the size must be capable of a cap."""
        terms = [ExponentiatedPauliTerm({i: "Z"}, 0.25) for i in range(4 * _HWP_BREAK_EVEN)]
        grouped = batch_equal_angles(terms, max_batch=_HWP_BREAK_EVEN)
        batches = [term for term in grouped if isinstance(term, BatchedExponentiatedPauliTerm)]
        assert len(batches) == 4
        assert {len(batch.pauli_terms) for batch in batches} == {_HWP_BREAK_EVEN}

    def test_keeps_the_identity_unbatched(self):
        """An identity factor has no representative qubit."""
        terms = [ExponentiatedPauliTerm({}, 0.25)]
        terms += [ExponentiatedPauliTerm({i: "Z"}, 0.25) for i in range(_HWP_BREAK_EVEN)]
        grouped = batch_equal_angles(terms)
        identity = next(term for term in grouped if isinstance(term, ExponentiatedPauliTerm))
        assert identity.pauli_term == {}

    def test_preserves_the_factors(self):
        """Reordering is only sound because it keeps exactly the same multiset."""
        terms = [ExponentiatedPauliTerm({i: "Z"}, 0.25) for i in range(_HWP_BREAK_EVEN)]
        terms += [ExponentiatedPauliTerm({99: "X"}, 0.9)]
        assert sorted(_term_signature(_expand_groups(batch_equal_angles(terms)))) == sorted(_term_signature(terms))


class TestPlaquetteBatchEmission:
    """What the builder actually emits as Hamming-weight batches, and whether it stays exact.

    Integration-level counterpart to ``TestBatchEqualAngles``: these run the builder and
    inspect the emitted step, or evolve it against ``exp(-i t H)``.

    The interesting batches come from a rotation-count optimization that hoists the
    plaquette phases. A section's plaquettes are vertex-disjoint on the lattice, so they
    act on disjoint fermionic modes and their Jordan-Wigner images are Majorana bilinears
    on disjoint Majorana indices, which commute. That lets each plaquette's two eigenvalue
    phases be pulled out of its Givens network and emitted together for the whole section,
    where equal-angle families on disjoint sites become Hamming-weight batches. Buried at
    positions 6-7 of every 14-term plaquette they never could be, because batching needs
    the equal-angle terms consecutive. These tests pin that the builder performs the hoist,
    batches the freed phases and the interaction diagonal, keeps every batch well formed,
    and that the reorder is an exact identity.
    """

    @staticmethod
    def _commute(left: dict[int, str], right: dict[int, str]) -> bool:
        """Return whether two sparse Pauli strings commute."""
        return sum(1 for qubit in set(left) & set(right) if left[qubit] != right[qubit]) % 2 == 0

    @pytest.mark.parametrize(
        ("side", "expected_sizes"),
        [
            (4, [4, 4, 4, 4] + [8] * 10),
            (6, [6, 6, 12, 12] + [18] * 10),
        ],
    )
    def test_batches_the_hopping_and_interaction_families(self, side, expected_sizes):
        """Jordan-Wigner gives 2L^2 single-Z factors and L^2 ZZ factors, each degenerate.

        Campbell's Eq. (E2) (arXiv:2012.09238v4, App. E) moves the interaction
        half-layers to the one-time boundaries, leaving one full interaction in the
        repeated body. Its two diagonal batches therefore have sizes ``[L^2, 2L^2]``.

        The hopping layers add their own batches. Each tiling application hoists the
        plaquettes' fused ``XX``/``YY`` phases together; ``XX`` and ``YY`` share a bond
        so they cannot occupy one register, but each axis batches across the section.

        The two pink applications reach the full ``L^2/2``. The gold tiling does not: its
        horizontally wrapping cycles put the fused bond at Jordan-Wigner distance
        ``L-1`` rather than 1, so those phases carry a parity string and overlap their
        neighbours. Python records each maximal disjoint group; Q# decides whether its
        size reaches the HWP break-even point.
        """
        container = _plaquette_container(side)
        sizes = sorted(len(batch.pauli_terms) for batch in _structured_batches(container.step_terms))
        assert sizes == expected_sizes

    @pytest.mark.parametrize("side", [4, 6])
    def test_emitted_batches_are_well_formed(self, side):
        """Every emitted batch is one applicable Hamming-weight block of a degenerate family.

        Re-homes the container's batch validation into a builder assertion, and folds
        in which families the batcher may target. Each batch must be single-angle, on
        pairwise-disjoint qubits, and never the identity. Batching must also never touch the fixed-angle Givens network:
        only the interaction layer's single ``Z`` and ``ZZ`` factors and the plaquettes'
        fused ``XX``/``YY`` phases carry the arbitrary angles worth batching, so every
        batched factor is controlled, off the fixed ``pi/8`` grid, and pure ``Z`` or a
        single off-diagonal axis. A ``pi/8`` network factor is already one T gate and
        would gain nothing.
        """
        container = _plaquette_container(side)
        eighth = math.pi / 8.0
        batches = _structured_batches(container.step_terms)
        assert batches, "the builder emitted no batch"
        for batch in batches:
            assert len(batch.pauli_terms) >= 2
            seen: set[int] = set()
            for pauli_term in batch.pauli_terms:
                assert pauli_term
                assert not np.isclose(batch.angle / eighth, round(batch.angle / eighth)), (
                    "a batch tagged a fixed-angle network factor, which saves nothing"
                )
                off_diagonal = {value for value in pauli_term.values() if value != "Z"}
                assert off_diagonal in ({"X"}, {"Y"}, set()), f"unexpected batched factor {pauli_term}"
                support = set(pauli_term)
                assert not (support & seen), "a batch reuses a qubit"
                seen |= support

    def test_the_freed_phases_are_batched(self):
        """Each pink application hoists its plaquette phases into two disjoint batches.

        On 4x4 a section evolves ``L^2/2 = 8`` plaquettes. After Campbell's fusion each
        contributes an ``XX`` and a ``YY`` rotation on its first bond, and since those
        two share a bond they cannot occupy one Hamming weight register: the section
        yields one batch per axis, eight members each, exactly the Q# HWP break-even size.

        Only the two pink applications reach that size. The gold tiling's horizontally
        wrapping cycles put the fused bond at Jordan-Wigner distance ``L-1``, so their
        phases carry a parity string and split into four-member structural batches,
        which Q# executes term by term.
        """
        side = 4
        container = _plaquette_container(side)
        hopping = [term for term in container.step_terms[:3] if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        pink_batches = [
            group
            for layer in (hopping[0], hopping[2])
            for group in layer.apply_terms
            if isinstance(group, BatchedExponentiatedPauliTerm)
        ]
        gold_batches = [group for group in hopping[1].apply_terms if isinstance(group, BatchedExponentiatedPauliTerm)]
        assert [len(batch.pauli_terms) for batch in pink_batches] == [8, 8, 8, 8]
        assert [len(batch.pauli_terms) for batch in gold_batches] == [4, 4, 4, 4]

    def test_batching_reduces_the_controlled_rotation_count(self):
        """The point of the whole exercise, measured rather than asserted."""
        container = _plaquette_container(4)
        plain = _unbatch_container(container)
        counts = []
        for candidate in (plain, container):
            mapper = create(
                "controlled_circuit_mapper",
                "pauli_sequence",
                control_indices=[0],
                target_indices=list(range(1, container.num_qubits + 1)),
            )
            estimate = mapper.run(UnitaryRepresentation(candidate)).estimate()
            counts.append(estimate["logicalCounts"]["rotationCount"])
        assert counts[1] < counts[0]

    def test_batched_and_loose_terms_agree_as_unitaries(self):
        """A phased family and its loose terms represent the same unitary."""
        raw = [ExponentiatedPauliTerm({i: "Z"}, 0.31) for i in range(_HWP_BREAK_EVEN)]
        grouped = batch_equal_angles(raw)
        assert isinstance(grouped[0], BatchedExponentiatedPauliTerm)
        grouped_container = PauliProductFormulaContainer(grouped, step_reps=1, num_qubits=_HWP_BREAK_EVEN)
        raw_container = PauliProductFormulaContainer(raw, step_reps=1, num_qubits=_HWP_BREAK_EVEN)
        mapper = create("circuit_mapper", "pauli_sequence")
        got = dense_matrix(mapper.run(UnitaryRepresentation(grouped_container))._qsharp_op, _HWP_BREAK_EVEN)
        want = dense_matrix(mapper.run(UnitaryRepresentation(raw_container))._qsharp_op, _HWP_BREAK_EVEN)
        assert np.max(np.abs(got - want)) < _TOL

    def test_hoisting_is_an_identity_even_when_the_strings_interleave(self):
        """Reordering the factors of two vertex-disjoint plaquettes cannot change the operator.

        The individual Pauli factors carry parity strings that thread through each
        other's sites, so their pairwise commutation is not obvious; but as Majorana
        bilinears on disjoint modes the plaquettes commute. This builds two plaquettes
        whose modes interleave -- the worst case for the strings -- and checks that the
        per-plaquette order and the hoisted order both reproduce ``exp(-i t H)``.
        """
        num_modes = 8
        plaq_a = (0, 2, 6, 4)
        plaq_b = (1, 3, 7, 5)
        time = 0.37

        per_plaquette = _plaquette_terms(plaq_a, 1.0, time) + _plaquette_terms(plaq_b, 1.0, time)
        hoisted = _single_spin_hop_layer([plaq_a, plaq_b], num_modes, 1.0, time)

        assert _term_signature(per_plaquette) != _term_signature(hoisted), "the hoist must actually move factors"
        assert sorted(_term_signature(per_plaquette)) == sorted(_term_signature(hoisted)), (
            "the hoist must keep the same factors"
        )

        section_h = _cycle_hamiltonian(plaq_a, num_modes) + _cycle_hamiltonian(plaq_b, num_modes)
        exact = scipy.linalg.expm(-1j * time * section_h)
        assert np.allclose(_unitary_from_terms(per_plaquette, num_modes), exact, atol=1e-10)
        assert np.allclose(_unitary_from_terms(hoisted, num_modes), exact, atol=1e-10)

    def test_cross_plaquette_factors_commute_on_a_real_section(self):
        """The identity above, checked non-vacuously on a real 4x4 section.

        A 4x4 section holds eight vertex-disjoint plaquettes whose Jordan-Wigner strings
        genuinely overlap. Every factor of one must commute with every factor of another
        for the hoist to be sound; here that is all 14x14 pairs across every plaquette
        pair, and none may anticommute.
        """
        side = 4
        num_sites = side * side

        for section in plaquette_sections(side, side):
            plaquettes = []
            for spin_offset in (0, num_sites):
                for cycle in section:
                    shifted = tuple(s + spin_offset for s in cycle)
                    plaquettes.append(_plaquette_terms(shifted, hopping=1.0, time=0.05))
            for i in range(len(plaquettes)):
                for j in range(i + 1, len(plaquettes)):
                    for fi in plaquettes[i]:
                        for fj in plaquettes[j]:
                            assert self._commute(fi.pauli_term, fj.pauli_term), "cross-plaquette factors anticommute"

    def test_the_batched_step_reproduces_the_exact_evolution(self):
        """On the 2x2 lattice the hoisted, batched step is exact."""
        side, time = 2, 0.29
        operator = _hubbard_operator(side, side, interaction=0.0)
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        exact = scipy.linalg.expm(-1j * time * pauli_to_dense_matrix(list(labels), list(coefficients)))
        container = _plaquette_builder(side, time=time).run(operator).get_container()
        got = _unitary_from_terms(container.step_terms, operator.num_qubits)
        assert np.allclose(got, exact, atol=1e-10)
