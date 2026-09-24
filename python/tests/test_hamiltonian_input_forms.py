"""Tests for the lattice / qubit Hamiltonian input forms accepted by builders and phase estimation.

Every Hamiltonian unitary builder accepts a :class:`~qdk_chemistry.data.QubitOperator`.
Only lattice-aware builders additionally accept an unmapped
:class:`~qdk_chemistry.data.Hamiltonian`; the rest must reject it. Phase estimation
forwards whichever form it is given to its nested builder and reports a mismatch.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import Counter

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter import PlaquetteTrotter
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.algorithms.state_preparation.identity import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, Hamiltonian, LatticeGraph, MajoranaMapping, QubitOperator
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
)
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_ppp_hamiltonian

#: Builders that only understand an already-mapped qubit Hamiltonian.
QUBIT_ONLY_BUILDERS = ["trotter", "qdrift", "partially_randomized", "zassenhaus", "lcu"]


def _hubbard_lattice(width: int, height: int, *, interaction: float, hopping: float = 1.0, epsilon: float = 0.0):
    """Return the periodic Hubbard model as an unmapped lattice Hamiltonian."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    return create_hubbard_hamiltonian(lattice, epsilon=epsilon, t=hopping, U=interaction)


def _to_qubits(hamiltonian: Hamiltonian, num_sites: int) -> QubitOperator:
    """Map a lattice Hamiltonian to its Jordan-Wigner qubit Hamiltonian."""
    return create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites))


def _leaf_counts(container) -> Counter:
    """Return the multiset of Pauli rotations a product-formula container applies.

    Conjugated blocks are expanded into their basis change, body, and inverse so the
    comparison is independent of how terms are grouped into equal-angle batches.
    """
    counts: Counter = Counter()

    def record(term, sign: float) -> None:
        if isinstance(term, ExponentiatedPauliTerm):
            key = (tuple(sorted(term.pauli_term.items())), round(sign * term.angle, 10))
            counts[key] += 1
        else:
            for pauli_term in term.pauli_terms:
                key = (tuple(sorted(pauli_term.items())), round(sign * term.angle, 10))
                counts[key] += 1

    def walk(terms) -> None:
        for term in terms:
            if isinstance(term, ConjugatedExponentiatedPauliTerm):
                for inner in term.within_terms:
                    record(inner, 1.0)
                for inner in term.apply_terms:
                    record(inner, 1.0)
                for inner in term.within_terms:
                    record(inner, -1.0)
            else:
                record(term, 1.0)

    walk(container.conjugating_terms)
    walk(container.step_terms)
    return counts


class TestPlaquetteAcceptsBothInputForms:
    """The plaquette builder must produce the same formula from either input form."""

    @pytest.mark.parametrize(("side", "interaction", "epsilon"), [(2, 4.0, 0.0), (2, 8.0, -4.0), (4, 4.0, 0.0)])
    def test_lattice_input_matches_the_mapped_qubit_input(self, side, interaction, epsilon):
        """Reading the sparse integrals reproduces the Jordan-Wigner Pauli path exactly."""
        lattice_hamiltonian = _hubbard_lattice(side, side, interaction=interaction, epsilon=epsilon)
        qubit_hamiltonian = _to_qubits(lattice_hamiltonian, side * side)

        def build(operand):
            return PlaquetteTrotter(
                lattice_width=side,
                lattice_height=side,
                time=0.2,
                num_divisions=3,
            ).run(operand)

        from_lattice = build(lattice_hamiltonian).get_container()
        from_qubits = build(qubit_hamiltonian).get_container()

        assert from_lattice.num_qubits == from_qubits.num_qubits == 2 * side * side
        assert from_lattice.step_reps == from_qubits.step_reps
        assert from_lattice.scale == pytest.approx(from_qubits.scale)
        assert _leaf_counts(from_lattice) == _leaf_counts(from_qubits)

    @pytest.mark.parametrize(("side", "interaction", "epsilon"), [(2, 4.0, 0.0), (4, 8.0, -4.0)])
    def test_auto_step_sizing_matches_across_input_forms(self, side, interaction, epsilon):
        """The error bound reads the same hopping and interaction from either input form."""
        lattice_hamiltonian = _hubbard_lattice(side, side, interaction=interaction, epsilon=epsilon)
        qubit_hamiltonian = _to_qubits(lattice_hamiltonian, side * side)
        builder = PlaquetteTrotter(
            lattice_width=side,
            lattice_height=side,
            time=1.0,
            target_accuracy=1e-3,
        )

        assert builder._interaction_strength(lattice_hamiltonian, side * side) == pytest.approx(
            builder._interaction_strength(qubit_hamiltonian, side * side)
        )
        assert builder._resolve_num_divisions(lattice_hamiltonian, 1.0) == builder._resolve_num_divisions(
            qubit_hamiltonian, 1.0
        )

    def test_lattice_input_needs_no_qubit_mapping(self):
        """A lattice Hamiltonian builds without ever constructing a QubitOperator."""
        lattice_hamiltonian = _hubbard_lattice(2, 2, interaction=4.0)

        container = (
            PlaquetteTrotter(lattice_width=2, lattice_height=2, time=0.1, num_divisions=1)
            .run(lattice_hamiltonian)
            .get_container()
        )

        assert container.num_qubits == 8
        assert container.step_reps == 1

    def test_declared_lattice_must_match_the_lattice_hamiltonian(self):
        """A lattice shape inconsistent with the Hamiltonian is rejected."""
        lattice_hamiltonian = _hubbard_lattice(4, 4, interaction=4.0)
        builder = PlaquetteTrotter(lattice_width=6, lattice_height=6, time=0.05, num_divisions=1)

        with pytest.raises(ValueError, match="needs 72 qubits"):
            builder.run(lattice_hamiltonian)

    def test_intersite_interaction_is_rejected(self):
        """The interaction layer only models an onsite Hubbard U, not an intersite potential."""
        lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
        intersite = np.full((4, 4), 0.5)
        np.fill_diagonal(intersite, 0.0)
        ppp = create_ppp_hamiltonian(
            lattice,
            epsilon=np.zeros(4),
            t=np.ones((4, 4)),
            U=np.full(4, 4.0),
            V=intersite,
            z=np.ones(4),
        )
        builder = PlaquetteTrotter(lattice_width=2, lattice_height=2, time=0.1, num_divisions=1)

        with pytest.raises(ValueError, match="off-site two-body integral"):
            builder.run(ppp)


def _make_builder(builder_name: str):
    """Create a unitary builder, setting an evolution time only where that setting exists."""
    builder = create("hamiltonian_unitary_builder", builder_name)
    if builder.settings().has("time"):
        builder.settings().set("time", 0.1)
    return builder


class TestQubitOnlyBuildersRejectLattice:
    """Every builder that cannot read a lattice must say so instead of failing obscurely."""

    @pytest.mark.parametrize("builder_name", QUBIT_ONLY_BUILDERS)
    def test_builder_rejects_a_lattice_hamiltonian(self, builder_name):
        """Passing an unmapped lattice Hamiltonian raises a TypeError naming the builder."""
        lattice_hamiltonian = _hubbard_lattice(2, 2, interaction=4.0)
        builder = _make_builder(builder_name)

        with pytest.raises(TypeError, match="lattice Hamiltonian"):
            builder.run(lattice_hamiltonian)

    @pytest.mark.parametrize("builder_name", QUBIT_ONLY_BUILDERS)
    def test_builder_still_accepts_a_qubit_hamiltonian(self, builder_name):
        """The rejection must not disturb the qubit path."""
        lattice_hamiltonian = _hubbard_lattice(2, 2, interaction=4.0)
        qubit_hamiltonian = _to_qubits(lattice_hamiltonian, 4)
        builder = _make_builder(builder_name)

        assert builder.run(qubit_hamiltonian).get_num_qubits() >= 8

    @pytest.mark.parametrize("builder_name", QUBIT_ONLY_BUILDERS)
    def test_builder_reports_declared_lattice_support(self, builder_name):
        """Qubit-only builders declare that they do not accept a lattice."""
        assert create("hamiltonian_unitary_builder", builder_name).accepts_lattice() is False

    def test_plaquette_declares_lattice_support(self):
        """The plaquette builder declares that it does accept a lattice."""
        assert create("hamiltonian_unitary_builder", "plaquette").accepts_lattice() is True

    def test_unsupported_type_is_rejected(self):
        """Anything that is neither input form is reported as such."""
        builder = _make_builder("trotter")

        with pytest.raises(TypeError, match="lattice Hamiltonian or a qubit Hamiltonian"):
            builder.run("not a Hamiltonian")


def _iqpe_with_builder(builder_ref: AlgorithmRef) -> IterativePhaseEstimation:
    """Return an IQPE instance configured with the given nested unitary builder."""
    iqpe = IterativePhaseEstimation(shots_per_bit=1)
    iqpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=1,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=builder_ref,
        ),
    )
    iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=1))
    return iqpe


class TestPhaseEstimationInputForms:
    """Phase estimation forwards either input form and reports a builder mismatch."""

    def test_lattice_input_with_a_qubit_only_builder_is_reported(self):
        """A lattice Hamiltonian under the Trotter builder names both sides of the mismatch."""
        lattice_hamiltonian = _hubbard_lattice(2, 2, interaction=4.0)
        iqpe = _iqpe_with_builder(AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=0.1))
        state_preparation = identity_state_prep(8)

        with pytest.raises(TypeError, match="lattice Hamiltonian"):
            iqpe.run(state_preparation=state_preparation, qubit_hamiltonian=lattice_hamiltonian)

    def test_mismatch_message_points_at_a_lattice_aware_builder(self):
        """The mismatch error tells the user how to resolve it."""
        lattice_hamiltonian = _hubbard_lattice(2, 2, interaction=4.0)
        iqpe = _iqpe_with_builder(AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=0.1))
        state_preparation = identity_state_prep(8)

        with pytest.raises(TypeError, match="plaquette"):
            iqpe.run(state_preparation=state_preparation, qubit_hamiltonian=lattice_hamiltonian)
