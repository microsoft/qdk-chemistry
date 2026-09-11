"""Tests for the plaquette Trotter builder, from classical structure to its Q# lowering."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
import scipy

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
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context


def _hubbard_operator(width: int, height: int, interaction: float, *, hopping: float = 1.0) -> QubitOperator:
    """Return the periodic Hubbard Hamiltonian in Jordan-Wigner encoding."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    return create("qubit_mapper").run(
        create_hubbard_hamiltonian(lattice, epsilon=0.0, t=hopping, U=interaction),
        mapping=MajoranaMapping.jordan_wigner(2 * width * height),
    )


class TestPlaquetteTrotterConfiguration:
    """Public configuration and validation behavior."""

    def test_explicit_num_divisions_is_used(self):
        """An explicit division count controls the repeated formula when auto sizing is disabled."""
        operator = _hubbard_operator(2, 2, interaction=4.0)
        container = (
            PlaquetteTrotter(
                lattice_width=2,
                lattice_height=2,
                order=2,
                time=0.2,
                target_accuracy=0.0,
                num_divisions=7,
            )
            .run(operator)
            .get_container()
        )

        assert container.step_reps == 7

    @pytest.mark.parametrize("order", [1, 3, 4])
    def test_rejects_unsupported_order(self, order):
        """The plaquette error bound and decomposition are second order only."""
        with pytest.raises(ValueError, match="order 2 only"):
            PlaquetteTrotter(order=order)

    @pytest.mark.parametrize(
        ("builder", "message"),
        [
            (PlaquetteTrotter(order=2, time=0.05, num_divisions=1), "lattice_width"),
            (
                PlaquetteTrotter(lattice_width=6, lattice_height=6, order=2, time=0.05, num_divisions=1),
                "needs 72 qubits",
            ),
        ],
    )
    def test_rejects_missing_or_mismatched_lattice(self, builder, message):
        """The declared lattice must identify the operator being decomposed."""
        with pytest.raises(ValueError, match=message):
            builder.run(_hubbard_operator(4, 4, interaction=4.0))


def _pauli_label(terms: dict[int, str], num_qubits: int) -> str:
    """Encode a sparse Pauli map as a big-endian label."""
    axes = ["I"] * num_qubits
    for qubit, axis in terms.items():
        axes[-qubit - 1] = axis
    return "".join(axes)


def _resolve_divisions(
    *,
    time: float,
    target_accuracy: float,
    num_divisions: int = 0,
    side: int = 8,
) -> int:
    """Resolve a division count for representative uniform Hubbard parameters."""
    builder = PlaquetteTrotter(
        lattice_width=side,
        lattice_height=side,
        time=time,
        target_accuracy=target_accuracy,
        num_divisions=num_divisions,
    )
    num_sites = side * side
    num_qubits = 2 * num_sites
    hopping, interaction = 1.0, 4.0
    # The bound reads only the uniform hopping and interaction, so a minimal operator suffices.
    operator = QubitOperator(
        pauli_strings=[
            _pauli_label({0: "X", 1: "X"}, num_qubits),
            _pauli_label({0: "Y", 1: "Y"}, num_qubits),
            _pauli_label({num_sites: "X", num_sites + 1: "X"}, num_qubits),
            _pauli_label({num_sites: "Y", num_sites + 1: "Y"}, num_qubits),
            _pauli_label({0: "Z", num_sites: "Z"}, num_qubits),
        ],
        coefficients=np.array([-hopping / 2.0, -hopping / 2.0, -hopping / 2.0, -hopping / 2.0, interaction / 4.0]),
    )
    return builder._resolve_num_divisions(operator, time)


class TestPlaquetteTrotterAutomaticDivisions:
    """Automatic and manual division-count behavior."""

    def test_computed_count_is_the_smallest_count_meeting_the_bound(self):
        """The selected step size satisfies the requested energy-error bound without over-rounding."""
        time = 0.75
        target_accuracy = 0.02
        divisions = _resolve_divisions(time=time, target_accuracy=target_accuracy)
        # A long probe time makes integer rounding of the division count negligible.
        probe_time = 1e9
        error_constant = (_resolve_divisions(time=probe_time, target_accuracy=1.0) / probe_time) ** 2

        assert error_constant * (time / divisions) ** 2 <= target_accuracy
        assert error_constant * (time / (divisions - 1)) ** 2 > target_accuracy

    def test_explicit_count_is_a_lower_bound_on_auto_sizing(self):
        """When both controls are supplied, the larger division count wins."""
        automatic = _resolve_divisions(time=1.0, target_accuracy=0.1)

        assert _resolve_divisions(time=1.0, target_accuracy=0.1, num_divisions=automatic - 1) == automatic
        assert _resolve_divisions(time=1.0, target_accuracy=0.1, num_divisions=automatic + 3) == automatic + 3

    @pytest.mark.parametrize(
        ("time_scale", "accuracy_scale", "expected_scale"),
        [(4.0, 1.0, 4.0), (1.0, 0.25, 2.0)],
    )
    def test_computed_count_has_second_order_scaling(self, time_scale, accuracy_scale, expected_scale):
        """The count scales as time/sqrt(accuracy), as required by the second-order bound."""
        baseline = _resolve_divisions(time=1.0, target_accuracy=1e-3)
        scaled = _resolve_divisions(time=time_scale, target_accuracy=accuracy_scale * 1e-3)

        assert scaled == pytest.approx(expected_scale * baseline, rel=0.02)


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron_all(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Return the Kronecker product in sequence order."""
    result = np.array([[1.0 + 0j]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def _expand_groups(
    groups: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm],
) -> list[ExponentiatedPauliTerm]:
    """Expand equal-angle groups into plain Pauli exponentials."""
    expanded: list[ExponentiatedPauliTerm] = []
    for group in groups:
        if isinstance(group, ExponentiatedPauliTerm):
            expanded.append(group)
        else:
            expanded.extend(ExponentiatedPauliTerm(term, group.angle) for term in group.pauli_terms)
    return expanded


def _expand_terms(
    terms: Sequence[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm],
) -> list[ExponentiatedPauliTerm]:
    """Expand batches and conjugated blocks into execution-order leaves."""
    expanded: list[ExponentiatedPauliTerm] = []
    for term in terms:
        if isinstance(term, ConjugatedExponentiatedPauliTerm):
            within = _expand_groups(term.within_terms)
            expanded.extend(within)
            expanded.extend(_expand_groups(term.apply_terms))
            expanded.extend(ExponentiatedPauliTerm(leaf.pauli_term, -leaf.angle) for leaf in reversed(within))
        else:
            expanded.extend(_expand_groups([term]))
    return expanded


class TestPlaquetteTrotterDecomposition:
    """Classical structure and matrix-level correctness of the emitted formula."""

    @pytest.mark.parametrize("shape", [(4, 4), (4, 6), (6, 8)])
    def test_sections_partition_the_periodic_lattice(self, shape):
        """The two vertex-disjoint sections cover each periodic bond exactly once."""
        width, height = shape
        sections = PlaquetteTrotter._plaquette_sections(width, height)
        bonds: list[frozenset[int]] = []
        for section in sections:
            seen: set[int] = set()
            for cycle in section:
                assert not seen.intersection(cycle)
                seen.update(cycle)
                bonds.extend(frozenset((cycle[index], cycle[(index + 1) % 4])) for index in range(4))

        assert len(bonds) == len(set(bonds)) == 2 * width * height

    @pytest.mark.parametrize("sites", [(0, 1, 4, 3), (4, 3, 0, 1)])
    def test_single_plaquette_decomposition_is_exact(self, sites):
        """The emitted basis change and phases reproduce one hopping cycle."""
        num_modes = 6
        time = 0.37
        layer = PlaquetteTrotter()._hop_layer([sites], num_sites=num_modes, hopping=1.0, time=time)
        assert layer is not None
        spin_up_terms = [term for term in _expand_terms([layer]) if all(qubit < num_modes for qubit in term.pauli_term)]

        lower = np.array([[0, 1], [0, 0]], dtype=complex)
        modes = [
            _kron_all([_PAULI["Z"]] * mode + [lower] + [_PAULI["I"]] * (num_modes - mode - 1))
            for mode in range(num_modes)
        ]
        hamiltonian = np.zeros((2**num_modes, 2**num_modes), dtype=complex)
        for index in range(4):
            left, right = sites[index], sites[(index + 1) % 4]
            hamiltonian -= modes[left].conj().T @ modes[right] + modes[right].conj().T @ modes[left]
        expected = scipy.linalg.expm(-1j * time * hamiltonian)

        actual = np.eye(2**num_modes, dtype=complex)
        for term in spin_up_terms:
            operator = _kron_all([_PAULI[term.pauli_term.get(mode, "I")] for mode in range(num_modes)])
            actual = scipy.linalg.expm(-1j * term.angle * operator) @ actual

        assert np.allclose(actual, expected, atol=1e-10)

    def test_repeated_formula_has_boundary_and_four_factor_body(self):
        """Campbell's rewrite emits one boundary layer around the repeated P-G-P-D body.

        The D layer is the interaction, whose equal-angle families are phased through
        Hamming weight registers: one batch per family, plus the identity factor
        carrying the Jordan-Wigner constant. Bounding ``max_batch`` would split those
        families into more batches without changing the three hopping conjugations.
        """
        operator = _hubbard_operator(4, 4, interaction=4.0)
        container = (
            PlaquetteTrotter(
                lattice_width=4,
                lattice_height=4,
                time=0.15,
                num_divisions=3,
            )
            .run(operator)
            .get_container()
        )

        assert container.step_reps == 3
        assert container.conjugating_terms
        hopping = [term for term in container.step_terms if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        assert len(hopping) == 3
        assert container.step_terms[:3] == hopping, "the hopping tilings must come first"
        assert all(
            isinstance(term, ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm)
            for term in container.step_terms[3:]
        ), "the interaction layer holds only plain and batched factors"


PauliGroup = ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm


def _qsharp_groups(groups: Sequence[PauliGroup]) -> str:
    """Serialize sparse Pauli groups as a Q# array."""
    serialized: list[str] = []
    for group in groups:
        pauli_terms = [group.pauli_term] if isinstance(group, ExponentiatedPauliTerm) else group.pauli_terms
        index_expression = "[" + ", ".join(str(list(term)) for term in pauli_terms) + "]"
        operations = [[f"Pauli{axis}" for axis in term.values()] for term in pauli_terms]
        operation_expression = "[" + ", ".join("[" + ", ".join(term) + "]" for term in operations) + "]"
        serialized.append(
            "new QDKChemistry.Utils.PauliExp.SparsePauliExpGroupParams { "
            f"pauliIndices = {index_expression}, pauliOps = {operation_expression}, angle = {group.angle!r} }}"
        )
    return "[" + ", ".join(serialized) + "]"


def _dump_container_state(
    container: PauliProductFormulaContainer,
    *,
    basis: int = 0,
    hadamards: Sequence[int] = (),
) -> np.ndarray:
    """Apply a product formula in an isolated context and return its statevector."""
    blocks: list[str] = []
    for block in container.step_terms:
        conjugated = isinstance(block, ConjugatedExponentiatedPauliTerm)
        within_groups = block.within_terms if conjugated else []
        apply_groups = block.apply_terms if conjugated else [block]
        blocks.append(
            "new QDKChemistry.Utils.PauliExp.ConjugatedSparsePauliExpParams { "
            f"withinGroups = {_qsharp_groups(within_groups)}, applyGroups = {_qsharp_groups(apply_groups)} }}"
        )
    params = (
        "new QDKChemistry.Utils.PauliExp.StructuredSparseRepPauliExpParams { "
        f"conjugatingGroups = {_qsharp_groups(container.conjugating_terms)}, "
        f"stepBlocks = [{', '.join(blocks)}], repetitions = {container.step_reps} }}"
    )

    context = create_qsharp_context()
    num_qubits = container.num_qubits
    context.eval(f"use qs = Qubit[{num_qubits}];")
    flipped = [qubit for qubit in range(num_qubits) if basis & (1 << (num_qubits - qubit - 1))]
    if flipped:
        context.eval(" ".join(f"X(qs[{qubit}]);" for qubit in flipped))
    if hadamards:
        context.eval(" ".join(f"H(qs[{qubit}]);" for qubit in hadamards))
    context.eval(f"QDKChemistry.Utils.PauliExp.StructuredSparseRepPauliExp({params}, qs);")
    state = np.asarray(context.dump_machine().as_dense_state(), dtype=complex)
    context.eval("ResetAll(qs);")
    return state


def _assert_same_state(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-9) -> None:
    """Assert statevector equality up to a physically irrelevant global phase."""
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) > atol else 1.0
    assert np.allclose(actual, phase * expected, atol=atol)


_HWP_BREAK_EVEN = 8


class TestHammingWeightPhasing:
    """HWP must preserve the state while reducing synthesized rotations."""

    @staticmethod
    def _containers() -> tuple[PauliProductFormulaContainer, PauliProductFormulaContainer]:
        raw_terms = [ExponentiatedPauliTerm({qubit: "Z"}, 0.31) for qubit in range(_HWP_BREAK_EVEN)]
        batched_terms = [BatchedExponentiatedPauliTerm([term.pauli_term for term in raw_terms], 0.31)]
        return (
            PauliProductFormulaContainer(batched_terms, step_reps=1, num_qubits=_HWP_BREAK_EVEN),
            PauliProductFormulaContainer(raw_terms, step_reps=1, num_qubits=_HWP_BREAK_EVEN),
        )

    def test_hwp_matches_loose_terms_on_the_resulting_state(self):
        """The HWP batch and its individual rotations produce the same superposition."""
        batched, loose = self._containers()
        hadamards = list(range(_HWP_BREAK_EVEN))

        _assert_same_state(
            _dump_container_state(batched, hadamards=hadamards),
            _dump_container_state(loose, hadamards=hadamards),
        )

    def test_hwp_reduces_the_rotation_count(self):
        """At the break-even batch size, HWP uses fewer rotations than loose exponentials."""
        batched, loose = self._containers()
        mapper = create("controlled_circuit_mapper", "pauli_sequence", control_indices=[0])
        counts = [
            mapper.run(UnitaryRepresentation(container)).estimate()["logicalCounts"]["rotationCount"]
            for container in (batched, loose)
        ]

        assert counts[0] < counts[1]

    def test_unbounded_batches_merge_the_interaction_families(self):
        """Without a cap each equal-angle family occupies one register, not several.

        A batch of ``m`` disjoint equal-angle terms costs ``ceil(log2(m+1))`` rotations,
        so splitting one family into ``k`` chunks pays that cost ``k`` times over. The
        default is therefore unbounded. Campbell's Table II instead budgets ``L^2/2``
        ancillas (arXiv:2012.09238v4), which is what ``max_batch`` reproduces: the same
        operator, fewer qubits, more rotations.
        """
        side = 4
        operator = _hubbard_operator(side, side, interaction=8.0)
        shapes = {}
        for cap in (0, side * side // 2):
            container = (
                PlaquetteTrotter(
                    lattice_width=side,
                    lattice_height=side,
                    time=0.05,
                    num_divisions=1,
                    max_batch=cap,
                )
                .run(operator)
                .get_container()
            )
            batches = [term for term in container.step_terms if isinstance(term, BatchedExponentiatedPauliTerm)]
            shapes[cap] = sorted(len(term.pauli_terms) for term in batches)

        unbounded, bounded = shapes[0], shapes[side * side // 2]
        assert sum(unbounded) == sum(bounded), "capping must not drop or add factors"
        assert len(unbounded) < len(bounded), "the unbounded form must use fewer registers"
        assert max(bounded) <= side * side // 2, "a bounded batch must respect the cap"


class TestPlaquetteTrotterBasisStates:
    """The emitted Q# plaquette step must reproduce exact basis-state evolution."""

    def test_resulting_basis_states_match_exact_hopping_evolution(self):
        """Vacuum, one-particle, and two-spin inputs match exp(-iHt) on one plaquette."""
        side = 2
        time = 0.29
        operator = _hubbard_operator(side, side, interaction=0.0)
        container = (
            PlaquetteTrotter(
                lattice_width=side,
                lattice_height=side,
                time=time,
                num_divisions=1,
            )
            .run(operator)
            .get_container()
        )
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        exact = scipy.linalg.expm(-1j * time * pauli_to_dense_matrix(list(labels), list(coefficients)))
        basis_states = [0, 1 << (operator.num_qubits - 1), (1 << (operator.num_qubits - 1)) | (1 << 3)]

        for basis in basis_states:
            _assert_same_state(_dump_container_state(container, basis=basis), exact[:, basis])


class TestPlaquetteTrotterPhaseEstimation:
    """End-to-end plaquette Trotterization under iterative phase estimation."""

    _SIDE = 2
    _ENERGY = -8.0
    _TIME = math.pi / 32.0
    _PHASE = 0.125
    _BITS: ClassVar[list[int]] = [0, 0, 1, 0]

    @classmethod
    def _operator(cls) -> QubitOperator:
        """Return the hopping-only single-plaquette Hamiltonian."""
        return _hubbard_operator(cls._SIDE, cls._SIDE, interaction=0.0)

    @classmethod
    def _state_preparation(cls, operator: QubitOperator) -> Circuit:
        """Prepare an exact ground-state eigenvector for phase estimation."""
        labels, coefficients = zip(*operator.get_real_coefficients(tolerance=1e-14), strict=True)
        values, vectors = np.linalg.eigh(pauli_to_dense_matrix(list(labels), list(coefficients)))
        assert np.isclose(values[0], cls._ENERGY)
        state = np.real(vectors[:, 0])
        state /= np.linalg.norm(state)
        num_qubits = operator.num_qubits
        params = {
            "rowMap": list(range(num_qubits - 1, -1, -1)),
            "stateVector": state.tolist(),
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
                parameter=params,
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params),
        )

    def test_qpe_recovers_the_exact_eigenvalue(self):
        """The full plaquette builder, controlled mapper, and IQPE recover the known phase."""
        operator = self._operator()
        iqpe = IterativePhaseEstimation(shots_per_bit=25)
        iqpe.settings().set(
            "qpe_circuit_builder",
            AlgorithmRef(
                "qpe_circuit_builder",
                "qdk_iterative",
                num_bits=4,
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
                unitary_builder=AlgorithmRef(
                    "hamiltonian_unitary_builder",
                    "plaquette",
                    time=self._TIME,
                    num_divisions=1,
                    order=2,
                    lattice_width=self._SIDE,
                    lattice_height=self._SIDE,
                ),
            ),
        )
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
        )

        result = iqpe.run(qubit_hamiltonian=operator, state_preparation=self._state_preparation(operator))

        assert list(result.bits_msb_first or []) == self._BITS
        assert np.isclose(result.phase_fraction, self._PHASE, atol=1e-9)
        assert np.isclose(result.raw_energy, self._ENERGY, atol=1e-9)
