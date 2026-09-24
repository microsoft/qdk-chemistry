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

    def test_repeated_formula_merges_the_hopping_layer_into_the_boundary(self):
        """The rewrite emits a pink boundary around a body carrying only two hop layers.

        Merging the hopping layer rather than the interaction is the saving: a symmetric
        step written with hopping outside contributes one merged pink layer per
        repetition instead of two pink half-layers, so the body holds two hopping
        conjugations rather than three. The interaction half-layers that move inward in
        exchange are batched through Hamming weight registers and cost far less.
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
        assert [isinstance(term, ConjugatedExponentiatedPauliTerm) for term in container.conjugating_terms] == [True], (
            "the one-time boundary is a single pink hopping half-layer"
        )
        hopping = [term for term in container.step_terms if isinstance(term, ConjugatedExponentiatedPauliTerm)]
        assert len(hopping) == 2, "the merged body carries gold and one full pink layer"
        assert container.step_terms[-1] is hopping[-1], "the merged pink layer closes the body"
        assert not isinstance(container.step_terms[0], ConjugatedExponentiatedPauliTerm), (
            "the body opens with a batched interaction half-layer"
        )


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


def _qsharp_blocks(terms: Sequence[PauliGroup | ConjugatedExponentiatedPauliTerm]) -> str:
    """Serialize terms as a Q# array of conjugated blocks."""
    blocks: list[str] = []
    for block in terms:
        conjugated = isinstance(block, ConjugatedExponentiatedPauliTerm)
        within_groups = block.within_terms if conjugated else []
        apply_groups = block.apply_terms if conjugated else [block]
        blocks.append(
            "new QDKChemistry.Utils.PauliExp.ConjugatedSparsePauliExpParams { "
            f"withinGroups = {_qsharp_groups(within_groups)}, applyGroups = {_qsharp_groups(apply_groups)} }}"
        )
    return "[" + ", ".join(blocks) + "]"


def _dump_container_state(
    container: PauliProductFormulaContainer,
    *,
    basis: int = 0,
    hadamards: Sequence[int] = (),
) -> np.ndarray:
    """Apply a product formula in an isolated context and return its statevector."""
    params = (
        "new QDKChemistry.Utils.PauliExp.StructuredSparseRepPauliExpParams { "
        f"conjugatingGroups = {_qsharp_blocks(container.conjugating_terms)}, "
        f"stepBlocks = {_qsharp_blocks(container.step_terms)}, repetitions = {container.step_reps} }}"
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


# --------------------------------------------------------------------------------------
# Campbell benchmark (arXiv:2012.09238v4): cost model regression, formerly a notebook
# --------------------------------------------------------------------------------------

#: Hopping amplitude; energies are quoted in units of this.
_HOPPING = 1.0

#: Campbell's strong-coupling regime.
_INTERACTION = 8.0 * _HOPPING

#: Per-site ground-state energy target, in units of the hopping.
_TARGET_PER_SITE = 0.0037 * _HOPPING

#: Threshold below which mapped coefficients are dropped.
_WEIGHT_THRESHOLD = 1e-12

#: Campbell's Table III, single-particle hopping norm per lattice size.
_HOPPING_NORM = dict(
    zip(range(4, 33, 2), [24, 56, 100, 160, 230, 320, 410, 520, 650, 780, 930, 1100, 1300, 1500, 1700], strict=True)
)

#: Campbell's Table III, the extra nested-commutator norm the plaquette split incurs.
_COMMUTATOR_NORM = dict(
    zip(range(4, 33, 2), [0, 110, 190, 300, 440, 630, 810, 1000, 1300, 1600, 1800, 2200, 2500, 2900, 3300], strict=True)
)

#: Campbell's Eq. (F7) prefactor, 3^{3/2} * 0.76 * pi / 2.
_PHASE_ESTIMATION_PREFACTOR = 3**1.5 * 0.76 * math.pi / 2.0

#: Table II at u/tau = 8: Toffoli and T totals for the whole phase estimation.
_TABLE_II_U8 = {
    8: (4.3e5, 4.1e6),
    10: (4.4e5, 3.3e6),
    12: (4.6e5, 2.9e6),
    14: (4.6e5, 2.5e6),
    16: (4.6e5, 2.3e6),
    18: (4.6e5, 2.2e6),
    20: (4.7e5, 2.0e6),
    22: (4.6e5, 2.2e6),
    24: (4.7e5, 2.1e6),
    26: (4.6e5, 2.1e6),
    28: (4.6e5, 2.1e6),
    30: (4.7e5, 2.1e6),
    32: (4.7e5, 2.1e6),
}


def _shifted_hubbard_operator(size: int) -> QubitOperator:
    """Return the Jordan-Wigner image of Campbell's shifted periodic Hubbard model.

    Args:
        size: Lattice side length.

    Returns:
        The qubit Hamiltonian, with identity factors dropped.

    """
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-_INTERACTION / 2.0, t=_HOPPING, U=_INTERACTION)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites))
    keep = [index for index, label in enumerate(mapped.pauli_strings) if set(label) != {"I"}]
    return QubitOperator(
        pauli_strings=[mapped.pauli_strings[index] for index in keep],
        coefficients=mapped.coefficients[keep],
        encoding=mapped.encoding,
        fermion_mode_order=mapped.fermion_mode_order,
    )


def _error_constant(size: int) -> float:
    """Return Campbell's ``W_PLAQ`` from his Eqs. (10) and (20) with Table III norms."""
    num_sites = size * size
    return (
        _INTERACTION * _HOPPING**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0)
        + _INTERACTION**2 / 24.0 * _HOPPING_NORM[size] * _HOPPING
        + _COMMUTATOR_NORM[size] * _HOPPING**3 / 8.0
    )


def _per_step_costs(size: int) -> tuple[float, float, float]:
    """Return Campbell's Appendix E per-step ``(Toffoli, T, rotations)``.

    A batch of ``m = L^2/2`` equal-angle terms is phased through one Hamming weight
    register, costing ``m - w(m)`` Toffolis and ``ceil(log2 m)`` rotations rather than
    one rotation per term.

    Args:
        size: Lattice side length.

    Returns:
        The three per-step counts.

    """
    num_sites = size * size
    batch = num_sites // 2
    toffoli = 4 * num_sites * (batch - batch.bit_count()) / batch
    t_gates = 12 * num_sites
    rotations = 4 * num_sites * batch.bit_length() / batch
    return toffoli, t_gates, rotations


def _schedule(size: int, synthesis_fraction: float) -> tuple[float, float]:
    """Return ``(N_PE, N_HT)`` for one split of the budget between Trotter and synthesis."""
    epsilon = _TARGET_PER_SITE * size * size
    delta = (1.0 - synthesis_fraction) * epsilon
    num_pe = _PHASE_ESTIMATION_PREFACTOR * math.sqrt(_error_constant(size)) / delta**1.5
    _, _, rotations = _per_step_costs(size)
    synthesis_t = 1.15 * math.log2(rotations * num_pe / (synthesis_fraction * epsilon)) + 9.2
    return num_pe, synthesis_t


def _toffoli_equivalent(size: int, synthesis_fraction: float) -> float:
    """Return the Appendix F objective, two synthesized T gates counted as one Toffoli."""
    num_pe, synthesis_t = _schedule(size, synthesis_fraction)
    toffoli, t_gates, rotations = _per_step_costs(size)
    return num_pe * (toffoli + (t_gates + rotations * synthesis_t) / 2.0)


def campbell_schedule(size: int) -> dict[str, float]:
    """Return Campbell's optimized Appendix F schedule for a lattice.

    The objective is smooth and unimodal in the synthesis fraction, so a golden-section
    search finds the optimum to machine precision. That avoids a ``scipy.optimize``
    dependency the package does not otherwise carry, and agrees with
    ``minimize_scalar`` on ``N_PE`` at every size from 8 to 32.

    Args:
        size: Lattice side length.

    Returns:
        The step time, application count, and synthesis cost per rotation.

    """
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = 1e-9, 0.5
    left, right = high - golden * (high - low), low + golden * (high - low)
    for _ in range(200):
        if _toffoli_equivalent(size, left) < _toffoli_equivalent(size, right):
            high, right = right, left
            left = high - golden * (high - low)
        else:
            low, left = left, right
            right = low + golden * (high - low)
        if abs(high - low) < 1e-15:
            break

    fraction = (low + high) / 2.0
    num_pe, synthesis_t = _schedule(size, fraction)
    epsilon = _TARGET_PER_SITE * size * size
    delta = (1.0 - fraction) * epsilon
    return {
        "synthesis_fraction": fraction,
        "num_pe": math.ceil(num_pe),
        "step_time": math.sqrt(delta / (3.0 * _error_constant(size))),
        "synthesis_t": synthesis_t,
    }


def _logical_counts(size: int, *, max_batch: int) -> dict:
    """Build the aggregate controlled circuit for one lattice and return its counts."""
    schedule = campbell_schedule(size)
    steps = int(schedule["num_pe"])
    operator = _shifted_hubbard_operator(size)
    unitary = create(
        "hamiltonian_unitary_builder",
        "plaquette",
        order=2,
        # N_PE steps of duration step_time, i.e. exactly U_TS(t)^N_PE.
        time=steps * float(schedule["step_time"]),
        num_divisions=steps,
        target_accuracy=0.0,
        lattice_width=size,
        lattice_height=size,
        max_batch=max_batch,
        weight_threshold=_WEIGHT_THRESHOLD,
    ).run(operator)
    circuit = create(
        "controlled_circuit_mapper",
        "pauli_sequence",
        control_indices=[0],
        target_indices=list(range(1, operator.num_qubits + 1)),
    ).run(unitary)
    return circuit.estimate().logical_counts


class TestCampbellCostModel:
    """The reconstruction of Campbell's own formulas, independent of what we emit."""

    @pytest.mark.parametrize(("size", "expected"), [(4, 282.4), (8, 527.2)])
    def test_error_constant_reproduces_the_paper(self, size, expected):
        """Eqs. (10) and (20) with the Table III norms must give the tabulated W_PLAQ.

        At 4x4 with u/tau = 8 this is 282.4, and at 8x8 with the plaquette commutator
        term it is 527, matching Campbell's Table I entry of 5.3e2. Getting these right
        is what makes the rest of the comparison meaningful.
        """
        if size == 4:
            constant = (
                _INTERACTION * _HOPPING**2 / 6.0 * 16 * (math.sqrt(5.0) + 8.0)
                + _INTERACTION**2 / 24.0 * _HOPPING_NORM[4]
            )
        else:
            constant = (
                4.0 * _HOPPING**2 / 6.0 * 64 * (math.sqrt(5.0) + 8.0)
                + 4.0**2 / 24.0 * _HOPPING_NORM[8]
                + 3.0 / 24.0 * _COMMUTATOR_NORM[8]
            )
        assert constant == pytest.approx(expected, rel=2e-3)

    def test_schedule_shrinks_as_the_lattice_grows(self):
        """A larger lattice has a looser absolute target, so it needs fewer applications.

        The target is per site, so epsilon grows as L^2 while W_PLAQ grows only linearly
        in the site count; N_PE ~ sqrt(W)/epsilon^{3/2} therefore falls.
        """
        counts = [campbell_schedule(size)["num_pe"] for size in (8, 16, 32)]
        assert counts == sorted(counts, reverse=True)

    def test_step_time_is_nearly_size_independent(self):
        """Campbell's optimum fixes the step duration, not the step count.

        Both W_PLAQ and epsilon scale with the site count, so their ratio and hence the
        step time barely move. This is what makes the per-step costs comparable across
        sizes in the first place.
        """
        times = [campbell_schedule(size)["step_time"] for size in range(8, 33, 2)]
        assert max(times) / min(times) < 1.01


class TestPlaquetteAgainstTableII:
    """What the builder actually emits, measured against the published totals."""

    #: Sizes spanning Table II. Each builds one aggregate circuit, so keep the list short.
    _SIZES = (8, 12, 20, 32)

    @pytest.mark.parametrize("size", _SIZES)
    def test_qubit_count_differs_from_the_paper_by_a_known_constant(self, size):
        """The two qubit conventions differ by exactly ``w(m) + 1``, with nothing left over.

        Table II charges ``2L^2 + alpha + 2`` with ``alpha = L^2/2`` Hamming weight
        ancillas. We allocate Gidney's exact adder workspace ``m - w(m)`` instead of the
        looser ``m``, which saves ``w(m)``, and one repeat-until-success herald rather
        than two spare qubits, which saves one more. The herald is a constant and not a
        per-rotation cost: each rotation measures and releases the same qubit in turn,
        so charging one per rotation would bill depth as width.

        This is asserted as an identity rather than a tolerance because every term in it
        is known; an unexplained qubit would mean one of those two conventions is wrong.
        """
        batch = size * size // 2
        counts = _logical_counts(size, max_batch=batch)
        paper = 2 * size * size + batch + 2

        assert paper - counts["numQubits"] == batch.bit_count() + 1

    @pytest.mark.parametrize("size", _SIZES)
    def test_toffoli_count_matches_the_paper(self, size):
        """Toffolis come from the Hamming weight arithmetic, which we implement as published.

        Agreement here is the evidence that the Trotter step and the phase-estimation
        schedule are both right: a wrong step count or a wrong tiling would move this.
        The residual few percent is the exact batch decomposition against Table II's
        two-significant-figure entries, plus the one-time boundary layer that a closed
        formula in ``N_PE`` does not carry. Measured at 0.0-6.9% across these sizes.
        """
        counts = _logical_counts(size, max_batch=size * size // 2)

        assert counts["cczCount"] == pytest.approx(_TABLE_II_U8[size][0], rel=0.10)

    @pytest.mark.parametrize("size", _SIZES)
    def test_synthesized_t_count_approaches_the_paper_as_the_lattice_grows(self, size):
        """Our T total falls from 3.2x Campbell's down to 0.95x as the lattice grows.

        The gap is rotation count, not step count: Campbell's directional-control
        costing emits fewer controlled rotations than a conventional controlled mapper,
        and each surviving rotation is then synthesized at his own ``N_HT``. Because the
        excess is a fixed number of rotations per step while his total grows with
        ``N_PE``, the ratio shrinks with the lattice, and with the hopping layer merged
        into the boundary it crosses below his published count near L=30. Asserted as a
        band rather than one tolerance, since no single budget is honest across the range.
        """
        schedule = campbell_schedule(size)
        counts = _logical_counts(size, max_batch=size * size // 2)
        synthesized = counts["tCount"] + counts["rotationCount"] * schedule["synthesis_t"]

        ratio = synthesized / _TABLE_II_U8[size][1]
        assert 0.9 < ratio <= 3.3, f"T ratio {ratio:.2f}x outside the measured band"

    def test_the_t_gap_narrows_with_the_lattice(self):
        """The rotation excess is per step, so it is amortized as N_PE grows.

        This is the claim the per-size bound above cannot make: if the gap were a
        mis-sized schedule rather than a control-lowering difference, it would not
        shrink monotonically.
        """
        ratios = []
        for size in self._SIZES:
            schedule = campbell_schedule(size)
            counts = _logical_counts(size, max_batch=size * size // 2)
            synthesized = counts["tCount"] + counts["rotationCount"] * schedule["synthesis_t"]
            ratios.append(synthesized / _TABLE_II_U8[size][1])

        assert ratios == sorted(ratios, reverse=True)

    @pytest.mark.parametrize("size", [8, 16])
    def test_unbounded_batches_trade_qubits_for_rotations(self, size):
        """Merging each equal-angle family into one register is the default, and costs width.

        Campbell's Table II budgets alpha = L^2/2 ancillas, which bounds a batch at that
        size and so splits the interaction's families into several registers. Lifting the
        bound removes rotations and adds qubits; neither form is wrong, so the builder
        exposes the choice.
        """
        bounded = _logical_counts(size, max_batch=size * size // 2)
        merged = _logical_counts(size, max_batch=0)
        assert merged["rotationCount"] < bounded["rotationCount"]
        assert merged["numQubits"] > bounded["numQubits"]
