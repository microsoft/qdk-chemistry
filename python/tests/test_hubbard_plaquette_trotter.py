"""Tests for the Hubbard plaquette Trotter builder and its Q# lowering."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
import scipy.linalg
from qdk import Result
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.hubbard_plaquette_trotter import (
    HubbardPlaquetteTrotter,
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
from qdk_chemistry.data.circuit import PhaseGradient, QsharpFactoryData
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _lattice_operator(width: int, height: int) -> QubitOperator:
    """Return a periodic square lattice as a lattice-backed qubit operator."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    return QubitOperator(container=LatticeContainer(lattice))


def _reference_hamiltonian(width: int, height: int, *, t: float, u: float) -> np.ndarray:
    """Return the dense Hamiltonian from an independent Jordan-Wigner mapping."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-0.5 * u, t=t, U=u)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * width * height))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    dense = pauli_to_dense_matrix(list(labels), list(coefficients))
    return dense + 0.25 * u * width * height * np.eye(dense.shape[0])


def _plaquette_parameters(container):
    """Return the Q# parameter struct for a plaquette container."""
    return QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
        width=container.width,
        height=container.height,
        interactionAngle=container.interaction_angle,
        hoppingAngle=container.hopping_angle,
        repetitions=container.step_reps,
    )


def _evolution_circuit(
    width: int,
    height: int,
    *,
    time: float,
    t: float = 1.0,
    u: float = 0.0,
    num_divisions: int = 1,
):
    """Return the uncontrolled plaquette evolution as a Q# callable."""
    builder = HubbardPlaquetteTrotter(order=2, time=time, t=t, u=u, num_divisions=num_divisions, target_accuracy=0.0)
    container = builder.run(_lattice_operator(width, height)).get_container()
    return QSHARP_UTILS.HubbardPlaquette.MakeRepPlaquetteExpOp(_plaquette_parameters(container))


def _applied_state(operation, state: np.ndarray) -> np.ndarray:
    """Return the state the operation produces from *state*."""
    num_qubits = round(math.log2(len(state)))
    amplitudes = [float(np.real(a)) for a in state]
    return np.asarray(
        dump_operation_on_state(operation, num_qubits, amplitudes, context=get_qsharp_context()),
        dtype=complex,
    )


def _random_state(num_qubits: int, seed: int) -> np.ndarray:
    """Return a normalized random state vector with real amplitudes.

    The simulator takes real amplitudes only. A real vector is still a general state
    for this purpose: it has support on every basis state, so nothing in the evolution
    goes unexercised.
    """
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**num_qubits)
    return (state / np.linalg.norm(state)).astype(complex)


def _infidelity(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return one minus the overlap magnitude, which ignores global phase."""
    return 1.0 - abs(np.vdot(expected, actual))


class TestHubbardPlaquetteContainer:
    """The emitted representation carries geometry and angles, and nothing that scales."""

    def test_builder_emits_a_plaquette_container(self):
        """The builder's representation is the plaquette container."""
        unitary = HubbardPlaquetteTrotter(order=2, time=0.1, t=1.0, u=4.0, num_divisions=3, target_accuracy=0.0).run(
            _lattice_operator(2, 2)
        )
        container = unitary.get_container()

        assert isinstance(container, HubbardPlaquetteContainer)
        assert (container.width, container.height) == (2, 2)
        assert container.num_qubits == 8
        assert container.step_reps == 3

    def test_representation_size_is_independent_of_the_lattice(self):
        """The payload is a fixed set of scalars, so it does not grow with the lattice."""
        payloads = [
            HubbardPlaquetteTrotter(order=2, time=0.1, t=1.0, u=4.0, num_divisions=1)
            .run(_lattice_operator(side, side))
            .get_container()
            .to_json()
            for side in (2, 4, 6)
        ]

        assert {frozenset(payload) for payload in payloads} == {frozenset(payloads[0])}

    def test_container_round_trips_through_json(self):
        """Serialization preserves every field the Q# lowering reads."""
        container = (
            HubbardPlaquetteTrotter(order=2, time=0.3, t=1.0, u=8.0, num_divisions=2)
            .run(_lattice_operator(4, 4))
            .get_container()
        )

        assert HubbardPlaquetteContainer.from_json(container.to_json()).to_json() == container.to_json()

    @pytest.mark.parametrize("order", [1, 3, 4])
    def test_rejects_unsupported_order(self, order):
        """The plaquette error bound and decomposition are second order only."""
        with pytest.raises(ValueError, match="order 2 only"):
            HubbardPlaquetteTrotter(order=order)

    def test_rejects_a_lattice_that_does_not_tile(self):
        """Open boundaries leave bonds the periodic plaquette tiling cannot cover."""
        open_lattice = QubitOperator(
            container=LatticeContainer(LatticeGraph.square(4, 4, periodic_x=False, periodic_y=False))
        )
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, t=1.0, u=4.0, num_divisions=1)

        with pytest.raises(ValueError, match="bond graph does not match"):
            builder.run(open_lattice)

    def test_rejects_a_mapped_qubit_operator(self):
        """The tiling needs the lattice structure, which a mapped operator discards."""
        lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
        mapped = create("qubit_mapper").run(
            create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0),
            mapping=MajoranaMapping.jordan_wigner(8),
        )
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, t=1.0, u=4.0, num_divisions=1)

        with pytest.raises(TypeError, match="LatticeContainer"):
            builder.run(mapped)


class TestPlaquetteTiling:
    """The Q# tilings must cover the lattice the way Campbell's decomposition requires."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (4, 6), (6, 6)])
    def test_tilings_cover_every_bond_exactly_once(self, width, height):
        """Together the pink and gold tilings reproduce the periodic lattice's bonds."""
        sites = width * height
        cycles = [
            [int(site) for site in cycle]
            for pink in ("true", "false")
            for cycle in get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({width},{height},{pink})"
            )
        ]
        tiled = [
            frozenset((cycle[index], cycle[(index + 1) % 4]))
            for cycle in cycles
            if cycle[0] < sites
            for index in range(4)
        ]
        expected = {
            frozenset((row * width + col, row * width + (col + 1) % width))
            for row in range(height)
            for col in range(width)
        } | {
            frozenset((row * width + col, ((row + 1) % height) * width + col))
            for row in range(height)
            for col in range(width)
        }

        assert len(tiled) == len(set(tiled)), "a bond may not appear in both tilings"
        assert set(tiled) == expected

    @pytest.mark.parametrize(("width", "height"), [(4, 4), (6, 6), (8, 8)])
    def test_each_tiling_is_vertex_disjoint(self, width, height):
        """Plaquettes within a tiling share no site, which is what lets them run together."""
        for pink in ("true", "false"):
            cycles = get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({width},{height},{pink})"
            )
            seen: set[int] = set()
            for cycle in cycles:
                sites = {int(site) for site in cycle}
                assert not seen & sites, "plaquettes within a tiling must be vertex disjoint"
                seen |= sites

    @pytest.mark.parametrize("side", [4, 6, 8])
    def test_routing_makes_each_plaquette_local(self, side):
        """After routing, each plaquette occupies four adjacent modes, whatever the size.

        The four land interleaved rather than in cycle order, which is what makes all
        three FFFT butterflies act on adjacent positions.
        """
        num_modes = 2 * side * side
        gold = [
            [int(site) for site in cycle]
            for cycle in get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({side},{side},false)"
            )
        ]
        swaps = [
            int(position)
            for position in get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.RoutingSwaps({gold}, {num_modes})"
            )
        ]

        # Replay the adjacent exchanges the network emits, which is what the circuit does.
        routed = list(range(num_modes))
        for position in swaps:
            assert 0 <= position < num_modes - 1, "every exchange must be adjacent"
            routed[position], routed[position + 1] = routed[position + 1], routed[position]

        assert sorted(routed) == list(range(num_modes)), "routing must be a permutation"
        for index, cycle in enumerate(gold):
            block = routed[4 * index : 4 * index + 4]
            interleaved = [cycle[0], cycle[2], cycle[1], cycle[3]]
            assert block == interleaved, "each plaquette must land contiguous and interleaved"

            # Every butterfly must act on adjacent positions, which is the invariant
            # `TwoModeFFFT` asserts: the two diagonals, then the surviving middle pair.
            for left, right in ((cycle[0], cycle[2]), (cycle[1], cycle[3]), (cycle[2], cycle[1])):
                assert abs(block.index(left) - block.index(right)) == 1, "butterflies must be local"


_FFFT = "QDKChemistry.Utils.HubbardPlaquette.TwoModeFFFT"
_FFFT_MODES = 4


def _occupied(*modes: int) -> np.ndarray:
    """Return the occupation basis state with *modes* filled.

    ``dump_operation_on_state`` numbers basis states big-endian, so qubit 0 is the
    most significant bit.
    """
    state = np.zeros(2**_FFFT_MODES)
    state[sum(1 << (_FFFT_MODES - 1 - mode) for mode in modes)] = 1.0
    return state


def _pauli_pair(pauli: np.ndarray, lo: int, hi: int, num_qubits: int = _FFFT_MODES) -> np.ndarray:
    """Return ``pauli`` on qubits *lo* and *hi* in the big-endian basis."""
    factors = [pauli if qubit in (lo, hi) else np.eye(2) for qubit in range(num_qubits)]
    matrix = np.array([[1.0 + 0j]])
    for factor in factors:
        matrix = np.kron(matrix, factor)
    return matrix


class TestTwoModeFFFT:
    """The radix-2 butterfly each plaquette's four-mode FFFT is built from."""

    @pytest.mark.parametrize("lo", [0, 1, 2])
    def test_splits_a_single_particle_evenly(self, lo):
        """One particle on either mode lands in an equal superposition of both."""
        hi = lo + 1
        operation = f"qs => {_FFFT}({lo}, {hi}, qs)"
        half = 1.0 / math.sqrt(2.0)

        cases = [
            (_occupied(), _occupied()),
            (_occupied(lo), half * (_occupied(lo) - _occupied(hi))),
            (_occupied(hi), half * (_occupied(lo) + _occupied(hi))),
            (_occupied(lo, hi), _occupied(lo, hi)),
        ]
        for initial, expected in cases:
            assert np.allclose(_applied_state(operation, initial), expected, atol=1e-10)

    @pytest.mark.parametrize("lo", [0, 1, 2])
    @pytest.mark.parametrize("theta", [0.3, 0.83, -1.7])
    def test_diagonalizes_two_mode_hopping(self, lo, theta):
        """Conjugating a number-difference phase by the butterfly is a hopping evolution."""
        hi = lo + 1
        operation = (
            f"qs => {{ within {{ Adjoint {_FFFT}({lo}, {hi}, qs); }} apply "
            f"{{ Exp([PauliZ], {theta / 2}, [qs[{lo}]]); Exp([PauliZ], {-theta / 2}, [qs[{hi}]]); }} }}"
        )
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        hopping = 0.5 * (_pauli_pair(pauli_x, lo, hi) + _pauli_pair(pauli_y, lo, hi))

        state = _random_state(_FFFT_MODES, seed=11 + lo)
        expected = scipy.linalg.expm(1j * theta * hopping) @ state
        assert np.allclose(_applied_state(operation, state), expected, atol=1e-10)


_PLAQUETTE = "QDKChemistry.Utils.HubbardPlaquette"
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _hopping_tower(angle: float, num_pairs: int) -> np.ndarray:
    """Return exp(i angle XX) exp(i angle YY) on every pair (2k, 2k + 1), which all commute."""
    num_qubits = 2 * num_pairs
    generator = sum(
        _pauli_pair(pauli, 2 * k, 2 * k + 1, num_qubits) for k in range(num_pairs) for pauli in (_PAULI_X, _PAULI_Y)
    )
    return scipy.linalg.expm(1j * angle * generator)


def _hopping_phases(angle: float, num_pairs: int, control: str | None = None) -> str:
    """Return Q# applying ``HoppingPhases`` on a catalyst prepared around it, optionally controlled.

    Only the tower is controlled, as in the evolution: the catalyst is prepared either way.
    """
    register = "qs" if control is None else "qs[1...]"
    arguments = f"({angle}, Std.Arrays.Chunks(2, {register}), catalyst)"
    tower = (
        f"{_PLAQUETTE}.HoppingPhases{arguments}"
        if control is None
        else f"Controlled {_PLAQUETTE}.HoppingPhases([{control}], {arguments})"
    )
    return (
        f"qs => {{ use catalyst = Qubit[{_PLAQUETTE}.TowerCatalystSize({2 * num_pairs})]; "
        f"within {{ {_PLAQUETTE}.PrepareTowerCatalyst({-2.0 * angle}, catalyst); }} apply {{ {tower}; }} }}"
    )


class TestHoppingPhases:
    """Every XX and YY term of a hopping tiling is phased as one equal-angle tower."""

    @pytest.mark.parametrize("num_pairs", [2, 4, 5])
    def test_tower_matches_the_separate_rotations(self, num_pairs):
        """Below the break-even the terms are applied directly; from 8 rotations on, through HWP."""
        angle = 0.41
        operation = _hopping_phases(angle, num_pairs)

        state = _random_state(2 * num_pairs, seed=num_pairs)
        expected = _hopping_tower(angle, num_pairs) @ state
        assert np.allclose(_applied_state(operation, state), expected, atol=1e-10)

    def test_controlled_tower_acts_only_when_the_control_is_set(self):
        """Under control a stray global phase of the tower would become a relative phase."""
        angle, num_pairs = 0.41, 4
        operation = _hopping_phases(angle, num_pairs, control="qs[0]")

        state = _random_state(2 * num_pairs + 1, seed=21)
        half = len(state) // 2
        expected = np.concatenate([state[:half], _hopping_tower(angle, num_pairs) @ state[half:]])
        assert np.allclose(_applied_state(operation, state), expected, atol=1e-10)


class TestInteractionLayer:
    """The on-site tower phases every site pair through a Hamming-weight register and its catalyst."""

    @pytest.mark.parametrize("angle", [0.37, -1.3])
    def test_matches_the_separate_pair_rotations(self, angle):
        """At eight sites the tower takes the catalyzed path; each basis state gets exp(-i angle sum Z Z)."""
        sites, num_qubits = 8, 16
        rng = np.random.default_rng(5)
        basis_states = rng.choice(2**num_qubits, size=24, replace=False)
        amplitudes = np.zeros(2**num_qubits)
        amplitudes[basis_states] = rng.normal(size=len(basis_states))
        amplitudes /= np.linalg.norm(amplitudes)

        expected = amplitudes.astype(complex)
        for index in basis_states:
            spins = [1 - 2 * ((index >> (num_qubits - 1 - qubit)) & 1) for qubit in range(num_qubits)]
            expected[index] *= np.exp(-1j * angle * sum(spins[s] * spins[s + sites] for s in range(sites)))

        operation = (
            f"qs => {{ use catalyst = Qubit[{_PLAQUETTE}.TowerCatalystSize({sites})]; "
            f"within {{ {_PLAQUETTE}.PrepareTowerCatalyst({2.0 * angle}, catalyst); }} "
            f"apply {{ {_PLAQUETTE}.InteractionLayer({angle}, {sites}, qs, catalyst); }} }}"
        )
        assert np.allclose(_applied_state(operation, amplitudes), expected, atol=1e-10)


class TestPlaquetteEvolutionOnAState:
    """The emitted circuit must act on a state the way exp(-iHt) does."""

    @pytest.mark.parametrize("basis_state", [0, 1, 0b10010110, 0b11111111])
    def test_hopping_only_evolution_is_exact_on_basis_states(self, basis_state):
        """A hopping-only plaquette evolution carries no Trotter error, so it is exact."""
        time = 0.17
        circuit = _evolution_circuit(2, 2, time=time)
        hamiltonian = _reference_hamiltonian(2, 2, t=1.0, u=0.0)

        state = np.zeros(2**8, dtype=complex)
        state[basis_state] = 1.0
        expected = scipy.linalg.expm(-1j * time * hamiltonian) @ state

        assert _infidelity(_applied_state(circuit, state), expected) < 1e-9

    def test_hopping_only_evolution_is_exact_on_a_superposition(self):
        """Exactness holds on an entangled superposition, not just basis states."""
        time = 0.23
        circuit = _evolution_circuit(2, 2, time=time)
        hamiltonian = _reference_hamiltonian(2, 2, t=1.0, u=0.0)
        state = _random_state(8, seed=7)
        expected = scipy.linalg.expm(-1j * time * hamiltonian) @ state

        assert _infidelity(_applied_state(circuit, state), expected) < 1e-9

    def test_gold_tiling_evolution_is_exact(self):
        """The gold tiling alone reproduces its own hopping evolution exactly."""
        num_modes, duration = 8, 0.23
        cycles = [[5, 6, 2, 1], [7, 4, 0, 3]]
        kappa = 2.0 * duration
        literal = "[" + ", ".join("[" + ", ".join(map(str, cycle)) + "]" for cycle in cycles) + "]"
        operation = get_qsharp_context().eval(
            f"qs => {{ use catalyst = Qubit[{_PLAQUETTE}.TowerCatalystSize({2 * len(cycles)})]; "
            f"within {{ {_PLAQUETTE}.PrepareTowerCatalyst({-kappa}, catalyst); }} "
            f"apply {{ {_PLAQUETTE}.HoppingLayer({kappa}, {literal}, qs, catalyst); }} }}"
        )

        annihilate = np.array([[0, 1], [0, 0]], dtype=complex)
        identity = np.eye(2)
        parity = np.diag([1, -1]).astype(complex)

        modes = []
        for index in range(num_modes):
            matrix = np.array([[1.0 + 0j]])
            for factor in [parity] * index + [annihilate] + [identity] * (num_modes - index - 1):
                matrix = np.kron(matrix, factor)
            modes.append(matrix)
        hamiltonian = np.zeros((2**num_modes, 2**num_modes), dtype=complex)
        for cycle in cycles:
            for index in range(4):
                left, right = cycle[index], cycle[(index + 1) % 4]
                hamiltonian -= modes[left].conj().T @ modes[right] + modes[right].conj().T @ modes[left]

        state = _random_state(num_modes, seed=3)
        expected = scipy.linalg.expm(-1j * duration * hamiltonian) @ state
        amplitudes = [float(np.real(a)) for a in state]
        actual = np.asarray(
            dump_operation_on_state(operation, num_modes, amplitudes, context=get_qsharp_context()),
            dtype=complex,
        )

        assert _infidelity(actual, expected) < 1e-9


class TestPlaquettePhaseEstimation:
    """Phase estimation over the plaquette evolution must recover the known eigenvalue."""

    @pytest.mark.parametrize(
        ("t", "u", "num_divisions", "expected_energy"),
        [
            pytest.param(1.0, 0.0, 1, -8.0, id="hopping-only"),
            pytest.param(0.5, 2.0, 4, -4.8284271247, id="weak-coupling"),
            pytest.param(1.0, 4.0, 8, -9.6568542495, id="strong-coupling"),
        ],
    )
    def test_iterative_qpe_recovers_the_ground_energy(self, t, u, num_divisions, expected_energy):
        """IQPE over the plaquette evolution recovers the 2x2 ground energy."""
        num_bits = 4
        time = 2 * np.pi / (2**num_bits * abs(expected_energy))
        hamiltonian = _reference_hamiltonian(2, 2, t=t, u=u)
        values, vectors = np.linalg.eigh(hamiltonian)
        assert values[0] == pytest.approx(expected_energy, abs=1e-9), "the reference energy pins the test"

        # Prepare the exact ground state, so the measured phase is unambiguous.
        ground_state = np.real(vectors[:, 0])
        ground_state /= np.linalg.norm(ground_state)
        num_qubits = 8
        params = {
            "rowMap": list(range(num_qubits - 1, -1, -1)),
            "stateVector": ground_state.tolist(),
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        preparation = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params),
        )

        iqpe = IterativePhaseEstimation(shots_per_bit=15)
        iqpe.settings().set(
            "qpe_circuit_builder",
            AlgorithmRef(
                "qpe_circuit_builder",
                "qdk_iterative",
                num_bits=num_bits,
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "hubbard_plaquette"),
                unitary_builder=AlgorithmRef(
                    "hamiltonian_unitary_builder",
                    "hubbard_plaquette",
                    order=2,
                    time=time,
                    t=t,
                    u=u,
                    num_divisions=num_divisions,
                    target_accuracy=0.0,
                ),
            ),
        )
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))

        result = iqpe.run(state_preparation=preparation, qubit_hamiltonian=_lattice_operator(2, 2))

        assert tuple(result.bits_msb_first or ()) == (0, 0, 0, 1), "phase 1/16 is exactly 0001"
        assert result.raw_energy == pytest.approx(expected_energy, rel=1e-6)


_CATALYST_WIDTH, _CATALYST_HEIGHT = 4, 2
_CATALYST_SITES = _CATALYST_WIDTH * _CATALYST_HEIGHT


def _plaquette_controlled(
    interaction_angle: float, hopping_angle: float, step_reps: int, width: int = 4, height: int = 2
) -> Circuit:
    """Return the controlled plaquette circuit the mapper emits for these exact angles."""
    container = HubbardPlaquetteContainer(
        width=width,
        height=height,
        interaction_angle=interaction_angle,
        hopping_angle=hopping_angle,
        step_reps=step_reps,
    )
    return create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0]).run(
        UnitaryRepresentation(container=container)
    )


def _one_electron_preparation() -> Circuit:
    """Return a preparation of one spin-up electron spread evenly over every site.

    Each plaquette maps the uniform vector to twice itself and the interaction is constant
    with a single electron, so this state is an exact eigenstate of every layer. Its phase
    per repetition is 2 kappa - u (sites - 2), which makes phase estimation deterministic.
    """
    amplitudes = [0.0] * 2**_CATALYST_SITES
    for site in range(_CATALYST_SITES):
        amplitudes[1 << site] = 1.0 / math.sqrt(_CATALYST_SITES)
    operation = get_qsharp_context().eval(
        f"qs => Std.StatePreparation.PreparePureStateD({amplitudes}, qs[0..{_CATALYST_SITES - 1}])"
    )
    return Circuit(qasm="OPENQASM 3.0;", qsharp_op=operation)


def _eigenphase(interaction_angle: float, hopping_angle: float, step_reps: int) -> float:
    """Return the phase the evolution multiplies ``_one_electron_preparation`` by."""
    return step_reps * (2.0 * hopping_angle - interaction_angle * (_CATALYST_SITES - 2))


def _run(circuit: Circuit, shots: int) -> list:
    """Run a factory circuit's Q# program in the shared context."""
    factory = circuit._qsharp_factory
    return get_qsharp_context().run(factory.program, shots, *factory.parameter.values())


def _standard_phase(results: list) -> float:
    """Decode one standard phase estimation shot as the executor does: the last result is the MSB."""
    bits = "".join("1" if result == Result.One else "0" for result in reversed(results))
    return int(bits, 2) / 2 ** len(bits)


# Angles whose eigenphase is exactly pi / 2, so every measured bit is deterministic.
_HOPPING = 0.5
_INTERACTION = (2.0 * _HOPPING - np.pi / 2) / (_CATALYST_SITES - 2)


class TestSharedPlaquetteCatalysts:
    """Phase estimation prepares the plaquette catalysts once and shares them across queries."""

    def test_mapper_declares_its_catalyst_gradients(self):
        """The towers need an interaction gradient and a hopping gradient with one extra qubit."""
        circuit = _plaquette_controlled(0.3, 0.2, 1)
        assert circuit.metadata.phase_gradients == (PhaseGradient(0.3, 4), PhaseGradient(-0.1, 5))
        assert circuit.num_qubits == 2 * _CATALYST_SITES + 9

    def test_a_lattice_below_the_break_even_declares_none(self):
        """The 2x2 towers rotate term by term, so there is nothing to share."""
        assert _plaquette_controlled(0.3, 0.2, 1, width=2, height=2).metadata.phase_gradients == ()

    @pytest.mark.slow
    @pytest.mark.parametrize(("feedback", "expected"), [(0.0, 0), (np.pi, 1)])
    def test_iterative_estimation_kicks_back_the_exact_phase(self, feedback, expected):
        """The phase qubit reads a fixed bit once the feedback cancels, or completes, the eigenphase."""
        builder = create("qpe_circuit_builder", "qdk_iterative", num_bits=1)
        controlled = _plaquette_controlled(_INTERACTION, _HOPPING, 1)
        phase = _eigenphase(_INTERACTION, _HOPPING, 1)
        circuit = builder._create_circuit_from_qsharp_op(
            _one_electron_preparation(), controlled, feedback - phase, 2 * _CATALYST_SITES
        )
        assert circuit._qsharp_factory.parameter["numSharedAncillas"] == 9

        outcomes = {int(shot[0] == Result.One) for shot in _run(circuit, shots=4)}
        assert outcomes == {expected}

    @pytest.mark.slow
    def test_standard_estimation_shares_one_register_across_powers(self):
        """Repetitions change the power but not the angles, so every query reuses the same catalysts.

        The eigenphase is pi / 2, so two phase bits read exactly one quarter.
        """
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        powers = [_plaquette_controlled(_INTERACTION, _HOPPING, reps) for reps in (2, 1)]
        circuit = builder._create_circuit_from_qsharp_op(_one_electron_preparation(), powers, 2, 2 * _CATALYST_SITES)
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSharedAncillas"] == 9
        assert parameters["prepareSharedOp"] is not QSHARP_UTILS.PrepSelPrep.NoOpPrepare

        assert {_standard_phase(shot) for shot in _run(circuit, shots=2)} == {0.25}

    @pytest.mark.slow
    def test_rescaled_powers_share_the_overlapping_catalyst_qubits(self):
        """Doubling the angles doubles each gradient's phase, which only drops its lowest qubit.

        So the two queries share 7 of their 9 catalyst qubits and each prepares the other 2 itself.
        """
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        powers = [
            _plaquette_controlled(2 * _INTERACTION, 2 * _HOPPING, 1),
            _plaquette_controlled(_INTERACTION, _HOPPING, 1),
        ]
        circuit = builder._create_circuit_from_qsharp_op(_one_electron_preparation(), powers, 2, 2 * _CATALYST_SITES)
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSharedAncillas"] == 7
        assert parameters["prepareSharedOp"] is not QSHARP_UTILS.PrepSelPrep.NoOpPrepare

        assert {_standard_phase(shot) for shot in _run(circuit, shots=2)} == {0.25}

    def test_a_circuit_without_catalysts_takes_none_from_the_pool(self):
        """Circuits may declare different gradients, including none; nothing is pooled that only one needs."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        circuits = [_plaquette_controlled(0.3, 0.2, 1), _plaquette_controlled(0.3, 0.2, 1, width=2, height=2)]
        ops, prepare, num_shared = builder._shared_register(circuits)
        assert len(ops) == 2
        assert num_shared == 0
        assert prepare is QSHARP_UTILS.PrepSelPrep.NoOpPrepare


def _reference_w_plaquette(width: int, height: int, *, t: float, u: float) -> float:
    """Recompute Campbell's W_PLAQ independently of the builder.

    Follows Eq. (10) of Campbell arXiv:2012.09238v4 for W_SO2, Eq. (D10) for the
    plaquette-splitting term, and Eq. (D6) for their sum.
    """
    num_sites = width * height
    matrices = []
    for cycles in HubbardPlaquetteTrotter._plaquette_sections(width, height):
        matrix = np.zeros((num_sites, num_sites))
        for cycle in cycles:
            for index in range(4):
                site_a, site_b = cycle[index], cycle[(index + 1) % 4]
                matrix[site_a, site_b] = matrix[site_b, site_a] = -1.0
        matrices.append(matrix)
    matrix_p, matrix_g = matrices

    inner = matrix_p @ matrix_g - matrix_g @ matrix_p
    outer = inner @ matrix_g - matrix_g @ inner
    hopping_norm = float(np.linalg.svd(matrix_p + matrix_g, compute_uv=False).sum()) * t
    commutator_norm = float(np.linalg.svd(outer, compute_uv=False).sum()) * t**3

    w_so2 = u * t**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + u**2 / 24.0 * hopping_norm
    return w_so2 + 3.0 / 24.0 * commutator_norm


def _auto_step_count(width: int, height: int, *, t: float, u: float, time: float, target_accuracy: float) -> int:
    """Ask the builder for the step count it derives from an accuracy target."""
    builder = HubbardPlaquetteTrotter(
        order=2,
        time=time,
        t=t,
        u=u,
        num_divisions=1,
        target_accuracy=target_accuracy,
    )
    return builder._step_count(t, width, height, time)


class TestAutomaticStepCount:
    """The ``target_accuracy`` path, which sizes the step count from the error bound."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (6, 6)])
    @pytest.mark.parametrize("phase", [1e-6, 0.25, 1.0, math.pi / 2])
    def test_matches_the_exact_rule_of_apel_algorithm_1(self, width, height, phase):
        """Reproduce r = ceil(sqrt(W_PLAQ tau^3 / (2 sin(eps tau / 2)))) against an independent W."""
        t, u, time = 1.0, 8.0, 3.0
        target_accuracy = phase / time

        count = _auto_step_count(width, height, t=t, u=u, time=time, target_accuracy=target_accuracy)

        w_plaquette = _reference_w_plaquette(width, height, t=t, u=u)
        expected = math.ceil(math.sqrt(w_plaquette * time**3 / (2.0 * math.sin(phase / 2.0))))
        assert count == expected

    def test_saturates_once_the_accuracy_target_exceeds_a_half_turn(self):
        """||Delta U|| <= 2 caps the arcsine, so the count stops falling at eps tau = pi."""
        t, u, time = 1.0, 8.0, 3.0

        at_pi = _auto_step_count(4, 4, t=t, u=u, time=time, target_accuracy=math.pi / time)
        beyond_pi = _auto_step_count(4, 4, t=t, u=u, time=time, target_accuracy=3.0 * math.pi / time)

        assert beyond_pi == at_pi

    def test_a_disabled_target_leaves_the_manual_count_alone(self):
        builder = HubbardPlaquetteTrotter(order=2, time=3.0, t=1.0, u=8.0, num_divisions=7, target_accuracy=0.0)
        assert builder._step_count(1.0, 4, 4, 3.0) == 7

    def test_the_hamiltonian_conserves_particle_number(self):
        """Justifies recovering the conventional energy by a classical shift."""
        width = height = 2
        num_qubits = 2 * width * height
        hamiltonian = _reference_hamiltonian(width, height, t=1.0, u=8.0)

        total_z = np.zeros((2**num_qubits, 2**num_qubits))
        for mode in range(num_qubits):
            factors = [np.eye(2)] * num_qubits
            factors[mode] = np.diag([1.0, -1.0])
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            total_z = total_z + term

        commutator = hamiltonian @ total_z - total_z @ hamiltonian
        assert np.max(np.abs(commutator)) < 1e-10

    @pytest.mark.parametrize("num_electrons", [0, 2, 4, 8])
    def test_the_electron_count_shifts_to_the_conventional_model(self, num_electrons):
        """``num_electrons`` adds the ``U*eta/2 - U*M/4`` offset during phase conversion."""
        width = height = 2
        u, time, divisions = 4.0, 0.3, 2
        operator = _lattice_operator(width, height)

        unshifted = (
            HubbardPlaquetteTrotter(order=2, time=time, t=1.0, u=u, num_divisions=divisions, target_accuracy=0.0)
            .run(operator)
            .get_container()
        )
        shifted = (
            HubbardPlaquetteTrotter(
                order=2,
                time=time,
                t=1.0,
                u=u,
                num_electrons=num_electrons,
                num_divisions=divisions,
                target_accuracy=0.0,
            )
            .run(operator)
            .get_container()
        )

        assert unshifted.constant_shift == 0.0, "an unset count leaves the symmetric energy alone"
        assert shifted.interaction_angle == unshifted.interaction_angle, "only the scalar moves"
        assert shifted.hopping_angle == unshifted.hopping_angle, "only the scalar moves"

        delta = time / divisions
        expected = u * (0.5 * num_electrons - 0.25 * width * height) * delta
        assert shifted.constant_shift == pytest.approx(expected)
        phase_fraction = 0.125
        expected_energy_shift = expected * divisions / time
        assert shifted.eigenvalue_from_phase(phase_fraction) == pytest.approx(
            unshifted.eigenvalue_from_phase(phase_fraction) + expected_energy_shift
        )
