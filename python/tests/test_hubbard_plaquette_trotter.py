"""Tests for the Hubbard plaquette Trotter builder and its Q# lowering.

The builder emits a :class:`HubbardPlaquetteContainer` carrying only the lattice shape
and the layer angles; the fermionic structure lives in Q#. These tests therefore check
the emitted *circuit* rather than any intermediate representation: what it does to a
state, and what phase estimation recovers from it.

The reference in every case is the same model mapped independently by ``qubit_mapper``
and exponentiated densely, so a shared mistake in the builder and its test cannot hide.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
import scipy.linalg
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_hubbard_plaquette_mapper import (
    plaquette_parameters,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.hubbard_plaquette_trotter import (
    HubbardPlaquetteTrotter,
)
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, LatticeGraph, MajoranaMapping, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context, use_qsharp_context


@pytest.fixture(scope="module")
def qsharp_context():
    """Return a Q# context shared by the module; interpreters are thread-affine."""
    return create_qsharp_context()


def _lattice_operator(width: int, height: int) -> QubitOperator:
    """Return a periodic square lattice as a lattice-backed qubit operator."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    return QubitOperator(container=LatticeContainer(lattice))


def _reference_hamiltonian(width: int, height: int, *, t: float, u: float, epsilon: float) -> np.ndarray:
    """Return the dense Hamiltonian from an independent Jordan-Wigner mapping."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=epsilon, t=t, U=u)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * width * height))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    return pauli_to_dense_matrix(list(labels), list(coefficients))


def _evolution_circuit(
    width: int,
    height: int,
    *,
    context,
    time: float,
    t: float = 1.0,
    u: float = 0.0,
    epsilon: float = 0.0,
    num_divisions: int = 1,
):
    """Return the uncontrolled plaquette evolution as a Q# callable.

    The mapper emits the controlled form, which phase estimation needs; the uncontrolled
    evolution is the same operation without its control, so it is taken directly from Q#.
    """
    builder = HubbardPlaquetteTrotter(
        order=2, time=time, t=t, U=u, epsilon=epsilon, num_divisions=num_divisions, target_accuracy=0.0
    )
    with use_qsharp_context(context):
        container = builder.run(_lattice_operator(width, height)).get_container()
    return QSHARP_UTILS.HubbardPlaquette.MakeRepPlaquetteExpOp(plaquette_parameters(container))


def _applied_state(operation, state: np.ndarray, context) -> np.ndarray:
    """Return the state the operation produces from *state*."""
    num_qubits = round(math.log2(len(state)))
    return np.asarray(
        dump_operation_on_state(operation, num_qubits, [float(np.real(a)) for a in state], context=context),
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
        unitary = HubbardPlaquetteTrotter(
            order=2, time=0.1, t=1.0, U=4.0, epsilon=-2.0, num_divisions=3, target_accuracy=0.0
        ).run(_lattice_operator(2, 2))
        container = unitary.get_container()

        assert isinstance(container, HubbardPlaquetteContainer)
        assert (container.width, container.height) == (2, 2)
        assert container.num_qubits == 8
        assert container.step_reps == 3

    def test_representation_size_is_independent_of_the_lattice(self):
        """The payload is a fixed set of scalars, so it does not grow with the lattice.

        This is the point of deriving the tilings in Q#: a Jordan-Wigner encoding would
        spell out one Pauli string per term, and the term count grows with the lattice.
        """
        payloads = [
            HubbardPlaquetteTrotter(order=2, time=0.1, t=1.0, U=4.0, epsilon=-2.0, num_divisions=1)
            .run(_lattice_operator(side, side))
            .get_container()
            .to_json()
            for side in (2, 4, 6)
        ]

        assert {frozenset(payload) for payload in payloads} == {frozenset(payloads[0])}

    def test_container_round_trips_through_json(self):
        """Serialization preserves every field the Q# lowering reads."""
        container = (
            HubbardPlaquetteTrotter(order=2, time=0.3, t=1.0, U=8.0, epsilon=-4.0, num_divisions=2)
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
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, t=1.0, U=4.0, num_divisions=1)

        with pytest.raises(ValueError, match="bond graph does not match"):
            builder.run(open_lattice)

    def test_rejects_a_mapped_qubit_operator(self):
        """The tiling needs the lattice structure, which a mapped operator discards."""
        lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
        mapped = create("qubit_mapper").run(
            create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0),
            mapping=MajoranaMapping.jordan_wigner(8),
        )
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, t=1.0, U=4.0, num_divisions=1)

        with pytest.raises(TypeError, match="LatticeContainer"):
            builder.run(mapped)


class TestPlaquetteTiling:
    """The Q# tilings must cover the lattice the way Campbell's decomposition requires."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (4, 6), (6, 6)])
    def test_tilings_cover_every_bond_exactly_once(self, width, height, qsharp_context):
        """Together the pink and gold tilings reproduce the periodic lattice's bonds."""
        sites = width * height
        cycles = [
            [int(site) for site in cycle]
            for pink in ("true", "false")
            for cycle in qsharp_context.eval(
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
    def test_each_tiling_is_vertex_disjoint(self, width, height, qsharp_context):
        """Plaquettes within a tiling share no site, which is what lets them run together."""
        for pink in ("true", "false"):
            cycles = qsharp_context.eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({width},{height},{pink})"
            )
            seen: set[int] = set()
            for cycle in cycles:
                sites = {int(site) for site in cycle}
                assert not seen & sites, "plaquettes within a tiling must be vertex disjoint"
                seen |= sites

    @pytest.mark.parametrize("side", [4, 6, 8])
    def test_routing_makes_each_plaquette_local(self, side, qsharp_context):
        """After routing, each plaquette occupies four adjacent modes, whatever the size.

        Without routing the Jordan-Wigner string of a wrap-around bond reaches across the
        whole spin sector, so no two plaquettes commute by disjointness.
        """
        num_modes = 2 * side * side
        gold = [
            [int(site) for site in cycle]
            for cycle in qsharp_context.eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({side},{side},false)"
            )
        ]
        swaps = [
            int(position)
            for position in qsharp_context.eval(
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
            assert routed[4 * index : 4 * index + 4] == cycle, "each plaquette must land contiguous"


class TestPlaquetteEvolutionOnAState:
    """The emitted circuit must act on a state the way exp(-iHt) does."""

    @pytest.mark.parametrize("basis_state", [0, 1, 0b10010110, 0b11111111])
    def test_hopping_only_evolution_is_exact_on_basis_states(self, basis_state, qsharp_context):
        """A hopping-only plaquette evolution carries no Trotter error, so it is exact."""
        time = 0.17
        circuit = _evolution_circuit(2, 2, time=time, context=qsharp_context)
        hamiltonian = _reference_hamiltonian(2, 2, t=1.0, u=0.0, epsilon=0.0)

        state = np.zeros(2**8, dtype=complex)
        state[basis_state] = 1.0
        expected = scipy.linalg.expm(-1j * time * hamiltonian) @ state

        assert _infidelity(_applied_state(circuit, state, qsharp_context), expected) < 1e-9

    def test_hopping_only_evolution_is_exact_on_a_superposition(self, qsharp_context):
        """Exactness holds on an entangled superposition, not just basis states."""
        time = 0.23
        circuit = _evolution_circuit(2, 2, time=time, context=qsharp_context)
        hamiltonian = _reference_hamiltonian(2, 2, t=1.0, u=0.0, epsilon=0.0)
        state = _random_state(8, seed=7)
        expected = scipy.linalg.expm(-1j * time * hamiltonian) @ state

        assert _infidelity(_applied_state(circuit, state, qsharp_context), expected) < 1e-9

    def test_interacting_evolution_converges_as_the_step_shrinks(self, qsharp_context):
        """With U nonzero the splitting is approximate, and the error falls with more steps.

        The interaction and the two hopping tilings do not commute, so only the limit of
        many steps reproduces exp(-iHt). Convergence is what distinguishes Trotter error
        from a wrong circuit.
        """
        time, t, u, epsilon = 0.17, 1.0, 4.0, -2.0
        hamiltonian = _reference_hamiltonian(2, 2, t=t, u=u, epsilon=epsilon)
        state = _random_state(8, seed=11)
        expected = scipy.linalg.expm(-1j * time * hamiltonian) @ state

        errors = [
            _infidelity(
                _applied_state(
                    _evolution_circuit(
                        2, 2, time=time, t=t, u=u, epsilon=epsilon, num_divisions=divisions, context=qsharp_context
                    ),
                    state,
                    qsharp_context,
                ),
                expected,
            )
            for divisions in (1, 2, 4)
        ]

        assert errors[0] > errors[1] > errors[2], f"error must fall with more steps, got {errors}"
        assert errors[2] < errors[0] / 10.0, f"second-order error should fall steeply, got {errors}"

    def test_gold_tiling_evolution_is_exact(self, qsharp_context):
        """The gold tiling alone reproduces its own hopping evolution exactly.

        A 2x2 lattice has an empty gold tiling, so this exercises 4x2 gold cycles, whose
        modes are neither adjacent nor ascending. Those are precisely the cycles that a
        lowering assuming contiguous modes gets wrong.

        The layer is checked directly rather than through a whole-lattice evolution: the
        smallest lattice carrying both a populated gold tiling and uniform edge weights
        is 4x4, whose state vector alone would need more memory than a test can use.
        """
        num_modes, duration = 8, 0.23
        cycles = [[5, 6, 2, 1], [7, 4, 0, 3]]
        literal = "[" + ", ".join("[" + ", ".join(map(str, cycle)) + "]" for cycle in cycles) + "]"
        operation = qsharp_context.eval(
            f"qs => QDKChemistry.Utils.HubbardPlaquette.HoppingLayer({2.0 * duration}, {literal}, qs)"
        )

        annihilate = np.array([[0, 1], [0, 0]], dtype=complex)
        identity = np.eye(2)
        parity = np.diag([1, -1]).astype(complex)

        def mode(index: int) -> np.ndarray:
            matrix = np.array([[1.0 + 0j]])
            for factor in [parity] * index + [annihilate] + [identity] * (num_modes - index - 1):
                matrix = np.kron(matrix, factor)
            return matrix

        modes = [mode(index) for index in range(num_modes)]
        hamiltonian = np.zeros((2**num_modes, 2**num_modes), dtype=complex)
        for cycle in cycles:
            for index in range(4):
                left, right = cycle[index], cycle[(index + 1) % 4]
                hamiltonian -= modes[left].conj().T @ modes[right] + modes[right].conj().T @ modes[left]

        state = _random_state(num_modes, seed=3)
        expected = scipy.linalg.expm(-1j * duration * hamiltonian) @ state
        actual = np.asarray(
            dump_operation_on_state(operation, num_modes, [float(np.real(a)) for a in state], context=qsharp_context),
            dtype=complex,
        )

        assert _infidelity(actual, expected) < 1e-9


class TestPlaquettePhaseEstimation:
    """Phase estimation over the plaquette evolution must recover the known eigenvalue."""

    @staticmethod
    def _ground_state_preparation(hamiltonian: np.ndarray, num_qubits: int) -> tuple[Circuit, float]:
        """Prepare the exact ground state, so the measured phase is unambiguous.

        Args:
            hamiltonian: The dense Hamiltonian to diagonalize.
            num_qubits: Width of the system register.

        Returns:
            The preparation circuit and the ground energy it was built from.

        """
        values, vectors = np.linalg.eigh(hamiltonian)
        state = np.real(vectors[:, 0])
        state /= np.linalg.norm(state)
        params = {
            "rowMap": list(range(num_qubits - 1, -1, -1)),
            "stateVector": state.tolist(),
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        circuit = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params),
        )
        return circuit, float(values[0])

    @pytest.mark.parametrize(
        ("t", "u", "epsilon", "num_divisions", "expected_energy"),
        [
            pytest.param(1.0, 0.0, 0.0, 1, -8.0, id="hopping-only"),
            pytest.param(0.5, 2.0, -1.0, 4, -6.8284271247, id="weak-coupling"),
            pytest.param(1.0, 4.0, -2.0, 8, -13.6568542495, id="strong-coupling"),
        ],
    )
    def test_iterative_qpe_recovers_the_ground_energy(
        self, t, u, epsilon, num_divisions, expected_energy, qsharp_context
    ):
        """IQPE over the plaquette evolution recovers the 2x2 ground energy.

        The evolution time is chosen so the phase lands exactly on a four-bit grid point,
        making the expected reading exact rather than approximate. The interacting cases
        need more Trotter steps, since only the hopping-only splitting is exact.

        The three cases span the regimes that stress different parts of the step: no
        interaction layer at all, a weak one, and one that dominates the hopping.
        """
        num_bits = 4
        time = 2 * np.pi / (2**num_bits * abs(expected_energy))
        hamiltonian = _reference_hamiltonian(2, 2, t=t, u=u, epsilon=epsilon)
        preparation, ground_energy = self._ground_state_preparation(hamiltonian, 8)
        assert ground_energy == pytest.approx(expected_energy, abs=1e-9), "the reference energy pins the test"

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
                    "plaquette",
                    order=2,
                    time=time,
                    t=t,
                    U=u,
                    epsilon=epsilon,
                    num_divisions=num_divisions,
                    target_accuracy=0.0,
                ),
            ),
        )
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))

        with use_qsharp_context(qsharp_context):
            result = iqpe.run(state_preparation=preparation, qubit_hamiltonian=_lattice_operator(2, 2))

        assert tuple(result.bits_msb_first or ()) == (0, 0, 0, 1), "phase 1/16 is exactly 0001"
        assert result.raw_energy == pytest.approx(expected_energy, rel=1e-6)

    def test_controlled_evolution_acts_only_when_the_control_is_set(self, qsharp_context):
        """The controlled circuit leaves the system untouched on the zero control branch.

        Phase estimation relies on this: the routing and basis changes must cancel when
        the control is off, or the ancilla would pick up a phase from the wrong branch.
        """
        builder = HubbardPlaquetteTrotter(
            order=2, time=0.19, t=1.0, U=4.0, epsilon=-2.0, num_divisions=1, target_accuracy=0.0
        )
        with use_qsharp_context(qsharp_context):
            unitary = builder.run(_lattice_operator(2, 2))
            # Built for its side effect: the mapper must accept this representation.
            create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0]).run(unitary)

        # The mapper's callable takes (control, systems); the simulator drives a single
        # register, so use the register-shaped form of the same operation.
        on_register = QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOnRegisterOp(
            plaquette_parameters(unitary.get_container())
        )
        system = _random_state(8, seed=13)
        control_off = np.kron([1.0, 0.0], system)
        result = np.asarray(
            dump_operation_on_state(on_register, 9, [float(np.real(a)) for a in control_off], context=qsharp_context),
            dtype=complex,
        )

        assert _infidelity(result, control_off) < 1e-9
