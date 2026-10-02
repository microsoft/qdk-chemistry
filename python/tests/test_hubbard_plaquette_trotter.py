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
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    LatticeGeometry,
    LatticeGraph,
    MajoranaMapping,
    QubitOperator,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _lattice_operator(width: int, height: int, *, t: float, u: float) -> QubitOperator:
    """Return a periodic square Fermi-Hubbard lattice as a lattice-backed qubit operator."""
    geometry = LatticeGeometry.square(width, height, periodic_x=True, periodic_y=True)
    return QubitOperator(container=LatticeContainer(geometry, couplings={"hopping": t, "interaction": u}))


#: Width of the phase gradient register the tests apply the Hamming-weight rotations through.
#: A b-qubit gradient rounds every rotation to the nearest multiple of 2*pi/2**b, so the tests
#: below keep every layer angle an exact multiple of that quantum. The construction is then exact
#: and the assertions can stay at 1e-10, while the register stays small enough to simulate: the
#: gradient is a uniform superposition, so it multiplies the simulated state's support by 2**b.
_GRADIENT_BITS = 6
_ANGLE_QUANTUM = 2.0 * math.pi / 2**_GRADIENT_BITS


def _reference_hamiltonian(width: int, height: int, *, t: float, u: float) -> np.ndarray:
    """Return the dense Hamiltonian from an independent Jordan-Wigner mapping."""
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-0.5 * u, t=t, U=u)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * width * height))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    dense = pauli_to_dense_matrix(list(labels), list(coefficients))
    return dense + 0.25 * u * width * height * np.eye(dense.shape[0])


def _conventional_hamiltonian(width: int, height: int, *, t: float, u: float) -> np.ndarray:
    r"""Return the unshifted :math:`U \sum_i n_{i\uparrow} n_{i\downarrow}` Hamiltonian, densely.

    This mirrors :func:`_reference_hamiltonian` and differs in exactly one place: the on-site
    energy is zero rather than :math:`-U/2`, and no :math:`UM/4` is added back. Those two terms
    are what carry the symmetric model onto the conventional one, so building both from the same
    mapper keeps the comparison between them an honest one.
    """
    lattice = LatticeGraph.square(width, height, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=t, U=u)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * width * height))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    return pauli_to_dense_matrix(list(labels), list(coefficients))


def _plaquette_parameters(container, max_batch_size: int = -1):
    """Return the Q# parameter struct for a plaquette container."""
    return QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
        width=container.width,
        height=container.height,
        interactionAngle=container.interaction_angle,
        hoppingAngle=container.hopping_angle,
        repetitions=container.step_reps,
        maxBatchSize=max_batch_size,
        rotationBitPrecision=_GRADIENT_BITS,
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
    builder = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=num_divisions, target_accuracy=0.0)
    container = builder.run(_lattice_operator(width, height, t=t, u=u)).get_container()
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
        unitary = HubbardPlaquetteTrotter(order=2, time=0.1, num_divisions=3, target_accuracy=0.0).run(
            _lattice_operator(2, 2, t=1.0, u=4.0)
        )
        container = unitary.get_container()

        assert isinstance(container, HubbardPlaquetteContainer)
        assert (container.width, container.height) == (2, 2)
        assert container.num_qubits == 8
        assert container.step_reps == 3

    def test_representation_size_is_independent_of_the_lattice(self):
        """The payload is a fixed set of scalars, so it does not grow with the lattice."""
        payloads = [
            HubbardPlaquetteTrotter(order=2, time=0.1, num_divisions=1)
            .run(_lattice_operator(side, side, t=1.0, u=4.0))
            .get_container()
            .to_json()
            for side in (2, 4, 6)
        ]

        geometry_fields = {"width", "height"}
        field_sets = [frozenset(payload) - geometry_fields for payload in payloads]

        assert field_sets == [field_sets[0]] * len(field_sets)

    def test_container_round_trips_through_json(self):
        """Serialization preserves every field the Q# lowering reads."""
        container = (
            HubbardPlaquetteTrotter(order=2, time=0.3, num_divisions=2)
            .run(_lattice_operator(4, 4, t=1.0, u=8.0))
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
            container=LatticeContainer(LatticeGeometry.square(4, 4, periodic_x=False, periodic_y=False))
        )
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, num_divisions=1)

        with pytest.raises(ValueError, match="periodic in both directions"):
            builder.run(open_lattice)

    @pytest.mark.parametrize(
        ("factory", "nx", "ny"),
        [("triangular", 4, 4), ("honeycomb", 2, 2), ("kagome", 2, 2)],
    )
    def test_rejects_a_geometry_that_is_not_a_square_grid(self, factory, nx, ny):
        """Only the square lattice carries the unit-spaced grid the plaquette tiling assumes."""
        geometry = getattr(LatticeGeometry, factory)(nx, ny, periodic_x=True, periodic_y=True)
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, num_divisions=1)

        with pytest.raises(ValueError, match="unit-spaced square lattice"):
            builder.run(QubitOperator(container=LatticeContainer(geometry)))

    def test_rejects_a_lattice_graph(self):
        """The container stores geometry, which a graph's renumberable edge list is not."""
        with pytest.raises(TypeError, match="LatticeGeometry"):
            LatticeContainer(LatticeGraph.square(4, 4, periodic_x=True, periodic_y=True))

    def test_rejects_a_mapped_qubit_operator(self):
        """The tiling needs the lattice structure, which a mapped operator discards."""
        lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
        mapped = create("qubit_mapper").run(
            create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0),
            mapping=MajoranaMapping.jordan_wigner(8),
        )
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, num_divisions=1)

        with pytest.raises(TypeError, match="LatticeContainer"):
            builder.run(mapped)

    @pytest.mark.parametrize(
        ("couplings", "message"),
        [
            ({"hopping": 1.0}, r"missing \['interaction'\]"),
            ({"interaction": 4.0}, r"missing \['hopping'\]"),
            ({}, r"missing \['hopping', 'interaction'\]"),
            ({"hopping": 1.0, "interaction": 4.0, "onsite": 0.5}, r"unsupported \['onsite'\]"),
        ],
    )
    def test_requires_exactly_the_hubbard_couplings(self, couplings, message):
        """A missing coupling has no safe default, and an extra one would silently drop from the evolution."""
        geometry = LatticeGeometry.square(4, 4, periodic_x=True, periodic_y=True)
        operator = QubitOperator(container=LatticeContainer(geometry, couplings=couplings))
        builder = HubbardPlaquetteTrotter(order=2, time=0.05, num_divisions=1)

        with pytest.raises(ValueError, match=message):
            builder.run(operator)

    def test_the_couplings_decide_the_evolution(self):
        """Same geometry, different interaction: distinct operators, distinct circuits."""
        weak = _lattice_operator(4, 4, t=1.0, u=0.0)
        strong = _lattice_operator(4, 4, t=1.0, u=8.0)
        builder = HubbardPlaquetteTrotter(order=2, time=0.1, num_divisions=1)

        assert weak.get_container().content_hash() != strong.get_container().content_hash()
        assert builder.run(weak).get_container().interaction_angle == 0.0
        assert builder.run(strong).get_container().interaction_angle == pytest.approx(0.25 * 8.0 * 0.1)

    @staticmethod
    def _step(time: float, width: int = 4, height: int = 4) -> HubbardPlaquetteContainer:
        builder = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=2)
        return builder.run(_lattice_operator(width, height, t=1.0, u=8.0)).get_container()

    def test_repetitions_of_one_body_combine_into_one(self):
        """Appending a body to itself adds the repetitions and keeps everything else."""
        step = self._step(0.1)

        combined = step.combine(step)

        assert isinstance(combined, HubbardPlaquetteContainer)
        assert combined.step_reps == 2 * step.step_reps
        fields = ("width", "height", "interaction_angle", "hopping_angle", "constant_shift", "scale")
        assert [getattr(combined, name) for name in fields] == [getattr(step, name) for name in fields]

    def test_the_rounding_of_a_split_interval_still_combines(self):
        """Euler splits 0.3 into steps of 0.1 and a residual of 0.3 - 0.2, which is 0.1 to within one ulp."""
        residual = 0.3 - 2 * 0.1
        assert residual != 0.1

        combined = self._step(0.1).combine(self._step(residual))

        assert combined.step_reps == 4

    def test_different_bodies_do_not_combine(self):
        """A rescaled power has different angles, so no single body can represent both."""
        with pytest.raises(ValueError, match="hopping_angle"):
            self._step(0.1).combine(self._step(0.2))

    def test_different_lattices_do_not_combine(self):
        with pytest.raises(ValueError, match="height"):
            self._step(0.1).combine(self._step(0.1, height=6))

    def test_only_plaquette_evolutions_combine(self):
        with pytest.raises(TypeError, match="plaquette"):
            self._step(0.1).combine(PauliProductFormulaContainer(step_terms=[], step_reps=1, num_qubits=32))


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
        _, positions = get_qsharp_context().eval(
            f"QDKChemistry.Utils.HubbardPlaquette.RoutingSwaps({gold}, {num_modes}, true)"
        )
        swaps = [int(position) for position in positions]

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


def _register_index(values: list[int], widths: list[int]) -> int:
    """Return the big-endian basis index of little-endian registers holding *values*."""
    index, offset = 0, sum(widths)
    for value, width in zip(values, widths, strict=True):
        for bit in range(width):
            offset -= 1
            index |= ((value >> bit) & 1) << offset
    return index


def _binary_gradient_words(phi: float, n: int, bits: int) -> list[int]:
    """Return the ``bits``-bit word of each place value, computed independently of Q#."""
    modulus = 1 << bits
    return [round(phi * 2**j * modulus / (4.0 * math.pi)) % modulus for j in range(n)]


class TestBinaryGradientWords:
    """The classical words Hamming-weight phasing loads, one per place value of the weight."""

    @pytest.mark.parametrize("phi", [0.37, -2.9, 0.75 * math.pi, 9.5])
    @pytest.mark.parametrize(("n", "bits"), [(1, 4), (3, 5), (4, 8)])
    def test_words_are_the_nearest_representable_rotation(self, phi, n, bits):
        """Word j rounds Rz(phi 2^j) onto the 4 pi / 2^bits lattice the gradient resolves."""
        actual = [int(word) for word in QSHARP_UTILS.HubbardPlaquette.BinaryGradientWords(phi, n, bits)]
        assert actual == _binary_gradient_words(phi, n, bits)
        assert all(0 <= word < 2**bits for word in actual), "a word must fit the gradient register"

    def test_a_lattice_angle_is_represented_exactly(self):
        """Angles that are multiples of 4 pi / 2^bits round to themselves, so the phasing is exact."""
        bits, k = 5, 3
        phi = 4.0 * math.pi * k / 2**bits
        actual = [int(word) for word in QSHARP_UTILS.HubbardPlaquette.BinaryGradientWords(phi, 4, bits)]
        assert actual == [(k * 2**j) % 2**bits for j in range(4)]


class TestPhaseByBinaryGradient:
    """Hamming-weight phasing applies e^{i phi w} and returns the gradient register prepared."""

    #: A phase on the 4 pi / 2^bits lattice, so every word is exact and the assertions can be tight.
    BITS = 4
    PHI = 4.0 * math.pi * 3 / 2**4

    @classmethod
    def _operation(cls, n: int, controlled: bool) -> str:
        """Return Q# applying the phasing, with the constant offset, on a gradient prepared around it."""
        words = _binary_gradient_words(cls.PHI, n, cls.BITS)
        offset = -sum(4.0 * math.pi * word / 2**cls.BITS for word in words)
        lead = 1 if controlled else 0
        weight = f"qs[{lead}..{lead + n - 1}]"
        gradient = f"qs[{lead + n}...]"
        body = (
            f"{_PLAQUETTE}.PhaseByBinaryGradient({words}, {weight}, {gradient}); R(PauliI, {offset}, {weight}[0]);"
            if not controlled
            else (
                f"Controlled {_PLAQUETTE}.PhaseByBinaryGradient([qs[0]], ({words}, {weight}, {gradient})); "
                f"Controlled R([qs[0]], (PauliI, {offset}, {weight}[0]));"
            )
        )
        return f"qs => {{ within {{ {_PLAQUETTE}.PreparePlaquetteGradient({gradient}); }} apply {{ {body} }} }}"

    @pytest.mark.parametrize("n", [1, 3])
    def test_phases_every_weight_and_restores_the_gradient(self, n):
        """Every weight picks up e^{i phi w}; the gradient ends in |0> after unpreparation."""
        widths = [n, self.BITS]
        amplitudes = np.zeros(2 ** sum(widths))
        for weight in range(2**n):
            amplitudes[_register_index([weight, 0], widths)] = 2 ** (-n / 2)
        actual = _applied_state(self._operation(n, controlled=False), amplitudes)

        expected = np.zeros_like(actual)
        for weight in range(2**n):
            expected[_register_index([weight, 0], widths)] = 2 ** (-n / 2) * np.exp(1j * self.PHI * weight)
        assert np.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3])
    def test_controlled_phasing_acts_only_when_the_control_is_set(self, n):
        """Loading the word under control leaves the addition, and the gradient, uncontrolled."""
        widths = [1, n, self.BITS]
        amplitudes = np.zeros(2 ** sum(widths))
        for control in (0, 1):
            for weight in range(2**n):
                amplitudes[_register_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2)
        actual = _applied_state(self._operation(n, controlled=True), amplitudes)

        expected = np.zeros_like(actual)
        for control in (0, 1):
            for weight in range(2**n):
                phase = np.exp(1j * self.PHI * weight) if control else 1.0
                expected[_register_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2) * phase
        assert np.allclose(actual, expected, atol=1e-10)


def _hopping_tower(angle: float, num_pairs: int) -> np.ndarray:
    """Return exp(i angle XX) exp(i angle YY) on every pair (2k, 2k + 1), which all commute."""
    num_qubits = 2 * num_pairs
    generator = sum(
        _pauli_pair(pauli, 2 * k, 2 * k + 1, num_qubits) for k in range(num_pairs) for pauli in (_PAULI_X, _PAULI_Y)
    )
    return scipy.linalg.expm(1j * angle * generator)


def _hopping_phases(angle: float, num_pairs: int, control: str | None = None) -> str:
    """Return Q# applying ``HoppingPhases`` on a gradient prepared around it, optionally controlled.

    Only the tower is controlled, as in the evolution: the gradient is prepared either way.
    """
    register = "qs" if control is None else "qs[1...]"
    arguments = f"({angle}, Std.Arrays.Chunks(2, {register}), -1, gradient)"
    tower = (
        f"{_PLAQUETTE}.HoppingPhases{arguments}"
        if control is None
        else f"Controlled {_PLAQUETTE}.HoppingPhases([{control}], {arguments})"
    )
    return (
        f"qs => {{ use gradient = Qubit[{_PLAQUETTE}.TowerGradientSize({2 * num_pairs}, -1, {_GRADIENT_BITS})]; "
        f"within {{ {_PLAQUETTE}.PreparePlaquetteGradient(gradient); }} apply {{ {tower}; }} }}"
    )


class TestHoppingPhases:
    """Every XX and YY term of a hopping tiling is phased as one equal-angle tower."""

    @pytest.mark.parametrize("num_pairs", [2, 4, 5])
    def test_tower_matches_the_separate_rotations(self, num_pairs):
        """Below the break-even the terms are applied directly; from 8 rotations on, through HWP."""
        angle = 4 * _ANGLE_QUANTUM
        operation = _hopping_phases(angle, num_pairs)

        state = _random_state(2 * num_pairs, seed=num_pairs)
        expected = _hopping_tower(angle, num_pairs) @ state
        assert np.allclose(_applied_state(operation, state), expected, atol=1e-10)

    def test_controlled_tower_acts_only_when_the_control_is_set(self):
        """Under control a stray global phase of the tower would become a relative phase."""
        angle, num_pairs = 4 * _ANGLE_QUANTUM, 4
        operation = _hopping_phases(angle, num_pairs, control="qs[0]")

        state = _random_state(2 * num_pairs + 1, seed=21)
        half = len(state) // 2
        expected = np.concatenate([state[:half], _hopping_tower(angle, num_pairs) @ state[half:]])
        assert np.allclose(_applied_state(operation, state), expected, atol=1e-10)


class TestInteractionLayer:
    """The on-site tower phases every site pair through a Hamming-weight register and the gradient."""

    @pytest.mark.parametrize("multiple", [4, -13])
    def test_matches_the_separate_pair_rotations(self, multiple):
        """At eight sites the tower takes the HWP path; each basis state gets exp(-i angle sum Z Z)."""
        angle = multiple * _ANGLE_QUANTUM
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
            f"qs => {{ use gradient = Qubit[{_PLAQUETTE}.TowerGradientSize({sites}, -1, {_GRADIENT_BITS})]; "
            f"within {{ {_PLAQUETTE}.PreparePlaquetteGradient(gradient); }} "
            f"apply {{ {_PLAQUETTE}.InteractionLayer({angle}, {sites}, qs, -1, gradient); }} }}"
        )
        assert np.allclose(_applied_state(operation, amplitudes), expected, atol=1e-10)


def _single_z_tower(angle: float, count: int, cap: int, controlled: bool = False) -> str:
    """Return Q# phasing ``count`` single-qubit Z terms as one capped tower, on a gradient prepared around it."""
    lead = 1 if controlled else 0
    arguments = (
        f"({angle}, [[PauliZ], size = {count}], Std.Arrays.Chunks(1, qs[{lead}..{lead + count - 1}]), {cap}, gradient)"
    )
    call = (
        f"Controlled {_PLAQUETTE}.HammingWeightPhase([qs[0]], {arguments})"
        if controlled
        else f"{_PLAQUETTE}.HammingWeightPhase{arguments}"
    )
    return (
        f"qs => {{ use gradient = Qubit[{_PLAQUETTE}.TowerGradientSize({count}, {cap}, {_GRADIENT_BITS})]; "
        f"within {{ {_PLAQUETTE}.PreparePlaquetteGradient(gradient); }} apply {{ {call}; }} }}"
    )


def _single_z_phases(angle: float, count: int, num_qubits: int, lead: int, basis_states) -> np.ndarray:
    """Return exp(-i angle sum_j Z_j) over ``count`` qubits starting at ``lead``, per basis state."""
    phases = np.ones(2**num_qubits, dtype=complex)
    for index in basis_states:
        spins = [1 - 2 * ((index >> (num_qubits - 1 - qubit)) & 1) for qubit in range(num_qubits)]
        phases[index] = np.exp(-1j * angle * sum(spins[lead : lead + count]))
    return phases


def _sparse_state(num_qubits: int, seed: int, support: int = 24):
    """Return a normalized real state on a few basis states, and which states those are."""
    rng = np.random.default_rng(seed)
    basis_states = rng.choice(2**num_qubits, size=support, replace=False)
    amplitudes = np.zeros(2**num_qubits)
    amplitudes[basis_states] = rng.normal(size=len(basis_states))
    return amplitudes / np.linalg.norm(amplitudes), basis_states


class TestHammingWeightBatchCap:
    """Capping the batch splits a tower into several Hamming-weight registers without changing it.

    The phases are additive over the split, so every configuration below must reproduce the same
    exp(-i angle sum_j Z_j). Each batch carries its own ``R(PauliI, _)`` constant, so a batch-count
    error in that constant shows up as a wrong phase here rather than as a wrong gate count.
    """

    @pytest.mark.parametrize(
        ("count", "cap", "batches"),
        [
            (12, -1, "one batch, uncapped"),
            (16, 8, "two batches, both above the break-even"),
            (12, 8, "one batch above the break-even and one below"),
            (10, 5, "two batches, both below the break-even"),
        ],
    )
    def test_the_split_preserves_the_tower(self, count, cap, batches):
        """Whatever the split, the tower is still exp(-i angle sum_j Z_j)."""
        angle = 3 * _ANGLE_QUANTUM
        amplitudes, basis_states = _sparse_state(count, seed=count + cap)
        expected = amplitudes.astype(complex) * _single_z_phases(angle, count, count, 0, basis_states)

        actual = _applied_state(_single_z_tower(angle, count, cap), amplitudes)
        assert np.allclose(actual, expected, atol=1e-10), f"capped tower differs for {batches}"

    def test_a_capped_tower_is_controlled_as_a_whole(self):
        """Each batch's identity phase is not global under control, so a stray one is visible here."""
        angle, count, cap = 3 * _ANGLE_QUANTUM, 12, 8
        num_qubits = count + 1
        amplitudes, basis_states = _sparse_state(num_qubits, seed=77)

        # The control is qubit 0, so it is the most significant bit of the big-endian index.
        half = 2 ** (num_qubits - 1)
        phases = _single_z_phases(angle, count, num_qubits, 1, basis_states)
        expected = amplitudes.astype(complex)
        expected[half:] *= phases[half:]

        actual = _applied_state(_single_z_tower(angle, count, cap, controlled=True), amplitudes)
        assert np.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("count", "cap", "expected"),
        [(16, -1, 16), (16, 8, 8), (16, 16, 16), (16, 32, 16), (5, 8, 5), (1, 1, 1)],
    )
    def test_the_batch_size_is_the_cap_until_the_tower_is_shorter(self, count, cap, expected):
        """A cap at or above the tower length leaves it whole; -1 never splits."""
        assert QSHARP_UTILS.HubbardPlaquette.HammingWeightBatchSize(count, cap) == expected

    @pytest.mark.parametrize(
        ("count", "cap", "phased"),
        [(16, -1, True), (16, 8, True), (16, 7, False), (12, 8, True), (5, -1, False), (0, -1, False)],
    )
    def test_the_gradient_follows_the_batch_rather_than_the_tower(self, count, cap, phased):
        """A cap that leaves every batch below the break-even must not allocate a gradient it never uses."""
        size = QSHARP_UTILS.HubbardPlaquette.TowerGradientSize(count, cap, _GRADIENT_BITS)
        assert size == (_GRADIENT_BITS if phased else 0)


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
            f"qs => {{ use gradient = Qubit[{_PLAQUETTE}.TowerGradientSize({2 * len(cycles)}, -1, {_GRADIENT_BITS})]; "
            f"within {{ {_PLAQUETTE}.PreparePlaquetteGradient(gradient); }} "
            f"apply {{ {_PLAQUETTE}.HoppingLayer({kappa}, {literal}, qs, -1, gradient); }} }}"
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
                    num_divisions=num_divisions,
                    target_accuracy=0.0,
                ),
            ),
        )
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))

        result = iqpe.run(state_preparation=preparation, qubit_hamiltonian=_lattice_operator(2, 2, t=t, u=u))

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
    return create(
        "controlled_circuit_mapper",
        "hubbard_plaquette",
        control_indices=[0],
        rotation_bit_precision=_GRADIENT_BITS,
    ).run(UnitaryRepresentation(container=container))


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


# Angles whose eigenphase is exactly pi / 2, so every measured bit is deterministic. The hopping
# angle is a multiple of 16 quanta, so every angle the layers derive from it -- halved for the
# boundary, halved again inside the tiling, doubled for the squared query -- stays on the gradient
# lattice and the Hamming-weight rotations are exact.
_HOPPING = 20 * _ANGLE_QUANTUM
_INTERACTION = (2.0 * _HOPPING - np.pi / 2) / (_CATALYST_SITES - 2)


class TestSharedPlaquetteCatalysts:
    """Phase estimation prepares the plaquette phase gradient once and shares it across queries."""

    def test_mapper_declares_its_phase_gradient(self):
        """Every tower phases through one binary gradient, whose state carries no angle."""
        circuit = _plaquette_controlled(0.3, 0.2, 1)
        assert circuit.metadata.num_phase_gradient_ancillas == _GRADIENT_BITS
        assert circuit.num_qubits == 2 * _CATALYST_SITES + _GRADIENT_BITS

    def test_a_lattice_below_the_break_even_declares_none(self):
        """The 2x2 towers rotate term by term, so there is nothing to share."""
        assert _plaquette_controlled(0.3, 0.2, 1, width=2, height=2).metadata.num_phase_gradient_ancillas == 0

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
        assert circuit._qsharp_factory.parameter["numSharedAncillas"] == _GRADIENT_BITS

        outcomes = {int(shot[0] == Result.One) for shot in _run(circuit, shots=4)}
        assert outcomes == {expected}

    @pytest.mark.slow
    def test_standard_estimation_shares_one_register_across_powers(self):
        """Repetitions change the power but not the gradient, so every query reuses the same register.

        The eigenphase is pi / 2, so two phase bits read exactly one quarter.
        """
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        powers = [_plaquette_controlled(_INTERACTION, _HOPPING, reps) for reps in (2, 1)]
        circuit = builder._create_circuit_from_qsharp_op(_one_electron_preparation(), powers, 2, 2 * _CATALYST_SITES)
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSharedAncillas"] == _GRADIENT_BITS
        assert parameters["prepareSharedOp"] is not QSHARP_UTILS.PrepSelPrep.NoOpPrepare

        assert {_standard_phase(shot) for shot in _run(circuit, shots=2)} == {0.25}

    @pytest.mark.slow
    def test_rescaled_powers_share_the_whole_register(self):
        """A binary gradient does not depend on the layer angles, so rescaled powers share all of it."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        powers = [
            _plaquette_controlled(2 * _INTERACTION, 2 * _HOPPING, 1),
            _plaquette_controlled(_INTERACTION, _HOPPING, 1),
        ]
        circuit = builder._create_circuit_from_qsharp_op(_one_electron_preparation(), powers, 2, 2 * _CATALYST_SITES)
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSharedAncillas"] == _GRADIENT_BITS
        assert parameters["prepareSharedOp"] is not QSHARP_UTILS.PrepSelPrep.NoOpPrepare

        assert {_standard_phase(shot) for shot in _run(circuit, shots=2)} == {0.25}

    def test_mismatched_gradient_requests_are_rejected(self):
        """One register cannot serve two widths, and silently dropping one would emit a broken circuit."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        circuits = [_plaquette_controlled(0.3, 0.2, 1), _plaquette_controlled(0.3, 0.2, 1, width=2, height=2)]
        with pytest.raises(ValueError, match="same phase gradient register"):
            builder._shared_register(circuits)

    def test_a_lattice_below_the_break_even_shares_nothing(self):
        """Nothing is prepared when no controlled unitary asks for a gradient."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        circuits = [_plaquette_controlled(0.3, 0.2, 1, width=2, height=2)] * 2
        ops, prepare, num_shared = builder._shared_register(circuits)
        assert len(ops) == 2
        assert num_shared == 0
        assert prepare is QSHARP_UTILS.PrepSelPrep.NoOpPrepare

    def test_the_mapper_defaults_to_an_uncapped_batch(self):
        """The cap is opt-in, so the default must leave the tower whole."""
        mapper = create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0])
        assert int(mapper.settings().get("max_hwp_batch_size")) == -1

    def test_the_mapper_rejects_a_zero_cap(self):
        """Zero terms per batch would phase nothing, and zero is inside the setting's range."""
        container = HubbardPlaquetteContainer(width=4, height=2, interaction_angle=0.3, hopping_angle=0.2, step_reps=1)
        mapper = create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0], max_hwp_batch_size=0)
        with pytest.raises(ValueError, match="max_hwp_batch_size must be -1 or a positive integer"):
            mapper.run(UnitaryRepresentation(container=container))

    def test_the_mapper_rejects_a_cap_below_the_no_cap_sentinel(self):
        """Only -1 means 'no cap', so the declared range refuses anything more negative."""
        with pytest.raises(ValueError, match="allowed range"):
            create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0], max_hwp_batch_size=-2)


def _tiling_matrices(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the pink and gold single-particle hopping matrices at unit amplitude."""
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
    return matrix_p, matrix_g


def _dense_commutator_trace_norm(width: int, height: int) -> float:
    """Return ``||[[R_p, R_g], R_g]||_1`` by a singular value decomposition of the dense matrix."""
    matrix_p, matrix_g = _tiling_matrices(width, height)
    inner = matrix_p @ matrix_g - matrix_g @ matrix_p
    outer = inner @ matrix_g - matrix_g @ inner
    return float(np.linalg.svd(outer, compute_uv=False).sum())


def _dense_hopping_trace_norm(width: int, height: int) -> float:
    """Return ``||R_p + R_g||_1`` by a singular value decomposition of the dense matrix."""
    matrix_p, matrix_g = _tiling_matrices(width, height)
    return float(np.linalg.svd(matrix_p + matrix_g, compute_uv=False).sum())


def _reference_w_plaquette(width: int, height: int, *, t: float, u: float) -> float:
    """Recompute Campbell's W_PLAQ independently of the builder.

    Follows Eq. (10) of Campbell arXiv:2012.09238v4 for W_SO2, Eq. (D10) for the
    plaquette-splitting term, and Eq. (D6) for their sum. Both norms come from dense
    singular value decompositions rather than the builder's closed forms.
    """
    num_sites = width * height
    commutator_norm = _dense_commutator_trace_norm(width, height) * t**3
    hopping_norm = _dense_hopping_trace_norm(width, height) * t

    w_so2 = u * t**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + u**2 / 24.0 * hopping_norm
    return w_so2 + 3.0 / 24.0 * commutator_norm


def _auto_step_count(width: int, height: int, *, t: float, u: float, time: float, target_accuracy: float) -> int:
    """Ask the builder for the step count it derives from an accuracy target."""
    builder = HubbardPlaquetteTrotter(
        order=2,
        time=time,
        num_divisions=1,
        target_accuracy=target_accuracy,
    )
    return builder._step_count(t, u, width, height, time)


class TestAutomaticStepCount:
    """The ``target_accuracy`` path, which sizes the step count from the error bound."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (6, 6), (8, 8)])
    @pytest.mark.parametrize("phase", [1e-6, 0.25, 1.0, math.pi / 2])
    def test_matches_the_exact_rule_of_apel_algorithm_1(self, width, height, phase):
        """Reproduce r = ceil(sqrt(W_PLAQ tau^3 / (2 sin(eps tau / 2)))) against an independent W."""
        t, u, time = 1.0, 8.0, 3.0
        target_accuracy = phase / time

        count = _auto_step_count(width, height, t=t, u=u, time=time, target_accuracy=target_accuracy)

        w_plaquette = _reference_w_plaquette(width, height, t=t, u=u)
        expected = math.ceil(math.sqrt(w_plaquette * time**3 / (2.0 * math.sin(phase / 2.0))))
        assert count == expected

    @pytest.mark.parametrize(
        ("width", "height"),
        [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (4, 6), (4, 8), (6, 8), (8, 12)],
    )
    def test_the_closed_form_norms_match_a_dense_decomposition(self, width, height):
        """Even cell counts are where a wrong Bloch phase shows, so they are covered on both axes."""
        assert HubbardPlaquetteTrotter._commutator_trace_norm(width, height) == pytest.approx(
            _dense_commutator_trace_norm(width, height), abs=1e-9
        )
        assert HubbardPlaquetteTrotter._hopping_trace_norm(width, height) == pytest.approx(
            _dense_hopping_trace_norm(width, height), abs=1e-9
        )

    def test_saturates_once_the_accuracy_target_exceeds_a_half_turn(self):
        """||Delta U|| <= 2 caps the arcsine, so the count stops falling at eps tau = pi."""
        t, u, time = 1.0, 8.0, 3.0

        at_pi = _auto_step_count(4, 4, t=t, u=u, time=time, target_accuracy=math.pi / time)
        beyond_pi = _auto_step_count(4, 4, t=t, u=u, time=time, target_accuracy=3.0 * math.pi / time)

        assert beyond_pi == at_pi

    def test_a_disabled_target_leaves_the_manual_count_alone(self):
        builder = HubbardPlaquetteTrotter(order=2, time=3.0, num_divisions=7, target_accuracy=0.0)
        assert builder._step_count(1.0, 8.0, 4, 4, 3.0) == 7

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
        operator = _lattice_operator(width, height, t=1.0, u=u)

        unshifted = (
            HubbardPlaquetteTrotter(order=2, time=time, num_divisions=divisions, target_accuracy=0.0)
            .run(operator)
            .get_container()
        )
        shifted = (
            HubbardPlaquetteTrotter(
                order=2,
                time=time,
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

    @pytest.mark.parametrize("num_electrons", [2, 4, 6])
    def test_the_shift_maps_the_symmetric_spectrum_onto_the_conventional_one(self, num_electrons):
        r"""Check the offset against exact diagonalization, not against its own formula.

        The circuit evolves the particle-hole-symmetric interaction
        :math:`U \sum_i (n_{i\uparrow} - 1/2)(n_{i\downarrow} - 1/2)`, which is pure :math:`ZZ`,
        and recovers the conventional :math:`U \sum_i n_{i\uparrow} n_{i\downarrow}` by adding
        :math:`U\eta/2 - UM/4` classically. That identity holds because the two differ by
        :math:`(U/2)\hat{N} - UM/4`, and :math:`\hat{N}` is conserved, so on a sector of fixed
        electron count the difference is a number rather than an operator.

        ``test_the_electron_count_shifts_to_the_conventional_model`` checks that
        ``constant_shift`` implements that formula. It cannot check the formula itself: it
        recomputes the same expression it is testing, so a sign error or a factor of two would
        pass. Diagonalizing both Hamiltonians in each particle-number sector does check it.
        """
        width = height = 2
        t, u = 1.0, 8.0
        sites = width * height
        num_qubits = 2 * sites

        symmetric = _reference_hamiltonian(width, height, t=t, u=u)
        conventional = _conventional_hamiltonian(width, height, t=t, u=u)

        sector = [state for state in range(2**num_qubits) if bin(state).count("1") == num_electrons]
        block = np.ix_(sector, sector)
        symmetric_levels = np.linalg.eigvalsh(symmetric[block])
        conventional_levels = np.linalg.eigvalsh(conventional[block])

        shift = u * (0.5 * num_electrons - 0.25 * sites)
        assert np.allclose(conventional_levels, symmetric_levels + shift, atol=1e-10), (
            f"the {num_electrons}-electron spectra differ by more than the classical shift {shift}"
        )

        # The sign matters and is easy to get backwards, so pin the direction too when the
        # sector has a nonzero offset. At eta = M/2 the two conventions coincide.
        if not np.isclose(shift, 0.0):
            assert (conventional_levels[0] > symmetric_levels[0]) == (shift > 0.0), "the shift has the wrong sign"


#: Settings of the ``examples/estimation_hubbard_2d.ipynb`` benchmark, repeated here so that the
#: pinned counts below depend on the library rather than on the notebook. The notebook explains
#: where each one comes from; the short version is that ``U = 8t`` is the strong-coupling regime,
#: the error budget is allocated per site, and minimizing the total Trotter step count
#: ``sum_k r_k ~ 1 / (f * sqrt(1 - f))`` over the phase-estimation share ``f`` gives ``f = 2/3``,
#: the same optimum used in Campbell (arXiv:2012.09238v4, App. F).
_BENCHMARK_HOPPING_T = 1.0
_BENCHMARK_U_OVER_T = 8.0
_BENCHMARK_FILLING = 0.875
_BENCHMARK_PRECISION_PER_SITE = 0.0051
_BENCHMARK_QPE_BITS = 10
_BENCHMARK_QPE_BUDGET_FRACTION = 2.0 / 3.0
_BENCHMARK_TROTTER_ORDER = 2


#: Rotation synthesis rounds transcendental angles, so these two counts drift by a few
#: units across platforms (the pins are Linux; other platforms have been seen up to +5, on
#: Windows ARM64). The phase gradient leaves only tens of thousands of rotations at L=4, where
#: a relative tolerance alone would not absorb that drift, so they are pinned to the larger of
#: a relative and an absolute tolerance; every other column stays exact.
_PLATFORM_SENSITIVE_COLUMNS = frozenset({"rotations", "rotation_depth"})
_PLATFORM_RELATIVE_TOLERANCE = 1e-4
_PLATFORM_ABSOLUTE_TOLERANCE = 16


#: L=2 has four sites, which is below the Hamming-weight-phasing break-even of eight terms, so
#: every tower rotates term by term: no adder tree, and therefore no Toffolis at all.
_HUBBARD_L2_FULL_CIRCUIT = {
    "L": 2,
    "sites": 4,
    "system_qubits": 8,
    "electrons": 4,
    "target_precision": 0.0204,
    "qpe_budget": 0.013600000000000001,
    "trotter_budget": 0.0068000000000000005,
    "qpe_bits": 10,
    "base_time": 0.2253660323553513,
    "logical_qubits": 18,
    "rotations": 1047435,
    "rotation_depth": 698444,
    "t_gates": 697995,
    "ccz_count": 0,
    "ccix_count": 0,
    "toffolis": 0,
    "measurements": 10,
}


#: L=4 has sixteen sites, so every tower is above the break-even and takes the
#: Hamming-weight-phasing path: an adder tree compresses sixteen same-angle rotations into a
#: five-bit weight, and each place value is rotated through the shared ten-qubit binary phase
#: gradient, which turns the place-value rotations into Toffolis. This is the case that
#: exercises the construction, which is why it is pinned.
_HUBBARD_L4_FULL_CIRCUIT = {
    "L": 4,
    "sites": 16,
    "system_qubits": 32,
    "electrons": 14,
    "target_precision": 0.0816,
    "qpe_budget": 0.054400000000000004,
    "trotter_budget": 0.027200000000000002,
    "qpe_bits": 10,
    "base_time": 0.056341508088837824,
    "logical_qubits": 87,
    "rotations": 24549,
    "rotation_depth": 24257,
    "t_gates": 758481,
    "ccz_count": 1421400,
    "ccix_count": 0,
    "toffolis": 1421400,
    "measurements": 1421410,
}


def _benchmark_schedule(size: int) -> tuple[float, float, float, float]:
    """Return the energy budgets and base evolution time the benchmark uses for one lattice."""
    energy_budget = _BENCHMARK_PRECISION_PER_SITE * size * size
    qpe_budget = _BENCHMARK_QPE_BUDGET_FRACTION * energy_budget
    trotter_budget = energy_budget - qpe_budget
    # A sine-windowed QPE phase state of N = 2^bits - 1 queries has spread
    # tan(pi / (N + 2)), which equals the required eps_QPE * tau.
    base_time = math.tan(math.pi / (2**_BENCHMARK_QPE_BITS - 1 + 2)) / qpe_budget
    return energy_budget, qpe_budget, trotter_budget, base_time


def _benchmark_logical_counts(size: int, max_batch_size: int = -1) -> dict:
    """Build the benchmark's phase-estimation circuit for one lattice and trace its gate counts."""
    operator = _lattice_operator(size, size, t=_BENCHMARK_HOPPING_T, u=_BENCHMARK_U_OVER_T * _BENCHMARK_HOPPING_T)
    energy_budget, qpe_budget, trotter_budget, base_time = _benchmark_schedule(size)

    unitary_builder = AlgorithmRef(
        "hamiltonian_unitary_builder",
        "hubbard_plaquette",
        order=_BENCHMARK_TROTTER_ORDER,
        time=base_time,
        # Bit k evolves for base_time * 2^k rather than repeating the block 2^k times.
        power_strategy="rescale",
        target_accuracy=trotter_budget,
    )
    circuit_builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=_BENCHMARK_QPE_BITS,
        unitary_builder=unitary_builder,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper", "hubbard_plaquette", max_hwp_batch_size=max_batch_size
        ),
    )
    circuit_builder.settings().set("phase_state", "sine")

    built = circuit_builder.run(identity_state_prep(num_qubits=operator.num_qubits), operator)[0]
    factory = built._qsharp_factory
    assert factory is not None, "the QPE circuit does not have Q# factory data"
    counts = dict(get_qsharp_context().logical_counts(factory.program, *factory.parameter.values()))

    ccz_count = int(counts.get("cczCount", 0))
    ccix_count = int(counts.get("ccixCount", 0))
    return {
        "L": size,
        "sites": size * size,
        "system_qubits": operator.num_qubits,
        "electrons": round(_BENCHMARK_FILLING * size * size),
        "target_precision": energy_budget,
        "qpe_budget": qpe_budget,
        "trotter_budget": trotter_budget,
        "qpe_bits": _BENCHMARK_QPE_BITS,
        "base_time": base_time,
        "logical_qubits": int(counts["numQubits"]),
        "rotations": int(counts.get("rotationCount", 0)),
        "rotation_depth": int(counts.get("rotationDepth", 0)),
        "t_gates": int(counts.get("tCount", 0)),
        "ccz_count": ccz_count,
        "ccix_count": ccix_count,
        "toffolis": ccz_count + ccix_count,
        "measurements": int(counts.get("measurementCount", 0)),
    }


def _compare_counts(actual: dict, expected: dict) -> list[str]:
    """Return one message per pinned column the traced circuit does not match."""
    mismatches = []
    for column, want in expected.items():
        got = actual[column]
        if column in _PLATFORM_SENSITIVE_COLUMNS:
            matches = got == pytest.approx(want, rel=_PLATFORM_RELATIVE_TOLERANCE, abs=_PLATFORM_ABSOLUTE_TOLERANCE)
            tolerance = f" (rel={_PLATFORM_RELATIVE_TOLERANCE}, abs={_PLATFORM_ABSOLUTE_TOLERANCE})"
        else:
            matches = got == pytest.approx(want) if isinstance(want, float) else got == want
            tolerance = ""
        if not matches:
            mismatches.append(f"  L={expected['L']} {column}: expected {want}{tolerance}, got {got}")
    return mismatches


class TestBenchmarkLogicalResources:
    """Pin the logical cost of the benchmark circuit that ``estimation_hubbard_2d.ipynb`` reports.

    These run unconditionally. The notebook's own end-to-end test is slow-gated and needs a
    Jupyter kernel, so it does not run on an ordinary push; this pin is the guard that does.
    """

    def test_lattice_below_the_break_even(self):
        """L=2 rotates term by term, so it must carry no adder tree and no Toffolis."""
        actual = _benchmark_logical_counts(2)
        mismatches = _compare_counts(actual, _HUBBARD_L2_FULL_CIRCUIT)
        assert not mismatches, "Mismatches found:\n" + "\n".join(mismatches)

        # Below the break-even zero Toffolis is the correct answer, not a collapsed circuit.
        assert actual["toffolis"] == 0, "the 2x2 lattice is below the break-even and phases term by term"

    def test_lattice_above_the_break_even(self):
        """L=4 is the case that exercises the adder tree and the Hamming-weight rotation ladder."""
        actual = _benchmark_logical_counts(4)
        mismatches = _compare_counts(actual, _HUBBARD_L4_FULL_CIRCUIT)
        assert not mismatches, "Mismatches found:\n" + "\n".join(mismatches)

        # A collapse back to the term-by-term fallback would silently erase the adder tree, which
        # is the only source of Toffolis in this circuit. Pin the sign of the count, not just its
        # value: this exact regression was golden-ed in once before it was caught.
        assert actual["toffolis"] > 0, "the 4x4 lattice must take the Hamming-weight-phasing path"

    def test_a_cap_at_least_the_tower_length_changes_nothing(self):
        """The 4x4 towers are sixteen terms long, so a cap of sixteen cannot split any of them."""
        assert _benchmark_logical_counts(4, max_batch_size=16) == _benchmark_logical_counts(4)

    def test_capping_trades_qubits_for_toffolis(self):
        """Halving the batch releases the adder-tree scratch sooner and pays for it in Toffolis."""
        uncapped = _benchmark_logical_counts(4)
        capped = _benchmark_logical_counts(4, max_batch_size=8)

        assert capped["logical_qubits"] < uncapped["logical_qubits"], "a shorter batch must hold fewer ancillas"
        assert capped["toffolis"] > uncapped["toffolis"], "each batch pays its own place-value gradient additions"

    def test_a_cap_below_the_break_even_falls_back(self):
        """No batch can reach eight terms, so the adder tree disappears entirely."""
        capped = _benchmark_logical_counts(4, max_batch_size=4)

        assert capped["toffolis"] == 0, "every batch is below the break-even, so nothing is phased through a tree"
        assert capped["logical_qubits"] < _benchmark_logical_counts(4)["logical_qubits"]
