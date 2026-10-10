"""Tests for the Hubbard plaquette Trotter builder and its Q# lowering."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Mapping

import h5py
import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.hubbard_plaquette_trotter import (
    HubbardPlaquetteTrotter,
)
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    FermiHubbardModelHamiltonianDescription,
    LatticeGeometry,
    MajoranaMapping,
    SettingTypeMismatch,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix, pauli_to_sparse_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

_PLAQUETTE = "QDKChemistry.Utils.HubbardPlaquette"
_HWP = "QDKChemistry.Utils.HammingWeightPhasing"

#: Width of the phase gradient the tests apply the Hamming-weight rotations through. A b-qubit
#: gradient rounds the rotation of a tower at angle a to a multiple of 2 pi / 2**b, so the tests
#: keep every tower angle on that lattice. The construction is then exact and the assertions can
#: stay at 1e-10, while the register stays small enough to simulate: the gradient is a uniform
#: superposition, so it multiplies the simulated state's support by 2**b.
_GRADIENT_BITS = 6
_ANGLE_QUANTUM = 2.0 * math.pi / 2**_GRADIENT_BITS

#: Run a Hamming-weight test both ways the place values can be rotated: through a phase gradient
#: of ``_GRADIENT_BITS`` qubits, and, with an empty gradient, as the original ladder of ``Rz``.
_GRADIENT_OR_LADDER = pytest.mark.parametrize("bits", [_GRADIENT_BITS, 0], ids=["phase_gradient", "rz_ladder"])


def _model(
    size: int, *, t: float = 1.0, u: float = 0.0, epsilon: float = 0.0, periodic: bool = True
) -> FermiHubbardModelHamiltonianDescription:
    """Return a Hubbard model on a ``size`` x ``size`` square lattice."""
    lattice = LatticeGeometry.square(size, size, periodic_x=periodic, periodic_y=periodic)
    return FermiHubbardModelHamiltonianDescription(lattice, t=t, u=u, epsilon=epsilon)


def _reference_hamiltonian(model: FermiHubbardModelHamiltonianDescription, *, symmetric: bool = True) -> np.ndarray:
    """Return the dense Jordan-Wigner matrix of the 2x2 ``model``, or of its particle-hole symmetric form.

    The 2x2 torus joins each pair through two periodic images, whose hoppings ``materialize()`` sums.
    """
    t, u, epsilon = (model.parameters[name] for name in ("t", "u", "epsilon"))
    reference = FermiHubbardModelHamiltonianDescription(
        model.lattice, t=t, u=u, epsilon=-0.5 * u if symmetric else epsilon
    )
    hamiltonian = reference.materialize()
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(8))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    dense = pauli_to_dense_matrix(list(labels), list(coefficients))
    return dense + 0.25 * u * model.lattice.num_sites * np.eye(len(dense)) if symmetric else dense


def _operation(
    name: str, args: str, *, controlled: bool = False, namespace: str = _PLAQUETTE, gradient: str | None = None
) -> str:
    """Return a Q# lambda calling ``name``, controlled on ``qs[0]`` if asked.

    With ``gradient``, a Q# expression for its width, the lambda allocates a phase gradient,
    prepares it around the call and passes it as ``gradient``. Only the call is controlled, as in
    phase estimation, which prepares the gradient once outside every query.
    """
    if controlled:
        call = f"Controlled {namespace}.{name}([qs[0]], ({args.format(qs='qs[1...]')}))"
    else:
        call = f"{namespace}.{name}({args.format(qs='qs')})"
    if gradient is None:
        return f"qs => {{ {call}; }}"
    return (
        f"qs => {{ use gradient = Qubit[{gradient}]; "
        f"within {{ {_PLAQUETTE}.PreparePlaquetteGradient(gradient); }} apply {{ {call}; }} }}"
    )


def _params(
    width: int,
    height: int,
    interaction_angle: float,
    hopping_angle: float,
    repetitions: int,
    *,
    cap: int = -1,
    bits: int = _GRADIENT_BITS,
) -> str:
    """Return a Q# ``HubbardPlaquetteParams``; ``bits`` of zero selects the ``Rz`` ladder."""
    return (
        f"{_PLAQUETTE}.HubbardPlaquetteParams({width}, {height}, {float(interaction_angle)}, "
        f"{float(hopping_angle)}, {repetitions}, {cap}, {'true' if bits else 'false'}, {bits or _GRADIENT_BITS})"
    )


def _evolution(params: str, *, controlled: bool = False) -> str:
    """Return a Q# lambda applying ``RepPlaquetteExp`` on the phase gradient ``params`` asks for."""
    return _operation(
        "RepPlaquetteExp",
        params + ", {qs}, gradient",
        controlled=controlled,
        gradient=f"{_PLAQUETTE}.PlaquetteGradientSize({params})",
    )


def _apply(operation: str, state: np.ndarray) -> np.ndarray:
    """Return the state ``operation`` produces from a real-amplitude ``state``.

    This dumps the whole machine rather than using ``dump_operation_on_state``, whose ``DumpRegister``
    rejects states with nonzero amplitudes below its zero cutoff as not separable, which the 16-qubit
    evolutions reach. Every helper qubit is released by then, so the machine is the register.
    """
    context = get_qsharp_context()
    if not hasattr(context.code, "_PlaquetteTestDumpMachine"):
        context.eval(
            "operation _PlaquetteTestDumpMachine(op : (Qubit[] => Unit), numQubits : Int, initial : Double[]) : Unit {"
            " use qubits = Qubit[numQubits];"
            " Std.StatePreparation.PreparePureStateD(initial, qubits);"
            " op(qubits);"
            " Std.Diagnostics.DumpMachine();"
            " ResetAll(qubits); }"
        )
    num_qubits = round(math.log2(len(state)))
    run = context.run(
        context.code._PlaquetteTestDumpMachine,
        1,
        context.eval(operation),
        num_qubits,
        state.tolist(),
        save_events=True,
    )
    result = np.zeros(len(state), dtype=complex)
    for index, amplitude in run[0]["events"][-1].state_dump().get_dict().items():
        result[index] = amplitude
    return result


def _random_state(num_qubits: int, seed: int, support: int | None = None) -> np.ndarray:
    """Return a normalized real state, dense or on ``support`` random basis states."""
    rng = np.random.default_rng(seed)
    state = np.zeros(2**num_qubits)
    indices = rng.choice(len(state), size=support, replace=False) if support else slice(None)
    state[indices] = rng.normal(size=state[indices].shape)
    return state / np.linalg.norm(state)


class TestHammingWeightPhasing:
    """Batched phase towers apply the same unitary as their separate rotations."""

    @_GRADIENT_OR_LADDER
    @pytest.mark.parametrize(
        ("count", "cap", "controlled"),
        [(12, -1, False), (16, 8, False), (12, 8, False), (10, 5, False), (12, 8, True)],
    )
    def test_z_tower(self, count, cap, controlled, bits):
        """exp(-i a sum_j Z_j) under any batch split, with or without control."""
        angle = 3 * _ANGLE_QUANTUM
        state = _random_state(count + controlled, seed=count + cap, support=24)
        weights = np.array([(index % 2**count).bit_count() for index in range(len(state))])
        phases = np.exp(-1j * angle * (count - 2 * weights))
        phases[: len(state) - 2**count] = 1.0  # The control, when present, is the most significant qubit.
        args = f"{angle}, [[PauliZ], size = {count}], Std.Arrays.Chunks(1, {{qs}}), {cap}, gradient"
        gradient = f"{_HWP}.TowerGradientSize({count}, {cap}, {bits})"
        actual = _apply(
            _operation("HammingWeightPhase", args, controlled=controlled, namespace=_HWP, gradient=gradient), state
        )
        assert np.allclose(actual, phases * state, atol=1e-10)

    def test_the_rz_ladder_is_exact_off_the_gradient_lattice(self):
        """Without a gradient nothing is rounded, so an angle no gradient could hold is still exact."""
        angle, count = 0.1234567, 12
        state = _random_state(count, seed=91, support=24)
        weights = np.array([index.bit_count() for index in range(len(state))])
        args = f"{angle}, [[PauliZ], size = {count}], Std.Arrays.Chunks(1, {{qs}}), -1, []"
        actual = _apply(_operation("HammingWeightPhase", args, namespace=_HWP), state)
        assert np.allclose(actual, np.exp(-1j * angle * (count - 2 * weights)) * state, atol=1e-10)

    @pytest.mark.parametrize(
        ("count", "cap", "expected"),
        [(16, -1, 16), (16, 8, 8), (16, 16, 16), (16, 32, 16), (5, 8, 5), (1, 1, 1)],
    )
    def test_batch_size(self, count, cap, expected):
        assert QSHARP_UTILS.HammingWeightPhasing.HammingWeightBatchSize(count, cap) == expected

    @pytest.mark.parametrize(
        ("count", "cap", "phased"),
        [(16, -1, True), (16, 8, True), (16, 7, False), (12, 8, True), (5, -1, False), (0, -1, False)],
    )
    def test_the_gradient_follows_the_batch_rather_than_the_tower(self, count, cap, phased):
        """A cap that leaves every batch below the break-even must not allocate a gradient it never uses."""
        size = QSHARP_UTILS.HammingWeightPhasing.TowerGradientSize(count, cap, _GRADIENT_BITS)
        assert size == (_GRADIENT_BITS if phased else 0)

    @_GRADIENT_OR_LADDER
    @pytest.mark.parametrize(("num_pairs", "controlled"), [(2, False), (5, False), (4, True)])
    def test_hopping_tower(self, num_pairs, controlled, bits):
        """exp(i a (XX + YY)) on every pair, below and above the eight-rotation break-even."""
        angle = 3 * _ANGLE_QUANTUM
        xx_plus_yy = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]) + np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
        pair_gate = scipy.linalg.expm(1j * angle * xx_plus_yy)
        state = _random_state(2 * num_pairs + controlled, seed=num_pairs, support=8 if num_pairs > 5 else None)

        expected = state.astype(complex).reshape(1 + controlled, -1)
        for pair in range(num_pairs):
            target = expected[-1].reshape(4**pair, 4, -1)
            expected[-1] = np.einsum("ij,ajb->aib", pair_gate, target).reshape(-1)
        args = f"{angle}, Std.Arrays.Chunks(2, {{qs}}), -1, gradient"
        gradient = f"{_HWP}.TowerGradientSize({2 * num_pairs}, -1, {bits})"
        actual = _apply(_operation("HoppingPhases", args, controlled=controlled, gradient=gradient), state)
        assert np.allclose(actual, expected.reshape(-1), atol=1e-10)

    @_GRADIENT_OR_LADDER
    @pytest.mark.parametrize("multiple", [4, -13])
    def test_interaction_tower(self, multiple, bits):
        """exp(-i a sum_s Z_s Z_{s+M}) over eight sites, which takes the batched path."""
        angle, sites = multiple * _ANGLE_QUANTUM, 8
        state = _random_state(2 * sites, seed=5, support=24)
        spins = 1 - 2 * ((np.arange(len(state))[:, None] >> np.arange(2 * sites - 1, -1, -1)) & 1)
        phases = np.exp(-1j * angle * (spins[:, :sites] * spins[:, sites:]).sum(axis=1))
        gradient = f"{_HWP}.TowerGradientSize({sites}, -1, {bits})"
        actual = _apply(
            _operation("InteractionLayer", f"{angle}, {sites}, {{qs}}, -1, gradient", gradient=gradient), state
        )
        assert np.allclose(actual, phases * state, atol=1e-10)


def _basis_index(bits: list[int]) -> int:
    """Return the dump index of the basis state with qubit ``q`` in ``bits[q]``; qubit 0 is the most significant."""
    return int("".join(map(str, bits)), 2)


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
        actual = [int(word) for word in QSHARP_UTILS.HammingWeightPhasing.BinaryGradientWords(phi, n, bits)]
        assert actual == _binary_gradient_words(phi, n, bits)
        assert all(0 <= word < 2**bits for word in actual), "a word must fit the gradient register"

    def test_a_lattice_angle_is_represented_exactly(self):
        """Angles that are multiples of 4 pi / 2^bits round to themselves, so the phasing is exact."""
        bits, k = 5, 3
        phi = 4.0 * math.pi * k / 2**bits
        actual = [int(word) for word in QSHARP_UTILS.HammingWeightPhasing.BinaryGradientWords(phi, 4, bits)]
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
            f"Controlled {_HWP}.PhaseByBinaryGradient([qs[0]], ({words}, {weight}, {gradient})); "
            f"Controlled R([qs[0]], (PauliI, {offset}, {weight}[0]));"
            if controlled
            else f"{_HWP}.PhaseByBinaryGradient({words}, {weight}, {gradient}); R(PauliI, {offset}, {weight}[0]);"
        )
        return f"qs => {{ within {{ {_PLAQUETTE}.PreparePlaquetteGradient({gradient}); }} apply {{ {body} }} }}"

    @staticmethod
    def _index(control: list[int], weight: int, n: int) -> int:
        """Return the dump index of ``control``, then little-endian ``weight``, then a gradient in |0>."""
        return _basis_index([*control, *((weight >> j) & 1 for j in range(n)), *[0] * TestPhaseByBinaryGradient.BITS])

    @pytest.mark.parametrize("n", [1, 3])
    def test_phases_every_weight_and_restores_the_gradient(self, n):
        """Every weight picks up e^{i phi w}; the gradient ends in |0> after unpreparation."""
        state = np.zeros(2 ** (n + self.BITS))
        expected = np.zeros(len(state), dtype=complex)
        for weight in range(2**n):
            state[self._index([], weight, n)] = 2 ** (-n / 2)
            expected[self._index([], weight, n)] = 2 ** (-n / 2) * np.exp(1j * self.PHI * weight)
        assert np.allclose(_apply(self._operation(n, controlled=False), state), expected, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3])
    def test_controlled_phasing_acts_only_when_the_control_is_set(self, n):
        """Loading the word under control leaves the addition, and the gradient, uncontrolled."""
        state = np.zeros(2 ** (1 + n + self.BITS))
        expected = np.zeros(len(state), dtype=complex)
        for control in (0, 1):
            for weight in range(2**n):
                index = self._index([control], weight, n)
                state[index] = 2 ** (-(n + 1) / 2)
                expected[index] = state[index] * (np.exp(1j * self.PHI * weight) if control else 1.0)
        assert np.allclose(_apply(self._operation(n, controlled=True), state), expected, atol=1e-10)


class TestPhaseEstimationAccuracy:
    """QPE over the plaquette evolution recovers the exact 2x2 ground energy."""

    @pytest.mark.parametrize(
        ("method", "power_strategy", "t", "u", "num_divisions", "energy"),
        [
            pytest.param("qdk_iterative", "repeat", 1.0, 0.0, 1, -8.0, id="iterative-hopping-only"),
            pytest.param("qdk_iterative", "repeat", 0.5, 2.0, 4, -4.8284271247, id="iterative-weak-coupling"),
            pytest.param("qdk_iterative", "repeat", 1.0, 4.0, 8, -9.6568542495, id="iterative-strong-coupling"),
            pytest.param("qdk_standard", "rescale", 1.0, 4.0, 8, -9.6568542495, id="standard-rescaled"),
        ],
    )
    def test_recovers_the_ground_energy(self, method, power_strategy, t, u, num_divisions, energy):
        num_bits = 4
        values, vectors = np.linalg.eigh(_reference_hamiltonian(_model(2, t=t, u=u)))
        assert values[0] == pytest.approx(energy, abs=1e-9)
        ground = np.real(vectors[:, 0]) / np.linalg.norm(np.real(vectors[:, 0]))
        params = {"rowMap": list(range(7, -1, -1)), "stateVector": ground.tolist(), "expansionOps": [], "numQubits": 8}
        preparation = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params),
        )

        unitary_builder = AlgorithmRef(
            "hamiltonian_unitary_builder",
            "hubbard_plaquette",
            order=2,
            # The ground phase is exactly 1/16, so every bit is deterministic.
            time=2 * np.pi / (2**num_bits * abs(energy)),
            num_divisions=num_divisions,
            power_strategy=power_strategy,
            target_accuracy=0.0,
        )
        shots = {"shots_per_bit": 15} if method == "qdk_iterative" else {"shots": 3}
        qpe = create(
            "phase_estimation",
            method,
            **shots,
            qpe_circuit_builder=AlgorithmRef(
                "qpe_circuit_builder",
                method,
                num_bits=num_bits,
                unitary_builder=unitary_builder,
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "hubbard_plaquette"),
            ),
            circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
        )
        result = qpe.run(state_preparation=preparation, qubit_hamiltonian=_model(2, t=t, u=u))
        assert result.raw_energy == pytest.approx(energy, rel=1e-6)


class TestPlaquetteCircuit:
    """The plaquette layers tile the lattice, route locally, and evolve states exactly."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (4, 6), (6, 6), (8, 8)])
    def test_tilings_cover_the_lattice_and_route_locally(self, width, height):
        """Tilings are vertex disjoint, cover each bond once, and gold routes from pink order onto adjacent modes."""
        sites = width * height
        pink, gold = (QSHARP_UTILS.HubbardPlaquette.PlaquetteSection(width, height, flag) for flag in (True, False))
        for tiling in (pink, gold):
            occupied = [site for cycle in tiling for site in cycle]
            assert len(occupied) == len(set(occupied)), "a tiling must be vertex disjoint"

        bonds = [frozenset((c[i], c[(i + 1) % 4])) for c in [*pink, *gold] if c[0] < sites for i in range(4)]
        horizontal = {
            frozenset((r * width + c, r * width + (c + 1) % width)) for r in range(height) for c in range(width)
        }
        vertical = {
            frozenset((r * width + c, (r + 1) % height * width + c)) for r in range(height) for c in range(width)
        }
        assert len(bonds) == len(set(bonds)), "a bond may not appear in both tilings"
        assert set(bonds) == horizontal | vertical

        # Each plaquette must land contiguous and interleaved for its FFFT, and pink order must keep
        # every on-site pair one register half apart for the interaction layer.
        pink_order, gold_order = (
            QSHARP_UTILS.HubbardPlaquette.TilingOrder(tiling, 2 * sites) for tiling in (pink, gold)
        )
        for tiling, order in ((pink, pink_order), (gold, gold_order)):
            for index, cycle in enumerate(tiling):
                assert order[4 * index : 4 * index + 4] == [cycle[0], cycle[2], cycle[1], cycle[3]]
        assert [mode + sites for mode in pink_order[:sites]] == pink_order[sites:]

        # Replay the swaps: into pink order once, then from pink order to gold order.
        routed = list(range(2 * sites))
        for start, target in ((list(routed), pink_order), (pink_order, gold_order)):
            _, swaps = QSHARP_UTILS.HubbardPlaquette.RoutingSwaps(start, target, True)
            for position in swaps:
                assert 0 <= position < 2 * sites - 1
                routed[position], routed[position + 1] = routed[position + 1], routed[position]
            assert routed == target

    @_GRADIENT_OR_LADDER
    def test_gold_layer_evolution_is_exact(self, bits):
        """At 4x2 the routed gold layer is nonempty, and the evolution is the exact PIG product of its pieces.

        The time keeps every tower angle on the gradient lattice: the interaction towers turn by
        ``U step / 8`` and the hopping towers by ``t step`` or, at the boundary, ``t step / 2``.
        """
        width, height, t, u, time, reps = 4, 2, 1.0, 4.0, 4 * _ANGLE_QUANTUM, 2
        sites, step = width * height, time / reps

        def hopping(tiling: list[list[int]]) -> scipy.sparse.csr_matrix:
            labels = []
            for cycle in tiling:
                for index in range(4):
                    low, high = sorted((cycle[index], cycle[(index + 1) % 4]))
                    for axis in "XY":
                        labels.append("I" * low + axis + "Z" * (high - low - 1) + axis + "I" * (2 * sites - high - 1))
            return pauli_to_sparse_matrix(labels, np.full(len(labels), -0.5 * t))

        pink, gold = (
            hopping(QSHARP_UTILS.HubbardPlaquette.PlaquetteSection(width, height, flag)) for flag in (True, False)
        )
        pairs = ["I" * s + "Z" + "I" * (sites - 1) + "Z" + "I" * (sites - s - 1) for s in range(sites)]
        interaction = pauli_to_sparse_matrix(pairs, np.full(sites, 0.25 * u))
        body = [(interaction, 0.5), (gold, 1.0), (interaction, 0.5)]
        layers = [(pink, 0.5), *([*body, (pink, 1.0)] * (reps - 1)), *body, (pink, 0.5)]

        state = _random_state(2 * sites, seed=13, support=32)
        expected = state.astype(complex)
        for hamiltonian, fraction in layers:
            expected = scipy.sparse.linalg.expm_multiply(-1j * fraction * step * hamiltonian, expected)
        params = _params(width, height, 0.25 * u * step, 2 * t * step, reps, bits=bits)
        actual = _apply(_evolution(params), state)
        assert np.allclose(actual, expected, atol=1e-8)

    @_GRADIENT_OR_LADDER
    def test_hamming_weight_phasing_matches_plain_rotations(self, bits):
        """At 4x2 every tower reaches the break-even, and phasing it under control matches plain rotations."""
        state = _random_state(17, seed=11, support=32)
        hwp, plain = (
            _apply(
                _evolution(
                    _params(4, 2, 10 * _ANGLE_QUANTUM, 12 * _ANGLE_QUANTUM, 2, cap=cap, bits=bits), controlled=True
                ),
                state,
            )
            for cap in (-1, 1)
        )
        assert np.allclose(hwp, plain, atol=1e-10)

    @pytest.mark.parametrize(("t", "u"), [(1.0, 0.0), (0.0, 4.0)])
    def test_single_term_evolution_is_exact(self, t, u):
        """With only hopping or only interaction the 2x2 step has no Trotter error, so it is exp(-iHt).

        Every 2x2 tower is below the break-even, so no gradient is used and the angles are arbitrary.
        """
        time = 0.23
        step = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=1, target_accuracy=0.0)
        c = step.run(_model(2, t=t, u=u)).get_container()
        params = _params(2, 2, c.interaction_angle, c.hopping_angle, c.step_reps)
        propagator = scipy.linalg.expm(-1j * time * _reference_hamiltonian(_model(2, t=t, u=u)))
        state = _random_state(8, seed=7)
        actual = _apply(_evolution(params), state)
        assert abs(np.vdot(propagator @ state, actual)) == pytest.approx(1.0, abs=1e-9)


# Logical counts of the benchmark circuit in examples/estimation_hubbard_2d.ipynb.
# At L=2 every tower is below the eight-rotation break-even, so the circuit has no Toffolis.
# At L=4 every tower takes an adder tree, and each place value of its weight is added into the
# shared ten-qubit phase gradient, which turns the place-value rotations into Toffolis.
_COUNT_KEYS = ("numQubits", "rotationCount", "rotationDepth", "tCount", "cczCount", "ccixCount", "measurementCount")
_BENCHMARK_COUNTS = {
    2: (18, 1047435, 698444, 697995, 0, 0, 10),
    4: (87, 24549, 24257, 758481, 1421400, 0, 1421410),
}

#: The same L=4 circuit with ``use_phase_gradient=False``: each place value is synthesized as its
#: own ``Rz``, the original rotation ladder, which needs no gradient register and pays in rotations.
_LADDER_BENCHMARK_COUNTS = {
    4: (57, 261233, 190086, 758427, 355350, 0, 355360),
}

#: Rotation synthesis rounds transcendental angles, so the rotation counts drift by a few units
#: across platforms. The phase gradient leaves only tens of thousands of rotations at L=4, where a
#: relative tolerance alone would not absorb that drift, so they also get an absolute tolerance.
_ROTATION_RELATIVE_TOLERANCE = 1e-4
_ROTATION_ABSOLUTE_TOLERANCE = 16


def _benchmark_counts(size: int, max_batch_size: int = -1, *, use_phase_gradient: bool = True) -> dict[str, int]:
    """Return the logical counts of the notebook's 10-bit sine-window QPE on an LxL lattice."""
    num_bits, energy_budget = 10, 0.0051 * size * size
    qpe_budget = 2.0 / 3.0 * energy_budget  # The optimal split of Campbell arXiv:2012.09238v4, App. F.
    builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=num_bits,
        phase_state="sine",
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "hubbard_plaquette",
            order=2,
            # A sine window over 2^bits - 1 queries has spread tan(pi / (2^bits + 1)).
            time=math.tan(math.pi / (2**num_bits + 1)) / qpe_budget,
            power_strategy="rescale",
            target_accuracy=energy_budget - qpe_budget,
        ),
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper",
            "hubbard_plaquette",
            max_hamming_weight_phasing_batch_size=max_batch_size,
            use_phase_gradient=use_phase_gradient,
        ),
    )
    model = _model(size, t=1.0, u=8.0)
    factory = builder.run(identity_state_prep(num_qubits=2 * size * size), model)[0]._qsharp_factory
    counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
    return {key: int(counts.get(key, 0)) for key in _COUNT_KEYS}


def _assert_counts_match_the_pins(
    counts: Mapping[str, float], size: int, pins: Mapping[int, tuple[int, ...]] = _BENCHMARK_COUNTS
) -> None:
    """Exact, except rotations get a tolerance for cross-platform synthesis drift."""
    mismatches = {
        key: (counts.get(key, 0), pinned)
        for key, pinned in zip(_COUNT_KEYS, pins[size], strict=True)
        if counts.get(key, 0)
        != (
            pytest.approx(pinned, rel=_ROTATION_RELATIVE_TOLERANCE, abs=_ROTATION_ABSOLUTE_TOLERANCE)
            if key.startswith("rotation")
            else pinned
        )
    }
    assert not mismatches, f"L={size} (actual, pinned): {mismatches}"


class TestBenchmarkResources:
    """Pin the logical cost the 2D Hubbard estimation notebook reports."""

    @pytest.mark.parametrize("size", [2, 4])
    def test_counts_match_the_pins(self, size):
        """``test_estimation_hubbard_2d`` holds the notebook's own output to the same pins."""
        _assert_counts_match_the_pins(_benchmark_counts(size), size)

    def test_the_rotation_ladder_is_still_available(self):
        """Turning the phase gradient off restores the original ladder of synthesized rotations."""
        _assert_counts_match_the_pins(_benchmark_counts(4, use_phase_gradient=False), 4, _LADDER_BENCHMARK_COUNTS)

    def test_batch_cap_trades_qubits_for_toffolis(self):
        """A cap of 8 keeps the adder tree on fewer qubits; a cap of 1 turns phasing off."""
        uncapped = _benchmark_counts(4)
        # At L=4 the interaction and hopping towers both have 16 terms, so 16 is the cap that splits nothing.
        assert _benchmark_counts(4, max_batch_size=16) == uncapped

        capped = _benchmark_counts(4, max_batch_size=8)
        assert capped["numQubits"] < uncapped["numQubits"]
        assert capped["cczCount"] > uncapped["cczCount"], "each batch pays its own place-value gradient additions"

        below = _benchmark_counts(4, max_batch_size=1)
        assert below["cczCount"] + below["ccixCount"] == 0
        assert below["numQubits"] < uncapped["numQubits"]

    def test_capping_the_ladder_trades_qubits_for_rotations(self):
        """Without the gradient the per-batch price is paid in synthesized place-value rotations."""
        uncapped = _benchmark_counts(4, use_phase_gradient=False)
        capped = _benchmark_counts(4, max_batch_size=8, use_phase_gradient=False)
        assert capped["numQubits"] < uncapped["numQubits"]
        assert capped["rotationCount"] > uncapped["rotationCount"]
        assert capped["cczCount"] > 0, "batches of eight are still at the break-even, so the tree survives"

    @pytest.mark.parametrize("use_phase_gradient", [True, False], ids=["phase_gradient", "rz_ladder"])
    @pytest.mark.parametrize(("cap", "uses_adders"), [(7, False), (8, True)])
    def test_adder_trees_start_at_eight_terms(self, cap, uses_adders, use_phase_gradient):
        """Batches of 7 equal-angle rotations stay plain rotations, and batches of 8 take an adder tree."""
        params = QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
            width=4,
            height=4,
            interactionAngle=0.1,
            hoppingAngle=0.2,
            repetitions=1,
            maxBatchSize=cap,
            usePhaseGradient=use_phase_gradient,
            rotationBitPrecision=10,
        )
        counts = get_qsharp_context().logical_counts(
            QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit, params, 0, list(range(1, 33))
        )
        assert (counts.get("cczCount", 0) + counts.get("ccixCount", 0) > 0) == uses_adders


def _plaquette_controlled(size: int = 4, *, use_phase_gradient: bool = True) -> Circuit:
    """Return the controlled plaquette circuit the mapper builds for a ``size`` x ``size`` container."""
    container = HubbardPlaquetteContainer(
        width=size, height=size, interaction_angle=0.3, hopping_angle=0.2, step_reps=1
    )
    mapper = create(
        "controlled_circuit_mapper",
        "hubbard_plaquette",
        control_indices=[0],
        use_phase_gradient=use_phase_gradient,
        rotation_bit_precision=_GRADIENT_BITS,
    )
    return mapper.run(UnitaryRepresentation(container=container))


class TestSharedPhaseGradient:
    """Phase estimation prepares the plaquette phase gradient once and shares it across queries."""

    def test_the_mapper_defaults_to_the_phase_gradient(self):
        """The rotation ladder is opt-in, so the default must phase through the gradient."""
        mapper = create("controlled_circuit_mapper", "hubbard_plaquette", control_indices=[0])
        assert mapper.settings().get("use_phase_gradient") is True
        assert int(mapper.settings().get("rotation_bit_precision")) == 10

    def test_the_mapper_declares_its_phase_gradient(self):
        """Every tower phases through one binary gradient, which sits after the system qubits."""
        circuit = _plaquette_controlled()
        assert circuit.metadata.num_phase_gradient_ancillas == _GRADIENT_BITS
        assert circuit.num_qubits == 32 + _GRADIENT_BITS

    def test_a_lattice_below_the_break_even_declares_none(self):
        """The 2x2 towers rotate term by term, so there is nothing to share."""
        assert _plaquette_controlled(2).metadata.num_phase_gradient_ancillas == 0

    def test_the_rotation_ladder_declares_none(self):
        """Without the phase gradient every place value is synthesized, so there is nothing to share."""
        assert _plaquette_controlled(use_phase_gradient=False).metadata.num_phase_gradient_ancillas == 0

    def test_mismatched_gradient_requests_are_rejected(self):
        """One register cannot serve two widths, and silently dropping one would emit a broken circuit."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        with pytest.raises(ValueError, match="same phase gradient register"):
            builder._shared_register([_plaquette_controlled(), _plaquette_controlled(2)])

    def test_a_lattice_below_the_break_even_shares_nothing(self):
        """Nothing is prepared when no controlled unitary asks for a gradient."""
        builder = create("qpe_circuit_builder", "qdk_standard", num_bits=2)
        ops, prepare, num_shared = builder._shared_register([_plaquette_controlled(2)] * 2)
        assert len(ops) == 2
        assert num_shared == 0
        assert prepare is QSHARP_UTILS.PrepSelPrep.NoOpPrepare

    @pytest.mark.parametrize("method", ["qdk_standard", "qdk_iterative"])
    @pytest.mark.parametrize("use_phase_gradient", [True, False], ids=["phase_gradient", "rz_ladder"])
    def test_phase_estimation_prepares_the_gradient_once(self, method, use_phase_gradient):
        """Each phase-estimation program allocates the register its controlled queries declare."""
        builder = create(
            "qpe_circuit_builder",
            method,
            num_bits=2,
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder",
                "hubbard_plaquette",
                order=2,
                time=0.05,
                num_divisions=1,
                target_accuracy=0.0,
            ),
            controlled_circuit_mapper=AlgorithmRef(
                "controlled_circuit_mapper",
                "hubbard_plaquette",
                use_phase_gradient=use_phase_gradient,
                rotation_bit_precision=_GRADIENT_BITS,
            ),
        )
        circuits = builder.run(identity_state_prep(num_qubits=32), _model(4, u=8.0))
        expected = _GRADIENT_BITS if use_phase_gradient else 0
        for circuit in circuits:
            parameters = circuit._qsharp_factory.parameter
            assert parameters["numSharedAncillas"] == expected
            prepared = parameters["prepareSharedOp"] is not QSHARP_UTILS.PrepSelPrep.NoOpPrepare
            assert prepared == use_phase_gradient


class TestStepCountAndShift:
    """The automatic step count and the conventional-model energy shift."""

    @pytest.mark.parametrize(("width", "height"), [(2, 2), (4, 4), (8, 8), (12, 12), (4, 6), (8, 12)])
    def test_step_count_matches_campbell_with_dense_norms(self, width, height):
        """Step count of Campbell arXiv:2012.09238 Eqs. (10), (D6), (D10), with norms from a dense SVD."""
        sites = width * height
        pink, gold = np.zeros((2, sites, sites))
        for matrix, cycles in zip(
            (pink, gold), HubbardPlaquetteTrotter._plaquette_sections(width, height), strict=True
        ):
            for cycle in cycles:
                for index in range(4):
                    a, b = cycle[index], cycle[(index + 1) % 4]
                    matrix[a, b] = matrix[b, a] = -1.0
        inner = pink @ gold - gold @ pink
        commutator = np.linalg.svd(inner @ gold - gold @ inner, compute_uv=False).sum()
        hopping = np.linalg.svd(pink + gold, compute_uv=False).sum()
        assert HubbardPlaquetteTrotter._commutator_trace_norm(width, height) == pytest.approx(commutator, abs=1e-9)
        assert HubbardPlaquetteTrotter._hopping_trace_norm(width, height) == pytest.approx(hopping, abs=1e-9)

        t, u, time = 1.0, 8.0, 3.0
        w_plaquette = u * t**2 / 6 * sites * (math.sqrt(5) + 8) + u**2 / 24 * hopping * t + 3 / 24 * commutator * t**3
        for phase in (1e-6, 0.25, 1.0, math.pi / 2, math.pi, 3 * math.pi):
            builder = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=1, target_accuracy=phase / time)
            expected = math.ceil(math.sqrt(w_plaquette * time**3 / (2 * math.sin(min(phase, math.pi) / 2))))
            assert builder._step_count(t, u, width, height, time) == expected
        manual = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=7, target_accuracy=0.0)
        assert manual._step_count(t, u, width, height, time) == 7

    @pytest.mark.parametrize(("num_electrons", "epsilon"), [(0, 0.0), (2, 0.0), (6, -1.5)])
    def test_electron_count_shifts_onto_the_described_spectrum(self, num_electrons, epsilon):
        """The shift equals the described-minus-symmetric gap from exact diagonalization."""
        model = _model(2, u=8.0, epsilon=epsilon)
        unshifted, shifted = (
            HubbardPlaquetteTrotter(order=2, time=0.3, num_divisions=2, target_accuracy=0.0, **extra)
            .run(model)
            .get_container()
            for extra in ({}, {"num_electrons": num_electrons})
        )
        assert unshifted.constant_shift == 0.0
        assert shifted.interaction_angle == unshifted.interaction_angle
        assert shifted.hopping_angle == unshifted.hopping_angle

        # N is conserved, so within its sector the two models differ by a constant.
        sector = np.flatnonzero([index.bit_count() == num_electrons for index in range(256)])
        block = np.ix_(sector, sector)
        described, symmetric = (
            np.linalg.eigvalsh(_reference_hamiltonian(model, symmetric=flag)[block]) for flag in (False, True)
        )
        gap = described - symmetric
        assert np.allclose(gap, gap[0], atol=1e-10)
        shift = shifted.eigenvalue_from_phase(0.125) - unshifted.eigenvalue_from_phase(0.125)
        assert shift == pytest.approx(gap[0])

    @pytest.mark.parametrize("power_strategy", ["repeat", "rescale"])
    def test_powered_evolution_decodes_the_unpowered_energy(self, power_strategy):
        """A phase of U^power decodes to the same shifted energy under either power strategy."""
        energy, time, power = -1.3, 0.3, 3
        container = (
            HubbardPlaquetteTrotter(
                order=2,
                time=time,
                power=power,
                power_strategy=power_strategy,
                num_divisions=2,
                target_accuracy=0.0,
                num_electrons=6,
            )
            .run(_model(2, u=8.0))
            .get_container()
        )
        phase = (-energy * time * power / (2 * math.pi)) % 1.0
        # (epsilon + U/2) eta - U M / 4 = 4 * 6 - 8 * 4 / 4 = 16.
        assert container.eigenvalue_from_phase(phase) == pytest.approx(energy + 16.0)


_BODY = {"width": 4, "height": 4, "interaction_angle": 0.1, "hopping_angle": 0.2, "step_reps": 1}


class TestContainerAndValidation:
    """Serialization and rejected inputs."""

    def test_serialization_round_trip(self, tmp_path):
        """A step survives JSON and HDF5 round trips."""
        step = HubbardPlaquetteTrotter(order=2, time=0.1, num_divisions=2).run(_model(4, u=8.0))
        with h5py.File(tmp_path / "step.h5", "w") as file:
            step.to_hdf5(file.create_group("step"))
        with h5py.File(tmp_path / "step.h5", "r") as file:
            restored = UnitaryRepresentation.from_hdf5(file["step"]).get_container()
        expected = step.get_container().to_json()
        from_json = UnitaryRepresentation.from_json(step.to_json()).get_container()
        assert from_json.to_json() == restored.to_json() == expected

    @pytest.mark.parametrize(
        ("build", "error", "match"),
        [
            pytest.param(lambda: HubbardPlaquetteTrotter(order=4), ValueError, "order 2 only", id="order"),
            pytest.param(
                lambda: HubbardPlaquetteTrotter(order=2, time=0.05).run(_model(4, periodic=False)),
                ValueError,
                "periodic in both directions",
                id="open-boundaries",
            ),
            pytest.param(
                lambda: HubbardPlaquetteTrotter(order=2, time=0.05).run(
                    FermiHubbardModelHamiltonianDescription(
                        LatticeGeometry.triangular(4, 4, periodic_x=True, periodic_y=True), t=1.0, u=0.0
                    )
                ),
                ValueError,
                "unit-spaced square lattice",
                id="not-a-square-grid",
            ),
            pytest.param(
                lambda: HubbardPlaquetteTrotter(order=2, time=0.05).run(
                    LatticeGeometry.square(4, 4, periodic_x=True, periodic_y=True)
                ),
                TypeError,
                "FermiHubbardModelHamiltonianDescription",
                id="bare-lattice",
            ),
            *[
                pytest.param(
                    lambda num_electrons=num_electrons: HubbardPlaquetteTrotter(
                        order=2, time=0.05, num_electrons=num_electrons
                    ).run(_model(4)),
                    ValueError,
                    "num_electrons",
                    id=f"num-electrons-{num_electrons}",
                )
                for num_electrons in (-2, 33)
            ],
            pytest.param(
                lambda: HubbardPlaquetteTrotter(order=2, num_electrons=1.9),
                SettingTypeMismatch,
                "num_electrons",
                id="num-electrons-float",
            ),
            pytest.param(
                lambda: HubbardPlaquetteContainer(**{**_BODY, "width": 2}),
                ValueError,
                "at least four",
                id="container-shape",
            ),
            pytest.param(
                lambda: HubbardPlaquetteContainer(**{**_BODY, "step_reps": 0}),
                ValueError,
                "step_reps must be a positive integer",
                id="container-zero-reps",
            ),
            pytest.param(
                lambda: HubbardPlaquetteContainer.from_json(
                    {**HubbardPlaquetteContainer(**_BODY).to_json(), "step_reps": 1.9}
                ),
                TypeError,
                "step_reps must be an integer",
                id="container-float-reps",
            ),
            pytest.param(
                lambda: create(
                    "controlled_circuit_mapper",
                    "hubbard_plaquette",
                    control_indices=[0],
                    max_hamming_weight_phasing_batch_size=0,
                ).run(UnitaryRepresentation(container=HubbardPlaquetteContainer(**_BODY))),
                ValueError,
                "max_hamming_weight_phasing_batch_size must be -1 or a positive integer",
                id="zero-batch-cap",
            ),
            pytest.param(
                lambda: HubbardPlaquetteContainer(**_BODY).combine(HubbardPlaquetteContainer(**_BODY)),
                NotImplementedError,
                "combine",
                id="combine",
            ),
        ],
    )
    def test_rejects_unsupported_inputs(self, build, error, match):
        with pytest.raises(error, match=match):
            build()
