"""Tests for the Hubbard plaquette Trotter builder and its Q# lowering."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import h5py
import numpy as np
import pytest
import scipy.linalg
from qdk.test_utils import dump_operation_on_state

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
    LatticeGraph,
    MajoranaMapping,
    SettingTypeMismatch,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

_PLAQUETTE = "QDKChemistry.Utils.HubbardPlaquette"
_HWP = "QDKChemistry.Utils.HammingWeightPhasing"


def _model(
    size: int, *, t: float = 1.0, u: float = 0.0, epsilon: float = 0.0, periodic: bool = True
) -> FermiHubbardModelHamiltonianDescription:
    """Return a Hubbard model on a ``size`` x ``size`` square lattice."""
    lattice = LatticeGeometry.square(size, size, periodic_x=periodic, periodic_y=periodic)
    return FermiHubbardModelHamiltonianDescription(lattice, t=t, u=u, epsilon=epsilon)


def _reference_hamiltonian(model: FermiHubbardModelHamiltonianDescription, *, symmetric: bool = True) -> np.ndarray:
    """Return the dense Jordan-Wigner matrix of the 2x2 ``model``, or of its particle-hole symmetric form.

    ``materialize()`` rejects the 2x2 torus, whose periodic images join each pair twice, so the
    graph comes from ``LatticeGraph.square``, which doubles those bonds.
    """
    t, u, epsilon = (model.parameters[name] for name in ("t", "u", "epsilon"))
    lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-0.5 * u if symmetric else epsilon, t=t, U=u)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(8))
    labels, coefficients = zip(*mapped.get_real_coefficients(tolerance=1e-14), strict=True)
    dense = pauli_to_dense_matrix(list(labels), list(coefficients))
    return dense + 0.25 * u * model.lattice.num_sites * np.eye(len(dense)) if symmetric else dense


def _operation(name: str, args: str, *, controlled: bool = False, namespace: str = _PLAQUETTE) -> str:
    """Return a Q# lambda calling ``name``, controlled on ``qs[0]`` if asked."""
    if controlled:
        return f"qs => {{ Controlled {namespace}.{name}([qs[0]], ({args.format(qs='qs[1...]')})); }}"
    return f"qs => {{ {namespace}.{name}({args.format(qs='qs')}); }}"


def _apply(operation: str, state: np.ndarray) -> np.ndarray:
    """Return the state ``operation`` produces from a real-amplitude ``state``."""
    num_qubits = round(math.log2(len(state)))
    return np.asarray(
        dump_operation_on_state(operation, num_qubits, state.tolist(), context=get_qsharp_context()), dtype=complex
    )


def _random_state(num_qubits: int, seed: int, support: int | None = None) -> np.ndarray:
    """Return a normalized real state, dense or on ``support`` random basis states."""
    rng = np.random.default_rng(seed)
    state = np.zeros(2**num_qubits)
    indices = rng.choice(len(state), size=support, replace=False) if support else slice(None)
    state[indices] = rng.normal(size=state[indices].shape)
    return state / np.linalg.norm(state)


class TestHammingWeightPhasing:
    """Batched phase towers apply the same unitary as their separate rotations."""

    @pytest.mark.parametrize(
        ("count", "cap", "controlled"),
        [(12, -1, False), (16, 8, False), (12, 8, False), (10, 5, False), (12, 8, True)],
    )
    def test_z_tower(self, count, cap, controlled):
        """exp(-i a sum_j Z_j) under any batch split, with or without control."""
        angle = 3 * np.pi / 32
        state = _random_state(count + controlled, seed=count + cap, support=24)
        weights = np.array([(index % 2**count).bit_count() for index in range(len(state))])
        phases = np.exp(-1j * angle * (count - 2 * weights))
        phases[: len(state) - 2**count] = 1.0  # The control, when present, is the most significant qubit.
        args = f"{angle}, [[PauliZ], size = {count}], Std.Arrays.Chunks(1, {{qs}}), {cap}"
        actual = _apply(_operation("HammingWeightPhase", args, controlled=controlled, namespace=_HWP), state)
        assert np.allclose(actual, phases * state, atol=1e-10)

    @pytest.mark.parametrize(
        ("count", "cap", "expected"),
        [(16, -1, 16), (16, 8, 8), (16, 16, 16), (16, 32, 16), (5, 8, 5), (1, 1, 1)],
    )
    def test_batch_size(self, count, cap, expected):
        assert QSHARP_UTILS.HammingWeightPhasing.HammingWeightBatchSize(count, cap) == expected

    @pytest.mark.parametrize(
        ("num_pairs", "controlled", "name"),
        [
            (2, False, "HoppingPhases"),
            (5, False, "HoppingPhases"),
            (4, True, "HoppingPhases"),
            (8, False, "HoppingPhasesWithForcedLegacyCostsForTest"),
        ],
    )
    def test_hopping_tower(self, num_pairs, controlled, name):
        """exp(i a (XX + YY)) on every pair, below and above the eight-rotation break-even."""
        angle = 3 * np.pi / 32
        xx_plus_yy = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]) + np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
        pair_gate = scipy.linalg.expm(1j * angle * xx_plus_yy)
        state = _random_state(2 * num_pairs + controlled, seed=num_pairs, support=8 if num_pairs > 5 else None)

        expected = state.astype(complex).reshape(1 + controlled, -1)
        for pair in range(num_pairs):
            target = expected[-1].reshape(4**pair, 4, -1)
            expected[-1] = np.einsum("ij,ajb->aib", pair_gate, target).reshape(-1)
        args = f"{angle}, Std.Arrays.Chunks(2, {{qs}}), -1"
        actual = _apply(_operation(name, args, controlled=controlled), state)
        assert np.allclose(actual, expected.reshape(-1), atol=1e-10)

    @pytest.mark.parametrize("angle", [np.pi / 8, -13 * np.pi / 32])
    def test_interaction_tower(self, angle):
        """exp(-i a sum_s Z_s Z_{s+M}) over eight sites, which takes the batched path."""
        sites = 8
        state = _random_state(2 * sites, seed=5, support=24)
        spins = 1 - 2 * ((np.arange(len(state))[:, None] >> np.arange(2 * sites - 1, -1, -1)) & 1)
        phases = np.exp(-1j * angle * (spins[:, :sites] * spins[:, sites:]).sum(axis=1))
        actual = _apply(_operation("InteractionLayer", f"{angle}, {sites}, {{qs}}, -1"), state)
        assert np.allclose(actual, phases * state, atol=1e-10)


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
        """Tilings are vertex disjoint, cover each bond once, and gold routes onto adjacent modes."""
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

        # Replay the swaps: each plaquette must land contiguous and interleaved for its FFFT.
        _, swaps = QSHARP_UTILS.HubbardPlaquette.RoutingSwaps(gold, 2 * sites, True)
        routed = list(range(2 * sites))
        for position in swaps:
            assert 0 <= position < 2 * sites - 1
            routed[position], routed[position + 1] = routed[position + 1], routed[position]
        for index, cycle in enumerate(gold):
            assert routed[4 * index : 4 * index + 4] == [cycle[0], cycle[2], cycle[1], cycle[3]]

    @pytest.mark.parametrize(("t", "u"), [(1.0, 0.0), (0.0, 4.0)])
    def test_single_term_evolution_is_exact(self, t, u):
        """With only hopping or only interaction the 2x2 step has no Trotter error, so it is exp(-iHt)."""
        time = 0.23
        step = HubbardPlaquetteTrotter(order=2, time=time, num_divisions=1, target_accuracy=0.0)
        c = step.run(_model(2, t=t, u=u)).get_container()
        params = (
            f"{_PLAQUETTE}.HubbardPlaquetteParams(2, 2, {c.interaction_angle}, {c.hopping_angle}, {c.step_reps}, -1)"
        )
        propagator = scipy.linalg.expm(-1j * time * _reference_hamiltonian(_model(2, t=t, u=u)))
        state = _random_state(8, seed=7)
        actual = _apply(_operation("RepPlaquetteExp", params + ", {qs}"), state)
        assert abs(np.vdot(propagator @ state, actual)) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_forced_legacy_evolution_adds_the_particle_number_phase(self, controlled):
        """The legacy cost path is the normal one times exp(-i (U N / 2 - U M / 4) t)."""
        time, u, sites = 0.05, 4.0, 4
        params = f"{_PLAQUETTE}.HubbardPlaquetteParams(2, 2, {0.25 * u * time}, {2.0 * time}, 1, -1), {{qs}}"
        normal, forced = (
            _operation(name, params, controlled=controlled)
            for name in ("RepPlaquetteExp", "RepPlaquetteExpWithForcedLegacyCostsForTest")
        )
        state = _random_state(2 * sites + controlled, seed=102)
        electrons = np.array([(index % 2 ** (2 * sites)).bit_count() for index in range(len(state))])
        phases = np.exp(-1j * (0.5 * u * electrons - 0.25 * u * sites) * time)
        phases[: len(state) - 2 ** (2 * sites)] = 1.0
        expected = phases * _apply(normal, state)
        assert abs(np.vdot(expected, _apply(forced, state))) == pytest.approx(1.0, abs=1e-12)


# Logical counts of the benchmark circuit in examples/estimation_hubbard_2d.ipynb.
# TEMPORARY (legacy parity): they include the legacy circuit's extra work, so L=2 still has Toffolis.
_COUNT_KEYS = ("numQubits", "rotationCount", "rotationDepth", "tCount", "cczCount", "ccixCount", "measurementCount")
_BENCHMARK_COUNTS = {
    2: (25, 2224986, 1483458, 697995, 610582, 0, 610592),
    4: (73, 729063, 551326, 758427, 710540, 0, 710550),
}


def _benchmark_counts(size: int, max_batch_size: int = -1) -> dict[str, int]:
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
            "controlled_circuit_mapper", "hubbard_plaquette", max_hamming_weight_phasing_batch_size=max_batch_size
        ),
    )
    model = _model(size, t=1.0, u=8.0)
    factory = builder.run(identity_state_prep(num_qubits=2 * size * size), model)[0]._qsharp_factory
    counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
    return {key: int(counts.get(key, 0)) for key in _COUNT_KEYS}


class TestBenchmarkResources:
    """Pin the logical cost the 2D Hubbard estimation notebook reports."""

    @pytest.mark.parametrize("size", [2, 4])
    def test_counts_match_the_pins(self, size):
        """Exact, except rotations get rel=1e-4 for cross-platform synthesis drift."""
        counts = _benchmark_counts(size)
        mismatches = {
            key: (counts[key], pinned)
            for key, pinned in zip(_COUNT_KEYS, _BENCHMARK_COUNTS[size], strict=True)
            if counts[key] != pytest.approx(pinned, rel=1e-4 if key.startswith("rotation") else 0)
        }
        assert not mismatches, f"L={size} (actual, pinned): {mismatches}"

    def test_batch_cap_trades_qubits_for_rotations(self):
        """A cap of 8 keeps the adder tree on fewer qubits; a cap of 1 turns phasing off."""
        uncapped = _benchmark_counts(4)
        # TEMPORARY (legacy parity): the legacy tower spans all 32 modes, so 32 is the cap that splits nothing.
        assert _benchmark_counts(4, max_batch_size=32) == uncapped

        capped = _benchmark_counts(4, max_batch_size=8)
        assert capped["numQubits"] < uncapped["numQubits"]
        assert capped["rotationCount"] > uncapped["rotationCount"]
        assert capped["cczCount"] > 0

        below = _benchmark_counts(4, max_batch_size=1)
        assert below["cczCount"] + below["ccixCount"] == 0
        assert below["numQubits"] < uncapped["numQubits"]


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
