"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.circuit_mapper.psp_mapper import PSPMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.algorithms.phase_estimation.circuit_builder import unary_phase_estimation_builder
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.unary_phase_estimation_builder import (
    QdkUnaryQpeCircuitBuilder,
    cosine_window_state,
)
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import (
    UnaryPhaseEstimation,
    _post_process_phase_estimation,
)
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, QubitOperator
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _address_qubits(num_actions: int) -> int:
    """Number of address qubits the Q# operations expect for ``num_actions`` values."""
    return QSHARP_UTILS.UnaryIteration.AddressQubits(num_actions)


def _dumped_address_index(address_value: int, num_address_qubits: int) -> int:
    """Map a little-endian address value onto its index in the dumped statevector."""
    if num_address_qubits == 0:
        return 0
    return int(format(address_value, f"0{num_address_qubits}b")[::-1], 2)


def _test_utils(context, module: str):
    """Resolve a test-only Q# module (staged from ``tests/qsharp``) from *context*."""
    return getattr(context.code.QDKChemistry.TestUtils, module)


def _dump_op(op, num_qubits: int, context) -> np.ndarray:
    """Simulate ``op`` on the all-zero state and return the resulting statevector.

    The context has to be the one the operation was resolved from, otherwise the helper
    that ``dump_operation_on_state`` evaluates cannot bind the callable.
    """
    return np.array(dump_operation_on_state(op, num_qubits, context=context))


def _decode(counts: dict[str, int], num_bits: int, *, resolve_positive_branch: bool = False):
    """Run the decoder against a walk whose block encoding has ``lambda = 1``."""
    return _post_process_phase_estimation(
        counts, num_bits, "qdk_unary", resolve_positive_branch, lambda phase: float(np.cos(2 * np.pi * phase))
    )


class TestUnaryIterationQsharp:
    """Statevector checks of the unary-iteration primitives against exact references."""

    @pytest.mark.parametrize("num_actions", [1, 2, 3, 5, 8])
    def test_one_hot_address_selects_exactly_one_action(self, qsharp_test_context, num_actions):
        """A basis-state address must fire the matching action and only that one."""
        num_address_qubits = _address_qubits(num_actions)
        for address_value in range(num_actions):
            op = _test_utils(qsharp_test_context, "UnaryIterationTests").TestMakeOneHotOp(num_actions, address_value)
            state = _dump_op(op, num_address_qubits + num_actions, qsharp_test_context)

            expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
            expected[1 << (num_actions - 1 - address_value)] = 1.0
            np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("num_actions", [2, 4, 8])
    def test_superposed_address_stays_coherent(self, qsharp_test_context, num_actions):
        """A superposed address must produce sum_a |a>|onehot(a)> with no ancilla residue."""
        num_address_qubits = _address_qubits(num_actions)
        op = _test_utils(qsharp_test_context, "UnaryIterationTests").TestMakeSuperposedAddressOp(num_actions)
        state = _dump_op(op, num_address_qubits + num_actions, qsharp_test_context)

        expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
        for address_value in range(num_actions):
            index = _dumped_address_index(address_value, num_address_qubits) << num_actions
            expected[index | (1 << (num_actions - 1 - address_value))] = 1.0 / np.sqrt(num_actions)
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("num_actions", "data"),
        [
            (2, [True, False]),
            (4, [True, False, False, True]),
            (7, [True, False, False, False, False, True, True]),
        ],
    )
    def test_exposed_control_is_an_equality_predicate(self, qsharp_test_context, num_actions, data):
        """Phasing the exposed control must imprint exactly the flagged sign pattern."""
        num_address_qubits = _address_qubits(num_actions)
        op = _test_utils(qsharp_test_context, "UnaryIterationTests").TestMakeControlPhasesOp(num_actions, data)
        state = _dump_op(op, num_address_qubits, qsharp_test_context)

        # Addresses at or above num_actions are outside the iteration's promise: the last
        # leaf fires on its control alone, so they alias onto a neighbouring action.
        amplitude = 1.0 / np.sqrt(1 << num_address_qubits)
        expected = np.array([-1.0 if flagged else 1.0 for flagged in data]) * amplitude
        actual = np.array([state[_dumped_address_index(a, num_address_qubits)] for a in range(num_actions)])
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestBlockEncodingAgnosticSchedule:
    """The signed-power schedule must work for any self-inverse block encoding."""

    @pytest.mark.parametrize(
        ("num_queries", "address_value"),
        [(p, t) for p in (1, 2, 3, 5) for t in range(p + 1)],
    )
    def test_psp_schedule_matches_the_explicit_walk_power(self, qsharp_test_context, num_queries, address_value):
        """A PREPARE-SELECT-PREPARE walk must obey the same ``W^(p - 2t)`` contract."""
        psp = qsharp_test_context.code.QDKChemistry.Utils.PrepSelPrep
        op = _test_utils(qsharp_test_context, "UnaryPhaseEstimationTests").TestMakeSignedPowerScheduleAgainstWalkOp(
            _test_utils(qsharp_test_context, "PrepSelPrepTests").TestMakeBlockEncodingOp(0.7),
            psp.MakeAncillaReflectionOp(1),
            num_queries,
            address_value,
            0.9,
        )
        num_address_qubits = _address_qubits(num_queries + 1)
        state = _dump_op(op, num_address_qubits + 2, qsharp_test_context)

        expected = np.zeros(1 << (num_address_qubits + 2), dtype=complex)
        expected[0] = np.cos(0.45)  # system |0>, ancilla |0>
        expected[2] = np.sin(0.45)  # system |1>, ancilla |0>
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("theta", [0.0, 1.3, np.pi / 2])
    def test_psp_schedule_holds_for_every_encoded_eigenvalue(self, qsharp_test_context, theta):
        """The contract must not depend on what the block encoding encodes."""
        psp = qsharp_test_context.code.QDKChemistry.Utils.PrepSelPrep
        op = _test_utils(qsharp_test_context, "UnaryPhaseEstimationTests").TestMakeSignedPowerScheduleAgainstWalkOp(
            _test_utils(qsharp_test_context, "PrepSelPrepTests").TestMakeBlockEncodingOp(theta),
            psp.MakeAncillaReflectionOp(1),
            3,
            1,
            0.9,
        )
        state = _dump_op(op, 4, qsharp_test_context)

        expected = np.zeros(1 << 4, dtype=complex)
        expected[0] = np.cos(0.45)
        expected[2] = np.sin(0.45)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestPhaseWindowState:
    """Window states prepared on the phase register."""

    @pytest.mark.parametrize("num_queries", [4, 9, 24, 25])
    def test_cosine_is_symmetric_and_single_lobed(self, num_queries):
        """The cosine window peaks in the middle and decays monotonically to both edges."""
        amplitudes = np.array(cosine_window_state(num_queries))[: num_queries + 1]
        np.testing.assert_allclose(amplitudes, amplitudes[::-1], rtol=1e-12, atol=1e-15)

        maxima = np.flatnonzero(np.isclose(amplitudes, amplitudes.max(), rtol=0.0, atol=1e-15))
        first, last = int(maxima[0]), int(maxima[-1])

        assert 0 < first <= last < len(amplitudes) - 1, "the lobe must be interior"
        assert last - first <= 1, f"a single lobe peaks over at most two slots, got {last - first + 1}"
        assert np.all(np.diff(amplitudes[: first + 1]) > 0.0)
        assert np.all(np.diff(amplitudes[last:]) < 0.0)

    def test_cosine_suppresses_spectral_leakage_relative_to_uniform(self):
        """The cosine window's phase spectrum has far lighter tails than a uniform one."""
        num_queries, oversampling = 31, 32
        windows = {
            "cosine": np.array(cosine_window_state(num_queries))[: num_queries + 1],
            "uniform": np.ones(num_queries + 1) / np.sqrt(num_queries + 1),
        }

        def tail_probability(window: str, bins: int) -> float:
            amplitudes = windows[window]
            spectrum = np.abs(np.fft.fft(amplitudes, amplitudes.size * oversampling)) ** 2
            spectrum /= spectrum.sum()
            offsets = np.arange(spectrum.size) - int(np.argmax(spectrum))
            distance = np.minimum(offsets % spectrum.size, (-offsets) % spectrum.size)
            return float(spectrum[distance > bins * oversampling].sum())

        assert tail_probability("cosine", 2) < 0.1 * tail_probability("uniform", 2)
        assert tail_probability("cosine", 4) < 1e-3


class TestPhaseDecoding:
    """Decoding of the doubled measured phase."""

    @pytest.mark.parametrize("resolve_positive_branch", [True, False])
    def test_dominant_phase_merges_conjugate_counts(self, resolve_positive_branch):
        """Conjugate bins are summed before the winner is selected."""
        counts = {"010": 3, "110": 3, "001": 5}  # 2/8 and 6/8 are conjugates, 1/8 is a separate bin
        result = _decode(counts, 3, resolve_positive_branch=resolve_positive_branch)
        expected = 0.125 if resolve_positive_branch else 0.375
        assert result.canonical_phase_fraction == pytest.approx(expected)
        # The reported bin and its fraction name the folded phase, so both branches agree.
        assert result.bitstring_msb_first == "010"
        assert result.phase_fraction == pytest.approx(0.25)

    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
    def test_the_two_sign_branches_partition_the_spectrum(self, num_bits):
        """The flag picks a branch of a sign ambiguity; it must not change the magnitude."""
        for value in range(1 << num_bits):
            counts = {format(value, f"0{num_bits}b"): 1}
            positive = _decode(counts, num_bits, resolve_positive_branch=True)
            negative = _decode(counts, num_bits, resolve_positive_branch=False)
            assert 0.0 <= positive.canonical_phase_fraction <= 0.25
            assert positive.canonical_phase_fraction + negative.canonical_phase_fraction == pytest.approx(0.5)
            assert positive.raw_energy == pytest.approx(-negative.raw_energy)

    def test_branching_is_still_two_candidates_at_the_degenerate_phase(self):
        """The half-way bin is the one that collapses, so pin it directly."""
        result = _decode({"10": 1}, 2, resolve_positive_branch=True)
        assert result.canonical_phase_fraction == pytest.approx(0.25)
        assert result.branching == pytest.approx((0.0, 0.0), abs=1e-12)

    @pytest.mark.parametrize("resolve_positive_branch", [True, False])
    def test_tied_counts_decode_independently_of_shot_order(self, resolve_positive_branch):
        """Bins that tie on counts must not decode differently just because they arrived in a different order."""
        tied = {"001": 4, "011": 4}
        forward = _decode(tied, 3, resolve_positive_branch=resolve_positive_branch)
        reversed_order = _decode(dict(reversed(tied.items())), 3, resolve_positive_branch=resolve_positive_branch)

        assert forward.canonical_phase_fraction == pytest.approx(reversed_order.canonical_phase_fraction)
        assert forward.bitstring_msb_first == reversed_order.bitstring_msb_first
        # The tie is broken toward the smaller phase fraction of the two candidates.
        expected = 0.0625 if resolve_positive_branch else 0.3125
        assert forward.canonical_phase_fraction == pytest.approx(expected)


class _Stub:
    """A nested-algorithm stand-in whose ``run`` hands back a fixed object."""

    def __init__(self, result) -> None:
        self._result = result

    def run(self, *_args, **_kwargs):
        """Return the fixed result."""
        return self._result


def _hamiltonian() -> QubitOperator:
    """``H = (X + Z)/2`` on a single qubit."""
    return QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))


def _make_builder(num_queries: int = 3, **nested) -> QdkUnaryQpeCircuitBuilder:
    """Build the unary builder with the ``nested`` algorithms substituted by setting name."""
    builder = QdkUnaryQpeCircuitBuilder(num_queries=num_queries)
    original = builder._create_nested
    builder._create_nested = lambda name, *a, **kw: nested[name] if name in nested else original(name, *a, **kw)
    return builder


def _run_builder(builder: QdkUnaryQpeCircuitBuilder, state_preparation: Circuit | None = None):
    """Run ``builder`` against ``H = (X + Z)/2``."""
    hamiltonian = _hamiltonian()
    return builder.run(
        state_preparation=state_preparation or identity_state_prep(hamiltonian.num_qubits),
        qubit_hamiltonian=hamiltonian,
    )


class TestMisconfigurationIsSurfaced:
    """Rejected configurations must raise, and overridden settings must warn."""

    @pytest.mark.parametrize("num_queries", [0, -1, -7])
    def test_a_non_positive_query_count_is_rejected(self, num_queries):
        """``num_queries`` sizes the whole schedule, so it must be positive."""
        with pytest.raises(ValueError, match="num_queries must be a positive integer"):
            QdkUnaryQpeCircuitBuilder(num_queries=num_queries).resolve_num_queries()

    def test_a_plain_block_encoding_is_rejected(self):
        """The schedule drops one reflection, so a bare LCU has nothing to drop."""
        with pytest.raises(ValueError, match="Requires a LCU walk unitary representation"):
            _run_builder(_make_builder(unitary_builder=LCUBuilder(quantum_walk=False)))

    @pytest.mark.parametrize(
        ("declared_width", "message"),
        [(None, "did not report num_qubits"), (1, "non-empty ancilla register"), (0, "non-empty ancilla register")],
    )
    def test_a_mapper_that_leaves_nothing_to_reflect_about_is_rejected(self, declared_width, message):
        """The builder reflects the ancilla tail that ``Circuit.num_qubits`` implies."""
        mapper = _Stub(Circuit(qasm="OPENQASM 3.0;\n", num_qubits=declared_width))
        with pytest.raises(ValueError, match=message):
            _run_builder(_make_builder(circuit_mapper=mapper))

    def test_a_state_preparation_without_a_qsharp_operation_is_rejected(self):
        """The schedule threads the state prep into a Q# operation, so QASM alone is not enough."""
        with pytest.raises(RuntimeError, match="State preparation has no Q# operation"):
            _run_builder(_make_builder(), state_preparation=Circuit(qasm="OPENQASM 3.0;\n"))

    @pytest.mark.parametrize(
        ("setting", "algorithm", "message"),
        [
            ("qpe_circuit_builder", "qdk_standard", "QdkUnaryQpeCircuitBuilder"),
            ("circuit_executor", "qdk_full_state_simulator", "qdk_sparse_state_simulator"),
        ],
    )
    def test_an_unsupported_nested_algorithm_is_rejected(self, setting, algorithm, message):
        """The doubled phase needs the unary builder, and the recursive Q# needs the sparse simulator."""
        qpe = UnaryPhaseEstimation(shots=8)
        qpe.settings().set(setting, AlgorithmRef(setting, algorithm))
        hamiltonian = _hamiltonian()
        with pytest.raises(TypeError, match=message):
            qpe.run(
                qubit_hamiltonian=hamiltonian,
                state_preparation=identity_state_prep(hamiltonian.num_qubits),
            )

    def test_a_configured_num_bits_is_ignored_with_a_warning(self, monkeypatch):
        """The query count fixes the register size, so ``num_bits`` cannot also set it."""
        warnings: list[str] = []
        monkeypatch.setattr(unary_phase_estimation_builder.Logger, "warn", warnings.append)
        builder = _make_builder()
        builder.settings().set("num_bits", 7)

        _run_builder(builder)

        assert any("num_bits=7 is ignored" in message for message in warnings), warnings

    def test_a_carried_walk_power_is_ignored_with_a_warning(self, monkeypatch):
        """The schedule picks its own power per slot, so a container power is meaningless."""
        warnings: list[str] = []
        monkeypatch.setattr(unary_phase_estimation_builder.Logger, "warn", warnings.append)
        container = LCUBuilder(quantum_walk=True).run(_hamiltonian()).get_container()
        powered = UnitaryRepresentation(
            container=LCUWalkContainer(container.block_encoding, power=2, scale=container.scale)
        )

        _run_builder(_make_builder(unitary_builder=_Stub(powered)))

        assert any("power 2 is ignored" in message for message in warnings), warnings


class TestRegisterSizeHasOneDefinition:
    """The phase register width must have a single definition across Python and Q#."""

    @pytest.mark.parametrize("num_queries", [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 63, 64])
    def test_python_and_qsharp_agree_on_the_phase_register_size(self, num_queries):
        """``resolve_num_queries``, ``PhaseRegisterSize`` and ``AddressQubits`` are one quantity.

        They agreed by coincidence of three separate formulas before; ``PhaseRegisterSize``
        now delegates to ``AddressQubits``, and this pins the Python side to the same value.
        """
        resolved, num_bits = QdkUnaryQpeCircuitBuilder(num_queries=num_queries).resolve_num_queries()
        assert resolved == num_queries, "the configured query count is applied as given"

        context = get_qsharp_context()
        qsharp_phase_bits = context.eval(f"QDKChemistry.Utils.UnaryPhaseEstimation.PhaseRegisterSize({resolved})")
        qsharp_address_bits = context.eval(f"QDKChemistry.Utils.UnaryIteration.AddressQubits({resolved + 1})")

        assert num_bits == qsharp_phase_bits == qsharp_address_bits
        assert (1 << num_bits) >= resolved + 1
        assert (1 << (num_bits - 1)) < resolved + 1, "the register must be the narrowest that addresses its slots"

    @pytest.mark.parametrize("num_actions", [1, 2, 3, 4, 5, 8, 9, 16, 17, 1024, 1025, 2**20, 2**29, 2**31])
    def test_address_width_is_exact_at_large_powers_of_two(self, num_actions):
        """``AddressQubits`` must be integer-exact where ``Ceiling(Lg(...))`` is not.

        The floating-point form returns 32 for ``2**31``, over-allocating the address
        register and tripping the power-of-two facts that guard the recursion. ``2**29``
        is the smallest power of two it gets wrong, so it is the case a realistic input
        could plausibly reach.
        """
        context = get_qsharp_context()
        computed = context.eval(f"QDKChemistry.Utils.UnaryIteration.AddressQubits({num_actions})")

        assert computed == (num_actions - 1).bit_length()
        assert (1 << computed) >= num_actions


class TestUnaryQpeEndToEnd:
    """End-to-end checks of ``MakeUnaryQPECircuit`` on a walk with an exact eigenphase."""

    @staticmethod
    def _measure(context, num_queries: int, k: int, system_angle: float) -> int:
        """Run the synthetic-walk QPE circuit from ``Ry(system_angle)|0>`` and decode the phase register."""
        num_states = num_queries + 1
        theta = -np.pi * k / num_states
        results = _test_utils(context, "UnaryPhaseEstimationTests").TestRunSyntheticWalkQpe(
            num_queries, theta, system_angle
        )
        return int("".join("1" if str(bit) == "One" else "0" for bit in reversed(results)), 2)

    @pytest.mark.parametrize(
        ("num_queries", "k", "system_state"),
        [(m, k, s) for m in (3, 7) for k in range(m + 1) for s in (0, 1)],
    )
    def test_measured_bin_is_twice_the_walk_phase(self, qsharp_test_context, num_queries, k, system_state):
        """The measured register must read exactly ``2*phi*N``."""
        num_states = num_queries + 1
        expected = (-k) % num_states if system_state == 1 else k
        assert self._measure(qsharp_test_context, num_queries, k, system_state * np.pi) == expected

    @pytest.mark.parametrize("system_angle", [np.pi / 2, 2 * np.pi / 3])
    def test_superposed_input_splits_across_both_eigenphase_bins(self, qsharp_test_context, system_angle):
        """A non-eigenstate must land in the two eigenphase bins with the overlap weights."""
        num_queries, k, shots = 7, 3, 256
        num_states = num_queries + 1
        ground_bin, excited_bin = k, (-k) % num_states

        measured = np.array([self._measure(qsharp_test_context, num_queries, k, system_angle) for _ in range(shots)])

        assert set(np.unique(measured).tolist()) == {ground_bin, excited_bin}
        assert np.mean(measured == excited_bin) == pytest.approx(np.sin(system_angle / 2) ** 2, abs=0.12)

    @pytest.mark.parametrize("resolve_positive_branch", [True, False])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_decoder_recovers_the_walk_phase(self, qsharp_test_context, k, resolve_positive_branch):
        """The measured bin, run through the decoder, returns the walk phase.

        The two eigenvectors carry conjugate phases ``+-phi`` and land in
        conjugate bins, which the decoder must fold onto the same answer. The sign flag
        only chooses which of the two mirror branches that answer is reported on.
        """
        num_queries = 7
        num_states = num_queries + 1
        num_bits = num_queries.bit_length()

        walk_phase = k / (2 * num_states)
        expected_phase = walk_phase if resolve_positive_branch else 0.5 - walk_phase
        for system_state in (0, 1):
            measured_bin = self._measure(qsharp_test_context, num_queries, k, system_state * np.pi)
            counts = {format(measured_bin, f"0{num_bits}b"): 1}
            result = _decode(counts, num_bits, resolve_positive_branch=resolve_positive_branch)
            assert result.canonical_phase_fraction == pytest.approx(expected_phase)

    @pytest.mark.parametrize("num_queries", [6, 8, 11, 23, 63])
    def test_builder_defaults_recover_the_ground_state_energy(self, num_queries):
        r"""The shipped defaults must recover :math:`H = (X + Z)/2` end to end."""
        num_bits = num_queries.bit_length()
        assert (num_queries + 1 < 1 << num_bits) == (num_queries != 63), "sweep must mix padded and exact registers"

        hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
        energies, vectors = np.linalg.eigh(hamiltonian.to_matrix())
        state_prep_params = {
            "rowMap": list(range(hamiltonian.num_qubits - 1, -1, -1)),
            "stateVector": np.real(vectors[:, 0]).tolist(),
            "expansionOps": [],
            "numQubits": hamiltonian.num_qubits,
        }
        state_preparation = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
                parameter=state_prep_params,
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params),
        )

        qpe = UnaryPhaseEstimation(shots=200)
        qpe.settings().set(
            "qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary", num_queries=num_queries)
        )
        qpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
        result = qpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_preparation)

        assert result.raw_energy == pytest.approx(float(energies[0]), abs=1e-9)


def test_the_builder_reflects_the_ancilla_tail_the_mapper_declared():
    """Every qubit the mapper exposes past the system register is reflected about."""
    hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
    rep = LCUBuilder(quantum_walk=True).run(hamiltonian)
    declared = PSPMapper().run(rep).num_qubits
    assert declared is not None

    builder = QdkUnaryQpeCircuitBuilder(num_queries=3)
    circuit = builder.run(
        state_preparation=identity_state_prep(hamiltonian.num_qubits),
        qubit_hamiltonian=hamiltonian,
    )[0]

    assert circuit._qsharp_factory.parameter["numAncillas"] == declared - hamiltonian.num_qubits
