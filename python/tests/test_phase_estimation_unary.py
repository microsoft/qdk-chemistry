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
from qdk_chemistry.data import AlgorithmRef, QubitOperator
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _identity_state_preparation(num_qubits: int) -> Circuit:
    """A state preparation that leaves the register in ``|0...0>``."""
    params = {
        "rowMap": list(range(num_qubits - 1, -1, -1)),
        "stateVector": [1.0] + [0.0] * (2**num_qubits - 1),
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


def _address_qubits(num_actions: int) -> int:
    """Number of address qubits the Q# operations allocate for ``num_actions`` values."""
    return int(np.ceil(np.log2(num_actions))) if num_actions > 1 else 0


def _dumped_address_index(address_value: int, num_address_qubits: int) -> int:
    """Map a little-endian address value onto its index in the dumped statevector."""
    if num_address_qubits == 0:
        return 0
    return int(format(address_value, f"0{num_address_qubits}b")[::-1], 2)


def _dump_op(op, num_qubits: int) -> np.ndarray:
    """Simulate ``op`` on the all-zero state and return the resulting statevector.

    The context has to be the one the operation was resolved from, otherwise the helper
    that ``dump_operation_on_state`` evaluates cannot bind the callable.
    """
    return np.array(dump_operation_on_state(op, num_qubits, context=get_qsharp_context()))


class TestUnaryIterationQsharp:
    """Statevector checks of the unary-iteration primitives against exact references."""

    @pytest.mark.parametrize(
        ("num_actions", "address_value"),
        [(n, a) for n in (1, 2, 3, 4, 5, 6, 7, 8, 11) for a in range(n)],
    )
    def test_selects_exactly_one_action_per_address(self, num_actions, address_value):
        """Address ``i`` must flip flag ``i`` and nothing else, for every valid address."""
        num_address_qubits = _address_qubits(num_actions)
        op = QSHARP_UTILS.UnaryIteration.MakeTestUnaryIterationOneHotOp(num_actions, address_value)
        state = _dump_op(op, num_address_qubits + num_actions)

        expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
        expected[1 << (num_actions - 1 - address_value)] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("num_actions", [2, 4, 8])
    def test_superposed_address_stays_coherent(self, num_actions):
        """A superposed address must produce sum_a |a>|onehot(a)> with no ancilla residue."""
        num_address_qubits = _address_qubits(num_actions)
        op = QSHARP_UTILS.UnaryIteration.MakeTestUnaryIterationSuperposedAddressOp(num_actions)
        state = _dump_op(op, num_address_qubits + num_actions)

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
            (8, [True, False, False, False, False, True, True, False]),
        ],
    )
    def test_exposed_control_is_an_equality_predicate(self, num_actions, data):
        """Phasing the exposed control must imprint exactly the flagged sign pattern."""
        num_address_qubits = _address_qubits(num_actions)
        op = QSHARP_UTILS.UnaryIteration.MakeTestUnaryIterationControlPhasesOp(num_actions, data)
        state = _dump_op(op, num_address_qubits)

        expected = np.zeros(1 << num_address_qubits, dtype=complex)
        for address_value in range(num_actions):
            sign = -1.0 if data[address_value] else 1.0
            expected[_dumped_address_index(address_value, num_address_qubits)] = sign / np.sqrt(num_actions)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestBlockEncodingAgnosticSchedule:
    """The signed-power schedule must work for any self-inverse block encoding."""

    @pytest.mark.parametrize(
        ("num_queries", "address_value"),
        [(p, t) for p in (1, 2, 3, 5) for t in range(p + 1)],
    )
    def test_psp_schedule_matches_the_explicit_walk_power(self, num_queries, address_value):
        """A PREPARE-SELECT-PREPARE walk must obey the same ``W^(p - 2t)`` contract."""
        psp = QSHARP_UTILS.PrepSelPrep
        op = QSHARP_UTILS.UnaryPhaseEstimation.MakeTestSignedPowerScheduleAgainstWalkOp(
            psp.MakeTestBlockEncodingOp(0.7), psp.MakeAncillaReflectionOp(1), num_queries, address_value, 0.9
        )
        num_address_qubits = _address_qubits(num_queries + 1)
        state = _dump_op(op, num_address_qubits + 2)

        expected = np.zeros(1 << (num_address_qubits + 2), dtype=complex)
        expected[0] = np.cos(0.45)  # system |0>, ancilla |0>
        expected[2] = np.sin(0.45)  # system |1>, ancilla |0>
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("theta", [0.0, 1.3, np.pi / 2])
    def test_psp_schedule_holds_for_every_encoded_eigenvalue(self, theta):
        """The contract must not depend on what the block encoding encodes."""
        psp = QSHARP_UTILS.PrepSelPrep
        op = QSHARP_UTILS.UnaryPhaseEstimation.MakeTestSignedPowerScheduleAgainstWalkOp(
            psp.MakeTestBlockEncodingOp(theta), psp.MakeAncillaReflectionOp(1), 3, 1, 0.9
        )
        state = _dump_op(op, 4)

        expected = np.zeros(1 << 4, dtype=complex)
        expected[0] = np.cos(0.45)
        expected[2] = np.sin(0.45)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestPhaseWindowState:
    """Window states prepared on the phase register."""

    @pytest.mark.parametrize("num_queries", [4, 9, 24])
    def test_cosine_is_symmetric_and_single_lobed(self, num_queries):
        """The cosine window peaks in the middle and decays monotonically to both edges."""
        amplitudes = np.array(cosine_window_state(num_queries))[: num_queries + 1]
        np.testing.assert_allclose(amplitudes, amplitudes[::-1], rtol=1e-12, atol=1e-15)
        peak = int(np.argmax(amplitudes))
        assert np.all(np.diff(amplitudes[: peak + 1]) > 0.0)
        assert np.all(np.diff(amplitudes[peak:]) < 0.0)

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

    def test_dominant_phase_merges_conjugate_counts(self):
        """Conjugate bins are summed before the winner is selected."""
        counts = {"010": 3, "110": 3, "001": 5}  # 2/8 and 6/8 are conjugates, 1/8 is a separate bin
        phase_fraction, bitstring, measured = _post_process_phase_estimation(counts, 3, use_positive_sign=True)
        assert phase_fraction == pytest.approx(0.125)
        assert bitstring in {"010", "110"}
        assert measured in {0.25, 0.75}

    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
    def test_the_shipped_default_reports_a_non_positive_eigenvalue(self, num_bits):
        r"""The default branch must fold every bin into :math:`[1/4, 1/2]`, i.e. :math:`E \le 0`.

        ``use_positive_sign`` defaults to ``False``, which is only correct for a ground state
        below zero. Pinning the reachable interval here documents that the default *cannot*
        report a positive energy, so a caller with a non-negative target must opt in.
        """
        for value in range(1 << num_bits):
            counts = {format(value, f"0{num_bits}b"): 1}
            phase_fraction, _, _ = _post_process_phase_estimation(counts, num_bits)
            assert 0.25 <= phase_fraction <= 0.5
            assert np.cos(2 * np.pi * phase_fraction) <= 1e-12

    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
    def test_the_two_sign_branches_partition_the_spectrum(self, num_bits):
        """The flag picks a branch of a sign ambiguity; it must not change the magnitude."""
        for value in range(1 << num_bits):
            counts = {format(value, f"0{num_bits}b"): 1}
            positive, _, _ = _post_process_phase_estimation(counts, num_bits, use_positive_sign=True)
            negative, _, _ = _post_process_phase_estimation(counts, num_bits, use_positive_sign=False)
            assert 0.0 <= positive <= 0.25
            assert positive + negative == pytest.approx(0.5)


class _FixedWidthMapper:
    """A circuit mapper that reports a chosen register width and nothing else.

    The builder locates the reflected qubits from ``Circuit.num_qubits``, so the width a
    mapper declares is the only thing these tests need to vary.
    """

    def __init__(self, num_qubits: int | None) -> None:
        self._num_qubits = num_qubits

    def run(self, _unitary: UnitaryRepresentation) -> Circuit:
        """Return a circuit that declares the configured width."""
        return Circuit(qasm="OPENQASM 3.0;\n", num_qubits=self._num_qubits)


class _RecordingLogger:
    """Collects ``Logger.warn`` messages so a warning path can be asserted.

    ``Logger`` lives in the compiled core, so the module-level name is replaced rather
    than an attribute of the extension type.
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(str(message))

    def __getattr__(self, _name: str):
        """Ignore every other logging call."""
        return lambda *_args, **_kwargs: None


class TestInvalidConfigurationIsRejected:
    """Each documented failure mode must raise rather than silently mis-size a register."""

    @staticmethod
    def _hamiltonian() -> QubitOperator:
        """``H = (X + Z)/2`` on a single qubit."""
        return QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))

    def _build_with_mapper(self, mapper, num_queries: int = 3):
        """Run the builder with ``mapper`` substituted for the configured circuit mapper."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=num_queries)
        hamiltonian = self._hamiltonian()
        original = builder._create_nested
        builder._create_nested = (
            lambda name, *args, **kwargs: mapper if name == "circuit_mapper" else original(name, *args, **kwargs)
        )
        return builder.run(
            state_preparation=_identity_state_preparation(hamiltonian.num_qubits),
            qubit_hamiltonian=hamiltonian,
        )

    @pytest.mark.parametrize("num_queries", [0, -1, -7])
    def test_a_non_positive_query_count_is_rejected(self, num_queries):
        """``num_queries`` sizes the whole schedule, so it must be positive."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=num_queries)
        with pytest.raises(ValueError, match="num_queries must be a positive integer"):
            builder.resolve_num_queries()

    def test_a_plain_block_encoding_is_rejected(self):
        """The schedule drops one reflection, so a bare LCU has nothing to drop."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=3)
        hamiltonian = self._hamiltonian()
        original = builder._create_nested
        builder._create_nested = (
            lambda name, *args, **kwargs: LCUBuilder(quantum_walk=False)
            if name == "unitary_builder"
            else original(name, *args, **kwargs)
        )
        with pytest.raises(ValueError, match="Requires a block encoding unitary representation"):
            builder.run(
                state_preparation=_identity_state_preparation(hamiltonian.num_qubits),
                qubit_hamiltonian=hamiltonian,
            )

    def test_a_mapper_that_hides_its_width_is_rejected(self):
        """Without ``num_qubits`` the builder cannot tell which qubits to reflect about."""
        with pytest.raises(ValueError, match="did not report num_qubits"):
            self._build_with_mapper(_FixedWidthMapper(None))

    @pytest.mark.parametrize("declared_width", [1, 0])
    def test_a_mapper_without_an_ancilla_register_is_rejected(self, declared_width):
        """A width that leaves no ancilla past the system register has no reflection."""
        with pytest.raises(ValueError, match="non-empty ancilla register"):
            self._build_with_mapper(_FixedWidthMapper(declared_width))

    def test_a_non_unary_circuit_builder_is_rejected(self):
        """The algorithm decodes a doubled phase, so it needs its own builder."""
        qpe = UnaryPhaseEstimation(shots=8)
        qpe.settings().set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_standard"))
        hamiltonian = self._hamiltonian()
        with pytest.raises(TypeError, match="QdkUnaryQpeCircuitBuilder"):
            qpe.run(
                qubit_hamiltonian=hamiltonian,
                state_preparation=_identity_state_preparation(hamiltonian.num_qubits),
            )


class TestIgnoredSettingsAreAnnounced:
    """Settings the schedule overrides must warn rather than silently take no effect."""

    @staticmethod
    def _hamiltonian() -> QubitOperator:
        """``H = (X + Z)/2`` on a single qubit."""
        return QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))

    def _run_with_recording_logger(self, builder, monkeypatch) -> list[str]:
        """Run ``builder`` with its module-level ``Logger`` replaced, returning the warnings."""
        recorder = _RecordingLogger()
        monkeypatch.setattr(unary_phase_estimation_builder, "Logger", recorder)
        hamiltonian = self._hamiltonian()
        builder.run(
            state_preparation=_identity_state_preparation(hamiltonian.num_qubits),
            qubit_hamiltonian=hamiltonian,
        )
        return recorder.warnings

    def test_a_configured_num_bits_is_ignored_with_a_warning(self, monkeypatch):
        """The query count fixes the register size, so ``num_bits`` cannot also set it."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=3)
        builder.settings().set("num_bits", 7)

        warnings = self._run_with_recording_logger(builder, monkeypatch)

        assert any("num_bits=7 is ignored" in message for message in warnings), warnings

    def test_a_carried_walk_power_is_ignored_with_a_warning(self, monkeypatch):
        """The schedule picks its own power per slot, so a container power is meaningless."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=3)
        walk = LCUBuilder(quantum_walk=True).run(self._hamiltonian())
        container = walk.get_container()
        powered = UnitaryRepresentation(
            container=LCUWalkContainer(container.block_encoding, power=2, scale=container.scale)
        )
        original = builder._create_nested
        builder._create_nested = (
            lambda name, *args, **kwargs: _PoweredUnitaryBuilder(powered)
            if name == "unitary_builder"
            else original(name, *args, **kwargs)
        )

        warnings = self._run_with_recording_logger(builder, monkeypatch)

        assert any("carries power 2" in message for message in warnings), warnings


class _PoweredUnitaryBuilder:
    """A unitary builder that hands back a prepared representation."""

    def __init__(self, representation: UnitaryRepresentation) -> None:
        self._representation = representation

    def run(self, _qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Return the prepared representation."""
        return self._representation


class TestRegisterSizeHasOneDefinition:
    """The phase register width must have a single definition across Python and Q#."""

    @pytest.mark.parametrize("num_queries", [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 63, 64])
    def test_python_and_qsharp_agree_on_the_phase_register_size(self, num_queries):
        """``resolve_num_queries``, ``PhaseRegisterSize`` and ``AddressQubits`` are one quantity.

        They agreed by coincidence of three separate formulas before; ``PhaseRegisterSize``
        now delegates to ``AddressQubits``, and this pins the Python side to the same value.
        """
        _, num_bits = QdkUnaryQpeCircuitBuilder(num_queries=num_queries).resolve_num_queries()
        context = get_qsharp_context()
        qsharp_phase_bits = context.eval(f"QDKChemistry.Utils.UnaryPhaseEstimation.PhaseRegisterSize({num_queries})")
        qsharp_address_bits = context.eval(f"QDKChemistry.Utils.UnaryIteration.AddressQubits({num_queries + 1})")

        assert num_bits == qsharp_phase_bits == qsharp_address_bits
        assert (1 << num_bits) >= num_queries + 1

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

    @pytest.mark.parametrize("num_queries", [1, 3, 63, 2**29 - 1, 2**31 - 1])
    def test_phase_register_size_is_exact_at_large_query_counts(self, num_queries):
        """``PhaseRegisterSize`` must keep delegating rather than compute its own width.

        ``test_python_and_qsharp_agree_on_the_phase_register_size`` only reaches 64, and
        an independent ``Ceiling(Lg(IntAsDouble(numQueries + 1)))`` agrees with the exact
        value everywhere below ``numQueries + 1 == 2**29``. A second definition would
        therefore be invisible to every other check in this module.
        """
        context = get_qsharp_context()
        computed = context.eval(f"QDKChemistry.Utils.UnaryPhaseEstimation.PhaseRegisterSize({num_queries})")

        assert computed == num_queries.bit_length()
        assert (1 << computed) >= num_queries + 1


class TestUnaryQpeEndToEnd:
    """End-to-end checks of ``MakeUnaryQPECircuit`` on a walk with an exact eigenphase."""

    @staticmethod
    def _measure(num_queries: int, k: int, system_angle: float) -> int:
        """Run the synthetic-walk QPE circuit from ``Ry(system_angle)|0>`` and decode the phase register."""
        num_states = num_queries + 1
        theta = -np.pi * k / num_states
        results = QSHARP_UTILS.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk(num_queries, theta, system_angle)
        return int("".join("1" if str(bit) == "One" else "0" for bit in reversed(results)), 2)

    @pytest.mark.parametrize(
        ("num_queries", "k", "system_state"),
        [(m, k, s) for m in (3, 7) for k in range(m + 1) for s in (0, 1)],
    )
    def test_measured_bin_is_twice_the_walk_phase(self, num_queries, k, system_state):
        """The measured register must read exactly ``2*phi*N``."""
        num_states = num_queries + 1
        expected = (-k) % num_states if system_state == 1 else k
        assert self._measure(num_queries, k, system_state * np.pi) == expected

    @pytest.mark.parametrize("system_angle", [np.pi / 2, 2 * np.pi / 3])
    def test_superposed_input_splits_across_both_eigenphase_bins(self, system_angle):
        """A non-eigenstate must land in the two eigenphase bins with the overlap weights.

        Both eigenphases sit exactly on the phase grid, so every shot has to read one of the
        two bins; their frequencies track the squared overlaps of ``Ry(system_angle)|0>``.
        """
        num_queries, k, shots = 7, 3, 256
        num_states = num_queries + 1
        ground_bin, excited_bin = k, (-k) % num_states

        measured = np.array([self._measure(num_queries, k, system_angle) for _ in range(shots)])

        assert set(np.unique(measured).tolist()) == {ground_bin, excited_bin}
        assert np.mean(measured == excited_bin) == pytest.approx(np.sin(system_angle / 2) ** 2, abs=0.12)

    def test_slot_sweep_grows_linearly_with_the_query_count(self):
        """The schedule must sweep ``num_queries + 1`` slots, not apply one controlled block."""
        theta = -np.pi / 4
        context = get_qsharp_context()
        counts = {
            num_queries: context.logical_counts(
                QSHARP_UTILS.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk,
                num_queries,
                theta,
                np.pi,
            )["measurementCount"]
            for num_queries in (3, 7, 15, 31)
        }

        queries = sorted(counts)
        assert [counts[num_queries] for num_queries in queries] == sorted(counts.values()), counts
        per_query = [counts[num_queries] / num_queries for num_queries in queries]
        assert min(per_query) > 0.75, counts
        assert max(per_query) / min(per_query) < 1.5, counts

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_decoder_recovers_the_walk_phase(self, k):
        """The measured bin, run through the decoder, returns the walk phase.

        The two eigenvectors carry conjugate phases ``+-phi`` and land in
        conjugate bins, which the decoder must fold onto the same answer.
        """
        num_queries = 7
        num_states = num_queries + 1
        num_bits = num_queries.bit_length()

        expected_phase = k / (2 * num_states)
        for system_state in (0, 1):
            measured_bin = self._measure(num_queries, k, system_state * np.pi)
            counts = {format(measured_bin, f"0{num_bits}b"): 1}
            phase_fraction, _, _ = _post_process_phase_estimation(counts, num_bits, use_positive_sign=True)
            assert phase_fraction == pytest.approx(expected_phase)

    @pytest.mark.parametrize("num_queries", [6, 11, 23, 63])
    def test_builder_defaults_recover_the_ground_state_energy(self, num_queries):
        r"""The shipped defaults must recover :math:`H = (X + Z)/2` end to end.

        Only ``63`` fills its phase register exactly. ``6``, ``11`` and ``23`` sweep ``7``,
        ``12`` and ``24`` reflection slots inside registers that hold ``8``, ``16`` and
        ``32``, so the window is zero-padded and the schedule addresses only part of the
        register — the case a power-of-two-only construction cannot express.
        """
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


class TestReflectionRegisterContract:
    """The builder locates the reflected qubits from the circuit, not from the mapper."""

    @staticmethod
    def _walk_representation() -> UnitaryRepresentation:
        """A quantum-walk representation of ``H = (X + Z)/2``."""
        hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
        return LCUBuilder(quantum_walk=True).run(hamiltonian)

    def test_the_mapper_declares_the_width_of_the_register_it_maps(self):
        """``Circuit.num_qubits`` has to agree with the container the mapper was handed.

        The builder subtracts the system size from it to name the reflected qubits, so a
        disagreement here would silently reflect about the wrong register.
        """
        unitary_rep = self._walk_representation()
        block_encoding = PSPMapper().run(unitary_rep)

        assert block_encoding.num_qubits == unitary_rep.get_num_qubits()

    def test_the_builder_reflects_the_ancilla_tail_the_mapper_declared(self):
        """Every qubit the mapper exposes past the system register is reflected about."""
        hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
        declared = PSPMapper().run(self._walk_representation()).num_qubits
        assert declared is not None

        builder = QdkUnaryQpeCircuitBuilder(num_queries=3)
        circuit = builder.run(
            state_preparation=_identity_state_preparation(hamiltonian.num_qubits),
            qubit_hamiltonian=hamiltonian,
        )[0]

        assert circuit._qsharp_factory.parameter["numAncillas"] == declared - hamiltonian.num_qubits
