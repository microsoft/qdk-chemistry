"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.unary_phase_estimation_builder import (
    QdkUnaryQpeCircuitBuilder,
    cosine_window_state,
    num_phase_bits,
)
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import (
    UnaryPhaseEstimation,
    _post_process_samples,
)
from qdk_chemistry.data import AlgorithmRef, QubitOperator
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _address_qubits(num_actions: int) -> int:
    """Number of address qubits the Q# operations allocate for ``num_actions`` values."""
    return int(np.ceil(np.log2(num_actions))) if num_actions > 1 else 0


def _dumped_address_index(address_value: int, num_address_qubits: int) -> int:
    """Map a little-endian address value onto its index in the dumped statevector.

    ``ApplyXorInPlace`` and the unary iteration address the register little-endian
    (``address[0]`` is the least significant bit) while ``dump_machine`` reads the
    first allocated qubit as the most significant bit, so the two differ by a bit
    reversal over the address register.
    """
    if num_address_qubits == 0:
        return 0
    return int(format(address_value, f"0{num_address_qubits}b")[::-1], 2)


def _matrix_power(matrix: np.ndarray, exponent: int) -> np.ndarray:
    """Signed matrix power for a self-inverse-product walk operator."""
    base = matrix if exponent >= 0 else np.linalg.inv(matrix)
    result = np.eye(matrix.shape[0], dtype=complex)
    for _ in range(abs(exponent)):
        result = base @ result
    return result


class TestUnaryIterationQsharp:
    """Statevector checks of the unary-iteration primitives against exact references."""

    @pytest.mark.parametrize(
        ("num_actions", "address_value"),
        [(n, a) for n in (1, 2, 3, 4, 5, 6, 7, 8, 11) for a in range(n)],
    )
    def test_selects_exactly_one_action_per_address(self, qdk_ctx, num_actions, address_value):
        """Address ``i`` must flip flag ``i`` and nothing else, for every valid address.

        Covers ``num_actions`` values that are not powers of two, where the iteration
        recurses into unequal halves. Comparing against the full basis state also rules
        out residual entanglement with the internal AND ancillas.
        """
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationOneHot(num_actions, address_value)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
        expected[1 << (num_actions - 1 - address_value)] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("num_actions", [2, 4, 8])
    def test_superposed_address_stays_coherent(self, qdk_ctx, num_actions):
        """A superposed address must produce sum_a |a>|onehot(a)> with no ancilla residue.

        A per-address test cannot detect ancillas that are left entangled with the
        address, because each computational-basis address leaves them in a product
        state; only a coherent superposition exposes that failure mode.
        """
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationSuperposedAddress(num_actions)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

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
    def test_exposed_control_is_an_equality_predicate(self, qdk_ctx, num_actions, data):
        """Phasing the exposed control must imprint exactly the flagged sign pattern.

        The reflection schedule uses the exposed control as a phase control rather than
        as a control on a target, so it has to be a clean ``[address == i]`` predicate.
        """
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationControlPhases(num_actions, data)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << num_address_qubits, dtype=complex)
        for address_value in range(num_actions):
            sign = -1.0 if data[address_value] else 1.0
            expected[_dumped_address_index(address_value, num_address_qubits)] = sign / np.sqrt(num_actions)
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("num_blocks", "address_value"),
        [(m, t) for m in (1, 2, 3, 4, 5, 6) for t in range(m + 1)],
    )
    def test_power_schedule_realizes_signed_powers(self, qdk_ctx, num_blocks, address_value):
        """Slot ``t`` must apply exactly ``W^(num_blocks - 2t)`` for ``W = Z.X``.

        The target starts in ``Ry(0.7)|0>``, which is an eigenstate of neither ``W`` nor
        any of its powers, so the comparison pins down the relative phase and therefore
        distinguishes every signed power in the schedule - including the negative ones.
        """
        walk = _PAULI_Z @ _PAULI_X
        initial = np.array([np.cos(0.35), np.sin(0.35)], dtype=complex)
        num_address_qubits = _address_qubits(num_blocks + 1)

        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryIterationSignedPower(num_blocks, address_value)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << (num_address_qubits + 1), dtype=complex)
        expected[:2] = _matrix_power(walk, num_blocks - 2 * address_value) @ initial
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestBlockEncodingAgnosticSchedule:
    """The signed-power schedule must work for any self-inverse block encoding."""

    @pytest.mark.parametrize(
        ("num_queries", "address_value"),
        [(p, t) for p in (1, 2, 3, 5) for t in range(p + 1)],
    )
    def test_psp_schedule_matches_the_explicit_walk_power(self, qdk_ctx, num_queries, address_value):
        """A PREPARE-SELECT-PREPARE walk must obey the same ``W^(p - 2t)`` contract.

        The Q# wrapper runs ``ApplySignedPowerSchedule`` at address ``t`` and then
        explicitly undoes ``W^(p - 2t)`` walk steps built from the same two callables.
        Whatever remains must be the untouched input state, so any mismatch in the power,
        the sign of the power, the reflection register, or the ancilla bookkeeping shows up
        as a deviation.

        This is the point of the generalization: the schedule never sees a container type or
        a PREPARE-SELECT-PREPARE symbol, only a block encoding and the reflection it pairs
        with, both supplied here from the ``PrepSelPrep`` module, so a completely different
        block encoding drives it unchanged.
        """
        psp = qdk_ctx.code.QDKChemistry.Utils.PrepSelPrep
        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestSignedPowerScheduleAgainstWalk(
            psp.MakeTestBlockEncodingOp(0.7), psp.MakeAncillaReflectionOp(1), num_queries, address_value, 2, 0.9
        )
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        num_address_qubits = _address_qubits(num_queries + 1)
        expected = np.zeros(1 << (num_address_qubits + 2), dtype=complex)
        expected[0] = np.cos(0.45)  # system |0>, ancilla |0>
        expected[2] = np.sin(0.45)  # system |1>, ancilla |0>
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("theta", [0.0, 1.3, np.pi / 2])
    def test_psp_schedule_holds_for_every_encoded_eigenvalue(self, qdk_ctx, theta):
        """The contract must not depend on what the block encoding encodes.

        ``PREPARE = Ry(theta)`` makes the encoded operator ``cos(theta)`` on the system
        qubit, sweeping the walk from a trivial reflection (``theta = 0``) to the
        maximally mixing case (``theta = pi/2``). A schedule that only happened to work
        for one spectrum would fail here.
        """
        psp = qdk_ctx.code.QDKChemistry.Utils.PrepSelPrep
        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestSignedPowerScheduleAgainstWalk(
            psp.MakeTestBlockEncodingOp(theta), psp.MakeAncillaReflectionOp(1), 3, 1, 2, 0.9
        )
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << 4, dtype=complex)
        expected[0] = np.cos(0.45)
        expected[2] = np.sin(0.45)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestPhaseRegisterSizing:
    """Phase register sizing for arbitrary query counts."""

    @pytest.mark.parametrize(
        ("num_queries", "expected_bits"),
        [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (7, 3), (8, 4), (25, 5)],
    )
    def test_num_phase_bits(self, num_queries, expected_bits):
        """The register must address num_queries + 1 reflection slots."""
        assert num_phase_bits(num_queries) == expected_bits
        assert (1 << num_phase_bits(num_queries)) >= num_queries + 1

    @pytest.mark.parametrize("num_queries", [0, -3])
    def test_non_positive_query_count_rejected(self, num_queries):
        """A non-positive query count is invalid."""
        with pytest.raises(ValueError, match="num_queries must be a positive integer"):
            num_phase_bits(num_queries)


class TestPhaseWindowState:
    """Window states prepared on the phase register."""

    @pytest.mark.parametrize("num_queries", [3, 5, 25])
    def test_padded_and_normalized(self, num_queries):
        """The window is unit-norm and zero on the unaddressed padding states."""
        amplitudes = np.array(cosine_window_state(num_queries))
        assert len(amplitudes) == 1 << num_phase_bits(num_queries)
        assert np.linalg.norm(amplitudes) == pytest.approx(1.0)
        assert np.all(amplitudes[num_queries + 1 :] == 0.0)
        assert np.all(amplitudes[: num_queries + 1] > 0.0)

    @pytest.mark.parametrize("num_queries", [3, 8, 25])
    def test_cosine_matches_babbush2018_control_state(self, num_queries):
        """The cosine window is sin(pi (t + 1) / (p + 2)) over the p + 1 slots."""
        amplitudes = np.array(cosine_window_state(num_queries))[: num_queries + 1]
        expected = np.sin(np.pi * (np.arange(num_queries + 1) + 1) / (num_queries + 2))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(amplitudes, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("num_queries", [4, 9, 24])
    def test_cosine_is_symmetric_and_single_lobed(self, num_queries):
        """The cosine window peaks in the middle and decays monotonically to both edges."""
        amplitudes = np.array(cosine_window_state(num_queries))[: num_queries + 1]
        np.testing.assert_allclose(amplitudes, amplitudes[::-1], rtol=1e-12, atol=1e-15)
        peak = int(np.argmax(amplitudes))
        assert np.all(np.diff(amplitudes[: peak + 1]) > 0.0)
        assert np.all(np.diff(amplitudes[peak:]) < 0.0)

    def test_cosine_suppresses_spectral_leakage_relative_to_uniform(self):
        r"""The cosine window's phase spectrum has far lighter tails than a uniform one.

        This is the property that makes it the Heisenberg-limited control state:
        the probability of a phase readout landing far from the true phase decays
        as :math:`1/\Delta^4` instead of the uniform window's :math:`1/\Delta^2`.
        """
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

    @pytest.mark.parametrize(
        ("bitstring", "expected_lower"), [("000", 0.0), ("010", 0.125), ("110", 0.125), ("100", 0.25)]
    )
    def test_conjugate_bins_fold_to_the_same_phase(self, bitstring, expected_lower):
        """Measured y and 1 - y describe the same walk phase."""
        counts = {bitstring: 1}
        lower, _, _ = _post_process_samples(counts, 3, "lower")
        upper, _, _ = _post_process_samples(counts, 3, "upper")
        assert lower == pytest.approx(expected_lower)
        assert upper == pytest.approx(0.5 - expected_lower)

    def test_dominant_phase_merges_conjugate_counts(self):
        """Conjugate bins are summed before the winner is selected."""
        counts = {"010": 3, "110": 3, "001": 5}  # 2/8 and 6/8 are conjugates, 1/8 is a separate bin
        phase_fraction, bitstring, measured = _post_process_samples(counts, 3, "lower")
        assert phase_fraction == pytest.approx(0.125)
        assert bitstring in {"010", "110"}
        assert measured in {0.25, 0.75}


class TestUnaryQpeEndToEnd:
    """End-to-end checks of ``MakeUnaryQPECircuit`` on a walk with an exact eigenphase."""

    @staticmethod
    def _measure(qdk_ctx, num_queries: int, k: int, system_state: int) -> int:
        """Run the synthetic-walk QPE circuit and decode the phase register.

        ``MakeUnaryQPECircuit`` returns the register least-significant bit first (the
        circuit-executor bitstring convention), so a direct Q# caller must reverse it.
        """
        num_states = num_queries + 1
        theta = -np.pi * k / num_states
        results = qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk(
            num_queries, theta, system_state
        )
        return int("".join("1" if str(bit) == "One" else "0" for bit in reversed(results)), 2)

    @pytest.mark.parametrize(
        ("num_queries", "k", "system_state"),
        [(m, k, s) for m in (3, 7) for k in range(m + 1) for s in (0, 1)],
    )
    def test_measured_bin_is_twice_the_walk_phase(self, qdk_ctx, num_queries, k, system_state):
        """The measured register must read exactly ``2*phi*N``.

        The synthetic walk is ``W = Rz(2*theta)`` built from two self-inverse
        reflections, so ``W|1> = e^{+i theta}|1>`` and ``W|0> = e^{-i theta}|0>``.
        Choosing ``theta = -pi*k/N`` with ``N = num_queries + 1`` puts the answer
        exactly on a bin boundary, making the outcome deterministic and the
        assertion exact rather than statistical.

        This pins the whole register chain at once: the big-endian window state,
        the little-endian unary addressing reached through ``Reversed``, the
        endianness of ``Adjoint ApplyQFT`` (which writes the phase little-endian
        because ``ApplyQFT`` maps a little-endian input to a big-endian output),
        and the least-significant-bit-first order of the returned results.
        """
        num_states = num_queries + 1
        expected = (-k) % num_states if system_state == 1 else k
        assert self._measure(qdk_ctx, num_queries, k, system_state) == expected

    def test_slot_sweep_grows_linearly_with_the_query_count(self, qdk_ctx):
        """The schedule must sweep ``num_queries + 1`` slots, not apply one controlled block.

        ``MakeUnaryQPECircuit`` hands the whole phase register to a single
        ``ApplySignedPowerSchedule`` call rather than repeating a walk step itself, because unary
        iteration fuses the slot sweep with the address decode. This pins that the repetition
        really happens inside that call. The synthetic walk block is Clifford, so the only
        non-Clifford cost is the unary-iteration AND ladder, one uncompute measurement per
        slot: the count must therefore grow proportionally to ``num_queries``. A schedule
        that applied a single block, or that ignored the query count it was built with, would
        give a flat count instead.
        """
        theta = -np.pi / 4
        counts = {
            num_queries: qdk_ctx.logical_counts(
                qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk,
                num_queries,
                theta,
                1,
            )["measurementCount"]
            for num_queries in (3, 7, 15, 31)
        }

        queries = sorted(counts)
        assert [counts[num_queries] for num_queries in queries] == sorted(counts.values()), counts
        per_query = [counts[num_queries] / num_queries for num_queries in queries]
        assert min(per_query) > 0.75, counts
        assert max(per_query) / min(per_query) < 1.5, counts

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_decoder_recovers_the_walk_phase(self, qdk_ctx, k):
        """The measured bin, run through the decoder, returns the walk phase.

        The two eigenvectors carry conjugate phases ``+-phi`` and land in
        conjugate bins, which the decoder must fold onto the same answer.
        """
        num_queries = 7
        num_states = num_queries + 1
        num_bits = num_phase_bits(num_queries)

        expected_phase = k / (2 * num_states)
        for system_state in (0, 1):
            measured_bin = self._measure(qdk_ctx, num_queries, k, system_state)
            counts = {format(measured_bin, f"0{num_bits}b"): 1}
            phase_fraction, _, _ = _post_process_samples(counts, num_bits, "lower")
            assert phase_fraction == pytest.approx(expected_phase)

    @pytest.mark.parametrize("bin_index", [1, 2, 3])
    def test_inverse_qft_leaves_the_answer_little_endian(self, qdk_ctx, bin_index):
        """After ``Adjoint ApplyQFT`` the answer sits little-endian in the phase register.

        This is the contract that decides the order in which ``MakeUnaryQPECircuit`` must
        return its results: a circuit executor emits the first measured ``Result`` as the
        right-most character of the bitstring, so the register has to be returned in the same
        little-endian order it is held in for ``int(bitstring, 2)`` to recover the bin.

        The check is at statevector level rather than through sampling, and the block
        encoding is deliberately a PREPARE-SELECT-PREPARE walk rather than the synthetic
        single-qubit one, because ``PREPARE = Ry(theta)`` with ``SELECT = c-Z`` encodes
        ``cos(theta)`` on ``|1>`` and so puts the answer on bin ``j`` exactly when
        ``theta = pi*j/N``. What has to move under bit reversal is the *pair* ``{j, N - j}``
        the two conjugate eigenvectors occupy, not ``j`` alone: ``j = 2`` is itself a fixed
        point of a three-bit reversal, but its pair still travels ``{2, 6} -> {2, 3}``. All
        three cases move (``{1, 7} -> {4, 7}``, ``{2, 6} -> {2, 3}``, ``{3, 5} -> {5, 6}``),
        so reading the register with the opposite endianness fails for every one of them.
        """
        num_queries = 7
        num_states = num_queries + 1
        psp = qdk_ctx.code.QDKChemistry.Utils.PrepSelPrep
        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestSchedulePhaseRamp(
            psp.MakeTestBlockEncodingOp(np.pi * bin_index / num_states),
            psp.MakeAncillaReflectionOp(1),
            num_queries,
            2,
            np.pi,
            True,
        )
        # dump_machine treats the first allocated qubit as most significant, so the row index
        # is the phase register read big-endian; reverse it to get the little-endian value.
        amplitudes = np.array(qdk_ctx.dump_machine().as_dense_state()).reshape(num_states, 4)
        probabilities = np.zeros(num_states)
        for row in range(num_states):
            little_endian = int(format(row, "03b")[::-1], 2)
            probabilities[little_endian] = np.sum(np.abs(amplitudes[row]) ** 2)

        # |1> is an equal superposition of the two walk eigenvectors, whose conjugate
        # phases land in bins j and N - j.
        expected = np.zeros(num_states)
        expected[bin_index] = 0.5
        expected[-bin_index] = 0.5
        assert probabilities == pytest.approx(expected, abs=1e-9)

    def test_builder_defaults_recover_the_ground_state_energy(self):
        r"""The shipped defaults must recover :math:`H = (X + Z)/2` end to end.

        This is the only test that drives the whole Python wiring -- default LCU unitary
        builder in quantum-walk mode, PSP mapper, circuit executor, phase decode -- rather
        than calling the Q# operations directly. It therefore covers the parts the direct
        Q# tests cannot: that the default block encoding really arrives as a walk container,
        that the ancilla count and preparation reach the Q# entry point, and that the
        executor's bitstring is decoded with the same endianness the circuit returns.

        The equal-coefficient case is exactly representable: :math:`E = -1/\sqrt{2}` with
        :math:`\lambda = 1` gives a walk phase of :math:`3/8`, so :math:`2\varphi = 3/4`
        lands on bin :math:`3N/4`. That bin is asymmetric under bit reversal, so reading the
        register the other way round decodes to :math:`+0.98` instead of :math:`-0.71`.
        """
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
        qpe.settings().set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary", num_queries=63))
        qpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
        result = qpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_preparation)

        assert result.raw_energy == pytest.approx(float(energies[0]), abs=1e-9)

    def test_a_non_walk_unitary_builder_is_rejected(self):
        """A unitary representation the schedule cannot drive must be refused, not stumbled over.

        The signed-power schedule interleaves the reflections itself, so it needs a mapper
        circuit that applies the block encoding exactly once -- which is what a quantum-walk
        container guarantees, since its power counts walk steps rather than block encodings.
        Swapping ``unitary_builder`` for any of the product-formula algorithms is a plausible
        thing to try, and without a guard here the container reaches an attribute access and
        dies as an ``AttributeError`` naming a private attribute rather than the setting the
        caller actually got wrong.
        """
        hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
        builder = QdkUnaryQpeCircuitBuilder(
            num_queries=4,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        )

        with pytest.raises(ValueError, match="quantum-walk"):
            builder._run_impl(Circuit(qasm="OPENQASM 3.0;"), hamiltonian)


class TestBaseProfileGuardrail:
    """Unary iteration must be unreachable from a ``TargetProfile.Base`` context.

    Base is not merely a weaker profile here, it is a wrong one. ``UnaryIterationWithControl``
    toggles the helper qubit of its AND ladder with a ``CNOT`` between the compute and the
    uncompute, so ``Adjoint AND`` has to read a measurement result to know which correction
    to apply. Base forbids mid-circuit measurement, so it lowers the uncompute to the unitary
    decomposition instead, which is only valid while the helper still holds the original AND.

    Nothing raises when that happens, and the damage is worse than a fixed wrong answer. For
    ``num_queries = 7`` the phase register comes back well-formed and in range, but equal to
    the correct bin XOR 2 or XOR 6 with roughly even odds: one phase bit is always flipped,
    and a second is left randomized by the corrupted uncompute. So the result is never
    accidentally right, and never reproducibly wrong either -- it is noise shaped like data,
    which no golden-value assertion downstream could pin.

    The only defence is refusing to load the sources at all. Omitting them from
    ``_BASE_PROFILE_FILES`` turns a silent wrong answer into an undefined-symbol error at the
    point of use.
    """

    @pytest.mark.parametrize("namespace", ["UnaryIteration", "UnaryPhaseEstimation"])
    def test_base_context_refuses_to_resolve_a_unary_operation(self, use_base_qdk_ctx, namespace):
        """Calling into the withheld sources must fail loudly rather than answer wrongly."""
        with pytest.raises(Exception, match="not found"):
            use_base_qdk_ctx.eval(f"QDKChemistry.Utils.{namespace}.AddressQubits(8)")


class TestQueryCountResolution:
    """The query count comes from the settings; a container power is ignored."""

    @staticmethod
    def _unitary_rep(power: int):
        """A minimal stand-in for a unitary representation carrying ``power``."""

        class _Container:
            def __init__(self) -> None:
                self.power = power

        class _UnitaryRep:
            def get_container(self) -> _Container:
                return _Container()

        return _UnitaryRep()

    def test_query_count_comes_from_settings(self):
        """The configured query count drives the schedule."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=25)
        assert builder.resolve_num_queries(self._unitary_rep(1)) == 25

    def test_container_power_is_ignored(self):
        """A power carried by the unitary representation is warned about and then ignored.

        The schedule fixes its length at build time, so there is no controlled power for
        the container's value to feed. Honouring it would silently resize the query chain
        out from under the ``num_queries`` the caller asked for.
        """
        builder = QdkUnaryQpeCircuitBuilder(num_queries=25)
        assert builder.resolve_num_queries(self._unitary_rep(11)) == 25

    def test_non_positive_query_count_is_rejected(self):
        """Without a usable setting there is nothing to fall back on."""
        builder = QdkUnaryQpeCircuitBuilder()
        with pytest.raises(ValueError, match="num_queries must be a positive integer"):
            builder.resolve_num_queries(self._unitary_rep(11))
