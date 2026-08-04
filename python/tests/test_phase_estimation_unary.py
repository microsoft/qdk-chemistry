"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.controlled_circuit_mapper import (
    ControlledPauliSequenceMapper,
    SOSSAMapper,
    UnaryIterationWalkMapper,
)
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.unary_phase_estimation_builder import (
    QdkUnaryQpeCircuitBuilder,
    num_phase_bits,
    phase_window_state,
)
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import (
    _select_dominant_decoded_phase,
)

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

        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationSignedPower(num_blocks, address_value)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << (num_address_qubits + 1), dtype=complex)
        expected[:2] = _matrix_power(walk, num_blocks - 2 * address_value) @ initial
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

    @pytest.mark.parametrize("window", ["cosine", "uniform"])
    @pytest.mark.parametrize("num_queries", [3, 5, 25])
    def test_padded_and_normalized(self, window, num_queries):
        """Windows are unit-norm and zero on the unaddressed padding states."""
        amplitudes = np.array(phase_window_state(num_queries, window))
        assert len(amplitudes) == 1 << num_phase_bits(num_queries)
        assert np.linalg.norm(amplitudes) == pytest.approx(1.0)
        assert np.all(amplitudes[num_queries + 1 :] == 0.0)
        assert np.all(amplitudes[: num_queries + 1] > 0.0)

    @pytest.mark.parametrize("num_queries", [3, 8, 25])
    def test_cosine_matches_babbush2018_control_state(self, num_queries):
        """The cosine window is sin(pi (t + 1) / (p + 2)) over the p + 1 slots."""
        amplitudes = np.array(phase_window_state(num_queries, "cosine"))[: num_queries + 1]
        expected = np.sin(np.pi * (np.arange(num_queries + 1) + 1) / (num_queries + 2))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(amplitudes, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("num_queries", [4, 9, 24])
    def test_cosine_is_symmetric_and_single_lobed(self, num_queries):
        """The cosine window peaks in the middle and decays monotonically to both edges."""
        amplitudes = np.array(phase_window_state(num_queries, "cosine"))[: num_queries + 1]
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

        def tail_probability(window: str, bins: int) -> float:
            amplitudes = np.array(phase_window_state(num_queries, window))[: num_queries + 1]
            spectrum = np.abs(np.fft.fft(amplitudes, amplitudes.size * oversampling)) ** 2
            spectrum /= spectrum.sum()
            offsets = np.arange(spectrum.size) - int(np.argmax(spectrum))
            distance = np.minimum(offsets % spectrum.size, (-offsets) % spectrum.size)
            return float(spectrum[distance > bins * oversampling].sum())

        assert tail_probability("cosine", 2) < 0.1 * tail_probability("uniform", 2)
        assert tail_probability("cosine", 4) < 1e-3

    def test_unknown_window_rejected(self):
        """Unsupported window names are rejected."""
        with pytest.raises(ValueError, match="window must be one of"):
            phase_window_state(4, "kaiser")


class TestMapperInterface:
    """The builder accepts any mapper exposing the unary-iteration walk interface."""

    def test_sossa_mapper_satisfies_the_protocol(self):
        """The SOSSA mapper is recognized structurally, not by class identity."""
        assert isinstance(SOSSAMapper(), UnaryIterationWalkMapper)

    def test_arbitrary_mapper_with_the_three_methods_satisfies_the_protocol(self):
        """A mapper unrelated to SOSSA qualifies as long as it implements the interface."""

        class ThirdPartyWalkMapper:
            def build_walk_op(self, _unitary, _num_queries, use_unary_iteration=True):  # noqa: ARG002
                return None

            def num_ancillary_qubits(self, _container):
                return 0

            def get_ancilla_prep_op(self):
                return None

        assert isinstance(ThirdPartyWalkMapper(), UnaryIterationWalkMapper)

    def test_mapper_missing_a_method_is_rejected(self):
        """Dropping any one of the three methods removes the capability."""

        class PartialWalkMapper:
            def build_walk_op(self, _unitary, _num_queries, use_unary_iteration=True):  # noqa: ARG002
                return None

            def num_ancillary_qubits(self, _container):
                return 0

        assert not isinstance(PartialWalkMapper(), UnaryIterationWalkMapper)
        assert not isinstance(ControlledPauliSequenceMapper(), UnaryIterationWalkMapper)

    def test_incapable_mapper_raises_and_names_the_missing_methods(self, monkeypatch):
        """A mapper without the interface is refused, and the error says exactly what is absent."""

        class StubUnitaryRepresentation:
            def get_container(self):
                return object()  # no ``power`` attribute, so the settings' count is used

        class StubUnitaryBuilder:
            def run(self, _qubit_hamiltonian):
                return StubUnitaryRepresentation()

        nested = {
            "unitary_builder": StubUnitaryBuilder(),
            "controlled_circuit_mapper": ControlledPauliSequenceMapper(),
        }
        builder = QdkUnaryQpeCircuitBuilder(num_queries=4)
        monkeypatch.setattr(builder, "_create_nested", lambda key: nested[key])

        with pytest.raises(TypeError) as excinfo:
            builder._run_impl(None, None)

        message = str(excinfo.value)
        assert "cannot drive unary-iteration phase estimation" in message
        assert "ControlledPauliSequenceMapper" in message
        for method in ("build_walk_op", "num_ancillary_qubits", "get_ancilla_prep_op"):
            assert method in message


class TestPhaseDecoding:
    """Decoding of the doubled measured phase."""

    @pytest.mark.parametrize(("measured", "expected_lower"), [(0.0, 0.0), (0.25, 0.125), (0.75, 0.125), (0.5, 0.25)])
    def test_conjugate_bins_fold_to_the_same_phase(self, measured, expected_lower):
        """Measured y and 1 - y describe the same walk phase."""
        builder = QdkUnaryQpeCircuitBuilder(num_queries=7, phase_band="lower")
        assert builder.phase_fraction_from_measurement(measured) == pytest.approx(expected_lower)

        upper_builder = QdkUnaryQpeCircuitBuilder(num_queries=7, phase_band="upper")
        assert upper_builder.phase_fraction_from_measurement(measured) == pytest.approx(0.5 - expected_lower)

    def test_dominant_phase_merges_conjugate_counts(self):
        """Conjugate bins are summed before the winner is selected."""
        counts = {"010": 3, "110": 3, "001": 5}  # 2/8 and 6/8 are conjugates, 1/8 is a separate bin
        builder = QdkUnaryQpeCircuitBuilder(num_queries=7, phase_band="lower")
        phase_fraction, bitstring, measured = _select_dominant_decoded_phase(
            counts, 3, builder.phase_fraction_from_measurement
        )
        assert phase_fraction == pytest.approx(0.125)
        assert bitstring in {"010", "110"}
        assert measured in {0.25, 0.75}


class TestUnaryQpeEndToEnd:
    """End-to-end checks of ``MakeUnaryQPECircuit`` on a walk with an exact eigenphase."""

    @staticmethod
    def _measure(qdk_ctx, num_queries: int, k: int, system_state: int) -> int:
        """Run the synthetic-walk QPE circuit and read the phase register MSB-first."""
        num_states = num_queries + 1
        theta = -np.pi * k / num_states
        results = qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk(
            num_queries, theta, system_state
        )
        return int("".join("1" if str(bit) == "One" else "0" for bit in results), 2)

    @pytest.mark.parametrize(
        ("num_queries", "k", "system_state"),
        [(m, k, s) for m in (3, 7) for k in range(m + 1) for s in (0, 1)],
    )
    def test_measured_bin_is_twice_the_walk_phase(self, qdk_ctx, num_queries, k, system_state):
        """The measured register must read exactly ``2*phi*N``, MSB-first.

        The synthetic walk is ``W = Rz(2*theta)`` built from two self-inverse
        reflections, so ``W|1> = e^{+i theta}|1>`` and ``W|0> = e^{-i theta}|0>``.
        Choosing ``theta = -pi*k/N`` with ``N = num_queries + 1`` puts the answer
        exactly on a bin boundary, making the outcome deterministic and the
        assertion exact rather than statistical.

        This pins the whole register chain at once: the big-endian window state,
        the little-endian unary addressing reached through ``Reversed``, the
        endianness of ``Adjoint ApplyQFT`` (which writes the phase little-endian
        because ``ApplyQFT`` maps a little-endian input to a big-endian output),
        and the most-significant-bit-first order of the returned results.
        """
        num_states = num_queries + 1
        expected = (-k) % num_states if system_state == 1 else k
        assert self._measure(qdk_ctx, num_queries, k, system_state) == expected

    def test_slot_sweep_grows_linearly_with_the_query_count(self, qdk_ctx):
        """The schedule must sweep ``num_queries + 1`` slots, not apply one controlled block.

        ``MakeUnaryQPECircuit`` hands the whole phase register to a single
        ``signedPowerSchedule`` call rather than repeating a walk step itself, because unary
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
        builder = QdkUnaryQpeCircuitBuilder(num_queries=num_queries, phase_band="lower")

        expected_phase = k / (2 * num_states)
        for system_state in (0, 1):
            measured = self._measure(qdk_ctx, num_queries, k, system_state) / num_states
            assert builder.phase_fraction_from_measurement(measured) == pytest.approx(expected_phase)


class TestRegistration:
    """Registry wiring for the unary phase estimation stack."""

    def test_query_count_falls_back_to_settings(self):
        """A unitary representation without a power uses the configured query count."""

        class _Container:
            power = 1

        class _UnitaryRep:
            def get_container(self):
                return _Container()

        builder = QdkUnaryQpeCircuitBuilder(num_queries=25)
        assert builder.resolve_num_queries(_UnitaryRep()) == 25

    def test_query_count_prefers_unitary_representation(self):
        """The power carried by the unitary representation wins over the setting."""

        class _Container:
            power = 11

        class _UnitaryRep:
            def get_container(self):
                return _Container()

        builder = QdkUnaryQpeCircuitBuilder(num_queries=25)
        assert builder.resolve_num_queries(_UnitaryRep()) == 11
