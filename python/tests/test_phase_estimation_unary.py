"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

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

    @pytest.mark.parametrize("window", ["kaiser", "cosine", "uniform"])
    @pytest.mark.parametrize("num_queries", [3, 5, 25])
    def test_padded_and_normalized(self, window, num_queries):
        """Windows are unit-norm and zero on the unaddressed padding states."""
        amplitudes = np.array(phase_window_state(num_queries, window))
        assert len(amplitudes) == 1 << num_phase_bits(num_queries)
        assert np.linalg.norm(amplitudes) == pytest.approx(1.0)
        assert np.all(amplitudes[num_queries + 1 :] == 0.0)
        assert np.all(amplitudes[: num_queries + 1] > 0.0)

    def test_kaiser_matches_bessel_definition(self):
        """The Kaiser window follows I0(pi*alpha*sqrt(1-x^2)) up to normalization."""
        num_queries, alpha = 7, 3.0
        amplitudes = np.array(phase_window_state(num_queries, "kaiser", alpha))[: num_queries + 1]
        indices = np.arange(num_queries + 1)
        x = 2.0 * indices / (num_queries + 1) - 1.0
        expected = np.i0(np.pi * alpha * np.sqrt(np.clip(1.0 - x**2, 0.0, None)))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(amplitudes, expected, rtol=1e-8)

    def test_larger_alpha_concentrates_the_window(self):
        """Increasing alpha lowers the side lobes, concentrating amplitude in the center."""
        peaked = np.array(phase_window_state(15, "kaiser", 6.0))
        flat = np.array(phase_window_state(15, "kaiser", 1.0))
        assert peaked.max() > flat.max()

    def test_unknown_window_rejected(self):
        """Unsupported window names are rejected."""
        with pytest.raises(ValueError, match="window must be one of"):
            phase_window_state(4, "hann")


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
