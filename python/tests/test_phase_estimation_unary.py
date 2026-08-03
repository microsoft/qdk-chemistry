"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.unary_phase_estimation_builder import (
    QdkUnaryQpeCircuitBuilder,
    num_phase_bits,
    phase_window_state,
)
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import (
    UnaryPhaseEstimation,
    _select_dominant_decoded_phase,
)


class TestUnaryIterationQsharp:
    """Q# level checks of the signed-power schedule."""

    @pytest.mark.parametrize("address_value", [0, 1, 2])
    def test_power_schedule_realizes_signed_powers(self, qdk_ctx, address_value):
        """With A = X, B = H and p = 2, slot t applies (XH)^(2-2t), which is the identity only at t = 1."""
        num_blocks = 2
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationPowerSchedule(num_blocks, address_value)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        probability_zero = abs(state[0]) ** 2
        if num_blocks - 2 * address_value == 0:
            assert probability_zero == pytest.approx(1.0, abs=1e-9)
        else:
            assert probability_zero == pytest.approx(0.0, abs=1e-9)


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

    def test_builder_is_registered(self):
        """The builder is available under its registry name."""
        builder = create("qpe_circuit_builder", "qdk_unary")
        assert isinstance(builder, QdkUnaryQpeCircuitBuilder)
        assert builder.settings().get("phase_window") == "kaiser"

    def test_algorithm_is_registered_with_matching_builder(self):
        """The algorithm defaults to the unary builder."""
        algorithm = create("phase_estimation", "qdk_unary", shots=17)
        assert isinstance(algorithm, UnaryPhaseEstimation)
        assert algorithm.settings().get("shots") == 17
        assert algorithm.settings().get("qpe_circuit_builder").algorithm_name == "qdk_unary"

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
