"""Tests for the alias sampling state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context, get_qsharp_context


def _run_alias_sampling_and_dump(
    coefficients: list[float],
    num_index_qubits: int,
    bits_precision: int,
) -> np.ndarray:
    """Run alias sampling state prep and return the full statevector."""
    total_qubits = 2 * num_index_qubits + 2 * bits_precision + 1
    params = QSHARP_UTILS.AliasSampling.AliasSamplingParams(
        coefficients=coefficients,
        bitsPrecision=bits_precision,
        numIndexQubits=num_index_qubits,
        numQubits=total_qubits,
    )
    op = QSHARP_UTILS.AliasSampling.MakeAliasSamplingOp(params)
    return np.array(dump_operation_on_state(op, total_qubits, context=get_qsharp_context()))


def _reverse_bits(values: np.ndarray, num_bits: int) -> np.ndarray:
    """Reverse the bit order of each entry of *values* within a *num_bits* field."""
    reversed_values = np.zeros_like(values)
    for k in range(num_bits):
        reversed_values |= ((values >> k) & 1) << (num_bits - 1 - k)
    return reversed_values


def _compute_marginal_probs(
    full_sv: np.ndarray,
    num_index_qubits: int,
) -> np.ndarray:
    """Compute marginal probabilities on the index register."""
    n_index = 2**num_index_qubits
    total_qubits = int(np.log2(len(full_sv)))
    shift = total_qubits - num_index_qubits

    be_index = (np.arange(len(full_sv)) >> shift) & (n_index - 1)
    probs = np.zeros(n_index)
    np.add.at(probs, _reverse_bits(be_index, num_index_qubits), np.abs(full_sv) ** 2)
    return probs


def _make_sparse_wavefunction(num_qubits: int, indices: list[int], amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction that occupies only *indices* of a ``num_qubits`` register."""
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")[::-1]) for idx in indices]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _make_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction from a list of amplitudes."""
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")[::-1]) for idx in range(len(amplitudes))]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _alias_atol(num_coefficients: int, bits_precision: int) -> float:
    """Tolerance on a marginal probability for an L-term, mu-bit alias table."""
    return 1.0 / (num_coefficients * 2**bits_precision)


class TestAliasSamplingStatePreparation:
    """Tests for the alias sampling state preparation algorithm."""

    def test_run_returns_circuit(self):
        """Test that run() returns a Circuit with ops set."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        circuit = prep.run(wf)
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_resource_counts(self):
        """Pin the logical resource counts so a costing regression is visible.

        These are now the counts of the circuit that executes. They used to be the counts of
        ``AliasSamplingPrepareLegacyResourceEstimate``, a branch ``AliasSamplingPrepare`` took
        under ``IsResourceEstimating()`` to reproduce an archived Fe2S2 headline; it omitted
        the comparator and the controlled index swap, so tracing it never touched the flag
        qubit and reported 14 qubits / 2 CCZ. With that branch deleted the trace is the
        runnable circuit throughout, which needs 17 / 8 -- exactly the figures the old
        docstring predicted for it.

        The conditional lookup is pinned at a representative shape: 90 conditions, 16 slots,
        and 21-bit words. At the optimal three swap bits, erasing the load by measurement
        rather than running it backwards takes its adjoint from 408 to 83 CCZ gates, and its
        round trip from 816 to 491.
        """
        prep = AliasSamplingStatePreparation(bits_precision=4)
        circuit = prep.run(_make_wavefunction([0.5, 0.3, 0.7, 0.1]))

        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == 17
        assert lc["cczCount"] == 8
        assert lc["tCount"] == 0
        assert lc["rotationCount"] == 0
        assert lc["measurementCount"] == 6

        context = create_qsharp_context()
        select_swap = context.code.QDKChemistry.Utils.SelectSwap
        lookup_data = [
            [[bool((17 * outer + 5 * inner + bit) % 7 < 3) for bit in range(21)] for inner in range(16)]
            for outer in range(90)
        ]
        num_swap_bits = select_swap.ComputeOptimalLambda2D(90, 16, 21, True)
        assert num_swap_bits == 3, f"expected three swap bits, got {num_swap_bits}"
        expected_lookup_counts = {
            "forward": (
                (True, False),
                {
                    "numQubits": 283,
                    "cczCount": 408,
                    "ccixCount": 0,
                    "tCount": 0,
                    "rotationCount": 0,
                    "measurementCount": 429,
                },
            ),
            "adjoint": (
                (False, True),
                {
                    "numQubits": 115,
                    "cczCount": 83,
                    "ccixCount": 0,
                    "tCount": 0,
                    "rotationCount": 0,
                    "measurementCount": 104,
                },
            ),
            "round_trip": (
                (True, True),
                {
                    "numQubits": 283,
                    "cczCount": 491,
                    "ccixCount": 0,
                    "tCount": 0,
                    "rotationCount": 0,
                    "measurementCount": 533,
                },
            ),
        }
        for direction, (flags, expected) in expected_lookup_counts.items():
            counts = context.logical_counts(
                select_swap.TestSelectSwap2DResourceProbe, lookup_data, num_swap_bits, True, *flags
            )
            actual = {name: counts[name] for name in expected}
            assert actual == expected, f"conditional alias lookup {direction}: {actual} != {expected}"

    def test_settings_expose_bits_precision(self):
        """The constructor argument is stored in settings so create() can reach it."""
        prep = AliasSamplingStatePreparation(bits_precision=6)
        assert prep.settings().get("bits_precision") == 6

        prep.settings().set("bits_precision", 8)
        assert prep.settings().get("bits_precision") == 8

    def test_negative_coefficients_rejected(self):
        """Alias sampling is a PREPARE oracle over magnitudes and cannot carry a sign."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        wf = _make_wavefunction([0.5, -0.3, 0.7, 0.1])
        with pytest.raises(ValueError, match="non-negative"):
            prep.run(wf)

    def test_sparse_wavefunction_uses_determinant_indices(self):
        """A coefficient belongs at its determinant's index, not its position in the list."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        weights, num_index_qubits = prep._sampling_weights(_make_sparse_wavefunction(2, [0, 3], [0.6, 0.8]))

        assert num_index_qubits == 2
        assert weights == pytest.approx([0.36, 0.0, 0.0, 0.64])

    def test_coefficients_that_overflow_when_squared_are_rejected(self):
        """The finiteness guard has to survive the squaring, not just precede it."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        with pytest.raises(ValueError, match="overflows to infinity"):
            prep.run(_make_wavefunction([1e200, 1.0]))

    def test_zero_coefficients_get_zero_probability(self):
        """A zero coefficient must receive exactly zero probability."""
        bits_precision = 4
        coefficients = [0.5, 0.3, 0.2, 0.0]
        num_index_qubits = 2

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        assert marginal_probs[3] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            marginal_probs,
            [0.5, 0.3, 0.2, 0.0],
            atol=_alias_atol(len(coefficients), bits_precision),
        )

    def test_largest_remainder_keeps_bound(self):
        """Reviewer counterexample: index-order residual fill-in broke 1/(L 2^mu)."""
        bits_precision = 1
        coefficients = [0.35, 0.30, 0.175, 0.175]
        num_index_qubits = 2

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        np.testing.assert_allclose(
            marginal_probs[: len(coefficients)],
            coefficients,
            atol=_alias_atol(len(coefficients), bits_precision),
        )

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_marginal_probs_random(self, num_coefficients):
        """Verify alias sampling marginal probabilities with random coefficients."""
        rng = np.random.default_rng(seed=42 + num_coefficients)
        coefficients = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_index_qubits = math.ceil(math.log2(num_coefficients))
        bits_precision = 6

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        np.testing.assert_allclose(
            marginal_probs[: len(coefficients)],
            expected_probs,
            atol=_alias_atol(num_coefficients, bits_precision),
        )

    def test_precision_setting_reduces_error(self):
        """Raising mu must measurably improve the prepared distribution."""
        coefficients = np.random.default_rng(seed=7).uniform(0.01, 1.0, size=6).tolist()
        expected = np.abs(coefficients) / np.sum(np.abs(coefficients))
        num_index_qubits = 3

        errors = {}
        for bits_precision in (3, 7):
            full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
            probs = _compute_marginal_probs(full_sv, num_index_qubits)
            errors[bits_precision] = np.max(np.abs(probs[: len(coefficients)] - expected))

        assert errors[7] < errors[3] / 4, f"mu=7 did not improve on mu=3: {errors}"


def _run_conditional_alias_fr_and_dump(
    coefficients: list[list[float]],
    free_rider_data: list[list[bool]],
    bits_precision: int,
    condition_value: int,
) -> np.ndarray:
    """Run conditional alias sampling with free-rider and return statevector."""
    n_cond = len(coefficients)
    n_coeffs = len(coefficients[0])
    n_index_bits = math.ceil(math.log2(n_coeffs))
    n_cond_bits = math.ceil(math.log2(n_cond))
    n_free_rider_bits = len(free_rider_data[0]) if free_rider_data else 0
    n_qrom_output = bits_precision + n_index_bits + 2
    total_qubits = n_cond_bits + n_index_bits + bits_precision + 1 + n_qrom_output + n_free_rider_bits
    op = QSHARP_UTILS.AliasSampling.MakeConditionalAliasSamplingPrepWithFreeRiderOp(
        coefficients, free_rider_data, bits_precision, condition_value
    )
    return np.array(dump_operation_on_state(op, total_qubits, context=get_qsharp_context()))


def _compute_conditional_marginal_probs(
    full_sv: np.ndarray,
    n_cond_bits: int,
    n_index_bits: int,
    condition_value: int,
) -> np.ndarray:
    """Compute marginal probabilities on the index register for a given condition."""
    total_qubits = int(np.log2(len(full_sv)))
    n_index = 2**n_index_bits
    indices = np.arange(len(full_sv))

    cond_be = (indices >> (total_qubits - n_cond_bits)) & ((1 << n_cond_bits) - 1)
    index_be = (indices >> (total_qubits - n_cond_bits - n_index_bits)) & (n_index - 1)
    on_condition = _reverse_bits(cond_be, n_cond_bits) == condition_value

    probs = np.zeros(n_index)
    np.add.at(
        probs,
        _reverse_bits(index_be[on_condition], n_index_bits),
        np.abs(full_sv[on_condition]) ** 2,
    )
    return probs


class TestConditionalAliasSamplingWithFreeRider:
    """Tests for conditional alias sampling with free-rider data."""

    @pytest.mark.parametrize("condition_value", [0, 1])
    def test_free_rider_register_holds_the_loaded_bits(self, condition_value):
        """The free rider depends only on the condition, so it comes out definite, not sampled."""
        n_cond, n_coeffs, n_fr_bits = 2, 4, 3
        rng = np.random.default_rng(seed=456 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        free_rider_data = [[bool(rng.integers(0, 2)) for _ in range(n_fr_bits)] for _ in range(n_cond)]

        full_sv = _run_conditional_alias_fr_and_dump(coefficients, free_rider_data, 6, condition_value)

        # The free rider occupies the last qubits, so its bits are the low field of the basis
        # index, read most significant first.
        loaded = sum(int(bit) << (n_fr_bits - 1 - j) for j, bit in enumerate(free_rider_data[condition_value]))
        fr_field = np.arange(len(full_sv)) & ((1 << n_fr_bits) - 1)
        off_pattern = float(np.sum(np.abs(full_sv[fr_field != loaded]) ** 2))
        assert off_pattern < 1e-12, f"amplitude escaped free_rider={free_rider_data[condition_value]}"


def _run_conditional_alias_and_dump(
    coefficients: list[list[float]],
    bits_precision: int,
    condition_value: int,
    num_swap_bits: int,
) -> np.ndarray:
    """Run conditional alias sampling (no free rider) and return statevector."""
    n_cond = len(coefficients)
    n_coeffs = len(coefficients[0])
    n_index_bits = math.ceil(math.log2(n_coeffs))
    n_cond_bits = math.ceil(math.log2(n_cond))
    n_qrom_output = bits_precision + n_index_bits + 2
    total_qubits = n_cond_bits + n_index_bits + bits_precision + 1 + n_qrom_output
    op = QSHARP_UTILS.AliasSampling.MakeConditionalAliasSamplingPrepOp(
        coefficients, bits_precision, condition_value, num_swap_bits
    )
    return np.array(dump_operation_on_state(op, total_qubits, context=get_qsharp_context()))


class TestConditionalAliasSampling:
    """Tests for conditional alias sampling without free-rider data."""

    @pytest.mark.parametrize("num_swap_bits", [0, -1, 1])
    @pytest.mark.parametrize("condition_value", [0, 1])
    def test_marginal_probs(self, condition_value, num_swap_bits):
        """Verify conditional marginal probs for no-swap, optimal, and select-swap loads."""
        n_cond, n_coeffs = 2, 4
        rng = np.random.default_rng(seed=789 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        bits_precision = 6
        n_index_bits = math.ceil(math.log2(n_coeffs))
        n_cond_bits = math.ceil(math.log2(n_cond))

        full_sv = _run_conditional_alias_and_dump(coefficients, bits_precision, condition_value, num_swap_bits)
        marginal_probs = _compute_conditional_marginal_probs(full_sv, n_cond_bits, n_index_bits, condition_value)

        abs_coeffs = np.abs(coefficients[condition_value])
        expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

        np.testing.assert_allclose(
            marginal_probs[:n_coeffs],
            expected_probs,
            atol=_alias_atol(n_coeffs, bits_precision),
            err_msg=f"cond={condition_value}, num_swap_bits={num_swap_bits}",
        )
