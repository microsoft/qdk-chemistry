"""Tests for the alias sampling state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
import qdk

from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction
from qdk_chemistry.utils.qsharp import create_qsharp_context


@pytest.fixture
def qdk_ctx() -> qdk.Context:
    """Fresh Q# context for a single test.

    Every test here simulates a program and reads the result off ``dump_machine``. A
    context that has already run one program reports that program's state on the next
    read, so sharing one across tests silently compares a vector against the previous
    test's output.
    """
    return create_qsharp_context()


def _run_alias_sampling_and_dump(
    ctx: qdk.Context,
    coefficients: list[float],
    num_index_qubits: int,
    bits_precision: int,
) -> np.ndarray:
    """Run alias sampling state prep via qdk.Context and return the full statevector.

    Loads the AliasSamplingStatePrep Q# sources and a thin wrapper that
    allocates qubits internally, then captures the statevector via
    ``ctx.dump_machine()``.
    """
    total_qubits = 2 * num_index_qubits + 2 * bits_precision + 1
    ctx.code.QDKChemistry.Utils.AliasSampling.RunAliasSamplingPrep(
        coefficients, bits_precision, num_index_qubits, total_qubits
    )
    state = ctx.dump_machine()
    return np.array(state.as_dense_state())


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
    """Compute marginal probabilities on the index register.

    The alias sampling register layout places the index register on the
    first ``num_index_qubits`` qubits (LE ordering).  dump_machine uses
    big-endian convention (qubit 0 = MSB), so the index register occupies
    the *top* bits of the statevector index.  We extract those bits and
    reverse them to recover the little-endian index value.
    """
    n_index = 2**num_index_qubits
    total_qubits = int(np.log2(len(full_sv)))
    shift = total_qubits - num_index_qubits

    be_index = (np.arange(len(full_sv)) >> shift) & (n_index - 1)
    probs = np.zeros(n_index)
    np.add.at(probs, _reverse_bits(be_index, num_index_qubits), np.abs(full_sv) ** 2)
    return probs


def _make_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction from a list of amplitudes.

    Zero amplitudes are kept so that determinant ``idx`` stays aligned with position ``idx``
    in the coefficient vector, which is the index the alias circuit addresses.
    """
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")) for idx in range(len(amplitudes))]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _alias_atol(num_coefficients: int, bits_precision: int) -> float:
    """Tolerance on a marginal probability for an L-term, mu-bit alias table.

    Babbush et al. (arXiv:1805.03662) Eq. (35) bounds the per-coefficient error on the
    normalized probability by 1 / (L 2^mu), and their Eq. (39) makes each prepared
    probability an exact multiple of that same quantum, so one differently-resolved tie in
    the classical table's rounding moves a coefficient by a full step. This implementation
    also pads the index register to a power of two, which the reference construction does
    not, and measurement shows up to 1.24x the reference bound at the worst padding ratio.
    The constant is therefore 2 rather than 1, which still rejects a wrong
    ``bits_precision``; the flat 2^-mu that the setting's description quotes would be a
    factor of L too loose to do so.
    """
    return 2.0 / (num_coefficients * 2**bits_precision)


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

        Alias sampling spends its budget entirely on the QROM lookup and the comparator, so
        there are no rotations at all; the whole circuit is Clifford + CCZ + measurement.
        """
        prep = AliasSamplingStatePreparation(bits_precision=4)
        circuit = prep.run(_make_wavefunction([0.5, 0.3, 0.7, 0.1]))

        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == 17
        assert lc["cczCount"] == 8
        assert lc["tCount"] == 0
        assert lc["rotationCount"] == 0
        assert lc["measurementCount"] == 6

    def test_settings_expose_bits_precision(self):
        """The constructor argument is stored in settings so create() can reach it."""
        prep = AliasSamplingStatePreparation(bits_precision=6)
        assert prep.settings().get("bits_precision") == 6

        prep.settings().set("bits_precision", 8)
        assert prep.settings().get("bits_precision") == 8

    def test_negative_coefficients_rejected(self):
        """Alias sampling is a PREPARE oracle over magnitudes and cannot carry a sign.

        Silently dropping the sign would prepare a different state than the caller asked
        for, so a negative coefficient is a hard error rather than a warning. This matches
        Qualtran's ``StatePreparationAliasSampling``, which raises for the same reason.
        """
        prep = AliasSamplingStatePreparation(bits_precision=4)
        wf = _make_wavefunction([0.5, -0.3, 0.7, 0.1])
        with pytest.raises(ValueError, match="non-negative"):
            prep.run(wf)

    def test_zero_coefficients_get_zero_probability(self, qdk_ctx):
        """A zero coefficient must receive exactly zero probability.

        The alias table stores a keep value out of a full bar height of 2^mu, so a
        coefficient of zero maps to keep = 0 and is always redirected to its alias. An
        off-by-one in the bar height would instead leave it a residual 2^-mu of probability.
        """
        bits_precision = 4
        coefficients = [0.5, 0.3, 0.2, 0.0]
        num_index_qubits = 2

        full_sv = _run_alias_sampling_and_dump(qdk_ctx, coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        assert marginal_probs[3] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            marginal_probs,
            [0.5, 0.3, 0.2, 0.0],
            atol=_alias_atol(len(coefficients), bits_precision),
        )

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_marginal_probs_random(self, qdk_ctx, num_coefficients):
        """Verify alias sampling marginal probabilities with random coefficients.

        ``AliasSamplingPrepare`` is the LCU PREPARE oracle of Babbush et al. Fig. 11: it
        samples the **1-norm** distribution p_l = |c_l| / Sum_k |c_k|, so the index register
        carries amplitude sqrt(p_l). This is deliberately different from
        ``ConditionalAliasSamplingPrepareWithFreeRider`` below, which squares its
        coefficients to realize the 2-norm state preparation instead.
        """
        rng = np.random.default_rng(seed=42 + num_coefficients)
        coefficients = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_index_qubits = math.ceil(math.log2(num_coefficients))
        bits_precision = 6

        full_sv = _run_alias_sampling_and_dump(qdk_ctx, coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        np.testing.assert_allclose(
            marginal_probs[: len(coefficients)],
            expected_probs,
            atol=_alias_atol(num_coefficients, bits_precision),
        )

    def test_precision_setting_reduces_error(self):
        """Raising mu must measurably improve the prepared distribution.

        The tolerance in the tests above scales as 1/(L 2^mu), but it still cannot separate
        neighbouring values of mu. This pins the setting from the other side: the discretization
        error has to actually shrink when more keep bits are requested, which fails immediately
        if ``bits_precision`` is ignored somewhere in the chain down to Q#.
        """
        coefficients = np.random.default_rng(seed=7).uniform(0.01, 1.0, size=6).tolist()
        expected = np.abs(coefficients) / np.sum(np.abs(coefficients))
        num_index_qubits = 3

        errors = {}
        for bits_precision in (3, 7):
            # A context that has already run a program replays that program's state, so each
            # precision needs its own.
            full_sv = _run_alias_sampling_and_dump(
                create_qsharp_context(), coefficients, num_index_qubits, bits_precision
            )
            probs = _compute_marginal_probs(full_sv, num_index_qubits)
            errors[bits_precision] = np.max(np.abs(probs[: len(coefficients)] - expected))

        assert errors[7] < errors[3] / 4, f"mu=7 did not improve on mu=3: {errors}"


def _run_conditional_alias_fr_and_dump(
    ctx: qdk.Context,
    coefficients: list[list[float]],
    free_rider_data: list[list[bool]],
    bits_precision: int,
    condition_value: int,
) -> np.ndarray:
    """Run conditional alias sampling with free-rider and return statevector."""
    ctx.code.QDKChemistry.Utils.AliasSampling.RunConditionalAliasSamplingPrepWithFreeRider(
        coefficients, free_rider_data, bits_precision, condition_value
    )
    state = ctx.dump_machine()
    return np.array(state.as_dense_state())


def _compute_conditional_marginal_probs(
    full_sv: np.ndarray,
    n_cond_bits: int,
    n_index_bits: int,
    condition_value: int,
) -> np.ndarray:
    """Compute marginal probabilities on the index register for a given condition.

    Register layout (LE): conditionalReg[nCond] + indexReg[nIdx] + ancilla.
    dump_machine uses BE: qubit 0 = MSB.
    """
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

    @pytest.mark.parametrize(
        ("n_cond", "n_coeffs", "condition_value"),
        [
            (2, 4, 0),
            (2, 4, 1),
        ],
    )
    def test_marginal_probs_with_free_rider(self, qdk_ctx, n_cond, n_coeffs, condition_value):
        """Verify marginal probs and free-rider data loading.

        Unlike ``AliasSamplingPrepare``, this variant squares its coefficients before
        building the alias table, so it samples the **2-norm** distribution
        p_l = c_l^2 / Sum_k c_k^2 and the index register carries amplitude |c_l| / ||c||_2.
        A QROM-loaded sign bit then applies a Z phase to recover the signed amplitude. The
        squared expectation below is therefore intentional, not a copy of the 1-norm
        assertion in ``test_marginal_probs_random``.
        """
        rng = np.random.default_rng(seed=456 + n_cond * 10 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        n_fr_bits = 3
        free_rider_data = [[bool(rng.integers(0, 2)) for _ in range(n_fr_bits)] for _ in range(n_cond)]
        bits_precision = 6
        n_index_bits = math.ceil(math.log2(n_coeffs))
        n_cond_bits = math.ceil(math.log2(n_cond))

        full_sv = _run_conditional_alias_fr_and_dump(
            qdk_ctx, coefficients, free_rider_data, bits_precision, condition_value
        )
        marginal_probs = _compute_conditional_marginal_probs(full_sv, n_cond_bits, n_index_bits, condition_value)

        abs_coeffs = np.abs(coefficients[condition_value])
        expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

        np.testing.assert_allclose(
            marginal_probs[:n_coeffs],
            expected_probs,
            atol=_alias_atol(n_coeffs, bits_precision),
            err_msg=f"cond={condition_value}, free_rider={free_rider_data[condition_value]}",
        )
