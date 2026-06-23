"""Tests for conditional alias sampling state preparation (2D QROM).

Standalone test file — only requires qdk, numpy, pytest (no qdk_chemistry build).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from pathlib import Path

import numpy as np
import pytest
import qdk
from qdk import qsharp

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════


def _run_conditional_alias_sampling_and_dump(
    coefficients: list[list[float]],
    bits_precision: int,
    condition_value: int,
) -> np.ndarray:
    """Run conditional alias sampling and return the full statevector."""
    qsharp.init(project_root=_PROJECT_ROOT)
    qdk.code.QDKChemistry.Utils.AliasSampling.RunConditionalAliasSamplingPrep(
        coefficients, bits_precision, condition_value
    )
    state = qsharp.dump_machine()
    return np.array(state.as_dense_state())


def _run_conditional_alias_fr_and_dump(
    coefficients: list[list[float]],
    free_rider_data: list[list[bool]],
    bits_precision: int,
    condition_value: int,
) -> np.ndarray:
    """Run conditional alias sampling with free-rider and return statevector."""
    qsharp.init(project_root=_PROJECT_ROOT)
    qdk.code.QDKChemistry.Utils.AliasSampling.RunConditionalAliasSamplingPrepWithFreeRider(
        coefficients, free_rider_data, bits_precision, condition_value
    )
    state = qsharp.dump_machine()
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
    probs = np.zeros(n_index)

    for i in range(len(full_sv)):
        amp = full_sv[i]
        if abs(amp) < 1e-15:
            continue
        bits = format(i, f"0{total_qubits}b")
        cond_be = bits[:n_cond_bits]
        cond_val = int(cond_be[::-1], 2)  # reverse for LE
        if cond_val != condition_value:
            continue
        idx_be = bits[n_cond_bits : n_cond_bits + n_index_bits]
        idx_val = int(idx_be[::-1], 2)  # reverse for LE
        probs[idx_val] += abs(amp) ** 2

    return probs


# ════════════════════════════════════════════════════════════════════════════
#  Tests
# ════════════════════════════════════════════════════════════════════════════


class TestConditionalAliasSamplingPrepare:
    """Tests for conditional alias sampling state preparation (2D)."""

    @pytest.mark.parametrize(
        ("n_cond", "n_coeffs", "condition_value"),
        [
            (2, 4, 0),
            (2, 4, 1),
            (3, 4, 2),
        ],
    )
    def test_marginal_probs(self, n_cond, n_coeffs, condition_value):
        """Verify conditional alias sampling marginal probs match expected.

        For each condition c, the index marginals should match
        |coeff[c][b]|² / Σ_j |coeff[c][j]|².
        """
        rng = np.random.default_rng(seed=123 + n_cond * 10 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        bits_precision = 6
        n_index_bits = math.ceil(math.log2(n_coeffs))
        n_cond_bits = math.ceil(math.log2(n_cond))

        full_sv = _run_conditional_alias_sampling_and_dump(coefficients, bits_precision, condition_value)
        marginal_probs = _compute_conditional_marginal_probs(full_sv, n_cond_bits, n_index_bits, condition_value)

        abs_coeffs = np.abs(coefficients[condition_value])
        expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(
            marginal_probs[:n_coeffs], expected_probs, atol=atol, err_msg=f"cond={condition_value}"
        )


class TestConditionalAliasSamplingWithFreeRider:
    """Tests for conditional alias sampling with free-rider data."""

    @pytest.mark.parametrize(
        ("n_cond", "n_coeffs", "condition_value"),
        [
            (2, 4, 0),
            (2, 4, 1),
        ],
    )
    def test_marginal_probs_with_free_rider(self, n_cond, n_coeffs, condition_value):
        """Verify marginal probs and free-rider data loading."""
        rng = np.random.default_rng(seed=456 + n_cond * 10 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        n_fr_bits = 3
        free_rider_data = [[bool(rng.integers(0, 2)) for _ in range(n_fr_bits)] for _ in range(n_cond)]
        bits_precision = 6
        n_index_bits = math.ceil(math.log2(n_coeffs))
        n_cond_bits = math.ceil(math.log2(n_cond))

        full_sv = _run_conditional_alias_fr_and_dump(coefficients, free_rider_data, bits_precision, condition_value)
        marginal_probs = _compute_conditional_marginal_probs(full_sv, n_cond_bits, n_index_bits, condition_value)

        abs_coeffs = np.abs(coefficients[condition_value])
        expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(
            marginal_probs[:n_coeffs],
            expected_probs,
            atol=atol,
            err_msg=f"cond={condition_value}, free_rider={free_rider_data[condition_value]}",
        )
