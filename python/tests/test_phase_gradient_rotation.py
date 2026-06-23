"""Tests for PhaseGradient.qs: RyViaPhaseGradient.

Validates that the phase gradient rotation operations produce correct
Ry rotations by comparing simulation statevectors against
analytically expected values.

The rotation angle is θ = 4π·x/2^b where x is the integer encoded
in angleQubits and b = Length(phaseGradient).
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

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"


def _make_ctx() -> qdk.Context:
    return qdk.Context(project_root=str(_QS_DIR))


def _reverse_bits(x: int, n: int) -> int:
    """Reverse the bit order of *x* within an *n*-bit field."""
    result = 0
    for k in range(n):
        if (x >> k) & 1:
            result |= 1 << (n - 1 - k)
    return result


def _target_amps(sv: np.ndarray, x: int, n_bits: int) -> tuple[complex, complex]:
    """Extract target qubit amplitudes from the full statevector.

    Qubit layout (BE in dump_machine): qubit 0 = MSB.
    Allocation order: target[0] (bit 2n), angle[0..n-1] (bits 2n-1..n), pg[0..n-1] (bits n-1..0).
    After uncomputing pg → |0⟩ and angle = |x⟩ (LE), the angle's LE bits
    map to descending bit positions, requiring bit-reversal of x.
    """
    angle_idx = _reverse_bits(x, n_bits) << n_bits
    idx_0 = angle_idx  # target = |0⟩
    idx_1 = angle_idx | (1 << (2 * n_bits))  # target = |1⟩
    return sv[idx_0], sv[idx_1]


class TestRyViaPhaseGradient:
    """Tests for the RyViaPhaseGradient operation."""

    @pytest.mark.parametrize(
        ("x", "n"),
        [
            (0, 4),  # θ = 0 → Ry = I
            (1, 4),  # θ = π/4
            (2, 4),  # θ = π/2
            (4, 4),  # θ = π → Ry|0⟩ = |1⟩
            (3, 5),  # θ = 3π/8
            (7, 4),  # θ = 7π/4
        ],
    )
    def test_rotation_probabilities(self, x, n):
        """P(|0⟩) = cos²(θ/2), P(|1⟩) = sin²(θ/2) with θ = 4πx/2^n."""
        ctx = _make_ctx()
        ctx.code.QDKChemistry.Utils.PhaseGradient.TestRy(x, n)
        sv = np.array(ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        np.testing.assert_allclose(abs(a0) ** 2, math.cos(theta / 2) ** 2, atol=1e-6)
        np.testing.assert_allclose(abs(a1) ** 2, math.sin(theta / 2) ** 2, atol=1e-6)

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (5, 5), (3, 4)])
    def test_adjoint_roundtrip(self, x, n):
        """Ry followed by Adjoint Ry returns target to |+⟩."""
        ctx = _make_ctx()
        ctx.code.QDKChemistry.Utils.PhaseGradient.TestRyRoundtrip(x, n)
        sv = np.array(ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        np.testing.assert_allclose(abs(a0), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(abs(a1), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(a0 / a1, 1.0, atol=1e-8)
