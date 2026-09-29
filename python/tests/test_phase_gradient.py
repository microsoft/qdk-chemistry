"""Tests for phase gradient states."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.utils.qsharp import get_qsharp_context

_PG = "QDKChemistry.Utils.PhaseGradient"


def _state(operation: str, num_qubits: int, amplitudes: list[float] | None = None) -> np.ndarray:
    """Return the state *operation* produces, numbered big-endian."""
    return np.asarray(
        dump_operation_on_state(operation, num_qubits, amplitudes, context=get_qsharp_context()), dtype=complex
    )


def _index(values: list[int], widths: list[int]) -> int:
    """Return the big-endian basis index of little-endian registers holding *values*."""
    index, offset = 0, sum(widths)
    for value, width in zip(values, widths, strict=True):
        for bit in range(width):
            offset -= 1
            index |= ((value >> bit) & 1) << offset
    return index


def _gradient_state(phase: float, num_qubits: int) -> np.ndarray:
    """Return the gradient state sum_k e^{-i phase k}|k> on little-endian qubits, big-endian numbered."""
    state = np.zeros(2**num_qubits, dtype=complex)
    for k in range(2**num_qubits):
        state[_index([k], [num_qubits])] = np.exp(-1j * phase * k)
    return state / np.sqrt(2**num_qubits)


class TestPreparePhaseGradientState:
    """The binary gradient register every rotation is added into."""

    @pytest.mark.parametrize("num_qubits", [1, 3, 4])
    def test_prepares_the_binary_gradient(self, num_qubits):
        """The state is sum_k e^{-2 pi i k / 2^n}|k> on little-endian qubits."""
        actual = _state(f"qs => {_PG}.PreparePhaseGradientState(qs)", num_qubits)
        assert np.allclose(actual, _gradient_state(2 * math.pi / 2**num_qubits, num_qubits), atol=1e-10)
