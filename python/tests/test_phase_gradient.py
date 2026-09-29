"""Tests for phase gradient states and Hamming-weight phasing through them."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

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


def _expected_words(phi: float, n: int, bits: int) -> list[int]:
    """Return the ``bits``-bit word of each place value, computed independently of Q#."""
    modulus = 1 << bits
    return [round(phi * 2**j * modulus / (4.0 * math.pi)) % modulus for j in range(n)]


class TestPreparePhaseGradientState:
    """The binary gradient register every rotation is added into."""

    @pytest.mark.parametrize("num_qubits", [1, 3, 4])
    def test_prepares_the_binary_gradient(self, num_qubits):
        """The state is sum_k e^{-2 pi i k / 2^n}|k> on little-endian qubits."""
        actual = _state(f"qs => {_PG}.PreparePhaseGradientState(qs)", num_qubits)
        assert np.allclose(actual, _gradient_state(2 * math.pi / 2**num_qubits, num_qubits), atol=1e-10)


class TestBinaryGradientWords:
    """The classical words Hamming-weight phasing loads, one per place value of the weight."""

    @pytest.mark.parametrize("phi", [0.37, -2.9, 0.75 * math.pi, 9.5])
    @pytest.mark.parametrize(("n", "bits"), [(1, 4), (3, 5), (4, 8)])
    def test_words_are_the_nearest_representable_rotation(self, phi, n, bits):
        """Word j rounds Rz(phi 2^j) onto the 4 pi / 2^bits lattice the gradient resolves."""
        actual = [int(word) for word in QSHARP_UTILS.PhaseGradient.BinaryGradientWords(phi, n, bits)]
        assert actual == _expected_words(phi, n, bits)
        assert all(0 <= word < 2**bits for word in actual), "a word must fit the gradient register"

    def test_a_lattice_angle_is_represented_exactly(self):
        """Angles that are multiples of 4 pi / 2^bits round to themselves, so the phasing is exact."""
        bits, k = 5, 3
        phi = 4.0 * math.pi * k / 2**bits
        actual = [int(word) for word in QSHARP_UTILS.PhaseGradient.BinaryGradientWords(phi, 4, bits)]
        assert actual == [(k * 2**j) % 2**bits for j in range(4)]

    def test_the_offset_cancels_the_rz_layer_constant(self):
        """Rz(a) = e^{-ia/2} R1(a), so the offset must be minus the sum of the realized angles."""
        bits = 5
        words = [3, 6, 12]
        offset = QSHARP_UTILS.PhaseGradient.BinaryGradientPhaseOffset(words, bits)
        assert offset == pytest.approx(-sum(4.0 * math.pi * word / 2**bits for word in words))


class TestPhaseByBinaryGradient:
    """Hamming-weight phasing applies e^{i phi w} and returns the gradient register prepared."""

    #: A phase on the 4 pi / 2^bits lattice, so every word is exact and the assertions can be tight.
    BITS = 4
    PHI = 4.0 * math.pi * 3 / 2**4

    @classmethod
    def _operation(cls, n: int, controlled: bool) -> str:
        """Return Q# applying the phasing, with the constant offset, on a gradient prepared around it."""
        words = _expected_words(cls.PHI, n, cls.BITS)
        offset = -sum(4.0 * math.pi * word / 2**cls.BITS for word in words)
        lead = 1 if controlled else 0
        weight = f"qs[{lead}..{lead + n - 1}]"
        gradient = f"qs[{lead + n}...]"
        body = (
            f"{_PG}.PhaseByBinaryGradient({words}, {weight}, {gradient}); R(PauliI, {offset}, {weight}[0]);"
            if not controlled
            else (
                f"Controlled {_PG}.PhaseByBinaryGradient([qs[0]], ({words}, {weight}, {gradient})); "
                f"Controlled R([qs[0]], (PauliI, {offset}, {weight}[0]));"
            )
        )
        return f"qs => {{ within {{ {_PG}.PreparePhaseGradientState({gradient}); }} apply {{ {body} }} }}"

    @pytest.mark.parametrize("n", [1, 3])
    def test_phases_every_weight_and_restores_the_gradient(self, n):
        """Every weight picks up e^{i phi w}; the gradient ends in |0> after unpreparation."""
        widths = [n, self.BITS]
        amplitudes = np.zeros(2 ** sum(widths))
        for weight in range(2**n):
            amplitudes[_index([weight, 0], widths)] = 2 ** (-n / 2)
        actual = _state(self._operation(n, controlled=False), sum(widths), amplitudes.tolist())

        expected = np.zeros_like(actual)
        for weight in range(2**n):
            expected[_index([weight, 0], widths)] = 2 ** (-n / 2) * np.exp(1j * self.PHI * weight)
        assert np.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3])
    def test_controlled_phasing_acts_only_when_the_control_is_set(self, n):
        """Loading the word under control leaves the addition, and the gradient, uncontrolled."""
        widths = [1, n, self.BITS]
        amplitudes = np.zeros(2 ** sum(widths))
        for control in (0, 1):
            for weight in range(2**n):
                amplitudes[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2)
        actual = _state(self._operation(n, controlled=True), sum(widths), amplitudes.tolist())

        expected = np.zeros_like(actual)
        for control in (0, 1):
            for weight in range(2**n):
                phase = np.exp(1j * self.PHI * weight) if control else 1.0
                expected[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2) * phase
        assert np.allclose(actual, expected, atol=1e-10)
