"""Tests for the QROM state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import cmath
import math
from typing import ClassVar

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _dump_op(op, num_qubits: int) -> np.ndarray:
    """Dump an operation in the shared Q# context."""
    return np.array(dump_operation_on_state(op, num_qubits, context=get_qsharp_context()))


def _run_qrom_state_prep_and_dump(amplitudes: list[float], num_qubits: int, bits: int = 10) -> np.ndarray:
    """Run QROM state preparation and return state-register amplitudes."""
    params = QSHARP_UTILS.QROMStatePrep.QROMStatePrepParams(
        amplitudes=amplitudes,
        rotationBitPrecision=bits,
        numStateQubits=num_qubits,
    )
    op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
        QSHARP_UTILS.QROMStatePrep.MakeQROMStatePrepOpWithPhaseGradient(params),
        QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState,
        bits,
    )
    sv = _dump_op(op, num_qubits + bits)
    return sv.reshape(2**num_qubits, 2**bits)[:, 0]


def _reverse_bits(x: int, n: int) -> int:
    """Reverse the bit order of *x* within an *n*-bit field."""
    result = 0
    for k in range(n):
        if (x >> k) & 1:
            result |= 1 << (n - 1 - k)
    return result


def _build_expected_from_amplitudes(amplitudes: list[float], num_qubits: int) -> np.ndarray:
    """Build the expected normalized statevector from input amplitudes.

    The state register is little-endian but state dumps index big-endian, so
    coefficient ``j`` appears at the bit-reversed dump position.
    """
    n_states = 2**num_qubits
    expected = np.zeros(n_states, dtype=complex)
    for j, amp in enumerate(amplitudes):
        if j < n_states:
            expected[_reverse_bits(j, num_qubits)] = amp
    norm = np.linalg.norm(expected)
    if norm > 0:
        expected /= norm
    return expected


def _make_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction from a list of amplitudes."""
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")[::-1]) for idx in range(len(amplitudes))]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _make_sparse_wavefunction(num_qubits: int, indices: list[int], amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction that occupies only *indices* of a ``num_qubits`` register."""
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")[::-1]) for idx in indices]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


class TestQROMStatePreparation:
    """Tests for the QROM-based state preparation algorithm."""

    def test_run_returns_circuit(self):
        """By default the gradient is internal, so the callable takes only the state."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        circuit = prep.run(wf)
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None
        assert circuit.num_qubits == 2
        assert circuit.metadata.num_phase_gradient_ancillas == 0

    def test_shared_gradient_widens_the_register_it_declares(self):
        """Opting out of internal allocation appends a gradient the caller must own."""
        prep = QROMStatePreparation(rotation_bit_precision=4, allocate_phase_gradient=False)
        circuit = prep.run(_make_wavefunction([0.5, 0.3, 0.7, 0.1]))

        assert circuit.num_qubits == 6
        assert circuit.metadata.num_phase_gradient_ancillas == 4

    def test_resource_counts(self):
        """Pin the logical resource counts so a costing regression is visible."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        circuit = prep.run(_make_wavefunction([0.5, 0.3, 0.7, 0.1]))

        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == 14
        assert lc["cczCount"] == 6
        assert lc["tCount"] == 18
        assert lc["rotationCount"] == 18
        assert lc["measurementCount"] == 10

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_fidelity_random(self, num_coefficients):
        """Verify QROM state prep fidelity with random amplitudes."""
        rng = np.random.default_rng(seed=42 + num_coefficients)
        amplitudes = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_qubits = math.ceil(math.log2(num_coefficients))
        actual_sv = _run_qrom_state_prep_and_dump(amplitudes, num_qubits)
        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-5)

    def test_precision_setting_reduces_error(self):
        """Raising the rotation precision must measurably improve the prepared state."""
        amplitudes = np.random.default_rng(seed=7).uniform(0.01, 1.0, size=6).tolist()
        num_qubits = 3
        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        infidelities = {}
        for bits in (4, 10):
            sv = _run_qrom_state_prep_and_dump(amplitudes, num_qubits, bits=bits)
            infidelities[bits] = 1.0 - abs(np.vdot(sv, expected))

        assert infidelities[10] < infidelities[4] / 100, f"bRot=10 did not improve on bRot=4: {infidelities}"

    def test_settings_expose_rotation_bit_precision(self):
        """The constructor argument is stored in settings so create() can reach it."""
        prep = QROMStatePreparation(rotation_bit_precision=6)
        assert prep.settings().get("rotation_bit_precision") == 6

        prep.settings().set("rotation_bit_precision", 8)
        assert prep.settings().get("rotation_bit_precision") == 8

    def test_empty_coefficients_rejected(self):
        """An empty coefficient vector is rejected rather than reaching log2(0)."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        with pytest.raises(ValueError, match="at least one coefficient"):
            prep._run_impl(_EmptyCoefficientWavefunction())

    def test_sparse_wavefunction_uses_determinant_indices(self):
        """A coefficient belongs at its determinant's index, not its position in the list."""
        prep = QROMStatePreparation(rotation_bit_precision=8)
        params = prep._build_params(_make_sparse_wavefunction(2, [0, 3], [0.6, 0.8]))

        assert params.numStateQubits == 2
        assert list(params.amplitudes) == pytest.approx([0.6, 0.0, 0.0, 0.8])

    @pytest.mark.parametrize("index", range(4))
    def test_index_register_is_little_endian(self, index):
        """Coefficient j must land at little-endian value j, the ordering SELECT decodes."""
        num_qubits = 2
        amplitudes = [0.0] * 4
        amplitudes[index] = 1.0
        sv = _run_qrom_state_prep_and_dump(amplitudes, num_qubits, bits=8)

        dump_index = int(np.argmax(np.abs(sv)))
        assert _reverse_bits(dump_index, num_qubits) == index


class _EmptyCoefficientWavefunction:
    """Stand-in wavefunction whose coefficient vector is empty."""

    def get_coefficients(self) -> np.ndarray:
        """Return an empty coefficient vector."""
        return np.array([])


class TestQROMSignedAmplitudes:
    """The QROM loader's handling of signed amplitudes."""

    SIGNED_AMPLITUDES: ClassVar[list[float]] = [0.5, -0.5, 0.5, 0.5]

    def test_magnitudes_are_correct(self):
        """Whatever happens to the signs, |amplitude| is still right."""
        num_qubits = 2
        sv = _run_qrom_state_prep_and_dump(self.SIGNED_AMPLITUDES, num_qubits, bits=4)
        actual = np.abs(sv)

        expected = np.abs(_build_expected_from_amplitudes(self.SIGNED_AMPLITUDES, num_qubits))

        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_signs_are_preserved(self):
        """The prepared state matches the signed target."""
        num_qubits = 2
        actual = _run_qrom_state_prep_and_dump(self.SIGNED_AMPLITUDES, num_qubits, bits=4)

        expected = _build_expected_from_amplitudes(self.SIGNED_AMPLITUDES, num_qubits)

        fidelity = abs(np.vdot(actual, expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    @pytest.mark.parametrize(
        "amplitudes",
        [
            [-0.5, -0.5, -0.5, -0.5],
            [0.1, -0.9, 0.3, -0.2],
            [-0.2, 0.4, 0.4, -0.8],
            [0.6, 0.3, -0.1, 0.0],
        ],
    )
    def test_sign_patterns_are_preserved(self, amplitudes):
        """Sign preservation holds for all-negative, mixed, and zero-containing vectors."""
        num_qubits = 2
        actual = _run_qrom_state_prep_and_dump(amplitudes, num_qubits, bits=8)

        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        fidelity = abs(np.vdot(actual, expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)


def _target_amps(sv: np.ndarray, x: int, n_bits: int) -> tuple[complex, complex]:
    """Extract target qubit amplitudes from the full statevector."""
    angle_idx = _reverse_bits(x, n_bits)
    state = sv.reshape(2, 2**n_bits, 2**n_bits)
    return state[0, angle_idx, 0], state[1, angle_idx, 0]


class TestRyViaPhaseGradient:
    """Tests for the RyViaPhaseGradient operation."""

    @pytest.mark.parametrize(
        ("x", "n"),
        [
            (0, 4),
            (1, 4),
            (2, 4),
            (4, 4),
            (3, 5),
            (7, 4),
        ],
    )
    def test_rotation_amplitudes(self, x, n):
        """Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩ with θ = 4πx/2^n."""
        op = QSHARP_UTILS.PhaseGradient.MakeTestRyOp(x, n)
        sv = _dump_op(op, 1 + 2 * n)
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        np.testing.assert_allclose(a0.real, math.cos(theta / 2), atol=1e-6)
        np.testing.assert_allclose(a1.real, math.sin(theta / 2), atol=1e-6)
        np.testing.assert_allclose(a0.imag, 0.0, atol=1e-6)
        np.testing.assert_allclose(a1.imag, 0.0, atol=1e-6)

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (5, 5), (3, 4)])
    def test_adjoint_roundtrip(self, x, n):
        """Ry followed by Adjoint Ry returns target to |+⟩."""
        op = QSHARP_UTILS.PhaseGradient.MakeTestRyRoundtripOp(x, n)
        sv = _dump_op(op, 1 + 2 * n)
        a0, a1 = _target_amps(sv, x, n)

        np.testing.assert_allclose(abs(a0), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(abs(a1), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(a0 / a1, 1.0, atol=1e-8)


class TestRzViaPhaseGradient:
    """Tests for the RzViaPhaseGradient operation."""

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (2, 4), (3, 4), (1, 5), (5, 5)])
    def test_polarity(self, x, n):
        """Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2}) with θ = +4πx/2^n."""
        op = QSHARP_UTILS.PhaseGradient.MakeTestRzOnPlusOp(x, n)
        sv = _dump_op(op, 1 + 2 * n)
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        expected = (theta + math.pi) % (2 * math.pi) - math.pi
        np.testing.assert_allclose(cmath.phase(a1 / a0), expected, atol=1e-6)
