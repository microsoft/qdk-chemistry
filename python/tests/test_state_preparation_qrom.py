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
import qdk

from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction
from qdk_chemistry.utils.qsharp import create_qsharp_context


@pytest.fixture
def qdk_ctx() -> qdk.Context:
    """Fresh Q# context for a single test."""
    return create_qsharp_context()


def _run_qrom_state_prep_and_dump(
    ctx: qdk.Context, amplitudes: list[float], num_qubits: int, bits: int = 10
) -> np.ndarray:
    """Run the QROM state preparation and return the statevector."""
    ctx.code.QDKChemistry.Utils.QROMStatePrep.RunQROMStatePrep(amplitudes, bits, num_qubits)
    state = ctx.dump_machine()
    return np.array(state.as_dense_state())


def _build_expected_from_amplitudes(amplitudes: list[float], num_qubits: int) -> np.ndarray:
    """Build the expected normalized statevector from input amplitudes."""
    n_states = 2**num_qubits
    expected = np.zeros(n_states, dtype=complex)
    for j, amp in enumerate(amplitudes):
        if j < n_states:
            expected[j] = amp
    norm = np.linalg.norm(expected)
    if norm > 0:
        expected /= norm
    return expected


def _make_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction from a list of amplitudes."""
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")) for idx in range(len(amplitudes))]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _reduced_state(sv: np.ndarray, num_qubits: int) -> np.ndarray:
    """Project the full statevector onto the ``num_qubits`` state qubits."""
    reduced = np.zeros(2**num_qubits, dtype=complex)
    stride = len(sv) // (2**num_qubits)
    for i, amp in enumerate(sv):
        if abs(amp) > 1e-12:
            reduced[i // stride] += amp
    return reduced


class TestQROMStatePreparation:
    """Tests for the QROM-based state preparation algorithm."""

    def test_run_returns_circuit(self):
        """Test that run() returns a Circuit with ops set."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        circuit = prep.run(wf)
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None
        assert circuit.num_qubits == 6

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
    def test_fidelity_random(self, qdk_ctx, num_coefficients):
        """Verify QROM state prep fidelity with random amplitudes."""
        rng = np.random.default_rng(seed=42 + num_coefficients)
        amplitudes = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_qubits = math.ceil(math.log2(num_coefficients))
        actual_sv = _run_qrom_state_prep_and_dump(qdk_ctx, amplitudes, num_qubits)
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
            sv = _run_qrom_state_prep_and_dump(create_qsharp_context(), amplitudes, num_qubits, bits=bits)
            infidelities[bits] = 1.0 - abs(np.vdot(_reduced_state(sv, num_qubits), expected))

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

    def test_negative_coefficients_are_accepted(self):
        """Signed coefficients are supported; only the imaginary part is refused."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, -0.5, 0.5, 0.5])
        assert prep.run(wf) is not None


class _EmptyCoefficientWavefunction:
    """Stand-in wavefunction whose coefficient vector is empty."""

    def get_coefficients(self) -> np.ndarray:
        """Return an empty coefficient vector."""
        return np.array([])


class TestQROMSignedAmplitudes:
    """The QROM loader's handling of signed amplitudes."""

    SIGNED_AMPLITUDES: ClassVar[list[float]] = [0.5, -0.5, 0.5, 0.5]
    SEEDS: ClassVar[list[int]] = [1, 2, 3, 4, 5, 6, 7, 8]

    def test_magnitudes_are_correct(self, qdk_ctx):
        """Whatever happens to the signs, |amplitude| is still right."""
        qdk_ctx.set_quantum_seed(1)
        num_qubits = 2
        sv = _run_qrom_state_prep_and_dump(qdk_ctx, self.SIGNED_AMPLITUDES, num_qubits, bits=4)
        actual = np.abs(_reduced_state(sv, num_qubits))

        expected = np.abs(np.array(self.SIGNED_AMPLITUDES, dtype=float))
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signs_are_preserved(self, qdk_ctx, seed):
        """The prepared state matches the signed target on every seed."""
        qdk_ctx.set_quantum_seed(seed)
        num_qubits = 2
        sv = _run_qrom_state_prep_and_dump(qdk_ctx, self.SIGNED_AMPLITUDES, num_qubits, bits=4)
        actual = _reduced_state(sv, num_qubits)

        expected = _build_expected_from_amplitudes(self.SIGNED_AMPLITUDES, num_qubits)

        fidelity = abs(np.vdot(actual, expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "amplitudes",
        [
            [-0.5, -0.5, -0.5, -0.5],
            [0.1, -0.9, 0.3, -0.2],
            [-0.2, 0.4, 0.4, -0.8],
            [0.6, 0.3, -0.1, 0.0],
        ],
    )
    def test_sign_patterns_are_preserved(self, qdk_ctx, amplitudes, seed):
        """Sign preservation holds for all-negative, mixed, and zero-containing vectors."""
        qdk_ctx.set_quantum_seed(seed)
        num_qubits = 2
        sv = _run_qrom_state_prep_and_dump(qdk_ctx, amplitudes, num_qubits, bits=8)
        actual = _reduced_state(sv, num_qubits)

        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        fidelity = abs(np.vdot(actual, expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)


def _reverse_bits(x: int, n: int) -> int:
    """Reverse the bit order of *x* within an *n*-bit field."""
    result = 0
    for k in range(n):
        if (x >> k) & 1:
            result |= 1 << (n - 1 - k)
    return result


def _target_amps(sv: np.ndarray, x: int, n_bits: int) -> tuple[complex, complex]:
    """Extract target qubit amplitudes from the full statevector."""
    angle_idx = _reverse_bits(x, n_bits) << n_bits
    idx_0 = angle_idx
    idx_1 = angle_idx | (1 << (2 * n_bits))
    return sv[idx_0], sv[idx_1]


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
    def test_rotation_amplitudes(self, qdk_ctx, x, n):
        """Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩ with θ = 4πx/2^n."""
        qdk_ctx.code.QDKChemistry.Utils.PhaseGradient.TestRy(x, n)
        sv = np.array(qdk_ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        np.testing.assert_allclose(a0.real, math.cos(theta / 2), atol=1e-6)
        np.testing.assert_allclose(a1.real, math.sin(theta / 2), atol=1e-6)
        np.testing.assert_allclose(a0.imag, 0.0, atol=1e-6)
        np.testing.assert_allclose(a1.imag, 0.0, atol=1e-6)

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (5, 5), (3, 4)])
    def test_adjoint_roundtrip(self, qdk_ctx, x, n):
        """Ry followed by Adjoint Ry returns target to |+⟩."""
        qdk_ctx.code.QDKChemistry.Utils.PhaseGradient.TestRyRoundtrip(x, n)
        sv = np.array(qdk_ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        np.testing.assert_allclose(abs(a0), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(abs(a1), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(a0 / a1, 1.0, atol=1e-8)


class TestRzViaPhaseGradient:
    """Tests for the RzViaPhaseGradient operation."""

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (2, 4), (3, 4), (1, 5), (5, 5)])
    def test_polarity(self, qdk_ctx, x, n):
        """Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2}) with θ = +4πx/2^n."""
        qdk_ctx.code.QDKChemistry.Utils.PhaseGradient.TestRzOnPlus(x, n)
        sv = np.array(qdk_ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        expected = (theta + math.pi) % (2 * math.pi) - math.pi
        np.testing.assert_allclose(cmath.phase(a1 / a0), expected, atol=1e-6)
