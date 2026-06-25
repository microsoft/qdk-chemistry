"""Tests for the QROM state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from pathlib import Path

import numpy as np
import pytest
import qdk

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction

# Path to the QSharp source directory used for test context initialization.
_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"


def _run_qrom_state_prep_and_dump(amplitudes: list[float], num_qubits: int) -> np.ndarray:
    """Run the QROM state preparation via qdk.Context and return the statevector.

    Creates a fresh Q# context, loads the QROMStatePrep Q# sources and a
    thin wrapper that allocates qubits internally, then captures the
    statevector via ``ctx.dump_machine()``.
    """
    ctx = qdk.Context(project_root=str(_QS_DIR))
    ctx.code.QDKChemistry.Utils.QROMStatePrep.RunQROMStatePrep(amplitudes, 10, num_qubits)
    state = ctx.dump_machine()
    return np.array(state.as_dense_state())


def _build_expected_from_amplitudes(amplitudes: list[float], num_qubits: int) -> np.ndarray:
    """Build the expected normalized statevector from input amplitudes.

    The QROM SBM decomposition prepares Σ_j (a_j/||a||) |j⟩ where j indexes
    the computational basis in Q# little-endian order (qubit k = bit k).
    """
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
    coeffs_list: list[float] = []
    dets: list[Configuration] = []
    for idx, amp in enumerate(amplitudes):
        if amp != 0.0:
            bitstring = format(idx, f"0{num_qubits}b")
            dets.append(Configuration.from_bitstring(bitstring))
            coeffs_list.append(float(amp))
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array(coeffs_list), dets, orbitals)
    return Wavefunction(container)


class TestQROMStatePreparation:
    """Tests for the QROM-based state preparation algorithm."""

    def test_name(self):
        """Test algorithm name."""
        prep = QROMStatePreparation()
        assert prep.name() == "qrom"

    def test_type_name(self):
        """Test algorithm type name."""
        prep = QROMStatePreparation()
        assert prep.type_name() == "state_prep"

    def test_rotation_bit_precision_custom(self):
        """Test custom rotation bit precision."""
        prep = QROMStatePreparation(rotation_bit_precision=8)
        assert prep.rotation_bit_precision == 8

    def test_run_returns_circuit(self):
        """Test that run() returns a Circuit with ops set."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        circuit = prep.run(wf)
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_registered_in_registry(self):
        """Test that qrom is registered in the algorithm registry."""
        prep = registry.create("state_prep", "qrom")
        assert isinstance(prep, QROMStatePreparation)

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_fidelity_random(self, num_coefficients):
        """Verify QROM state prep fidelity with random amplitudes.

        The SBM decomposition should prepare:
          |ψ⟩ = Σ_j (a_j / ||a||) |j⟩
        with quantized Ry rotations via phase gradient (bRot=10), so fidelity ≈ 1.
        """
        rng = np.random.default_rng(seed=42 + num_coefficients)
        amplitudes = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_qubits = math.ceil(math.log2(num_coefficients))
        actual_sv = _run_qrom_state_prep_and_dump(amplitudes, num_qubits)
        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

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
