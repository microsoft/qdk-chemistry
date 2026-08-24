"""Tests for the QROM state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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
    """Fresh Q# context for a single test.

    Every test here simulates a program and reads the result off ``dump_machine``. A
    context that has already run one program reports that program's state on the next
    read, so sharing one across tests silently compares a vector against the previous
    test's output.
    """
    return create_qsharp_context()


def _run_qrom_state_prep_and_dump(ctx: qdk.Context, amplitudes: list[float], num_qubits: int) -> np.ndarray:
    """Run the QROM state preparation via qdk.Context and return the statevector.

    Loads the QROMStatePrep Q# sources and a thin wrapper that allocates
    qubits internally, then captures the statevector via ``ctx.dump_machine()``.
    """
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
    """Create a Wavefunction from a list of amplitudes.

    Zero amplitudes are kept so that determinant ``idx`` stays aligned with position ``idx``
    in the coefficient vector, which is the index the QROM circuit addresses.
    """
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")) for idx in range(len(amplitudes))]
    orbitals = ModelOrbitals(num_qubits)
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, orbitals)
    return Wavefunction(container)


def _reduced_state(sv: np.ndarray, num_qubits: int) -> np.ndarray:
    """Project the full statevector onto the ``num_qubits`` state qubits.

    ``dump_machine`` is big-endian (qubit 0 = MSB) and the state register is allocated
    first, so it occupies the top ``num_qubits`` bits of the dense index.
    """
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

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_fidelity_random(self, qdk_ctx, num_coefficients):
        """Verify QROM state prep fidelity with random amplitudes.

        The SBM decomposition should prepare:
          |ψ⟩ = Σ_j (a_j / ||a||) |j⟩
        with quantized Ry rotations via phase gradient (bRot=10), so fidelity ≈ 1.
        """
        rng = np.random.default_rng(seed=42 + num_coefficients)
        amplitudes = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_qubits = math.ceil(math.log2(num_coefficients))
        actual_sv = _run_qrom_state_prep_and_dump(qdk_ctx, amplitudes, num_qubits)
        expected = _build_expected_from_amplitudes(amplitudes, num_qubits)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    def test_settings_expose_rotation_bit_precision(self):
        """The constructor argument is stored in settings so create() can reach it."""
        prep = QROMStatePreparation(rotation_bit_precision=6)
        assert prep.rotation_bit_precision == 6
        assert prep.settings.get("rotation_bit_precision") == 6

        prep.settings.set("rotation_bit_precision", 8)
        assert prep.rotation_bit_precision == 8

    def test_empty_coefficients_rejected(self):
        """An empty coefficient vector is rejected rather than reaching log2(0)."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        with pytest.raises(ValueError, match="at least one coefficient"):
            prep._run_impl(_EmptyCoefficientWavefunction())

    def test_negative_coefficients_rejected(self):
        """Negative coefficients produce a seed-dependent wrong state, so they are refused."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, -0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="negative coefficients"):
            prep.run(wf)

    def test_non_negative_coefficients_are_accepted(self):
        """The negative-coefficient guard must not fire on ordinary input."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        assert prep.run(wf) is not None


class _EmptyCoefficientWavefunction:
    """Stand-in wavefunction whose coefficient vector is empty.

    ``StateVectorContainer`` will not build a determinant-free wavefunction, so the empty
    input guard is exercised through the minimal interface ``_run_impl`` actually uses.
    """

    def get_coefficients(self) -> np.ndarray:
        """Return an empty coefficient vector."""
        return np.array([])


class TestQROMNegativeAmplitudes:
    """The QROM loader's handling of signed amplitudes.

    :class:`QROMStatePreparation` refuses negative coefficients, so this drives the Q#
    operation directly. Magnitudes come from the multiplexed Ry rotations and are always
    correct. Signs are applied by a separate QROM-loaded ``Z`` phase kickback whose
    uncompute is not a faithful adjoint, so the sign ancilla is implicitly measured on
    release and the sign pattern collapses at random -- which is why the Python entry
    point rejects negative coefficients instead of silently returning a wrong state.
    """

    NEGATIVE_AMPLITUDES: ClassVar[list[float]] = [0.5, -0.5, 0.5, 0.5]

    def test_magnitudes_are_correct(self, qdk_ctx):
        """Whatever happens to the signs, |amplitude| is still right."""
        qdk_ctx.set_quantum_seed(1)
        num_qubits = 2
        sv = _run_qrom_state_prep_and_dump(qdk_ctx, self.NEGATIVE_AMPLITUDES, num_qubits)
        actual = np.abs(_reduced_state(sv, num_qubits))

        expected = np.abs(np.array(self.NEGATIVE_AMPLITUDES, dtype=float))
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(actual, expected, atol=1e-3)


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
    def test_rotation_probabilities(self, qdk_ctx, x, n):
        """P(|0⟩) = cos²(θ/2), P(|1⟩) = sin²(θ/2) with θ = 4πx/2^n."""
        qdk_ctx.code.QDKChemistry.Utils.PhaseGradient.TestRy(x, n)
        sv = np.array(qdk_ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        theta = 4.0 * math.pi * x / (1 << n)
        np.testing.assert_allclose(abs(a0) ** 2, math.cos(theta / 2) ** 2, atol=1e-6)
        np.testing.assert_allclose(abs(a1) ** 2, math.sin(theta / 2) ** 2, atol=1e-6)

    @pytest.mark.parametrize(("x", "n"), [(1, 4), (5, 5), (3, 4)])
    def test_adjoint_roundtrip(self, qdk_ctx, x, n):
        """Ry followed by Adjoint Ry returns target to |+⟩."""
        qdk_ctx.code.QDKChemistry.Utils.PhaseGradient.TestRyRoundtrip(x, n)
        sv = np.array(qdk_ctx.dump_machine().as_dense_state())
        a0, a1 = _target_amps(sv, x, n)

        np.testing.assert_allclose(abs(a0), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(abs(a1), 1 / math.sqrt(2), atol=1e-8)
        np.testing.assert_allclose(a0 / a1, 1.0, atol=1e-8)
