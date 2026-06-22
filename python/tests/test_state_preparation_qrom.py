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

# Q# source directory (relative to this test file, not the installed package).
_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction


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
        assert prep.name() == "qrom_state_prep"

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
        """Test that qrom_state_prep is registered in the algorithm registry."""
        prep = registry.create("state_prep", "qrom_state_prep")
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
