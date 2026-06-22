"""Tests for the alias sampling state preparation algorithm."""

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
from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.data import Configuration, ModelOrbitals, StateVectorContainer, Wavefunction

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)


def _run_alias_sampling_and_dump(
    coefficients: list[float],
    num_index_qubits: int,
    bits_precision: int,
) -> np.ndarray:
    """Run alias sampling state prep via qdk.Context and return the full statevector.

    Creates a fresh Q# context, loads the AliasSamplingStatePrep Q# sources
    and a thin wrapper that allocates qubits internally, then captures the
    statevector via ``ctx.dump_machine()``.
    """
    total_qubits = 2 * num_index_qubits + 2 * bits_precision + 1
    ctx = qdk.Context(project_root=_PROJECT_ROOT)
    ctx.code.QDKChemistry.Utils.AliasSampling.RunAliasSamplingPrep(
        coefficients, bits_precision, num_index_qubits, total_qubits
    )
    state = ctx.dump_machine()
    return np.array(state.as_dense_state())


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
    probs = np.zeros(n_index)
    for i in range(len(full_sv)):
        # Extract top num_index_qubits bits (BE) and reverse for LE value
        be_idx = (i >> shift) & (n_index - 1)
        index_val = int("{:0{w}b}".format(be_idx, w=num_index_qubits)[::-1], 2)
        probs[index_val] += abs(full_sv[i]) ** 2
    return probs


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


class TestAliasSamplingStatePreparation:
    """Tests for the alias sampling state preparation algorithm."""

    def test_name(self):
        """Test algorithm name."""
        prep = AliasSamplingStatePreparation()
        assert prep.name() == "alias_sampling"

    def test_type_name(self):
        """Test algorithm type name."""
        prep = AliasSamplingStatePreparation()
        assert prep.type_name() == "state_prep"

    def test_bits_precision_custom(self):
        """Test custom bits precision."""
        prep = AliasSamplingStatePreparation(bits_precision=7)
        assert prep.bits_precision == 7

    def test_registered_in_registry(self):
        """Test that alias_sampling is registered in the algorithm registry."""
        prep = registry.create("state_prep", "alias_sampling")
        assert isinstance(prep, AliasSamplingStatePreparation)

    def test_run_returns_circuit(self):
        """Test that run() returns a Circuit with ops set."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        wf = _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        circuit = prep.run(wf)
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    @pytest.mark.parametrize("num_coefficients", range(3, 10, 3))
    def test_marginal_probs_random(self, num_coefficients):
        """Verify alias sampling marginal probabilities with random coefficients.

        The alias sampling circuit prepares:
          |0⟩ → Σ_ℓ √(p̃_ℓ) |ℓ⟩|garbage_ℓ⟩
        where p̃_ℓ ≈ |c_ℓ| / Σ|c_j| is the discretized probability.

        We verify the marginal probability on the index register matches the
        expected distribution within the discretization tolerance.
        """
        rng = np.random.default_rng(seed=42 + num_coefficients)
        coefficients = rng.uniform(0.01, 1.0, size=num_coefficients).tolist()
        num_index_qubits = math.ceil(math.log2(num_coefficients))
        bits_precision = 6

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(marginal_probs[: len(coefficients)], expected_probs, atol=atol)
