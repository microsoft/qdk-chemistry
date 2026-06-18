"""Tests for the alias sampling state preparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import qdk

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)

# Q# wrapper: allocates qubits via QIR.Runtime so they persist for dump_machine
# (Qubit values cannot cross the Python ↔ Q# boundary).
_ALIAS_WRAPPER_QS = """
operation RunAliasSamplingPrep(
    coefficients : Double[],
    bitsPrecision : Int,
    numIndexQubits : Int,
    numQubits : Int,
) : Unit {
    let qs = QIR.Runtime.AllocateQubitArray(numQubits);
    let params = new QDKChemistry.Utils.AliasSampling.AliasSamplingParams {
        coefficients = coefficients,
        bitsPrecision = bitsPrecision,
        numIndexQubits = numIndexQubits,
        numQubits = numQubits,
    };
    QDKChemistry.Utils.AliasSampling.AliasSamplingPrepare(params, qs);
}
"""


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
    ctx.eval(_ALIAS_WRAPPER_QS)
    ctx.code.RunAliasSamplingPrep(coefficients, bits_precision, num_index_qubits, total_qubits)
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
        index_val = int('{:0{w}b}'.format(be_idx, w=num_index_qubits)[::-1], 2)
        probs[index_val] += abs(full_sv[i]) ** 2
    return probs


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

    def test_prepare_from_statevector_returns_circuit(self):
        """Test that prepare_from_statevector returns a Circuit with ops set."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        statevector = np.array([0.5, 0.3, 0.7, 0.1])
        circuit = prep.prepare_from_statevector(
            statevector=statevector,
            num_qubits=2,
            qubit_indices=[0, 1],
        )
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None
    
    def test_circuit_prepare_correct_state(self):
        """Test that the prepared circuit creates the correct state.

        The alias sampling circuit prepares:
          |0⟩ → Σ_ℓ √(p̃_ℓ) |ℓ⟩|garbage_ℓ⟩
        where p̃_ℓ ≈ |c_ℓ| / Σ|c_j| is the discretized probability.

        We verify the marginal probability on the index register matches the
        expected distribution within the discretization tolerance.
        """
        coefficients = [0.5, 0.3, 0.7, 0.1]
        num_index_qubits = 2
        bits_precision = 4

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        # Expected: p(ℓ) = |c_ℓ| / Σ|c_j|
        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        # Discretization tolerance: O(1 / 2^bits_precision) per coefficient
        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(marginal_probs[:len(coefficients)], expected_probs, atol=atol)

    def test_marginal_probs_uniform(self):
        """Verify alias sampling produces a uniform distribution for equal coefficients."""
        coefficients = [1.0, 1.0, 1.0, 1.0]
        num_index_qubits = 2
        bits_precision = 4

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        expected_probs = np.array([0.25, 0.25, 0.25, 0.25])

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(marginal_probs, expected_probs, atol=atol)

    def test_marginal_probs_eight_components(self):
        """Verify alias sampling marginal probabilities for an 8-component distribution."""
        coefficients = [0.5, 0.3, 0.7, 0.1, 0.4, 0.2, 0.6, 0.8]
        num_index_qubits = 3
        bits_precision = 6

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(marginal_probs, expected_probs, atol=atol)

    def test_marginal_probs_skewed(self):
        """Verify alias sampling with a heavily skewed distribution."""
        coefficients = [0.99, 0.005, 0.003, 0.002]
        num_index_qubits = 2
        bits_precision = 6

        full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bits_precision)
        marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(marginal_probs, expected_probs, atol=atol)

    def test_higher_precision_improves_fidelity(self):
        """Verify that increasing bits_precision improves the marginal probability accuracy."""
        coefficients = [0.5, 0.3, 0.7, 0.1]
        num_index_qubits = 2
        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        errors = []
        for bp in [3, 6]:
            full_sv = _run_alias_sampling_and_dump(coefficients, num_index_qubits, bp)
            marginal_probs = _compute_marginal_probs(full_sv, num_index_qubits)
            error = np.max(np.abs(marginal_probs[:len(coefficients)] - expected_probs))
            errors.append(error)

        # Higher precision should give smaller error
        assert errors[1] <= errors[0]