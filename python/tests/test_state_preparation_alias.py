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
          |0> -> Sum_l sqrt(p_l) |l>|garbage_l>
        where p_l ~ |c_l| / Sum|c_j| is the discretized probability.

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


def _run_conditional_alias_fr_and_dump(
    coefficients: list[list[float]],
    free_rider_data: list[list[bool]],
    bits_precision: int,
    condition_value: int,
) -> np.ndarray:
    """Run conditional alias sampling with free-rider and return statevector."""
    qdk.init(project_root=_PROJECT_ROOT)
    qdk.code.QDKChemistry.Utils.AliasSampling.RunConditionalAliasSamplingPrepWithFreeRider(
        coefficients, free_rider_data, bits_precision, condition_value
    )
    state = qdk.dump_machine()
    return np.array(state.as_dense_state())


def _compute_conditional_marginal_probs(
    full_sv: np.ndarray,
    n_cond_bits: int,
    n_index_bits: int,
    condition_value: int,
) -> np.ndarray:
    """Compute marginal probabilities on the index register for a given condition.

    Register layout (LE): conditionalReg[nCond] + indexReg[nIdx] + ancilla.
    dump_machine uses BE: qubit 0 = MSB.
    """
    total_qubits = int(np.log2(len(full_sv)))
    n_index = 2**n_index_bits
    probs = np.zeros(n_index)

    for i in range(len(full_sv)):
        amp = full_sv[i]
        if abs(amp) < 1e-15:
            continue
        bits = format(i, f"0{total_qubits}b")
        cond_be = bits[:n_cond_bits]
        cond_val = int(cond_be[::-1], 2)  # reverse for LE
        if cond_val != condition_value:
            continue
        idx_be = bits[n_cond_bits : n_cond_bits + n_index_bits]
        idx_val = int(idx_be[::-1], 2)  # reverse for LE
        probs[idx_val] += abs(amp) ** 2

    return probs


class TestConditionalAliasSamplingWithFreeRider:
    """Tests for conditional alias sampling with free-rider data."""

    @pytest.mark.parametrize(
        ("n_cond", "n_coeffs", "condition_value"),
        [
            (2, 4, 0),
            (2, 4, 1),
        ],
    )
    def test_marginal_probs_with_free_rider(self, n_cond, n_coeffs, condition_value):
        """Verify marginal probs and free-rider data loading."""
        rng = np.random.default_rng(seed=456 + n_cond * 10 + condition_value)
        coefficients = rng.uniform(-1.0, 1.0, size=(n_cond, n_coeffs)).tolist()
        n_fr_bits = 3
        free_rider_data = [[bool(rng.integers(0, 2)) for _ in range(n_fr_bits)] for _ in range(n_cond)]
        bits_precision = 6
        n_index_bits = math.ceil(math.log2(n_coeffs))
        n_cond_bits = math.ceil(math.log2(n_cond))

        full_sv = _run_conditional_alias_fr_and_dump(coefficients, free_rider_data, bits_precision, condition_value)
        marginal_probs = _compute_conditional_marginal_probs(full_sv, n_cond_bits, n_index_bits, condition_value)

        abs_coeffs = np.abs(coefficients[condition_value])
        expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

        atol = 2.0 / (2**bits_precision)
        np.testing.assert_allclose(
            marginal_probs[:n_coeffs],
            expected_probs,
            atol=atol,
            err_msg=f"cond={condition_value}, free_rider={free_rider_data[condition_value]}",
        )
