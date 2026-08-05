"""Tests for unary-iteration phase estimation with arbitrary query counts."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.unary_phase_estimation_builder import (
    QdkUnaryQpeCircuitBuilder,
    cosine_window_state,
    num_phase_bits,
)
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import (
    UnaryPhaseEstimation,
    _post_process_phase_estimation,
)
from qdk_chemistry.data import AlgorithmRef, QubitOperator
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _address_qubits(num_actions: int) -> int:
    """Number of address qubits the Q# operations allocate for ``num_actions`` values."""
    return int(np.ceil(np.log2(num_actions))) if num_actions > 1 else 0


def _dumped_address_index(address_value: int, num_address_qubits: int) -> int:
    """Map a little-endian address value onto its index in the dumped statevector."""
    if num_address_qubits == 0:
        return 0
    return int(format(address_value, f"0{num_address_qubits}b")[::-1], 2)

class TestUnaryIterationQsharp:
    """Statevector checks of the unary-iteration primitives against exact references."""

    @pytest.mark.parametrize(
        ("num_actions", "address_value"),
        [(n, a) for n in (1, 2, 3, 4, 5, 6, 7, 8, 11) for a in range(n)],
    )
    def test_selects_exactly_one_action_per_address(self, qdk_ctx, num_actions, address_value):
        """Address ``i`` must flip flag ``i`` and nothing else, for every valid address."""
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationOneHot(num_actions, address_value)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
        expected[1 << (num_actions - 1 - address_value)] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("num_actions", [2, 4, 8])
    def test_superposed_address_stays_coherent(self, qdk_ctx, num_actions):
        """A superposed address must produce sum_a |a>|onehot(a)> with no ancilla residue."""
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationSuperposedAddress(num_actions)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << (num_address_qubits + num_actions), dtype=complex)
        for address_value in range(num_actions):
            index = _dumped_address_index(address_value, num_address_qubits) << num_actions
            expected[index | (1 << (num_actions - 1 - address_value))] = 1.0 / np.sqrt(num_actions)
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("num_actions", "data"),
        [
            (2, [True, False]),
            (4, [True, False, False, True]),
            (8, [True, False, False, False, False, True, True, False]),
        ],
    )
    def test_exposed_control_is_an_equality_predicate(self, qdk_ctx, num_actions, data):
        """Phasing the exposed control must imprint exactly the flagged sign pattern."""
        num_address_qubits = _address_qubits(num_actions)
        qdk_ctx.code.QDKChemistry.Utils.UnaryIteration.TestUnaryIterationControlPhases(num_actions, data)
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << num_address_qubits, dtype=complex)
        for address_value in range(num_actions):
            sign = -1.0 if data[address_value] else 1.0
            expected[_dumped_address_index(address_value, num_address_qubits)] = sign / np.sqrt(num_actions)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestBlockEncodingAgnosticSchedule:
    """The signed-power schedule must work for any self-inverse block encoding."""

    @pytest.mark.parametrize(
        ("num_queries", "address_value"),
        [(p, t) for p in (1, 2, 3, 5) for t in range(p + 1)],
    )
    def test_psp_schedule_matches_the_explicit_walk_power(self, qdk_ctx, num_queries, address_value):
        """A PREPARE-SELECT-PREPARE walk must obey the same ``W^(p - 2t)`` contract."""
        psp = qdk_ctx.code.QDKChemistry.Utils.PrepSelPrep
        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestSignedPowerScheduleAgainstWalk(
            psp.MakeTestBlockEncodingOp(0.7), psp.MakeAncillaReflectionOp(1), num_queries, address_value, 2, 0.9
        )
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        num_address_qubits = _address_qubits(num_queries + 1)
        expected = np.zeros(1 << (num_address_qubits + 2), dtype=complex)
        expected[0] = np.cos(0.45)  # system |0>, ancilla |0>
        expected[2] = np.sin(0.45)  # system |1>, ancilla |0>
        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.parametrize("theta", [0.0, 1.3, np.pi / 2])
    def test_psp_schedule_holds_for_every_encoded_eigenvalue(self, qdk_ctx, theta):
        """The contract must not depend on what the block encoding encodes."""
        psp = qdk_ctx.code.QDKChemistry.Utils.PrepSelPrep
        qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestSignedPowerScheduleAgainstWalk(
            psp.MakeTestBlockEncodingOp(theta), psp.MakeAncillaReflectionOp(1), 3, 1, 2, 0.9
        )
        state = np.array(qdk_ctx.dump_machine().as_dense_state())

        expected = np.zeros(1 << 4, dtype=complex)
        expected[0] = np.cos(0.45)
        expected[2] = np.sin(0.45)
        np.testing.assert_allclose(state, expected, atol=1e-10)


class TestPhaseWindowState:
    """Window states prepared on the phase register."""

    @pytest.mark.parametrize("num_queries", [4, 9, 24])
    def test_cosine_is_symmetric_and_single_lobed(self, num_queries):
        """The cosine window peaks in the middle and decays monotonically to both edges."""
        amplitudes = np.array(cosine_window_state(num_queries))[: num_queries + 1]
        np.testing.assert_allclose(amplitudes, amplitudes[::-1], rtol=1e-12, atol=1e-15)
        peak = int(np.argmax(amplitudes))
        assert np.all(np.diff(amplitudes[: peak + 1]) > 0.0)
        assert np.all(np.diff(amplitudes[peak:]) < 0.0)

    def test_cosine_suppresses_spectral_leakage_relative_to_uniform(self):
        """The cosine window's phase spectrum has far lighter tails than a uniform one."""
        num_queries, oversampling = 31, 32
        windows = {
            "cosine": np.array(cosine_window_state(num_queries))[: num_queries + 1],
            "uniform": np.ones(num_queries + 1) / np.sqrt(num_queries + 1),
        }

        def tail_probability(window: str, bins: int) -> float:
            amplitudes = windows[window]
            spectrum = np.abs(np.fft.fft(amplitudes, amplitudes.size * oversampling)) ** 2
            spectrum /= spectrum.sum()
            offsets = np.arange(spectrum.size) - int(np.argmax(spectrum))
            distance = np.minimum(offsets % spectrum.size, (-offsets) % spectrum.size)
            return float(spectrum[distance > bins * oversampling].sum())

        assert tail_probability("cosine", 2) < 0.1 * tail_probability("uniform", 2)
        assert tail_probability("cosine", 4) < 1e-3


class TestPhaseDecoding:
    """Decoding of the doubled measured phase."""

    def test_dominant_phase_merges_conjugate_counts(self):
        """Conjugate bins are summed before the winner is selected."""
        counts = {"010": 3, "110": 3, "001": 5}  # 2/8 and 6/8 are conjugates, 1/8 is a separate bin
        phase_fraction, bitstring, measured = _post_process_phase_estimation(counts, 3, use_positive_sign=True)
        assert phase_fraction == pytest.approx(0.125)
        assert bitstring in {"010", "110"}
        assert measured in {0.25, 0.75}


class TestUnaryQpeEndToEnd:
    """End-to-end checks of ``MakeUnaryQPECircuit`` on a walk with an exact eigenphase."""

    @staticmethod
    def _measure(qdk_ctx, num_queries: int, k: int, system_state: int) -> int:
        """Run the synthetic-walk QPE circuit and decode the phase register."""
        num_states = num_queries + 1
        theta = -np.pi * k / num_states
        results = qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk(
            num_queries, theta, system_state
        )
        return int("".join("1" if str(bit) == "One" else "0" for bit in reversed(results)), 2)

    @pytest.mark.parametrize(
        ("num_queries", "k", "system_state"),
        [(m, k, s) for m in (3, 7) for k in range(m + 1) for s in (0, 1)],
    )
    def test_measured_bin_is_twice_the_walk_phase(self, qdk_ctx, num_queries, k, system_state):
        """The measured register must read exactly ``2*phi*N``."""
        num_states = num_queries + 1
        expected = (-k) % num_states if system_state == 1 else k
        assert self._measure(qdk_ctx, num_queries, k, system_state) == expected

    def test_slot_sweep_grows_linearly_with_the_query_count(self, qdk_ctx):
        """The schedule must sweep ``num_queries + 1`` slots, not apply one controlled block."""
        theta = -np.pi / 4
        counts = {
            num_queries: qdk_ctx.logical_counts(
                qdk_ctx.code.QDKChemistry.Utils.UnaryPhaseEstimation.TestUnaryQpeSyntheticWalk,
                num_queries,
                theta,
                1,
            )["measurementCount"]
            for num_queries in (3, 7, 15, 31)
        }

        queries = sorted(counts)
        assert [counts[num_queries] for num_queries in queries] == sorted(counts.values()), counts
        per_query = [counts[num_queries] / num_queries for num_queries in queries]
        assert min(per_query) > 0.75, counts
        assert max(per_query) / min(per_query) < 1.5, counts

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_decoder_recovers_the_walk_phase(self, qdk_ctx, k):
        """The measured bin, run through the decoder, returns the walk phase.

        The two eigenvectors carry conjugate phases ``+-phi`` and land in
        conjugate bins, which the decoder must fold onto the same answer.
        """
        num_queries = 7
        num_states = num_queries + 1
        num_bits = num_phase_bits(num_queries)

        expected_phase = k / (2 * num_states)
        for system_state in (0, 1):
            measured_bin = self._measure(qdk_ctx, num_queries, k, system_state)
            counts = {format(measured_bin, f"0{num_bits}b"): 1}
            phase_fraction, _, _ = _post_process_phase_estimation(counts, num_bits, use_positive_sign=True)
            assert phase_fraction == pytest.approx(expected_phase)


    def test_builder_defaults_recover_the_ground_state_energy(self):
        r"""The shipped defaults must recover :math:`H = (X + Z)/2` end to end."""
        hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=np.array([0.5, 0.5]))
        energies, vectors = np.linalg.eigh(hamiltonian.to_matrix())
        state_prep_params = {
            "rowMap": list(range(hamiltonian.num_qubits - 1, -1, -1)),
            "stateVector": np.real(vectors[:, 0]).tolist(),
            "expansionOps": [],
            "numQubits": hamiltonian.num_qubits,
        }
        state_preparation = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
                parameter=state_prep_params,
            ),
            qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params),
        )

        qpe = UnaryPhaseEstimation(shots=200)
        qpe.settings().set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary", num_queries=63))
        qpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
        result = qpe.run(qubit_hamiltonian=hamiltonian, state_preparation=state_preparation)

        assert result.raw_energy == pytest.approx(float(energies[0]), abs=1e-9)