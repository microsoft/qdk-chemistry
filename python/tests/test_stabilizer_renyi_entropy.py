"""Independent mathematical and wavefunction tests for the real 2-SRE backend."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from itertools import combinations, product
from math import fsum, log2
from unittest.mock import MagicMock

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import ReplicaStabilizerRenyiEntropy
from qdk_chemistry.algorithms import stabilizer_renyi_entropy as sre
from qdk_chemistry.data import (
    AmplitudeContainer,
    AmplitudeType,
    Configuration,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.utils import Logger
from tests.helpers import (
    create_dense_wavefunction,
    create_h2_molecule,
    create_sparse_wavefunction,
    create_test_orbitals,
)


def _pauli_entropy(state: np.ndarray) -> float:
    """Enumerate all phase-free Paulis, including identity, for small tests only."""
    paulis = (
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    )
    n = state.size.bit_length() - 1
    moments = []
    for factors in product(paulis, repeat=n):
        operator = np.ones((1, 1))
        for factor in factors:
            operator = np.kron(operator, factor)
        moments.append(float(abs(np.vdot(state, operator @ state)) ** 4))
    return -log2(fsum(moments) / state.size)


def _ordered_replica_moment(state: np.ndarray) -> float:
    """Sum every ordered quadruple without index permutation optimizations."""
    amplitudes = {x: float(c) for x, c in enumerate(state) if c != 0}
    return fsum(
        amplitudes[x]
        * amplitudes[y]
        * amplitudes[z]
        * amplitudes[w]
        * amplitudes.get(x ^ y ^ z, 0.0)
        * amplitudes.get(x ^ y ^ w, 0.0)
        * amplitudes.get(x ^ z ^ w, 0.0)
        * amplitudes.get(y ^ z ^ w, 0.0)
        for x, y, z, w in product(amplitudes, repeat=4)
    )


def _bitwise_pauli_entropy(state: np.ndarray) -> float:
    """Evaluate every Pauli expectation with an independent Walsh transform."""
    indices = np.arange(state.size)
    moments = []
    for flip in range(state.size):
        overlaps = state * state[indices ^ flip]
        width = 1
        while width < state.size:
            blocks = overlaps.reshape(-1, 2 * width)
            left = blocks[:, :width].copy()
            right = blocks[:, width:].copy()
            blocks[:, :width] = left + right
            blocks[:, width:] = left - right
            width *= 2
        moments.append(float(np.sum(overlaps**4)))
    return -log2(fsum(moments) / state.size)


def _calculate(state, *, truncate=False, cutoff=1e-12):
    algorithm = ReplicaStabilizerRenyiEntropy()
    algorithm.settings().set("truncate", truncate)
    algorithm.settings().set("coefficient_cutoff", cutoff)
    return algorithm.run(create_dense_wavefunction(state))


@pytest.mark.parametrize(
    "state",
    [
        [1.0, 0.0],
        [0.0, -1.0],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        [0.5, 0.5, 0.5, -0.5],
        np.ones(8) / np.sqrt(8),
    ],
)
def test_stabilizer_states(state):
    assert _calculate(state).entropy == pytest.approx(0.0, abs=1e-12)
    assert _pauli_entropy(np.asarray(state)) == pytest.approx(0.0, abs=1e-12)


def test_real_magic_state():
    state = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])
    result = _calculate(state)
    assert result.entropy == pytest.approx(log2(4 / 3), abs=1e-12)
    assert result.alpha == result.log_base == 2
    assert result.encoding == "jordan-wigner"
    assert result.fermion_mode_order == "blocked"


@pytest.mark.parametrize("truncate", [False, True])
def test_calculation_and_result_do_not_trace_function_entry(monkeypatch, truncate):
    state = np.array([0.6, -0.8])
    wavefunction = create_dense_wavefunction(state)
    algorithm = algorithms.create("stabilizer_renyi_entropy", truncate=truncate, coefficient_cutoff=0.7)
    expected = 0.0 if truncate else _pauli_entropy(state)
    trace_entering = MagicMock()
    monkeypatch.setattr(Logger, "trace_entering", trace_entering)

    result = algorithm.run(wavefunction)
    restored = type(result).from_json(result.to_json())

    trace_entering.assert_not_called()
    assert result.entropy == pytest.approx(expected, abs=1e-12)
    assert restored.content_hash() == result.content_hash()


@pytest.mark.parametrize("seed", [7, 29, 91])
@pytest.mark.parametrize("support_size", [1, 2, 3, 4, 5, 8, 11, 16])
@pytest.mark.parametrize("batched", [False, True])
def test_three_independent_real_state_calculations(seed, support_size, batched, monkeypatch):
    if batched:
        monkeypatch.setattr(sre, "_SCALAR_CONFIGURATION_LIMIT", 0)
    rng = np.random.default_rng(seed)
    indices = rng.choice(16, support_size, replace=False)
    coefficients = rng.standard_normal(support_size)
    coefficients /= np.linalg.norm(coefficients)
    state = np.zeros(16)
    state[indices] = coefficients
    expected = _pauli_entropy(state)
    assert -log2(_ordered_replica_moment(state)) == pytest.approx(expected, abs=1e-12)
    assert _calculate(state).entropy == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", [42, 91])
@pytest.mark.parametrize("num_configurations", [17, 24, 50, 100, 256])
def test_batched_replica_against_all_bitwise_paulis(seed, num_configurations):
    rng = np.random.default_rng(seed)
    labels = rng.choice(256, num_configurations, replace=False)
    state = np.zeros(256)
    state[labels] = rng.standard_normal(num_configurations)
    state /= np.linalg.norm(state)
    assert _calculate(state).entropy == pytest.approx(_bitwise_pauli_entropy(state), abs=1e-12)


@pytest.mark.parametrize("signs", [False, True])
def test_batched_replica_dense_stabilizer(signs):
    state = np.ones(256) / 16
    if signs:
        state *= np.array([-1 if (x & 3) == 3 else 1 for x in range(256)])
    assert _calculate(state).entropy == pytest.approx(0.0, abs=1e-12)


def test_batched_replica_fixed_particle_number():
    sector = [sum(1 << i for i in occupied) for occupied in combinations(range(4), 2)]
    labels = [a | (b << 4) for a in sector for b in sector]
    state = np.zeros(256)
    state[labels] = np.random.default_rng(42).standard_normal(len(labels))
    state /= np.linalg.norm(state)
    assert _calculate(state).entropy == pytest.approx(_bitwise_pauli_entropy(state), abs=1e-12)


@pytest.mark.parametrize("positions", [[0, 1, 2, 3, 63], [0, 7, 63, 64, 129]])
def test_batched_replica_arbitrary_width_labels_and_input_order(positions):
    state = np.random.default_rng(42).standard_normal(32)
    state /= np.linalg.norm(state)
    labels = [sum(((x >> i) & 1) << bit for i, bit in enumerate(positions)) for x in range(32)]
    wavefunction = create_sparse_wavefunction(max(positions) + 1, labels, state)
    original_hash = wavefunction.content_hash()
    result = ReplicaStabilizerRenyiEntropy().run(wavefunction)
    assert result.num_qubits == max(positions) + 1
    assert result.entropy == pytest.approx(_bitwise_pauli_entropy(state), abs=1e-12)
    permuted = create_sparse_wavefunction(max(positions) + 1, labels[::-1], state[::-1])
    assert ReplicaStabilizerRenyiEntropy().run(permuted).to_json() == result.to_json()
    np.testing.assert_array_equal(wavefunction.get_coefficients(), state)
    assert wavefunction.content_hash() == original_hash


def test_batched_replica_without_distinct_xor_partners():
    labels = [1 << i for i in range(24)]
    coefficients = np.random.default_rng(42).standard_normal(len(labels))
    coefficients /= np.linalg.norm(coefficients)
    fourth = [float(c) ** 4 for c in coefficients]
    expected = fsum((14 * fsum(a * b for a, b in combinations(fourth, 2)), fsum(c * c for c in fourth)))
    assert sre._replica_moment(labels, coefficients) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("gate", ["h", "cnot", "swap"])
def test_real_clifford_invariance(gate):
    state = np.random.default_rng(42).standard_normal(8)
    state /= np.linalg.norm(state)
    transformed = np.empty_like(state)
    if gate == "h":
        transformed[0::2] = (state[0::2] + state[1::2]) / np.sqrt(2)
        transformed[1::2] = (state[0::2] - state[1::2]) / np.sqrt(2)
    else:
        for x, coefficient in enumerate(state):
            target = x ^ ((x & 1) << 2) if gate == "cnot" else x ^ (((x & 1) ^ ((x >> 2) & 1)) * 5)
            transformed[target] = coefficient
    assert _calculate(transformed).entropy == pytest.approx(_calculate(state).entropy, abs=1e-12)


def test_global_sign_and_tensor_additivity():
    first = np.array([np.cos(0.21), -np.sin(0.21)])
    second = np.array([np.cos(0.37), np.sin(0.37)])
    assert _calculate(-first).entropy == pytest.approx(_calculate(first).entropy, abs=1e-12)
    assert _calculate(np.kron(first, second)).entropy == pytest.approx(
        _calculate(first).entropy + _calculate(second).entropy, abs=1e-12
    )


def test_sparse_input_order_and_repeated_calls():
    labels = [1, 4, 6, 9, 15]
    coefficients = np.array([1.0, -2.0, 3.0, -4.0, 5.0]) / np.sqrt(55)
    algorithm = ReplicaStabilizerRenyiEntropy()
    first = create_sparse_wavefunction(4, labels, coefficients)
    permuted = create_sparse_wavefunction(4, labels[::-1], coefficients[::-1])
    expected = algorithm.run(first)
    algorithm.run(create_dense_wavefunction([1.0, 0.0]))
    assert algorithm.run(permuted).to_json() == expected.to_json()
    assert algorithm.run(first).content_hash() == expected.content_hash()


def test_blocked_jordan_wigner_labels():
    orbitals = create_test_orbitals(2)
    determinants = [Configuration.from_spin_half_string("20"), Configuration.from_spin_half_string("02")]
    wavefunction = Wavefunction(StateVectorContainer(np.array([0.6, -0.8]), determinants, orbitals, "electrons"))
    n, labels, coefficients = sre._sparse_coefficients(wavefunction)
    assert n == 4
    assert labels == [5, 10]
    np.testing.assert_array_equal(coefficients, [0.6, -0.8])
    state = np.zeros(16)
    state[[5, 10]] = coefficients
    assert ReplicaStabilizerRenyiEntropy().run(wavefunction).entropy == pytest.approx(_pauli_entropy(state), abs=1e-12)


@pytest.mark.parametrize("labels", [[0, 1], [0, 1 << 69]])
def test_register_width_and_labels_above_int64(labels):
    wavefunction = create_sparse_wavefunction(70, labels, [np.cos(np.pi / 8), np.sin(np.pi / 8)])
    n, extracted, _ = sre._sparse_coefficients(wavefunction)
    assert n == 70
    assert extracted == labels
    result = ReplicaStabilizerRenyiEntropy().run(wavefunction)
    assert result.num_qubits == 70
    assert result.entropy == pytest.approx(log2(4 / 3), abs=1e-12)


def test_truncation_disabled_preserves_every_nonzero_coefficient():
    state = [1.0, 1e-13, -2e-13, 0.0]
    result = _calculate(state, cutoff=0.9)
    assert not result.truncation_enabled
    assert result.coefficient_cutoff is None
    assert result.num_configurations == result.num_retained_configurations == 3
    assert result.discarded_weight == 0


def test_default_cutoff_and_tiny_discarded_probability():
    wavefunction = create_dense_wavefunction([1.0, 1e-13, -2e-13, 1e-10])
    original = wavefunction.get_coefficients().copy()
    original_hash = wavefunction.content_hash()
    algorithm = ReplicaStabilizerRenyiEntropy()
    assert algorithm.settings().get("coefficient_cutoff") == 1e-12
    algorithm.settings().set("truncate", True)
    result = algorithm.run(wavefunction)
    assert result.truncation_enabled
    assert result.coefficient_cutoff == 1e-12
    assert result.num_configurations == 4
    assert result.num_retained_configurations == 2
    assert result.discarded_weight == pytest.approx(5e-26, rel=1e-12, abs=0)
    np.testing.assert_array_equal(wavefunction.get_coefficients(), original)
    assert wavefunction.content_hash() == original_hash


def test_custom_cutoff_renormalizes_and_records_weight():
    state = np.array([np.sqrt(0.75), 0.2, -0.4, np.sqrt(0.05)])
    result = _calculate(state, truncate=True, cutoff=0.2)
    retained = state.copy()
    retained[np.abs(retained) <= 0.2] = 0
    retained /= np.linalg.norm(retained)
    assert result.num_configurations == 4
    assert result.num_retained_configurations == 3
    assert result.discarded_weight == pytest.approx(0.04, abs=1e-15)
    assert result.entropy == pytest.approx(_pauli_entropy(retained), abs=1e-12)


def test_strict_cutoff_boundary():
    result = _calculate([np.sqrt(0.5), 0.5, -0.5, 0.0], truncate=True, cutoff=0.5)
    assert result.num_configurations == 3
    assert result.num_retained_configurations == 1
    assert result.discarded_weight == pytest.approx(0.5, abs=1e-15)
    assert result.entropy == 0


def test_zero_cutoff_and_no_discarded_support():
    state = [0.6, -0.8, 0.0, 0.0]
    result = _calculate(state, truncate=True, cutoff=0.0)
    assert result.num_configurations == result.num_retained_configurations == 2
    assert result.discarded_weight == 0
    assert result.entropy == pytest.approx(_calculate(state).entropy, abs=1e-12)


@pytest.mark.parametrize("cutoff", [-1.0, float("nan"), float("inf")])
def test_invalid_cutoff_only_applies_when_enabled(cutoff):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _calculate([1.0, 0.0], truncate=True, cutoff=cutoff)
    assert _calculate([1.0, 0.0], cutoff=cutoff).entropy == 0


def test_all_coefficients_discarded():
    with pytest.raises(ValueError, match="removed every"):
        _calculate([0.6, -0.8], truncate=True, cutoff=0.8)


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("state", [[0.0, 0.0], [0.6, 0.7], [np.sqrt(1 + 2e-10), 0.0], [1e200, 0.0]])
def test_unnormalized_input_rejected_before_truncation(state, truncate):
    with pytest.raises(ValueError, match="must be normalized"):
        _calculate(state, truncate=truncate, cutoff=2.0)


def test_roundoff_normalization_is_bounded_and_does_not_mutate():
    state = [np.sqrt(1 + 0.5e-10), 0.0]
    wavefunction = create_dense_wavefunction(state)
    assert ReplicaStabilizerRenyiEntropy().run(wavefunction).entropy == 0
    np.testing.assert_array_equal(wavefunction.get_coefficients(), state)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficients(value):
    with pytest.raises(ValueError, match="must be finite"):
        _calculate([1.0, value])


def test_empty_expansion():
    wavefunction = Wavefunction(StateVectorContainer(np.array([]), [], ModelOrbitals(1)))
    with pytest.raises(ValueError, match="must not be empty"):
        ReplicaStabilizerRenyiEntropy().run(wavefunction)


def test_duplicate_configurations_rejected():
    wavefunction = create_sparse_wavefunction(1, [0, 0], [0.6, 0.8])
    with pytest.raises(ValueError, match="duplicate configurations"):
        ReplicaStabilizerRenyiEntropy().run(wavefunction)


@pytest.mark.parametrize("imaginary", [0.5, 1e-30])
def test_genuinely_complex_coefficients_rejected(imaginary):
    real = create_dense_wavefunction([1.0, 0.0])
    coefficients = np.array([1.0, imaginary * 1j])
    coefficients /= np.linalg.norm(coefficients)
    wavefunction = Wavefunction(StateVectorContainer(coefficients, real.get_active_determinants(), real.get_orbitals()))
    with pytest.raises(ValueError, match="only real coefficients"):
        ReplicaStabilizerRenyiEntropy().run(wavefunction)


def test_complex_storage_with_real_values():
    real = create_dense_wavefunction([0.6, -0.8])
    coefficients = np.array([0.6, -0.8], dtype=complex)
    wavefunction = Wavefunction(StateVectorContainer(coefficients, real.get_active_determinants(), real.get_orbitals()))
    assert ReplicaStabilizerRenyiEntropy().run(wavefunction).entropy == pytest.approx(_calculate([0.6, -0.8]).entropy)


def test_amplitude_container_rejected():
    orbitals = create_test_orbitals(2)
    reference = Wavefunction(StateVectorContainer(Configuration.from_spin_half_string("20"), orbitals, "electrons"))
    amplitudes = Wavefunction(AmplitudeContainer(orbitals, reference, AmplitudeType.CoupledCluster))
    with pytest.raises(TypeError, match="coefficient expansion"):
        ReplicaStabilizerRenyiEntropy().run(amplitudes)


def test_non_wavefunction_rejected():
    with pytest.raises(TypeError, match="must be a Wavefunction"):
        ReplicaStabilizerRenyiEntropy().run([1.0, 0.0])


@pytest.mark.parametrize("moment", [0.0, -0.1, np.nan, np.inf, 1.01, 2.0**-10])
def test_invalid_replica_results_raise(monkeypatch, moment):
    monkeypatch.setattr(sre, "_replica_moment", lambda *_args: moment)
    with pytest.raises(RuntimeError, match="replica calculation produced"):
        _calculate([1.0, 0.0])


def test_bounded_moment_roundoff(monkeypatch):
    monkeypatch.setattr(sre, "_replica_moment", lambda *_args: 1 + 0.5e-12)
    assert _calculate([1.0, 0.0]).entropy == 0


def test_h2_ground_state_against_all_paulis():
    _, mean_field = algorithms.create("scf_solver").run(create_h2_molecule(), 0, 1, "sto-3g")
    hamiltonian = algorithms.create("hamiltonian_constructor").run(mean_field.get_orbitals())
    _, ground_state = algorithms.create("multi_configuration_calculator", "macis_cas").run(hamiltonian, 1, 1)
    state = np.zeros(16)
    for determinant, coefficient in zip(
        ground_state.get_active_determinants(), ground_state.get_coefficients(), strict=True
    ):
        index = int("".join(str(bit) for bit in reversed(determinant.to_bits(4))), 2)
        state[index] = coefficient
    result = algorithms.create("stabilizer_renyi_entropy", "replica").run(ground_state)
    assert result.num_qubits == 4
    assert result.entropy == pytest.approx(_pauli_entropy(state), abs=1e-10)
    assert result.entropy > 0
