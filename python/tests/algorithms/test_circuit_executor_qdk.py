"""Test QDK/Chemistry circuit executor with QDK full state and sparse state simulator."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms.circuit_executor.qdk import (
    QdkFullStateSimulator,
    QdkSparseStateSimulator,
)
from qdk_chemistry.data import Circuit, QuantumErrorProfile


@pytest.fixture
def test_circuit_1() -> Circuit:
    """Create a test circuit."""
    return Circuit(
        qasm="""
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        """,
    )


@pytest.fixture
def test_circuit_2() -> Circuit:
    """Create a test circuit with fixed bitstring outcomes."""
    return Circuit(
        qasm="""
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
        """,
    )


class TestQdkFullStateCircuitExecutor:
    """Test suite for QDK full state circuit executor."""

    def test_initialization(self):
        """Test initialization of the executor."""
        executor = QdkFullStateSimulator()
        assert executor.settings().get("seed") == 42
        assert executor.settings().get("type") == "cpu"
        executor.settings().update("type", "clifford")
        assert executor.settings().get("type") == "clifford"
        executor.settings().update("type", "gpu")
        assert executor.settings().get("type") == "gpu"

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        executor = QdkFullStateSimulator(simulator_type="clifford", seed=123)
        assert executor.settings().get("seed") == 123
        assert executor.settings().get("type") == "clifford"

    def test_executor_name(self):
        """Test executor name."""
        executor = QdkFullStateSimulator()
        assert executor.name() == "qdk_full_state_simulator"

    def test_circuit_executor_qdk_full_state(self, test_circuit_1: Circuit, test_circuit_2: Circuit):
        """Test the QDK full state circuit executor."""
        executor = QdkFullStateSimulator()

        # Test circuit 1, which will return "00" and "11" outcomes
        result_1 = executor.run(test_circuit_1, shots=10)
        counts_1 = result_1.bitstring_counts
        assert counts_1.get("00", 0) > 0
        assert counts_1.get("11", 0) > 0
        assert counts_1.get("01", 0) == 0
        assert counts_1.get("10", 0) == 0

        # Test circuit 2, which will always return "10" outcome
        result_2 = executor.run(test_circuit_2, shots=10)
        counts_2 = result_2.bitstring_counts
        raw_data = result_2.get_executor_metadata()
        assert counts_2.get("01", 0) == 10  # "10" in qubit order is "01" in bitstring order
        assert counts_2.get("00", 0) == 0
        assert counts_2.get("10", 0) == 0
        assert counts_2.get("11", 0) == 0
        assert len(raw_data) == 10
        for outcome in raw_data:
            assert "One, Zero" in str(outcome)  # Raw data bitstring outcomes should be in qubit order

    def test_noise_with_depolarizing_error(self, test_circuit_1, simple_error_profile):
        """Test execution with a depolarizing noise model."""
        executor = QdkFullStateSimulator()
        result = executor.run(test_circuit_1, shots=1000, noise=simple_error_profile)
        counts = result.bitstring_counts
        # With noise, we expect some erroneous outcomes ("01" or "10") in addition to "00" and "11"
        total = sum(counts.values())
        assert total == 1000
        error_count = counts.get("01", 0) + counts.get("10", 0)
        assert error_count > 0, "Noise should introduce some erroneous outcomes"

    def test_noise_high_error_rate(self, test_circuit_2):
        """Test execution with a high noise rate produces many errors."""
        high_noise = QuantumErrorProfile(
            name="high_noise",
            description="High noise for testing",
            errors={
                "x": {"type": "depolarizing_error", "rate": 0.5, "num_qubits": 1},
                "h": {"type": "depolarizing_error", "rate": 0.5, "num_qubits": 1},
                "cx": {"type": "depolarizing_error", "rate": 0.5, "num_qubits": 2},
            },
        )
        executor = QdkFullStateSimulator()
        result = executor.run(test_circuit_2, shots=1000, noise=high_noise)
        counts = result.bitstring_counts
        assert sum(counts.values()) == 1000
        # High noise should produce multiple unique outcomes
        assert len(counts) > 1


class TestQdkSparseStateCircuitExecutor:
    """Test suite for QDK sparse state circuit executor."""

    def test_initialization(self):
        """Test initialization of the sparse state executor."""
        executor = QdkSparseStateSimulator()
        assert executor.settings().get("seed") == 42

    def test_executor_name(self):
        """Test executor name."""
        executor = QdkSparseStateSimulator()
        assert executor.name() == "qdk_sparse_state_simulator"

    def test_deterministic_circuit(self, test_circuit_2):
        """Test the sparse state executor with a deterministic circuit."""
        executor = QdkSparseStateSimulator()
        result = executor.run(test_circuit_2, shots=10)
        counts = result.bitstring_counts
        assert sum(counts.values()) == 10
        assert result.total_shots == 10
        assert result.executor == "qdk_sparse_state_simulator"
        # x q[0] -> measure gives "01" in little-endian bitstring order
        assert len(counts) == 1

    def test_bell_state_circuit(self, test_circuit_1):
        """Test the sparse state executor with a Bell state circuit."""
        executor = QdkSparseStateSimulator()
        result = executor.run(test_circuit_1, shots=100)
        counts = result.bitstring_counts
        assert sum(counts.values()) == 100
        # Bell state should produce only correlated outcomes
        for bs in counts:
            assert bs in ("00", "11")

    def test_executor_metadata_populated(self, test_circuit_2):
        """Test that executor metadata is populated."""
        executor = QdkSparseStateSimulator()
        result = executor.run(test_circuit_2, shots=5)
        assert result.executor == "qdk_sparse_state_simulator"
        assert result.get_executor_metadata() is not None
        assert all("10" in str(outcome) for outcome in result.get_executor_metadata())

    def test_gate_noise_profile_raises(self, test_circuit_1, simple_error_profile):
        """Test that passing a QuantumErrorProfile noise raises NotImplementedError."""
        executor = QdkSparseStateSimulator()
        with pytest.raises(NotImplementedError, match="Gate specific noise is not yet supported"):
            executor.run(test_circuit_1, shots=10, noise=simple_error_profile)
