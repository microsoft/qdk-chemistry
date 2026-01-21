"""Test QDK/Chemistry circuit executor with QDK full state simulator."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms.circuit_executor.qdk import QdkFullStateSimulator
from qdk_chemistry.data import Circuit


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
            assert "Zero, One" in str(outcome)
