"""Test for circuit executor in QDK/Chemistry Azure Quantum plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os

import pytest

from qdk_chemistry.data import Circuit, QuantumErrorProfile
from qdk_chemistry.plugins.azure_quantum import QDK_CHEMISTRY_HAS_AZURE_QUANTUM

if QDK_CHEMISTRY_HAS_AZURE_QUANTUM:
    from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumEmulator


pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_AZURE_QUANTUM, reason="azure-quantum not available")

_SUBSCRIPTION_ID = os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", "")
_RESOURCE_GROUP = os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", "")
_WORKSPACE_NAME = os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", "")
_LOCATION = os.environ.get("AZURE_QUANTUM_LOCATION", "")
_TARGET_NAME = os.environ.get("AZURE_QUANTUM_TARGET_NAME", "")
_HAS_WORKSPACE_CONFIG = all([_SUBSCRIPTION_ID, _RESOURCE_GROUP, _WORKSPACE_NAME, _LOCATION, _TARGET_NAME])


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


class TestAzureQuantumEmulatorCircuitExecutor:
    """Test suite for Azure Quantum Emulator circuit executor."""

    def test_initialization(self):
        """Test initialization of the executor."""
        executor = AzureQuantumEmulator()
        assert executor.settings().get("seed") == 42
        assert executor.settings().get("simulation_type") == "cliffordrounding"

    @pytest.mark.skipif(not _HAS_WORKSPACE_CONFIG, reason="Azure Quantum workspace env vars not set")
    def test_circuit_executor_azure_quantum(self, test_circuit_1: Circuit, test_circuit_2: Circuit):
        """Test the Azure Quantum Emulator circuit executor."""
        executor = AzureQuantumEmulator(
            subscription_id=_SUBSCRIPTION_ID,
            resource_group=_RESOURCE_GROUP,
            workspace_name=_WORKSPACE_NAME,
            location=_LOCATION,
            target_name=_TARGET_NAME,
            seed=-1,
        )

        # Test circuit 1 (Bell state: H+CX, all Clifford) — should return "00" and "11" outcomes
        result_1 = executor.run(test_circuit_1, shots=100)
        counts_1 = result_1.bitstring_counts
        assert counts_1.get("00", 0) > 0
        assert counts_1.get("11", 0) > 0
        assert counts_1.get("01", 0) + counts_1.get("10", 0) == 0

        # Test circuit 2, which will always return "10" outcome
        result_2 = executor.run(test_circuit_2, shots=100)
        counts_2 = result_2.bitstring_counts
        total_counts_2 = sum(counts_2.values())
        assert total_counts_2 == 100

    def test_circuit_executor_with_error_profile(
        self, test_circuit_1: Circuit, simple_error_profile: QuantumErrorProfile
    ):
        """Test that passing a noise profile raises NotImplementedError."""
        executor = AzureQuantumEmulator()
        with pytest.raises(NotImplementedError, match="Custom noise profiles are not yet supported"):
            executor.run(test_circuit_1, shots=10, noise=simple_error_profile)
