"""Tests for the ResourceEstimator algorithm in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import pytest
import qsharp

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.resource_estimator.base import ResourceEstimator
from qdk_chemistry.algorithms.resource_estimator.qdk import QdkQreV1
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.resource_estimator_data import (
    CircuitCounts,
    ErrorBudget,
    EstimationConfig,
    LogicalCounts,
    LogicalQubit,
    PhysicalCounts,
    ResourceEstimatorData,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


class TestResourceEstimatorRegistry:
    """Test that the ResourceEstimator is properly registered."""

    def test_create_default_resource_estimator(self):
        """Test creating the default resource estimator via the registry."""
        estimator = create("resource_estimator")
        assert isinstance(estimator, ResourceEstimator)
        assert isinstance(estimator, QdkQreV1)

    def test_create_by_name(self):
        """Test creating a resource estimator by explicit name."""
        estimator = create("resource_estimator", "qdk_qre_v1")
        assert isinstance(estimator, QdkQreV1)

    def test_algorithm_name(self):
        """Test the algorithm name property."""
        estimator = QdkQreV1()
        assert estimator.name() == "qdk_qre_v1"

    def test_algorithm_type_name(self):
        """Test the algorithm type name property."""
        estimator = QdkQreV1()
        assert estimator.type_name() == "resource_estimator"

    def test_default_settings(self):
        """Test that default settings are applied."""
        estimator = QdkQreV1()
        assert estimator.settings().get("error_budget") == 0.001


class TestQdkQreV1:
    """Test cases for QdkQreV1 execution."""

    def test_estimate_from_factory(self):
        """Test resource estimation with Q# factory data."""
        state_prep_params = {
            "rowMap": [1, 0],
            "stateVector": [0.6, 0.0, 0.0, 0.8],
            "expansionOps": [],
            "numQubits": 2,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        circuit = Circuit(qsharp_factory=qsharp_factory)
        estimator = QdkQreV1()
        result = estimator.run(circuit)
        assert isinstance(result, ResourceEstimatorData)
        assert result.estimator == "qdk_qre_v1"
        assert result.logical_counts is not None
        assert result.logical_counts.num_qubits >= 0
        assert result.physical_counts.physical_qubits >= 0

    def test_estimate_from_qasm(self):
        """Test resource estimation with QASM representation."""
        qasm_with_t = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            h q[0];
            t q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        """
        circuit = Circuit(qasm=qasm_with_t)
        estimator = QdkQreV1()
        result = estimator.run(circuit)
        assert isinstance(result, ResourceEstimatorData)
        assert result.logical_counts.t_count >= 0
        assert result.status == "success"
        # New fields populated by QdkQreV1
        assert result.physical_counts.algorithm_qubits > 0
        assert result.physical_counts.factory_qubits >= 0
        assert result.physical_counts.runtime_unit == "ns"
        assert result.error > 0.0
        assert result.config is not None
        assert result.config.qec_scheme == "surface_code"
        assert result.config.qubit_model != ""

    def test_estimate_raises_with_qir_only(self):
        """Test that estimation raises when only QIR representation is available."""
        qir = qsharp.openqasm.compile("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        """)
        circuit = Circuit(qir=qir)
        estimator = QdkQreV1()
        with pytest.raises(RuntimeError, match="Cannot estimate resources"):
            estimator.run(circuit)

    def test_estimate_with_custom_error_budget(self):
        """Test resource estimation with custom error budget setting."""
        qasm_with_t = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            h q[0];
            t q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        """
        circuit = Circuit(qasm=qasm_with_t)
        estimator = QdkQreV1()
        estimator.settings().set("error_budget", 0.01)
        result = estimator.run(circuit)
        assert isinstance(result, ResourceEstimatorData)


class TestResourceEstimatorData:
    """Test cases for ResourceEstimatorData DataClass."""

    def _make_sample_data(self) -> ResourceEstimatorData:
        return ResourceEstimatorData(
            logical_counts=LogicalCounts(
                num_qubits=4, t_count=10, rotation_count=0,
                rotation_depth=0, ccz_count=0, ccix_count=0, measurement_count=2,
            ),
            physical_counts=PhysicalCounts(
                physical_qubits=1000, runtime=500, runtime_unit="ns",
                rqops=100, algorithm_qubits=600, factory_qubits=400,
                algorithmic_logical_depth=3, logical_depth=10,
            ),
            logical_qubit=LogicalQubit(
                code_distance=7, logical_cycle_time=2800,
                logical_error_rate=3e-6, physical_qubits=98,
            ),
            error_budget=ErrorBudget(
                logical=0.0005, rotations=0.0, tstates=0.0005,
            ),
            estimator="qdk_qre_v1",
            status="success",
            error=0.001,
            circuit_counts=CircuitCounts(
                depth=5, num_gates=8,
                num_single_qubit_clifford=3, num_two_qubit_clifford=2,
                num_non_clifford=3,
            ),
            config=EstimationConfig(
                qubit_model="qubit_gate_ns_e3",
                qec_scheme="surface_code",
                error_budget=0.001,
            ),
        )

    def test_properties(self):
        """Test data class properties."""
        data = self._make_sample_data()
        assert data.status == "success"
        assert data.error == 0.001
        assert data.logical_counts.num_qubits == 4
        assert data.logical_counts.t_count == 10
        assert data.physical_counts.physical_qubits == 1000
        assert data.physical_counts.runtime == 500
        assert data.physical_counts.runtime_unit == "ns"
        assert data.physical_counts.algorithm_qubits == 600
        assert data.physical_counts.factory_qubits == 400
        assert data.physical_counts.algorithmic_logical_depth == 3
        assert data.physical_counts.logical_depth == 10
        assert data.logical_qubit.code_distance == 7
        assert data.error_budget.logical == 0.0005
        assert data.estimator == "qdk_qre_v1"
        assert data.circuit_counts is not None
        assert data.circuit_counts.depth == 5
        assert data.circuit_counts.num_non_clifford == 3
        assert data.config is not None
        assert data.config.qec_scheme == "surface_code"

    def test_immutability(self):
        """Test that the data class is immutable after init."""
        data = self._make_sample_data()
        with pytest.raises(AttributeError):
            data.estimator = "changed"

    def test_get_summary(self):
        """Test human-readable summary."""
        data = self._make_sample_data()
        summary = data.get_summary()
        assert "Resource Estimator Data" in summary
        assert "qdk_qre_v1" in summary
        assert "success" in summary
        assert "4" in summary   # logical qubits
        assert "10" in summary  # T-count

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization roundtrip."""
        original = self._make_sample_data()
        json_data = original.to_json()
        restored = ResourceEstimatorData.from_json(json_data)
        assert restored.logical_counts == original.logical_counts
        assert restored.physical_counts == original.physical_counts
        assert restored.logical_qubit == original.logical_qubit
        assert restored.error_budget == original.error_budget
        assert restored.estimator == original.estimator
        assert restored.status == original.status
        assert restored.error == original.error
        assert restored.circuit_counts == original.circuit_counts
        assert restored.config == original.config

    def test_hdf5_roundtrip(self):
        """Test HDF5 serialization/deserialization roundtrip."""
        original = self._make_sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.resource_estimator_data.h5"
            with h5py.File(filepath, "w") as f:
                original.to_hdf5(f)
            with h5py.File(filepath, "r") as f:
                restored = ResourceEstimatorData.from_hdf5(f)
        assert restored.logical_counts == original.logical_counts
        assert restored.physical_counts == original.physical_counts
        assert restored.logical_qubit == original.logical_qubit
        assert restored.error_budget == original.error_budget
        assert restored.estimator == original.estimator

    def test_json_file_roundtrip(self):
        """Test to_json_file and from_json_file."""
        original = self._make_sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.resource_estimator_data.json"
            original.to_json_file(str(filepath))
            restored = ResourceEstimatorData.from_json_file(str(filepath))
        assert restored.logical_counts == original.logical_counts
        assert restored.estimator == original.estimator
