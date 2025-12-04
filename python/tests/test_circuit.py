"""Tests for the Circuit data class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import re
import tempfile
from pathlib import Path

import h5py
import pytest
from qiskit import QuantumCircuit, qasm3

from qdk_chemistry.data import Circuit


def strip_ws(s: str) -> str:
    """Normalize whitespace to make string matching more robust."""
    return re.sub(r"\s+", " ", s).strip()


class TestCircuitConstruction:
    """Test cases for Circuit construction and validation."""

    def test_circuit_construction_with_qasm(self):
        """Test that Circuit can be constructed with QASM string."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        circuit = Circuit(qasm=qasm)
        assert circuit.qasm is not None
        assert "h q[0];" in circuit.qasm

    def test_circuit_construction_without_qasm_raises(self):
        """Test that Circuit construction without QASM raises RuntimeError."""
        with pytest.raises(RuntimeError, match="The quantum circuit in QASM format is not set"):
            Circuit(qasm=None)

    def test_get_qasm(self):
        """Test get_qasm method returns the QASM string."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)
        retrieved_qasm = circuit.get_qasm()
        assert retrieved_qasm == qasm
        assert "h q[0];" in retrieved_qasm


class TestCircuitTrimming:
    """Test cases for circuit trimming functionality."""

    def test_trim_removes_idle_qubits(self):
        """Test that _trim_circuit removes idle qubits correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[1] c;
        h q[0];
        cx q[0], q[2];
        c[0] = measure q[0];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=False)

        # idle qubit q[1] removed → new indices → q[0], q[1]
        norm = strip_ws(trimmed)

        assert "h q[0];" in norm
        assert "cx q[0], q[1];" in norm
        assert "c[0] = measure q[0];" in norm

        # ensure q[2] never appears (idle qubit eliminated)
        assert "q[2]" not in norm

    def test_trim_removes_classical_qubits(self):
        """Test that _trim_circuit removes classical qubits correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        h q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=True)
        norm = strip_ws(trimmed)

        # classical qubit q[0] removed
        assert "h q[0];" in norm  # reindexed: old q[1] → new q[0]
        assert "c[0] = measure q[0];" in norm

        # everything referencing q[2] or old q[0] is gone
        assert "q[2]" not in norm
        assert "measure q[1]" not in norm

    def test_measurement_dropped_if_qubit_filtered(self):
        """Test that measurements are dropped if the qubit is filtered out."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[1] c;
        h q[0];
        c[0] = measure q[1];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=True)
        norm = strip_ws(trimmed)

        assert "h q[0];" in norm
        assert "measure" not in norm  # measurement removed

    def test_control_gate_removed_if_control_is_classical(self):
        """Test that control gates are removed if control qubit is classical."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[1] c;
        c[0] = measure q[0];
        h q[1];
        cx q[0], q[2];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=False, remove_classical_qubits=True)
        norm = strip_ws(trimmed)

        assert "h q[0];" in norm
        assert "cx" not in norm  # control removed
        assert "measure" not in norm  # measure removed because q0 removed entirely

    def test_trim_reindexing_correct(self):
        """Test that _trim_circuit reindexes qubits correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        bit[4] c;
        h q[1];
        cx q[1], q[2];
        c[3] = measure q[3];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=True)
        norm = strip_ws(trimmed)

        # only q1 and q2 survive -> new q[0], q[1]
        assert "h q[0];" in norm
        assert "cx q[0], q[1];" in norm
        assert "measure" not in norm

    def test_raises_if_all_removed(self):
        """Test that ValueError is raised if all qubits are removed."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        c[0] = measure q[0];
        """
        circuit = Circuit(qasm=qasm)
        with pytest.raises(ValueError, match="No qubits remain after filtering"):
            circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=True)

    def test_no_removal_returns_same_circuit(self):
        """Test that no removal options returns the same circuit structure."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        bit[1] c;
        qubit[2] q;
        h q[0];
        c[0] = measure q[0];
        x q[1];
        """
        circuit = Circuit(qasm=qasm)
        trimmed = circuit._trim_circuit(remove_idle_qubits=False, remove_classical_qubits=False)

        # Do whitespace normalization and compare
        assert strip_ws(trimmed) == strip_ws(qasm)

    def test_trim_circuit_with_invalid_qasm_raises(self):
        """Test that _trim_circuit raises ValueError for invalid QASM."""
        invalid_qasm = "INVALID QASM CODE"
        circuit = Circuit(qasm=invalid_qasm)

        with pytest.raises(ValueError, match="Invalid QASM3 syntax provided"):
            circuit._trim_circuit(remove_idle_qubits=True, remove_classical_qubits=True)


class TestGetQsharpCircuit:
    """Test cases for get_qsharp method."""

    def test_get_qsharp_basic(self):
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        x q[2];
        """
        circuit = Circuit(qasm=qasm)
        qdk_circuit = circuit.get_qsharp(remove_idle_qubits=True, remove_classical_qubits=True)

        # The resulting circuit should have 2 qubits (q0 and q1)
        qdk_circuit_info = json.loads(qdk_circuit.json())
        assert len(qdk_circuit_info["qubits"]) == 2

    def test_get_qsharp_no_removal(self):
        """Test get_qsharp with no removal options."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = qasm3.dumps(qc)

        circuit = Circuit(qasm=qasm)
        qdk_circuit = circuit.get_qsharp(remove_idle_qubits=False, remove_classical_qubits=False)

        circuit_info = json.loads(qdk_circuit.json())
        assert len(circuit_info["qubits"]) == 2

    def test_get_qsharp_with_idle_removal_only(self):
        """Test get_qsharp with only idle qubit removal."""
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qasm = qasm3.dumps(qc)

        circuit = Circuit(qasm=qasm)
        qdk_circuit = circuit.get_qsharp(remove_idle_qubits=True, remove_classical_qubits=False)

        circuit_info = json.loads(qdk_circuit.json())
        # Should have 2 qubits (idle q[2] removed)
        assert len(circuit_info["qubits"]) == 2


class TestCircuitSerialization:
    """Test cases for Circuit serialization and deserialization."""

    def test_to_json(self):
        """Test that to_json creates correct dictionary."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        circuit = Circuit(qasm=qasm)
        json_data = circuit.to_json()

        assert "qasm" in json_data
        assert json_data["qasm"] == qasm

    def test_from_json(self):
        """Test that from_json reconstructs Circuit correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        json_data = {"qasm": qasm, "version": Circuit._serialization_version}
        circuit = Circuit.from_json(json_data)

        assert circuit.qasm == qasm

    def test_json_roundtrip(self):
        """Test that JSON serialization and deserialization preserves data."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        original = Circuit(qasm=qasm)
        json_data = original.to_json()
        reconstructed = Circuit.from_json(json_data)

        assert reconstructed.qasm == original.qasm

    def test_to_hdf5(self):
        """Test that to_hdf5 saves Circuit correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                group = f.create_group("circuit")
                circuit.to_hdf5(group)

            with h5py.File(tmp_path, "r") as f:
                group = f["circuit"]
                assert "qasm" in group.attrs
                assert group.attrs["qasm"] == qasm
        finally:
            tmp_path.unlink()

    def test_from_hdf5(self):
        """Test that from_hdf5 loads Circuit correctly."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                group = f.create_group("circuit")
                group.attrs["version"] = Circuit._serialization_version
                group.attrs["qasm"] = qasm

            with h5py.File(tmp_path, "r") as f:
                group = f["circuit"]
                circuit = Circuit.from_hdf5(group)

            assert circuit.qasm == qasm
        finally:
            tmp_path.unlink()

    def test_hdf5_roundtrip(self):
        """Test that HDF5 serialization and deserialization preserves data."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        original = Circuit(qasm=qasm)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                group = f.create_group("circuit")
                original.to_hdf5(group)

            with h5py.File(tmp_path, "r") as f:
                group = f["circuit"]
                reconstructed = Circuit.from_hdf5(group)

            assert reconstructed.qasm == original.qasm
        finally:
            tmp_path.unlink()

    def test_to_json_file(self):
        """Test that to_json_file saves Circuit to a file."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)

        with tempfile.NamedTemporaryFile(suffix=".circuit.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            circuit.to_json_file(tmp_path)

            with open(tmp_path) as f:
                loaded_data = json.load(f)

            assert "qasm" in loaded_data
            assert loaded_data["qasm"] == qasm
        finally:
            tmp_path.unlink()

    def test_from_json_file(self):
        """Test that from_json_file loads Circuit from a file."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".circuit.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            json.dump({"qasm": qasm, "version": Circuit._serialization_version}, tmp)

        try:
            circuit = Circuit.from_json_file(tmp_path)
            assert circuit.qasm == qasm
        finally:
            tmp_path.unlink()

    def test_to_hdf5_file(self):
        """Test that to_hdf5_file saves Circuit to a file."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)

        with tempfile.NamedTemporaryFile(suffix=".circuit.h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            circuit.to_hdf5_file(tmp_path)

            with h5py.File(tmp_path, "r") as f:
                assert "qasm" in f.attrs
                assert f.attrs["qasm"] == qasm
        finally:
            tmp_path.unlink()

    def test_from_hdf5_file(self):
        """Test that from_hdf5_file loads Circuit from a file."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        with tempfile.NamedTemporaryFile(suffix=".circuit.h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["version"] = Circuit._serialization_version
                f.attrs["qasm"] = qasm

            circuit = Circuit.from_hdf5_file(tmp_path)
            assert circuit.qasm == qasm
        finally:
            tmp_path.unlink()


class TestCircuitSummary:
    """Test cases for Circuit summary functionality."""

    def test_get_summary(self):
        """Test that get_summary returns a descriptive string."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)
        summary = circuit.get_summary()

        assert "Circuit" in summary
        assert "QASM string" in summary
        assert qasm in summary

    def test_get_summary_format(self):
        """Test that get_summary has correct format."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        x q[0];
        """
        circuit = Circuit(qasm=qasm)
        summary = circuit.get_summary()

        lines = summary.split("\n")
        assert len(lines) >= 1
        assert lines[0] == "Circuit"


class TestCircuitImmutability:
    """Test cases for Circuit immutability."""

    def test_circuit_is_immutable_after_construction(self):
        """Test that Circuit attributes cannot be modified after construction."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            circuit.qasm = "modified"

    def test_cannot_add_new_attributes(self):
        """Test that new attributes cannot be added to Circuit."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        """
        circuit = Circuit(qasm=qasm)

        # Attempting to add a new attribute should raise an error
        with pytest.raises(AttributeError):
            circuit.new_attr = "value"
