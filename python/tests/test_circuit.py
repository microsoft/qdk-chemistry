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
import qsharp

from qdk_chemistry.data import Circuit
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT


def strip_ws(s: str) -> str:
    """Normalize whitespace to make string matching more robust."""
    return re.sub(r"\s+", " ", s).strip()


@pytest.fixture
def simple_qasm() -> str:
    """A simple QASM string for testing."""
    return """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[0] c;
    h q[0];
    cx q[0], q[1];
    """


@pytest.fixture
def simple_qir(simple_qasm) -> str:
    """The QIR representation of the simple QASM string."""
    return str(qsharp.openqasm.compile(simple_qasm))


class TestCircuitConstruction:
    """Test cases for Circuit construction and validation."""

    def test_circuit_construction_with_qasm(self):
        """Test that Circuit can be constructed with QASM string."""
        qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        """
        qir = qsharp.openqasm.compile(qasm)
        qsharp_circuit = qsharp.openqasm.circuit(qasm)
        circuit = Circuit(qasm=qasm, qir=qir, qsharp=qsharp_circuit)
        assert circuit.qasm is not None
        assert "h q[0];" in circuit.qasm
        assert isinstance(circuit.qir, qsharp._qsharp.QirInputData)
        assert isinstance(circuit.qsharp, qsharp._native.Circuit)

    def test_circuit_construction_raises(self):
        """Test that Circuit construction without QASM raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No representation of the quantum circuit"):
            Circuit()

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


class TestGetQsharpCircuit:
    """Test cases for get_qsharp method."""

    def test_get_qsharp_and_qir(self):
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
        qdk_circuit = circuit.get_qsharp_circuit()
        qdk_circuit_info = json.loads(qdk_circuit.json())
        assert len(qdk_circuit_info["qubits"]) == 3
        qir = circuit.get_qir()
        assert isinstance(qir, qsharp._qsharp.QirInputData)


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
class TestGetQiskitConversion:
    """Test cases for get_qiskit_circuit method."""

    def test_circuit_qasm_to_qiskit(self, simple_qasm):
        """Test that a QASM string can be converted to a Qiskit QuantumCircuit."""
        circuit = Circuit(qasm=simple_qasm)
        qiskit_circuit = circuit.get_qiskit_circuit()
        assert qiskit_circuit is not None
        assert len(qiskit_circuit.qubits) == 2
        assert len(qiskit_circuit.data) == 2  # H and CX gates

    def test_circuit_qir_to_qiskit(self, simple_qir):
        """Test that a QIR representation can be converted to a Qiskit QuantumCircuit."""
        circuit = Circuit(qir=simple_qir)
        qiskit_circuit = circuit.get_qiskit_circuit()
        assert qiskit_circuit is not None
        assert len(qiskit_circuit.qubits) == 2
        assert len(qiskit_circuit.data) == 2  # H and CX gates


class TestCircuitSerialization:
    """Test cases for Circuit serialization and deserialization."""

    def test_to_json(self, simple_qasm, simple_qir):
        """Test that to_json creates correct dictionary."""
        circuit = Circuit(qasm=simple_qasm, qir=simple_qir)
        json_data = circuit.to_json()

        assert "qasm" in json_data
        assert json_data["qasm"] == simple_qasm
        assert "qir" in json_data
        assert json_data["qir"] == str(simple_qir)

    def test_from_json(self, simple_qasm, simple_qir):
        """Test that from_json reconstructs Circuit correctly."""
        json_data = {"qasm": simple_qasm, "qir": str(simple_qir), "version": Circuit._serialization_version}
        circuit = Circuit.from_json(json_data)

        assert circuit.qasm == simple_qasm
        assert circuit.qir == simple_qir

    def test_json_roundtrip(self, simple_qasm, simple_qir):
        """Test that JSON serialization and deserialization preserves data."""
        original = Circuit(qasm=simple_qasm, qir=simple_qir)
        json_data = original.to_json()
        reconstructed = Circuit.from_json(json_data)

        assert reconstructed.qasm == original.qasm
        assert reconstructed.qir == original.qir

    def test_to_hdf5(self, simple_qasm, simple_qir):
        """Test that to_hdf5 saves Circuit correctly."""
        qasm = simple_qasm
        qir = simple_qir
        circuit = Circuit(qasm=qasm, qir=qir)

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
                assert "qir" in group.attrs
                assert group.attrs["qir"] == str(qir)
        finally:
            tmp_path.unlink()

    def test_from_hdf5(self, simple_qasm, simple_qir):
        """Test that from_hdf5 loads Circuit correctly."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                group = f.create_group("circuit")
                group.attrs["version"] = Circuit._serialization_version
                group.attrs["qasm"] = simple_qasm
                group.attrs["qir"] = simple_qir

            with h5py.File(tmp_path, "r") as f:
                group = f["circuit"]
                circuit = Circuit.from_hdf5(group)

            assert circuit.qasm == simple_qasm
            assert circuit.qir == simple_qir
        finally:
            tmp_path.unlink()

    def test_hdf5_roundtrip(self, simple_qasm, simple_qir):
        """Test that HDF5 serialization and deserialization preserves data."""
        original = Circuit(qasm=simple_qasm, qir=simple_qir)

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
            assert reconstructed.qir == original.qir

        finally:
            tmp_path.unlink()

    def test_to_json_file(self, simple_qasm, simple_qir):
        """Test that to_json_file saves Circuit to a file."""
        circuit = Circuit(qasm=simple_qasm, qir=simple_qir)

        with tempfile.NamedTemporaryFile(suffix=".circuit.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            circuit.to_json_file(tmp_path)

            with open(tmp_path) as f:
                loaded_data = json.load(f)

            assert "qasm" in loaded_data
            assert loaded_data["qasm"] == simple_qasm
            assert "qir" in loaded_data
            assert loaded_data["qir"] == simple_qir

        finally:
            tmp_path.unlink()

    def test_from_json_file(self, simple_qasm, simple_qir):
        """Test that from_json_file loads Circuit from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".circuit.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            json.dump({"qasm": simple_qasm, "qir": simple_qir, "version": Circuit._serialization_version}, tmp)

        try:
            circuit = Circuit.from_json_file(tmp_path)
            assert circuit.qasm == simple_qasm
            assert circuit.qir == simple_qir

        finally:
            tmp_path.unlink()

    def test_to_hdf5_file(self, simple_qasm, simple_qir):
        """Test that to_hdf5_file saves Circuit to a file."""
        circuit = Circuit(qasm=simple_qasm, qir=simple_qir)
        with tempfile.NamedTemporaryFile(suffix=".circuit.h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            circuit.to_hdf5_file(tmp_path)

            with h5py.File(tmp_path, "r") as f:
                assert "qasm" in f.attrs
                assert f.attrs["qasm"] == simple_qasm
                assert "qir" in f.attrs
                assert f.attrs["qir"] == simple_qir
        finally:
            tmp_path.unlink()

    def test_from_hdf5_file(self, simple_qasm, simple_qir):
        """Test that from_hdf5_file loads Circuit from a file."""
        with tempfile.NamedTemporaryFile(suffix=".circuit.h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["version"] = Circuit._serialization_version
                f.attrs["qasm"] = simple_qasm
                f.attrs["qir"] = simple_qir

            circuit = Circuit.from_hdf5_file(tmp_path)
            assert circuit.qasm == simple_qasm
            assert circuit.qir == simple_qir

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
