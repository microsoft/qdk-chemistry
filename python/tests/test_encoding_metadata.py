"""Tests for encoding metadata and validation in Circuit and QubitHamiltonian."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, EncodingMismatchError, QubitHamiltonian, validate_encoding_compatibility

from .test_helpers import create_test_hamiltonian


def test_circuit_encoding_metadata():
    """Test that Circuit properly stores and retrieves encoding metadata."""
    # Test with encoding specified
    circuit_jw = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")
    assert circuit_jw.encoding == "jordan-wigner"

    # Test with no encoding specified
    circuit_none = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding=None)
    assert circuit_none.encoding is None

    # Test different encodings
    circuit_bk = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="bravyi-kitaev")
    assert circuit_bk.encoding == "bravyi-kitaev"

    circuit_parity = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="parity")
    assert circuit_parity.encoding == "parity"


def test_qubit_hamiltonian_encoding_metadata():
    """Test that QubitHamiltonian properly stores and retrieves encoding metadata."""
    pauli_strings = ["II", "ZI", "IZ", "ZZ"]
    coefficients = np.array([1.0, 0.5, 0.5, 0.25])

    # Test with encoding specified
    ham_jw = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")
    assert ham_jw.encoding == "jordan-wigner"

    # Test with no encoding specified
    ham_none = QubitHamiltonian(pauli_strings, coefficients, encoding=None)
    assert ham_none.encoding is None

    # Test different encodings
    ham_bk = QubitHamiltonian(pauli_strings, coefficients, encoding="bravyi-kitaev")
    assert ham_bk.encoding == "bravyi-kitaev"

    ham_parity = QubitHamiltonian(pauli_strings, coefficients, encoding="parity")
    assert ham_parity.encoding == "parity"


def test_circuit_encoding_serialization_json():
    """Test that Circuit encoding is preserved through JSON serialization."""
    circuit = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")

    # Serialize to JSON
    json_data = circuit.to_json()
    assert "encoding" in json_data
    assert json_data["encoding"] == "jordan-wigner"

    # Deserialize from JSON
    circuit_restored = Circuit.from_json(json_data)
    assert circuit_restored.encoding == "jordan-wigner"
    assert circuit_restored.qasm == circuit.qasm


def test_circuit_encoding_serialization_hdf5(tmp_path):
    """Test that Circuit encoding is preserved through HDF5 serialization."""
    circuit = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")

    # Save to HDF5
    hdf5_path = tmp_path / "circuit.h5"
    with h5py.File(hdf5_path, "w") as f:
        circuit.to_hdf5(f)

    # Load from HDF5
    with h5py.File(hdf5_path, "r") as f:
        circuit_restored = Circuit.from_hdf5(f)

    assert circuit_restored.encoding == "jordan-wigner"
    assert circuit_restored.qasm == circuit.qasm


def test_qubit_hamiltonian_encoding_serialization_json():
    """Test that QubitHamiltonian encoding is preserved through JSON serialization."""
    pauli_strings = ["II", "ZI", "IZ"]
    coefficients = np.array([1.0, 0.5, 0.5])
    ham = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")

    # Serialize to JSON
    json_data = ham.to_json()
    assert "encoding" in json_data
    assert json_data["encoding"] == "jordan-wigner"

    # Deserialize from JSON
    ham_restored = QubitHamiltonian.from_json(json_data)
    assert ham_restored.encoding == "jordan-wigner"
    assert ham_restored.pauli_strings == pauli_strings
    assert np.array_equal(ham_restored.coefficients, coefficients)


def test_qubit_hamiltonian_encoding_serialization_hdf5(tmp_path):
    """Test that QubitHamiltonian encoding is preserved through HDF5 serialization."""
    pauli_strings = ["II", "ZI", "IZ"]
    coefficients = np.array([1.0, 0.5, 0.5])
    ham = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")

    # Save to HDF5
    hdf5_path = tmp_path / "hamiltonian.h5"
    with h5py.File(hdf5_path, "w") as f:
        ham.to_hdf5(f)

    # Load from HDF5
    with h5py.File(hdf5_path, "r") as f:
        ham_restored = QubitHamiltonian.from_hdf5(f)

    assert ham_restored.encoding == "jordan-wigner"
    assert ham_restored.pauli_strings == pauli_strings
    assert np.array_equal(ham_restored.coefficients, coefficients)


def test_validate_encoding_compatibility_matching():
    """Test that validation passes when encodings match."""
    circuit = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")
    pauli_strings = ["II", "ZI"]
    coefficients = np.array([1.0, 0.5])
    hamiltonian = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")

    # Should not raise an error
    validate_encoding_compatibility(circuit, hamiltonian)


def test_validate_encoding_compatibility_none():
    """Test that validation fails when encodings are None."""
    circuit_none = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding=None)
    circuit_jw = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")
    pauli_strings = ["II", "ZI"]
    coefficients = np.array([1.0, 0.5])
    ham_none = QubitHamiltonian(pauli_strings, coefficients, encoding=None)
    ham_jw = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")

    # Circuit with None encoding should raise error
    with pytest.raises(EncodingMismatchError) as exc_info:
        validate_encoding_compatibility(circuit_none, ham_jw)
    assert "Circuit encoding is not specified" in str(exc_info.value)

    # Hamiltonian with None encoding should raise error
    with pytest.raises(EncodingMismatchError) as exc_info:
        validate_encoding_compatibility(circuit_jw, ham_none)
    assert "QubitHamiltonian encoding is not specified" in str(exc_info.value)

    # Both None should raise error (circuit is checked first)
    with pytest.raises(EncodingMismatchError) as exc_info:
        validate_encoding_compatibility(circuit_none, ham_none)
    assert "Circuit encoding is not specified" in str(exc_info.value)


def test_validate_encoding_compatibility_mismatch():
    """Test that validation fails when encodings don't match."""
    circuit = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")
    pauli_strings = ["II", "ZI"]
    coefficients = np.array([1.0, 0.5])
    hamiltonian = QubitHamiltonian(pauli_strings, coefficients, encoding="bravyi-kitaev")

    # Should raise EncodingMismatchError
    with pytest.raises(EncodingMismatchError) as exc_info:
        validate_encoding_compatibility(circuit, hamiltonian)

    error_msg = str(exc_info.value)
    assert "jordan-wigner" in error_msg
    assert "bravyi-kitaev" in error_msg
    assert "incompatible" in error_msg.lower()


def test_state_preparation_injects_jordan_wigner_encoding(wavefunction_4e4o):
    """Test that StatePreparation algorithms inject Jordan-Wigner encoding."""
    # Test sparse_isometry_gf2x
    prep_gf2x = create("state_prep", "sparse_isometry_gf2x")
    circuit_gf2x = prep_gf2x.run(wavefunction_4e4o)
    assert circuit_gf2x.encoding == "jordan-wigner"

    # Test qiskit_regular_isometry
    prep_regular = create("state_prep", "qiskit_regular_isometry")
    circuit_regular = prep_regular.run(wavefunction_4e4o)
    assert circuit_regular.encoding == "jordan-wigner"


def test_qubit_mapper_injects_encoding():
    """Test that QubitMapper injects the correct encoding."""
    hamiltonian = create_test_hamiltonian(2)

    # Test Jordan-Wigner
    mapper_jw = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
    qubit_ham_jw = mapper_jw.run(hamiltonian)
    assert qubit_ham_jw.encoding == "jordan-wigner"

    # Test Bravyi-Kitaev
    mapper_bk = create("qubit_mapper", "qiskit", encoding="bravyi-kitaev")
    qubit_ham_bk = mapper_bk.run(hamiltonian)
    assert qubit_ham_bk.encoding == "bravyi-kitaev"

    # Test Parity
    mapper_parity = create("qubit_mapper", "qiskit", encoding="parity")
    qubit_ham_parity = mapper_parity.run(hamiltonian)
    assert qubit_ham_parity.encoding == "parity"


def test_qdk_qubit_mapper_injects_encoding():
    """Test that QDK QubitMapper injects the correct encoding."""
    hamiltonian = create_test_hamiltonian(2)

    # Test Jordan-Wigner
    mapper_jw = create("qubit_mapper", "qdk", encoding="jordan-wigner")
    qubit_ham_jw = mapper_jw.run(hamiltonian)
    assert qubit_ham_jw.encoding == "jordan-wigner"

    # Test Bravyi-Kitaev
    mapper_bk = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")
    qubit_ham_bk = mapper_bk.run(hamiltonian)
    assert qubit_ham_bk.encoding == "bravyi-kitaev"


def test_group_commuting_preserves_encoding():
    """Test that group_commuting preserves the encoding metadata."""
    pauli_strings = ["II", "ZI", "IZ", "ZZ", "XX", "YY"]
    coefficients = np.array([1.0, 0.5, 0.5, 0.25, 0.3, 0.3])
    ham = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")

    # Group into commuting subsets
    grouped = ham.group_commuting(qubit_wise=True)

    # Each group should preserve the encoding
    for group in grouped:
        assert group.encoding == "jordan-wigner"


def test_end_to_end_workflow_compatible_encodings(wavefunction_4e4o):
    """Test end-to-end workflow with compatible encodings (both Jordan-Wigner)."""
    hamiltonian = create_test_hamiltonian(2)

    # Create QubitHamiltonian with Jordan-Wigner encoding
    mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
    qubit_ham = mapper.run(hamiltonian)

    # Create Circuit with state preparation (should be Jordan-Wigner)
    prep = create("state_prep", "sparse_isometry_gf2x")
    circuit = prep.run(wavefunction_4e4o)

    # Validation should pass
    validate_encoding_compatibility(circuit, qubit_ham)


def test_end_to_end_workflow_incompatible_encodings(wavefunction_4e4o):
    """Test end-to-end workflow with incompatible encodings."""
    hamiltonian = create_test_hamiltonian(2)

    # Create QubitHamiltonian with Bravyi-Kitaev encoding
    mapper = create("qubit_mapper", "qiskit", encoding="bravyi-kitaev")
    qubit_ham = mapper.run(hamiltonian)

    # Create Circuit with state preparation (should be Jordan-Wigner)
    prep = create("state_prep", "sparse_isometry_gf2x")
    circuit = prep.run(wavefunction_4e4o)

    # Validation should fail
    with pytest.raises(EncodingMismatchError):
        validate_encoding_compatibility(circuit, qubit_ham)


def test_circuit_summary_includes_encoding():
    """Test that Circuit.get_summary() includes encoding information."""
    circuit = Circuit(qasm="OPENQASM 3.0; qubit[2] q;", encoding="jordan-wigner")
    summary = circuit.get_summary()

    assert "jordan-wigner" in summary
    assert "Encoding" in summary


def test_qubit_hamiltonian_summary_includes_encoding():
    """Test that QubitHamiltonian.get_summary() includes encoding information."""
    pauli_strings = ["II", "ZI"]
    coefficients = np.array([1.0, 0.5])
    ham = QubitHamiltonian(pauli_strings, coefficients, encoding="jordan-wigner")
    summary = ham.get_summary()

    assert "jordan-wigner" in summary
    assert "Encoding" in summary
