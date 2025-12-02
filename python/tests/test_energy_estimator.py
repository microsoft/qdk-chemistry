"""Tests for energy estimation in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, PauliList

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.energy_estimator.energy_estimator import (
    EnergyEstimator,
    _build_measurement_circuit,
    _compute_expval_and_variance_from_bitstrings,
    _determine_measurement_basis,
    _parity,
    _paulis_to_indices,
)
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.data.estimator_data import MeasurementData

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

try:
    from qiskit_aer import AerSimulator

    from qdk_chemistry.plugins.qiskit.energy_estimator import QiskitEnergyEstimator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QiskitEnergyEstimator = None

try:
    from qdk_chemistry.algorithms.energy_estimator import QDKEnergyEstimator

    QSHARP_AVAILABLE = True
except ImportError:
    QSHARP_AVAILABLE = False
    QDKEnergyEstimator = None


def test_parity():
    """Test parity calculation."""
    assert _parity(0b0000) == 0  # This is the integer 0 in binary notation
    assert _parity(0b0001) == 1
    assert _parity(0b0011) == 0
    assert _parity(0b0101) == 0
    assert _parity(0b1111) == 0
    assert _parity(0b1011) == 1


def test_determine_measurement_basis():
    """Test measurement basis determination."""
    pauli = PauliList(["IZII", "YZIZ"])
    basis = _determine_measurement_basis(pauli)
    assert basis.to_label() == "YZIZ"


def test_determine_measurement_basis_not_qubit_wise_commuting():
    """Test measurement basis determination for non-qubit-wise commuting Pauli operators will raise ValueError."""
    pauli = PauliList(["XX", "YY"])
    with pytest.raises(
        ValueError,
        match=r"Paulis are not qubit-wise commuting\. Please group them first to generate a valid measurement basis\.",
    ):
        _determine_measurement_basis(pauli)


def test_measurement_circuit_z_basis():
    """Test that Z basis adds only measurement (no rotation)."""
    basis = Pauli("Z")
    qc = _build_measurement_circuit(basis)
    ops = [inst.operation.name for inst in qc.data]

    # Expect only one measurement, no H or Sdg
    assert ops.count("measure") == 1
    assert "h" not in ops
    assert "sdg" not in ops
    assert qc.num_qubits == 1
    assert qc.num_clbits == 1


def test_measurement_circuit_x_basis():
    """Test that X basis adds H rotation before measurement."""
    basis = Pauli("X")
    qc = _build_measurement_circuit(basis)
    ops = [inst.operation.name for inst in qc.data]

    # Expect H + measure
    assert "h" in ops
    assert "measure" in ops
    assert "sdg" not in ops
    assert qc.num_clbits == 1


def test_measurement_circuit_y_basis():
    """Test that Y basis adds Sdg + H before measurement."""
    basis = Pauli("Y")
    qc = _build_measurement_circuit(basis)
    ops = [inst.operation.name for inst in qc.data]

    # Expect Sdg + H + measure
    assert "sdg" in ops
    assert "h" in ops
    assert "measure" in ops


def test_measurement_circuit_identity_only():
    """Test that I-only basis not generates any measurements."""
    basis = Pauli("II")
    qc = _build_measurement_circuit(basis)

    assert qc.num_qubits == 2
    assert not any(inst.operation.name == "measure" for inst in qc.data)


def test_measurement_circuit_mixed_basis():
    """Test correct mapping for a multi-qubit Pauli basis."""
    basis = Pauli("XYIZ")
    qc = _build_measurement_circuit(basis)
    ops = [inst.operation.name for inst in qc.data]

    # Expect Sdg + H for Y, H for X, measure on active qubits only
    assert ops.count("measure") == 3  # X, Y, Z active
    assert qc.num_clbits == 3
    assert qc.num_qubits == 4


def test_create_measurement_circuits_basic():
    """Test measurement circuit generation for a simple observable."""
    # Prepare input circuit
    circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        x q[0];
        cx q[0], q[1];
        """

    # Define observable
    observable = [
        QubitHamiltonian(["ZI", "IZ", "ZZ"], np.array([1.0, 1.0, 1.0])),
        QubitHamiltonian(["XX"], np.array([1.0])),
        QubitHamiltonian(["YY"], np.array([1.0])),
    ]

    # Call function
    circuits = EnergyEstimator._create_measurement_circuits(circuit_qasm, observable)

    # There should be one measurement circuit per observable
    assert isinstance(circuits, list)
    assert len(circuits) == 3
    assert all(isinstance(circ, str) for circ in circuits)
    assert "measure" in circuits[0]  # Z basis
    assert circuits[0].count("measure") == 2
    assert "h q" not in circuits[0]  # No basis change for Z
    assert "h q" in circuits[1]  # X basis change
    assert circuits[1].count("h q") == 2  # One H gate added for X basis for each qubit
    assert circuits[0].count("measure") == 2
    assert "sdg q" in circuits[2]  # Y basis change
    assert circuits[2].count("sdg q") == 2  # One Sdg gate added for Y basis for each qubit
    assert "h q" in circuits[2]  # Y basis change
    assert circuits[2].count("h q") == 2  # One H gate added for Y basis for each qubit
    assert circuits[0].count("measure") == 2


def test_create_measurement_circuits_qubit_mismatch():
    """Test measurement circuit generation raises ValueError on qubit number mismatch."""
    # Prepare input circuit with 2 qubits
    circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        x q[0];
        cx q[0], q[1];
        """

    # Define observable with 3 qubits
    observable = [
        QubitHamiltonian(["ZII", "IZI", "IIZ"], np.array([1.0, 1.0, 1.0])),
    ]

    # Call function and expect ValueError
    with pytest.raises(
        ValueError,
        match=(
            r"Number of qubits in the base circuit \(2\) does not match "
            r"the number of qubits in the Hamiltonian \(3\)\."
        ),
    ):
        EnergyEstimator._create_measurement_circuits(circuit_qasm, observable)


@pytest.mark.parametrize(
    ("counts", "paulis", "expected_expvals", "expected_vars"),
    [
        (
            {"0x0": 30, "0x1": 70},
            PauliList(["ZZ", "ZI"]),
            np.array([-0.4, 1.0]),
            np.array([0.0084, 0.0]),
        ),
        (
            {"0x0": 30, "0x1": 70},
            PauliList(["XX", "XI"]),
            np.array([-0.4, 1.0]),
            np.array([0.0084, 0.0]),
        ),
        (
            {"0x2": 1579, "0x1": 48421},
            PauliList(["IIIIIIZI", "IIIIIZII"]),
            np.array([-0.93684, 0.93684]),
            np.array([2.44661629e-06, 2.44661629e-06]),
        ),
    ],
)
def test_compute_expval_and_variance_for_paulis_from_bitstring_counts(counts, paulis, expected_expvals, expected_vars):
    """Test the statistics (mean and variance) calculation for Pauli observables."""
    expvals, vars_ = _compute_expval_and_variance_from_bitstrings(counts, paulis)
    assert np.allclose(
        expvals, expected_expvals, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        vars_, expected_vars, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )


def test_compute_expval_invalid_bitstring_format():
    """Test _compute_expval_and_variance with invalid bitstring formats."""
    # Test with invalid hex format
    counts = {"invalid_hex": 50, "0x1": 50}
    paulis = PauliList(["Z"])

    with pytest.raises(ValueError, match="Unsupported bitstring format"):
        _compute_expval_and_variance_from_bitstrings(counts, paulis)


def test_compute_expval_mixed_bitstring_formats():
    """Test _compute_expval_and_variance with mixed valid bitstring formats."""
    # Test with both hex and binary formats
    counts = {"0x0": 30, "1": 70}  # Mix of hex and binary
    paulis = PauliList(["Z"])

    expvals, vars_ = _compute_expval_and_variance_from_bitstrings(counts, paulis)

    # Should handle both formats correctly
    assert len(expvals) == 1
    assert len(vars_) == 1
    assert isinstance(expvals[0], float)
    assert isinstance(vars_[0], float)
    assert np.isclose(
        expvals[0], -0.4, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )
    assert np.isclose(
        vars_[0], 0.0084, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )


def test_compute_expval_empty_counts() -> None:
    """Test _compute_expval_and_variance with empty bitstring counts."""
    counts: dict = {}
    paulis = PauliList(["Z"])

    with pytest.raises(ValueError, match=r"Bitstring counts are empty\."):
        _compute_expval_and_variance_from_bitstrings(counts, paulis)


def test_paulis_to_indices():
    """Test conversion of a list of Pauli strings to indices."""
    inds = _paulis_to_indices(PauliList(["IZ", "ZX", "YZ", "ZY"]))
    assert np.array_equal(inds, [1, 3, 3, 3])


def test_compute_energy_expectation_from_bitstrings_mismatched_lengths():
    """Test calculate_energy_expval_and_variance with mismatched input lengths."""
    bitstring_counts = [{"0": 50, "1": 50}]
    observables = [QubitHamiltonian(["Z"], [1.0]), QubitHamiltonian(["X"], [1.0])]  # Extra observable

    with pytest.raises(ValueError, match="Expected 2 bitstring result sets, got 1"):
        EnergyEstimator._compute_energy_expectation_from_bitstrings(observables, bitstring_counts)


def test_calculate_energy_expval_variance_none_counts():
    """Test calculate_energy_expval_and_variance with None in bitstring_counts."""
    bitstring_counts = [None, {"0": 50, "1": 50}]
    observables = [QubitHamiltonian(["Z"], [1.0]), QubitHamiltonian(["X"], [1.0])]

    result = EnergyEstimator._compute_energy_expectation_from_bitstrings(observables, bitstring_counts)

    # Should handle None entries gracefully
    assert np.isclose(
        result.energy_expectation_value,
        0.0,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )


def test_measurement_data_to_json():
    """Test MeasurementData.to_json method."""
    measurement_results = MeasurementData(
        hamiltonians=[QubitHamiltonian(["Z"], np.array([1.0]))],
        bitstring_counts=[{"0": 50, "1": 50}],
        shots_list=[100],
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".measurement_data.json", delete=False) as f:
        temp_path = f.name

    try:
        measurement_results.to_json_file(temp_path)

        # Verify file was created and contains expected structure
        assert Path(temp_path).exists()
        with open(temp_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        # Should have one entry for the pauli group
        assert len(data) == 1
    finally:
        Path(temp_path).unlink()


def test_create_energy_estimator_qiskit():
    """Test factory function for creating Qiskit energy estimator."""
    estimator = create(
        "energy_estimator",
        "qiskit_aer_simulator",
    )
    assert isinstance(estimator, EnergyEstimator)
    assert isinstance(estimator, QiskitEnergyEstimator)
    assert isinstance(estimator.backend, AerSimulator)
    assert estimator.backend.options.seed_simulator == 42


def test_create_energy_estimator_qdk():
    """Test factory function for creating QDK energy estimator."""
    estimator = create("energy_estimator", "qdk_base_simulator")
    assert isinstance(estimator, EnergyEstimator)
    assert isinstance(estimator, QDKEnergyEstimator)
    assert estimator.seed == 42
    assert estimator.noise_model is None
