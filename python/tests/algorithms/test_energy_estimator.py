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

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.energy_estimator.qdk import (
    QdkEnergyEstimator,
    _append_measurement_to_circuit,
    _compute_expval_and_variance_from_bitstrings,
    _determine_measurement_basis,
    _parity,
    _paulis_to_nonid_masks,
)
from qdk_chemistry.data import Circuit, MeasurementData, QubitHamiltonian
from qdk_chemistry.data.qubit_hamiltonian import filter_and_group_pauli_ops_from_wavefunction
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT, QDK_CHEMISTRY_HAS_QISKIT_AER

from ..reference_tolerances import (
    estimator_energy_tolerance,
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)


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
    pauli_strings = ["IZII", "YZIZ"]
    basis = _determine_measurement_basis(pauli_strings)
    assert basis == "YZIZ"


def test_determine_measurement_basis_not_qubit_wise_commuting():
    """Test measurement basis determination for non-qubit-wise commuting Pauli operators will raise ValueError."""
    pauli_strings = ["XX", "YY"]
    with pytest.raises(
        ValueError,
        match=r"Paulis are not qubit-wise commuting\. Please group them first to generate a valid measurement basis\.",
    ):
        _determine_measurement_basis(pauli_strings)


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
@pytest.mark.parametrize(
    ("basis", "n_qubits", "expect_measure", "expect_h", "expect_sdg", "measure_count"),
    [
        ("Z", 1, True, False, False, 1),
        ("X", 1, True, True, False, 1),
        ("Y", 1, True, True, True, 1),
        ("II", 2, False, False, False, 0),
        ("XYIZ", 4, True, True, True, 3),
    ],
    ids=["Z-basis", "X-basis", "Y-basis", "identity-only", "mixed-basis"],
)
def test_append_measurement_to_circuit_qasm(basis, n_qubits, expect_measure, expect_h, expect_sdg, measure_count):
    """Test that _append_measurement_to_circuit applies correct rotations and measurements."""
    base_qasm = f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n_qubits}] q;\n'
    base_circuit = Circuit(qasm=base_qasm)
    result = _append_measurement_to_circuit(base_circuit, basis)
    qasm = result.get_qasm()

    assert ("measure" in qasm) == expect_measure
    assert ("h q" in qasm) == expect_h
    assert ("sdg q" in qasm) == expect_sdg
    assert qasm.count("measure") == measure_count


def test_create_measurement_circuits_basic(wavefunction_4e4o):
    """Test measurement circuit generation for a simple observable."""
    state_prep = create("state_prep", "sparse_isometry_gf2x")
    circuit = state_prep.run(wavefunction_4e4o)

    # Define observable
    observable = [
        QubitHamiltonian(["ZIIIIIII", "IZIIIIII", "ZZIIIIII"], np.array([1.0, 1.0, 1.0])),
        QubitHamiltonian(["XXIIIIII"], np.array([1.0])),
        QubitHamiltonian(["YYIIIIII"], np.array([1.0])),
    ]

    # Call function
    circuits = QdkEnergyEstimator._create_measurement_circuits(circuit, observable)
    qsc_json = [circ.get_qsharp_circuit().json() for circ in circuits]
    # There should be one measurement circuit per observable
    assert isinstance(circuits, list)
    assert len(circuits) == 3
    assert all(isinstance(circ, Circuit) for circ in circuits)
    assert all(qsc.count("measure") == 2 for qsc in qsc_json)


@pytest.mark.parametrize(
    ("counts", "paulis", "expected_expvals", "expected_vars"),
    [
        (
            {"0x0": 30, "0x1": 70},
            ["ZZ", "ZI"],
            np.array([-0.4, 1.0]),
            np.array([0.0084, 0.0]),
        ),
        (
            {"0x0": 30, "0x1": 70},
            ["XX", "XI"],
            np.array([-0.4, 1.0]),
            np.array([0.0084, 0.0]),
        ),
        (
            {"0x2": 1579, "0x1": 48421},
            ["IIIIIIZI", "IIIIIZII"],
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
    paulis = ["Z"]

    with pytest.raises(ValueError, match="Unsupported bitstring format"):
        _compute_expval_and_variance_from_bitstrings(counts, paulis)


def test_compute_expval_mixed_bitstring_formats():
    """Test _compute_expval_and_variance with mixed valid bitstring formats."""
    # Test with both hex and binary formats
    counts = {"0x0": 30, "1": 70}  # Mix of hex and binary
    paulis = ["Z"]

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
    paulis = ["Z"]

    with pytest.raises(ValueError, match=r"Bitstring counts are empty\."):
        _compute_expval_and_variance_from_bitstrings(counts, paulis)


def test_paulis_to_nonid_masks():
    """Test conversion of a list of Pauli strings to non-identity bitmasks."""
    masks = _paulis_to_nonid_masks(["IZ", "ZX", "YZ", "ZY"])
    assert masks == [1, 3, 3, 3]


def test_compute_energy_expectation_from_bitstrings_mismatched_lengths():
    """Test calculate_energy_expval_and_variance with mismatched input lengths."""
    bitstring_counts = [{"0": 50, "1": 50}]
    observables = [QubitHamiltonian(["Z"], [1.0]), QubitHamiltonian(["X"], [1.0])]  # Extra observable

    with pytest.raises(ValueError, match="Expected 2 bitstring result sets, got 1"):
        QdkEnergyEstimator._compute_energy_expectation_from_bitstrings(observables, bitstring_counts)


def test_calculate_energy_expval_variance_none_counts():
    """Test calculate_energy_expval_and_variance with None in bitstring_counts."""
    bitstring_counts = [None, {"0": 50, "1": 50}]
    observables = [QubitHamiltonian(["Z"], [1.0]), QubitHamiltonian(["X"], [1.0])]

    result = QdkEnergyEstimator._compute_energy_expectation_from_bitstrings(observables, bitstring_counts)

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
        # Should have one entry for the pauli group plus version field
        assert len(data) == 2
        assert "version" in data
        assert "0" in data
    finally:
        Path(temp_path).unlink()


def test_create_energy_estimator_qdk():
    """Test factory function for creating QDK energy estimator."""
    estimator = create("energy_estimator", "qdk")
    assert isinstance(estimator, QdkEnergyEstimator)


def test_estimator_fewer_shots():
    """Test estimator raises error when total shots is less than number of observables."""
    qasm = """
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    """
    circuit = Circuit(qasm=qasm)
    observable = [
        QubitHamiltonian(["ZZ"], np.array([2])),
        QubitHamiltonian(["XX"], np.array([3])),
        QubitHamiltonian(["YY"], np.array([4])),
    ]
    executor = create("circuit_executor", "qdk_full_state_simulator")
    estimator = QdkEnergyEstimator()
    with pytest.raises(ValueError, match=r"Total shots .* is less than the number of observables .*"):
        estimator.run(circuit, observable, executor, total_shots=1)


@pytest.mark.parametrize(
    "executor_name",
    [
        "qdk_full_state_simulator",
        pytest.param(
            "qiskit_aer_simulator",
            marks=pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT_AER, reason="Qiskit Aer not available"),
        ),
    ],
    ids=["qdk-full-state", "qiskit-aer"],
)
def test_estimator_run_4e4o(executor_name, hamiltonian_4e4o, wavefunction_4e4o, ref_energy_4e4o):
    """Functional test: energy estimation on the 4e4o ethylene problem with different circuit executors."""
    state_prep = create("state_prep", "sparse_isometry_gf2x")
    state_prep_circuit = state_prep.run(wavefunction_4e4o)
    filtered_hamiltonian, classical_coeffs = filter_and_group_pauli_ops_from_wavefunction(
        hamiltonian_4e4o, wavefunction_4e4o
    )
    executor = create("circuit_executor", executor_name)
    estimator = QdkEnergyEstimator()
    energy_result, _ = estimator.run(
        state_prep_circuit,
        filtered_hamiltonian,
        executor,
        total_shots=50000,
    )
    assert np.isclose(
        energy_result.energy_expectation_value + sum(classical_coeffs),
        ref_energy_4e4o,
        rtol=float_comparison_relative_tolerance,
        atol=estimator_energy_tolerance,
    )
