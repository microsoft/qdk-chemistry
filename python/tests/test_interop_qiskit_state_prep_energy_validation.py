"""Tests for validating state prep energy result and interoperability between QDK/Chemistry and Qiskit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from itertools import combinations

import numpy as np
import pytest
from qiskit import qasm3, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation.sparse_isometry import (
    SparseIsometryGF2XStatePreparation,
)
from qdk_chemistry.data import Circuit

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_energy_agreement_between_state_prep_methods(wavefunction_4e4o, hamiltonian_4e4o, ref_energy_4e4o):
    """Compare isometry method energy expectation values.

    Test whether sparse and regular isometry methods yield the same energy values.
    """
    # Create both state preparation instances
    basis_gates = ["cx", "rz", "ry", "rx", "h", "x", "z"]
    sparse_prep_gf2x = create(
        "state_prep", algorithm_name="sparse_isometry_gf2x", transpile_optimization_level=1, basis_gates=basis_gates
    )
    regular_prep = create(
        "state_prep", algorithm_name="qiskit_regular_isometry", transpile_optimization_level=1, basis_gates=basis_gates
    )

    sparse_gf2x_circuit = qasm3.loads(sparse_prep_gf2x.run(wavefunction_4e4o).get_qasm())
    regular_circuit = qasm3.loads(regular_prep.run(wavefunction_4e4o).get_qasm())

    # Create estimator and calculate energy for both circuits
    estimator = AerEstimator()

    sparse_gf2x_job = estimator.run([(sparse_gf2x_circuit, hamiltonian_4e4o.pauli_ops)])
    result = sparse_gf2x_job.result()
    sparse_gf2x_energy = result[0].data.evs

    regular_job = estimator.run([(regular_circuit, hamiltonian_4e4o.pauli_ops)])
    result = regular_job.result()
    regular_energy = result[0].data.evs

    energy_diff = []
    for energy_a, energy_b in combinations([ref_energy_4e4o, sparse_gf2x_energy, regular_energy], 2):
        energy_diff.append(abs(energy_a - energy_b))
    assert np.allclose(
        energy_diff, 0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    ), f"Energy difference {energy_diff} exceeds tolerance. "


def test_sparse_isometry_gf2x_energy_validation(wavefunction_10e6o, hamiltonian_10e6o, ref_energy_10e6o):
    """Test SparseIsometryGF2XStatePreparation energy validation for 10e6o F2."""
    # Create SparseIsometryGF2XStatePreparation instance for F2 test
    sparse_prep = create(
        "state_prep",
        algorithm_name="sparse_isometry_gf2x",
        basis_gates=["cx", "rz", "ry", "rx", "h", "x", "z"],
        transpile_optimization_level=1,
    )

    # Create circuit qasm and convert to QuantumCircuit
    circuit = qasm3.loads(sparse_prep.run(wavefunction_10e6o).get_qasm())

    # Calculate circuit energy using the estimator
    estimator = AerEstimator()
    job = estimator.run([(circuit, hamiltonian_10e6o.pauli_ops)])
    result = job.result()
    circuit_energy = result[0].data.evs
    # Basic validation: energy should be negative
    assert circuit_energy < 0, f"Circuit energy should be negative for electronic systems, got {circuit_energy:.6f}"
    # For exact 10e6o f2 case: circuit should match reference
    energy_diff = abs(circuit_energy - ref_energy_10e6o)
    assert np.isclose(
        energy_diff, 0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    ), (
        f"For 10e6o f2 wavefunction, circuit energy should match "
        f"reference energy. Got energy difference: {energy_diff:.8f} Hartree"
    )


def test_sparse_isometry_gf2x_circuit_efficiency(wavefunction_4e4o):
    """Compare isometry resource requirements.

    Test that SparseIsometryGF2XStatePrep creates more circuits using fewer resources
    than regular isometry.
    """
    # Create both state preparation instances
    basis_gates = ["cx", "rz", "ry", "rx", "h", "x", "z"]
    sparse_prep = create(
        "state_prep", algorithm_name="sparse_isometry_gf2x", transpile_optimization_level=1, basis_gates=basis_gates
    )
    regular_prep = create(
        "state_prep", algorithm_name="qiskit_regular_isometry", transpile_optimization_level=1, basis_gates=basis_gates
    )

    # Create circuits using both methods
    transpiled_sparse_circuit = qasm3.loads(sparse_prep.run(wavefunction_4e4o).get_qasm())
    transpiled_regular_circuit = qasm3.loads(regular_prep.run(wavefunction_4e4o).get_qasm())

    # Compare circuit metrics
    sparse_depth = transpiled_sparse_circuit.depth()
    regular_depth = transpiled_regular_circuit.depth()

    sparse_size = transpiled_sparse_circuit.size()
    regular_size = transpiled_regular_circuit.size()

    # Sparse isometry should be more efficient (fewer gates and lower depth)
    # Allow for cases where they might be equal if the problem is already very simple
    assert sparse_depth <= regular_depth, (
        f"Sparse isometry depth ({sparse_depth}) should be <= regular isometry depth ({regular_depth})"
    )
    assert sparse_size <= regular_size, (
        f"Sparse isometry size ({sparse_size}) should be <= regular isometry size ({regular_size})"
    )


def get_bitstring(circuit: Circuit) -> str:
    """Get the measurement result bitstring from a quantum circuit.

    Args:
        circuit: The quantum circuit to measure

    Returns:
        The measured bitstring (in Qiskit's little-endian format)

    """
    # Add measurements
    meas_circuit = qasm3.loads(circuit.get_qasm())
    meas_circuit.measure_all()

    # Simulate
    simulator = AerSimulator()
    compiled_circuit = transpile(meas_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=1).result()
    counts = result.get_counts()

    # There should be only one result with 100% probability
    assert len(counts) == 1, f"Expected deterministic state, got multiple results: {counts}"

    # Return the bitstring (in Qiskit's little-endian format)
    return next(iter(counts.keys()))


@pytest.mark.parametrize("bitstring", ["1010", "0000", "1111", "101001", "1", "0"])
def test_single_reference_state_basic(bitstring):
    """Test basic single reference state preparation with various bitstrings."""
    test_cls = SparseIsometryGF2XStatePreparation()
    circuit = test_cls._prepare_single_reference_state(bitstring)
    result_bitstring = get_bitstring(circuit)
    assert result_bitstring == bitstring, f"Expected {bitstring}, got {result_bitstring}"
