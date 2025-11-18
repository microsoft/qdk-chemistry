"""Tests for energy estimator in QDK/Chemistry qsharp interop."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qsharp import BitFlipNoise, DepolarizingNoise, PauliNoise

from qdk_chemistry.algorithms.energy_estimator import QDKEnergyEstimator
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian, filter_and_group_pauli_ops_from_wavefunction

from .reference_tolerances import (
    estimator_energy_tolerance,
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)


@pytest.fixture
def circuit_4e4o(test_data_files_path):
    """Fixture to create the test circuit for 4e4o ethylene problem."""
    with open(test_data_files_path / "4e4o-ethylene_2det-can-7967f80e_2_1.qasm") as f:
        return f.read()


class TestQSharpEnergyEstimator:
    """Tests for QSharp Energy Estimator."""

    def test_estimator_initialize(self):
        """Test initialization of EnergyEstimator."""
        estimator = QDKEnergyEstimator()
        assert estimator.seed == 42
        assert estimator.noise_model is None
        assert np.isclose(
            estimator.qubit_loss,
            0.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_estimator_run(self, circuit_4e4o, wavefunction_4e4o, hamiltonian_4e4o, ref_energy_4e4o):
        """Functional test for expectation value calculation using Estimator.

        4e4o ethylene problem.
        """
        filtered_hamiltonian, classical_coeffs = filter_and_group_pauli_ops_from_wavefunction(
            hamiltonian_4e4o, wavefunction_4e4o
        )

        estimator = QDKEnergyEstimator()
        energy_expectations, _ = estimator.run(
            circuit_4e4o,
            filtered_hamiltonian,
            total_shots=10000,
            classical_coeffs=classical_coeffs,
        )

        # Limited shots within 1 mHartree
        assert np.isclose(
            energy_expectations.energy_expectation_value,
            ref_energy_4e4o,
            rtol=float_comparison_relative_tolerance,
            atol=estimator_energy_tolerance,
        )

    def test_estimator_fewer_shots(self):
        """Test estimator raises error when total shots less than number of observables."""
        circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """

        simple_observable = [
            QubitHamiltonian(["ZZ"], np.array([2])),
            QubitHamiltonian(["XX"], np.array([3])),
            QubitHamiltonian(["YY"], np.array([4])),
        ]

        estimator = QDKEnergyEstimator()
        with pytest.raises(ValueError, match=r"Total shots .* is less than the number of observables .*"):
            estimator.run(circuit_qasm, simple_observable, total_shots=1)  # Only 1 shot for 3 observables

    def test_simulate_with_noise(self):
        """Test estimator with different noise models."""
        circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        x q[0];
        """

        simple_observable = [QubitHamiltonian(["ZZ"], np.array([1]))]
        expected_expectation_noiseless = -1.0  # expected value is -1

        # Test without noise
        noiseless_estimator = QDKEnergyEstimator()
        noiseless_results, _ = noiseless_estimator.run(circuit_qasm, simple_observable, total_shots=10000)

        # Verify noiseless case is close to theoretical value
        noiseless_error = abs(noiseless_results.energy_expectation_value - expected_expectation_noiseless)
        assert np.isclose(
            noiseless_error, 0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )  # the bitstring for this test circuit is deterministic in noiseless case

        qubit_loss_estimator = QDKEnergyEstimator(qubit_loss=0.05)
        assert np.isclose(
            qubit_loss_estimator.qubit_loss,
            0.05,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        qubit_loss_results, _ = qubit_loss_estimator.run(circuit_qasm, simple_observable, total_shots=10000)
        qubit_loss_error = abs(qubit_loss_results.energy_expectation_value - expected_expectation_noiseless)
        assert qubit_loss_error > noiseless_error

        # Test with different noise models
        noise_models = [
            BitFlipNoise(0.1),
            DepolarizingNoise(0.05),
            PauliNoise(x=0.05, y=0.05, z=0.05),
        ]

        for noise_model in noise_models:
            noisy_estimator = QDKEnergyEstimator(noise_model=noise_model)
            noisy_results, _ = noisy_estimator.run(circuit_qasm, simple_observable, total_shots=10000)
            noisy_error = abs(noisy_results.energy_expectation_value - expected_expectation_noiseless)

            # Noise should increase the error (with some tolerance for statistical fluctuations)
            assert noisy_error > noiseless_error

    def test_save_measurement_data(self, tmp_path):
        """Test saving measurement data to JSON."""
        circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """

        simple_observable = [QubitHamiltonian(["ZZ"], np.array([1])), QubitHamiltonian(["IX"], np.array([1]))]

        estimator = QDKEnergyEstimator()
        _, measurement_data = estimator.run(circuit_qasm, simple_observable, total_shots=1000)

        json_file = tmp_path / "test.measurement_data.json"
        measurement_data.to_json_file(json_file)

        # Verify the file was created
        assert json_file.exists()
