"""Tests for energy estimator in QDK/Chemistry qiskit interop.

Test functionality related to measurement circuit generation
and expectation value calculations for quantum circuits and observables.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian, filter_and_group_pauli_ops_from_wavefunction
from qdk_chemistry.plugins.qiskit.energy_estimator import QiskitEnergyEstimator

from .reference_tolerances import estimator_energy_tolerance, float_comparison_relative_tolerance


@pytest.fixture
def circuit_4e4o(test_data_files_path):
    """Fixture to create the test circuit for 4e4o ethylene problem."""
    with open(test_data_files_path / "4e4o-ethylene_2det-can-7967f80e_2_1.qasm") as f:
        return f.read()


def test_estimator_initialize():
    """Test initialization of EnergyEstimator."""
    estimator = QiskitEnergyEstimator()
    assert isinstance(estimator.backend, AerSimulator)

    qiskit_aer_backend = AerSimulator()
    estimator_aer = QiskitEnergyEstimator(backend=qiskit_aer_backend)
    assert isinstance(estimator_aer.backend, AerSimulator)
    assert estimator_aer.backend.options.seed_simulator == 42


def test_estimator_from_backend():
    """Test EnergyEstimator from backend."""
    # Qiskit AerSimulator
    estimator = QiskitEnergyEstimator.from_backend_options(backend_options={"shots": 5000})
    assert isinstance(estimator.backend, AerSimulator)
    assert estimator.backend.options.seed_simulator == 42
    assert estimator.backend.options.shots == 5000

    # No options
    estimator = QiskitEnergyEstimator.from_backend_options()
    assert isinstance(estimator.backend, AerSimulator)
    assert estimator.backend.options.shots == 1024  # Default
    assert estimator.backend.options.seed_simulator == 42  # Default


def test_estimator_run():
    """Functional test for expectation value calculation using Estimator.

    Bell state test.
    """
    circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """

    simple_observable = [
        QubitHamiltonian(["ZI", "IZ", "ZZ"], np.array([0.2, 0.3, 0.4])),
        QubitHamiltonian(["XX"], np.array([0.5])),
    ]
    estimator = QiskitEnergyEstimator.from_backend_options(backend_options={"seed_simulator": 42})
    results, _ = estimator.run(circuit_qasm, simple_observable, 100000)
    # For Bell state, <ZI>=0, <IZ>=0, <ZZ>=1, <XX>=1
    # So expected value = 0.2*0 + 0.3*0 + 0.4*1 + 0.5*1 = 0.9
    expected = [[0.0, 0.0, 1.0], [1.0]]
    assert all(
        np.allclose(e, a, rtol=float_comparison_relative_tolerance, atol=estimator_energy_tolerance)
        for e, a in zip(expected, results.expvals_each_term, strict=True)
    )
    assert np.isclose(
        results.energy_expectation_value,
        0.9,
        rtol=float_comparison_relative_tolerance,
        atol=estimator_energy_tolerance,
    )
    assert np.less(results.energy_variance, 1e-5)


def test_estimator_run_with_noise_model():
    """Test EnergyEstimator.run with a noise model."""
    circuit_qasm = """
        include "stdgates.inc";
        qubit[3] q;
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        x q[2];
        """

    observable = [QubitHamiltonian(["IZZ"], np.array([1.0]))]

    # Create a simple depolarizing noise model
    noise_model = NoiseModel()  # Default basis gates are ['id', 'rz', 'sx', 'cx']
    error = depolarizing_error(0.05, 2)
    noise_model.add_all_qubit_quantum_error(error, ["cx"])

    estimator = QiskitEnergyEstimator.from_backend_options(
        backend_options={"seed_simulator": 42, "noise_model": noise_model}
    )
    estimator.run(circuit_qasm, observable, total_shots=10)


def test_estimator_for_4e4o_2det_problem(hamiltonian_4e4o, wavefunction_4e4o, circuit_4e4o, ref_energy_4e4o):
    """Functional test for expectation value from Estimator for demo problem 4e4o-ethylene_2det-can-7967f80e.

    Reference energy is -4.023112557011, noiseless test with 100000 shots should return
    energy expectation value within chemical accuracy 0.001.
    """
    filtered_hamiltonian, classical_coeffs = filter_and_group_pauli_ops_from_wavefunction(
        hamiltonian_4e4o, wavefunction_4e4o
    )
    backend = AerSimulator(seed_simulator=42)
    estimator = QiskitEnergyEstimator(backend=backend)
    results, _ = estimator.run(
        circuit_4e4o,
        filtered_hamiltonian,
        total_shots=100000,
        classical_coeffs=classical_coeffs,
    )
    assert np.isclose(
        results.energy_expectation_value,
        ref_energy_4e4o,
        rtol=float_comparison_relative_tolerance,
        atol=estimator_energy_tolerance,
    )
    assert np.less(results.energy_variance, 1e-5)


def test_energy_estimator_with_fewer_shots_than_observables():
    """Test EnergyEstimator.run with fewer total shots than observables."""
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
    estimator = QiskitEnergyEstimator.from_backend_options(backend_options={"seed_simulator": 42})

    with pytest.raises(ValueError, match=r"Total shots .* is less than the number of observables .*"):
        estimator.run(circuit_qasm, simple_observable, total_shots=2)  # Only 2 shots for 3 observables


def test_estimator_save_results_to_json(tmp_path):
    """Test saving measurement results to JSON file."""
    estimator = QiskitEnergyEstimator()

    # Run a simple estimation first
    circuit_qasm = """
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
    obs = [QubitHamiltonian(["Z"], np.array([1.0]))]
    _, measurement_data = estimator.run(circuit_qasm, obs, total_shots=100)
    json_file = tmp_path / "test.measurement_data.json"
    measurement_data.to_json_file(json_file)

    # Verify the file was created
    assert json_file.exists()
