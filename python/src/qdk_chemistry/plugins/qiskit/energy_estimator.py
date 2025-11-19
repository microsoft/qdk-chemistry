"""QDK/Chemistry energy estimator module.

This module defines a custom `EnergyEstimator` class for evaluating expectation values of quantum circuits
with respect to Hamiltonian. The estimator leverages Qiskit backends to execute quantum circuits and
collect bitstring outcomes.

Key Features:
    - Accepts a quantum circuit (as a QASM string) and observables (as a list of QubitHamiltonian).
    - Generates measurement circuits for each observable term.
    - Executes measurement circuits on a simulator backend with a specified number of shots.
    - Collects bitstring counts and computes expectation values and variances.
    - Supports noise simulations and classical error analysis.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging
from typing import Any

import numpy as np
from qiskit import qasm3, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qdk_chemistry.algorithms import register
from qdk_chemistry.algorithms.energy_estimator import (
    EnergyEstimator,
    compute_energy_expectation_from_bitstrings,
    create_measurement_circuits,
)
from qdk_chemistry.data import EnergyExpectationResult, MeasurementData, QubitHamiltonian

_LOGGER = logging.getLogger(__name__)


class QiskitEnergyEstimator(EnergyEstimator):
    """Custom Estimator to estimate expectation values of quantum circuits with respect to a given observable."""

    def __init__(self, backend: BackendV2 | None = None, seed: int = 42):
        """Initialize the Estimator with a backend and optional transpilation settings.

        Args:
            backend: Backend simulator to run circuits. Default to use qiskit AerSimulator.
            seed: Seed for the simulator to ensure reproducibility. Default is 42.
                This argument takes priority over the seed specified in the Backend configuration.

        """
        super().__init__()
        if backend is None:
            self.backend = AerSimulator(seed_simulator=seed)
        else:
            self.backend = backend
            # Reset the seed in the backend if applicable
            if isinstance(self.backend, AerSimulator):
                self.backend.set_options(seed_simulator=seed)

    @classmethod
    def from_backend_options(cls, seed: int = 42, backend_options: dict[str, Any] | None = None) -> "EnergyEstimator":
        """Create an EnergyEstimator from specified backend options for Aer simulator.

        Args:
            seed: Seed for the simulator to ensure reproducibility. Default is 42.
            This argument takes priority over the seed specified in the Backend configuration/options.
            backend_options: Backend-specific configuration dictionary. Frequently used options include
            ``{"seed_simulator": int, "noise_model": NoiseModel, ...}``

        References: `Qiskit Aer Simulator <https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py>`_.

        """
        if backend_options is None:
            backend_options = {}
        backend = AerSimulator(**backend_options)
        return cls(backend=backend, seed=seed)

    def _run_measurement_circuits_and_get_bitstring_counts(
        self, measurement_circuits: list[QuantumCircuit], shots_list: list[int]
    ) -> list[dict[str, int] | None]:
        """Run the measurement circuits and return the bitstring counts.

        Args:
            measurement_circuits: list of measurement circuits to run.
            shots_list: list of shots allocated for each measurement circuit.

        Returns:
            list of dictionaries containing the bitstring counts for each measurement circuit.
            A list of dictionaries containing the bitstring counts for each measurement circuit.

        """
        bitstring_counts: list[dict[str, int] | None] = []
        for i, meas_circuit in enumerate(measurement_circuits):
            shots = shots_list[i]
            _LOGGER.debug(f"Running backend with circuit {i} and {shots} shots")
            result = self.backend.run(meas_circuit, shots=shots).result().results[0].data.counts
            bitstring_counts.append(result)
        return bitstring_counts

    def _get_measurement_data(
        self,
        measurement_circuits: list[QuantumCircuit],
        qubit_hamiltonians: list[QubitHamiltonian],
        shots_list: list[int],
    ) -> MeasurementData:
        """Get measurement data objects from running measurement circuits.

        Args:
            measurement_circuits: A list of measurement circuits to run.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to be measured.
            shots_list: A list of shots allocated for each measurement circuit.

        Returns:
            ``MeasurementData`` containing the measurement counts and observable data.

        """
        counts = self._run_measurement_circuits_and_get_bitstring_counts(measurement_circuits, shots_list)
        return MeasurementData(bitstring_counts=counts, hamiltonians=qubit_hamiltonians, shots_list=shots_list)

    def _run_impl(
        self,
        circuit_qasm: str,
        qubit_hamiltonians: list[QubitHamiltonian],
        total_shots: int,
        classical_coeffs: list | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of Hamiltonians.

        Args:
            circuit_qasm: OpenQASM3 string of the quantum circuit to be evaluated.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across Hamiltonian terms.
            classical_coeffs: Optional list of coefficients for classical Pauli terms to calculate energy offset.

        Returns:
            A dictionary containing the energy expectation value, variance, and per-observable expectation values
            and variances.

        ... note::
            - Measurement circuits are generated for each observable term.
            - Parameterized circuits are not supported.
            - Only one circuit is supported per run.
            - If NoiseModel is provided in the backend options, it will be used in simulation,
                and the circuit will be transpiled into the basis gates defined by the noise model.

        """
        num_observables = len(qubit_hamiltonians)
        if total_shots < num_observables:
            raise ValueError(
                f"Total shots {total_shots} is less than the number of observables {num_observables}. "
                "Please increase total shots to ensure each observable is measured."
            )

        # Evenly distribute shots across all observables
        shots_list = [total_shots // num_observables] * num_observables
        _LOGGER.debug(f"Shots allocated: {shots_list}")

        energy_offset = sum(classical_coeffs) if classical_coeffs else 0.0

        # Check once for basis gates
        basis_gates = None
        if (
            isinstance(self.backend, AerSimulator)
            and hasattr(self.backend.options, "noise_model")
            and isinstance(self.backend.options.noise_model, NoiseModel)
        ):
            basis_gates = self.backend.options.noise_model.basis_gates

        # Create measurement circuits
        measurement_circuits_qasm = create_measurement_circuits(
            circuit_qasm=circuit_qasm,
            grouped_hamiltonians=qubit_hamiltonians,
        )

        # Load and optionally transpile circuits into basis gates defined by noise model
        measurement_circuits = []
        for qasm in measurement_circuits_qasm:
            circuit = qasm3.loads(qasm)
            if basis_gates is not None:
                circuit = transpile(circuit, basis_gates=basis_gates)
            measurement_circuits.append(circuit)

        measurement_data = self._get_measurement_data(
            measurement_circuits=measurement_circuits,
            qubit_hamiltonians=qubit_hamiltonians,
            shots_list=shots_list,
        )

        return compute_energy_expectation_from_bitstrings(
            qubit_hamiltonians, measurement_data.bitstring_counts, energy_offset
        ), measurement_data

    def name(self) -> str:
        """Get the name of the estimator backend."""
        return "qiskit_aer_simulator"


register(lambda: QiskitEnergyEstimator())

if __name__ == "__main__":
    """Example usage of the Estimator from different backends."""
    from qiskit import qasm3
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    logging.basicConfig(level=logging.WARNING)
    _LOGGER = logging.getLogger(__name__)
    _LOGGER.setLevel(logging.INFO)

    circuit_qasm = """
        include "stdgates.inc";
        qubit[2] q;
        rz(pi) q[0];
        x q[0];
        cx q[0], q[1];
        """
    qubit_hamiltonians = [QubitHamiltonian(["ZZ"], np.array([1.0]))]

    # Example usage: qiskit aer simulator
    estimator = QiskitEnergyEstimator()
    results = estimator.run(circuit_qasm, qubit_hamiltonians, total_shots=1000)
    _LOGGER.info(f"Energy expectation value from AerSimulator: {results['energy_expectation_value']}")

    # Example usage: qiskit aer simulator with noise model
    noise_model = NoiseModel(basis_gates=["rz", "sx", "cx", "measure"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rz", "sx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.007, 2), ["cx"])
    backend_options = {"noise_model": noise_model}  # add noise
    estimator = QiskitEnergyEstimator.from_backend_options(backend_options=backend_options)
    results = estimator.run(circuit_qasm, qubit_hamiltonians, total_shots=1000)
    _LOGGER.info(f"Energy expectation value from AerSimulator with noise: {results['energy_expectation_value']}")
