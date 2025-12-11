"""QDK/Chemistry energy estimator module.

This module defines a custom `EnergyEstimator` class for evaluating expectation values of quantum circuits
with respect to Hamiltonian.
The estimator leverages Qiskit :cite:`Javadi-Abhari2024` backends to execute quantum circuits and collect
bitstring outcomes.

Key Features:
    - Accepts a quantum circuit (as a Circuit) and observables (as a list of QubitHamiltonian).
    - Generates measurement circuits for each observable term.
    - Executes measurement circuits on a simulator backend with a specified number of shots.
    - Collects bitstring counts and computes expectation values and variances.
    - Supports noise simulations and classical error analysis.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit import qasm3, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

if TYPE_CHECKING:
    import qiskit
    from qiskit.circuit import QuantumCircuit

from qdk_chemistry.algorithms.energy_estimator import (
    EnergyEstimator,
)
from qdk_chemistry.data import Circuit, EnergyExpectationResult, MeasurementData, QubitHamiltonian, Settings
from qdk_chemistry.utils import Logger

__all__ = ["QiskitEnergyEstimator", "QiskitEnergyEstimatorSettings"]


class QiskitEnergyEstimatorSettings(Settings):
    """Settings configuration for a QiskitEnergyEstimator.

    QiskitEnergyEstimator-specific settings:
        seed (int, default=42): Random seed for reproducibility.

    """

    def __init__(self):
        """Initialize QiskitEnergyEstimatorSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("seed", "int", 42)


class QiskitEnergyEstimator(EnergyEstimator):
    """Custom Estimator to estimate expectation values of quantum circuits with respect to a given observable."""

    def __init__(
        self,
        seed: int = 42,
        backend_options: dict[str, Any] | None = None,
        backend: qiskit.providers.backend.BackendV2 | None = None,
    ):
        """Initialize the Estimator with a backend and optional transpilation settings.

        Args:
            seed: Seed for the simulator to ensure reproducibility. Default is 42.
                This argument takes priority over the seed specified in the Backend configuration/options.
            backend_options: Backend-specific configuration dictionary for Qiskit AerSimulator.
                Frequently used options include ``{"seed_simulator": int, "noise_model": NoiseModel, ...}``.
                The backend option "seed_simulator" is overwritten by seed.
                The backend option "noise_model" is overwritten when a noise model is provided as argument in the
                ``run`` function.
                This keyword argument is not compatible with qdk_chemistry.algorithms.create.
            backend: Backend simulator to run circuits. Defaults to Qiskit AerSimulator.
                ``backend`` and ``backend_options`` are mutually exclusive; provide only one.
                This keyword argument is not compatible with qdk_chemistry.algorithms.create.

        References: `Qiskit Aer Simulator <https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py>`_.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitEnergyEstimatorSettings()
        self._settings.set("seed", seed)

        if backend is None:
            if backend_options is None:
                backend_options = {}
            backend_options["seed_simulator"] = self._settings.seed
            self.backend = AerSimulator(**backend_options)
        else:
            if backend_options is not None:
                raise ValueError("backend and backend_options are mutually exclusive; provide only one.")
            self.backend = backend
            # Reset the seed in the backend if applicable
            if isinstance(self.backend, AerSimulator):
                self.backend.set_options(seed_simulator=self._settings.seed)

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
        Logger.trace_entering()
        bitstring_counts: list[dict[str, int] | None] = []
        for i, meas_circuit in enumerate(measurement_circuits):
            shots = shots_list[i]
            Logger.debug(f"Running backend with circuit {i} and {shots} shots")
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
        Logger.trace_entering()
        counts = self._run_measurement_circuits_and_get_bitstring_counts(measurement_circuits, shots_list)
        return MeasurementData(bitstring_counts=counts, hamiltonians=qubit_hamiltonians, shots_list=shots_list)

    def _run_impl(
        self,
        circuit: Circuit,
        qubit_hamiltonians: list[QubitHamiltonian],
        total_shots: int,
        noise_model: NoiseModel | None = None,
        classical_coeffs: list | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of Hamiltonians.

        Args:
            circuit: Circuit that provides an OpenQASM3 string of the quantum circuit to be evaluated.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across Hamiltonian terms.
            noise_model: NoiseModel to be used in simulation, and the circuit will be transpiled into the basis gates
                defined by the noise model.
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
        Logger.trace_entering()
        if noise_model is not None:
            if isinstance(self.backend, AerSimulator):
                self.backend.set_options(noise_model=noise_model)
            else:
                raise NotImplementedError("A noise model can only be set for an AerSimulator.")

        num_observables = len(qubit_hamiltonians)
        if total_shots < num_observables:
            raise ValueError(
                f"Total shots {total_shots} is less than the number of observables {num_observables}. "
                "Please increase total shots to ensure each observable is measured."
            )

        # Evenly distribute shots across all observables
        shots_list = [total_shots // num_observables] * num_observables
        Logger.debug(f"Shots allocated: {shots_list}")

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
        measurement_circuits = self._create_measurement_circuits(
            circuit=circuit,
            grouped_hamiltonians=qubit_hamiltonians,
        )

        # Load and optionally transpile circuits into basis gates defined by noise model
        measurement_circuits_qiskit = []
        for measurement_circuit in measurement_circuits:
            circuit = qasm3.loads(measurement_circuit.get_qasm())
            if basis_gates is not None:
                circuit = transpile(circuit, basis_gates=basis_gates)
            measurement_circuits_qiskit.append(circuit)

        measurement_data = self._get_measurement_data(
            measurement_circuits=measurement_circuits_qiskit,
            qubit_hamiltonians=qubit_hamiltonians,
            shots_list=shots_list,
        )

        return self._compute_energy_expectation_from_bitstrings(
            qubit_hamiltonians, measurement_data.bitstring_counts, energy_offset
        ), measurement_data

    def name(self) -> str:
        """Get the name of the estimator backend."""
        Logger.trace_entering()
        return "qiskit_aer_simulator"
