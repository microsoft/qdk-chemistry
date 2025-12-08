"""Energy Estimator using QDK simulator."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import Counter

import qsharp
from qsharp.openqasm import run

from qdk_chemistry.algorithms.energy_estimator.energy_estimator import EnergyEstimator
from qdk_chemistry.data import Circuit, EnergyExpectationResult, MeasurementData, QubitHamiltonian, Settings
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class QDKEnergyEstimatorSettings(Settings):
    """Settings configuration for a QDKEnergyEstimator.

    QDKEnergyEstimator-specific settings:
        seed (int, default=42): Random seed for reproducibility.
        qubit_loss (float, default=0.0): Probability of qubit loss in simulation.

    """

    def __init__(self):
        """Initialize QDKEnergyEstimatorSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("seed", "int", 42)
        self._set_default("qubit_loss", "double", 0.0)


class QDKEnergyEstimator(EnergyEstimator):
    """Energy Estimator to estimate expectation values of quantum circuits with respect to a given observable.

    This class uses a QDK base simulator backend to run quantum circuits and estimate
    the expectation values of qubit Hamiltonians. It supports optional noise models for noise simulation.
    """

    def __init__(
        self,
        seed: int = 42,
        qubit_loss: float = 0.0,
    ):
        """Initialize the Estimator with optional settings.

        Args:
            seed: Random seed for reproducibility.
            qubit_loss: Probability of qubit loss in simulation.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QDKEnergyEstimatorSettings()
        self._settings.set("seed", seed)
        self._settings.set("qubit_loss", qubit_loss)

    def _run_measurement_circuits_and_get_bitstring_counts(
        self,
        measurement_circuits: list[Circuit],
        shots_list: list[int],
        noise_model: qsharp.DepolarizingNoise
        | qsharp.BitFlipNoise
        | qsharp.PauliNoise
        | qsharp.PhaseFlipNoise
        | None = None,
    ) -> list[dict[str, int]]:
        """Run the measurement circuits and return the bitstring counts.

        Args:
            measurement_circuits: A list of Circuits that provide measurement circuits in OpenQASM3 format to run.
            shots_list: A list of shots allocated for each measurement circuit.
            noise_model: Optional noise model to simulate noise in the quantum circuit.

        Returns:
            A list of dictionaries containing the bitstring counts for each measurement circuit.

        """
        all_bitstring_counts: list[dict[str, int]] = []
        for circuit, shots in zip(measurement_circuits, shots_list, strict=True):
            result = run(
                circuit.get_qasm(),
                shots=shots,
                noise=noise_model,
                qubit_loss=self._settings.qubit_loss,
                as_bitstring=True,
                seed=self._settings.seed,
            )
            bitstring_count = {
                bitstring[::-1]: count for bitstring, count in Counter(result).items()
            }  # Reverse bitstrings to match Little-Endian convention
            all_bitstring_counts.append(bitstring_count)
        return all_bitstring_counts

    def _get_measurement_data(
        self,
        measurement_circuits: list[Circuit],
        qubit_hamiltonians: list[QubitHamiltonian],
        shots_list: list[int],
        noise_model: qsharp.DepolarizingNoise
        | qsharp.BitFlipNoise
        | qsharp.PauliNoise
        | qsharp.PhaseFlipNoise
        | None = None,
    ) -> MeasurementData:
        """Get ``MeasurementData`` from running measurement circuits.

        Args:
            measurement_circuits: A list of measurement circuits to run.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to be evaluated.
            shots_list: A list of shots allocated for each measurement circuit.
            noise_model: Optional noise model to simulate noise in the quantum circuit.

        Returns:
            MeasurementData: Measurement counts paired with their corresponding ``QubitHamiltonian`` objects.

        """
        counts = self._run_measurement_circuits_and_get_bitstring_counts(measurement_circuits, shots_list, noise_model)
        return MeasurementData(bitstring_counts=counts, hamiltonians=qubit_hamiltonians, shots_list=shots_list)

    def _run_impl(
        self,
        circuit: Circuit,
        qubit_hamiltonians: list[QubitHamiltonian],
        total_shots: int,
        noise_model: qsharp.DepolarizingNoise
        | qsharp.BitFlipNoise
        | qsharp.PauliNoise
        | qsharp.PhaseFlipNoise
        | None = None,
        classical_coeffs: list | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of Hamiltonians.

        Args:
            circuit: Circuit that provides an OpenQASM3 string of the quantum circuit to be evaluated.
            qubit_hamiltonians: List of ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across the observable terms.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            classical_coeffs: Optional list of coefficients for classical Pauli terms to calculate energy offset.

        Returns:
            tuple[EnergyExpectationResult, MeasurementData]: Tuple containing:

                * ``energy_result``: Energy expectation value and variance for the provided Hamiltonians.
                * ``measurement_data``: Raw measurement counts and metadata used to compute the expectation value.

        Note:
            * Measurement circuits are generated for each QubitHamiltonian term.
            * Parameterized circuits are not supported.
            * Only one circuit is supported per run.

        """
        Logger.trace_entering()
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

        # Create measurement circuits
        measurement_circuits = self._create_measurement_circuits(
            circuit=circuit,
            grouped_hamiltonians=qubit_hamiltonians,
        )

        measurement_data = self._get_measurement_data(
            measurement_circuits=measurement_circuits,
            qubit_hamiltonians=qubit_hamiltonians,
            shots_list=shots_list,
            noise_model=noise_model,
        )

        return self._compute_energy_expectation_from_bitstrings(
            qubit_hamiltonians, measurement_data.bitstring_counts, energy_offset
        ), measurement_data

    def name(self) -> str:
        """Get the name of the estimator for registry purposes."""
        return "qdk_base_simulator"
