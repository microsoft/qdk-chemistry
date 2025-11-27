"""Energy Estimator using QDK simulator."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging
from collections import Counter

from qsharp import BitFlipNoise, DepolarizingNoise, PauliNoise, PhaseFlipNoise
from qsharp.openqasm import run

from qdk_chemistry.algorithms.energy_estimator.energy_estimator import (
    EnergyEstimator,
    compute_energy_expectation_from_bitstrings,
    create_measurement_circuits,
)
from qdk_chemistry.data import EnergyExpectationResult, MeasurementData, QubitHamiltonian

_LOGGER = logging.getLogger(__name__)


class QDKEnergyEstimator(EnergyEstimator):
    """Energy Estimator to estimate expectation values of quantum circuits with respect to a given observable.

    This class uses a QDK base simulator backend to run quantum circuits and estimate
    the expectation values of qubit Hamiltonians. It supports optional noise models for noise simulation.
    """

    def __init__(
        self,
        seed: int = 42,
        noise_model: DepolarizingNoise | BitFlipNoise | PauliNoise | PhaseFlipNoise | None = None,
        qubit_loss: float = 0.0,
    ):
        """Initialize the Estimator with a backend and optional transpilation settings.

        Args:
            seed: Random seed for reproducibility.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            qubit_loss: Probability of qubit loss in simulation.

        """
        super().__init__()
        self.seed = seed
        self.noise_model = noise_model
        self.qubit_loss = qubit_loss

    def _run_measurement_circuits_and_get_bitstring_counts(
        self,
        measurement_circuits: list[str],
        shots_list: list[int],
    ) -> list[dict[str, int]]:
        """Run the measurement circuits and return the bitstring counts.

        Args:
            measurement_circuits: A list of measurement circuits in OpenQASM3 format to run.
            shots_list: A list of shots allocated for each measurement circuit.

        Returns:
            A list of dictionaries containing the bitstring counts for each measurement circuit.

        """
        all_bitstring_counts: list[dict[str, int]] = []
        for circuit_qasm, shots in zip(measurement_circuits, shots_list, strict=True):
            result = run(
                circuit_qasm,
                shots=shots,
                noise=self.noise_model,
                qubit_loss=self.qubit_loss,
                as_bitstring=True,
                seed=self.seed,
            )
            bitstring_count = {
                bitstring[::-1]: count for bitstring, count in Counter(result).items()
            }  # Reverse bitstrings to match Little-Endian convention
            all_bitstring_counts.append(bitstring_count)
        return all_bitstring_counts

    def _get_measurement_data(
        self,
        measurement_circuits: list[str],
        qubit_hamiltonians: list[QubitHamiltonian],
        shots_list: list[int],
    ) -> MeasurementData:
        """Get ``MeasurementData`` from running measurement circuits.

        Args:
            measurement_circuits: A list of measurement circuits to run.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to be evaluated.
            shots_list: A list of shots allocated for each measurement circuit.

        Returns:
            ``MeasurementData`` containing the measurement counts and Hamiltonian data.

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
            qubit_hamiltonians: List of ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across the observable terms.
            classical_coeffs: Optional list of coefficients for classical Pauli terms to calculate energy offset.

        Returns:
            ``EnergyExpectationResult`` containing the energy expectation value and variance.

        Note:
            * Measurement circuits are generated for each QubitHamiltonian term.
            * Parameterized circuits are not supported.
            * Only one circuit is supported per run.

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

        # Create measurement circuits
        measurement_circuits_qasm = create_measurement_circuits(
            circuit_qasm=circuit_qasm,
            grouped_hamiltonians=qubit_hamiltonians,
        )

        measurement_data = self._get_measurement_data(
            measurement_circuits=measurement_circuits_qasm,
            qubit_hamiltonians=qubit_hamiltonians,
            shots_list=shots_list,
        )

        return compute_energy_expectation_from_bitstrings(
            qubit_hamiltonians, measurement_data.bitstring_counts, energy_offset
        ), measurement_data

    def name(self) -> str:
        """Get the name of the estimator for registry purposes."""
        return "qdk_base_simulator"
