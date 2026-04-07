"""QDK/Chemistry energy estimator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms import CircuitExecutor
from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
)

__all__: list[str] = ["EnergyEstimator", "EnergyEstimatorFactory"]


class EnergyEstimator(Algorithm):
    """Abstract base class for energy estimator algorithms."""

    def __init__(self):
        """Initialize the EnergyEstimator."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    @abstractmethod
    def _run_impl(
        self,
        circuit: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        circuit_executor: CircuitExecutor,
        total_shots: int,
        noise_model: QuantumErrorProfile | None = None,
        device_backend_name: str | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of the Hamiltonian.

        Args:
            circuit: Circuit.
            qubit_hamiltonian: ``QubitHamiltonian`` to estimate.
            circuit_executor: An instance of ``CircuitExecutor`` to run quantum circuits.
            total_shots: Total number of shots to allocate across the observable terms.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            device_backend_name: Optional device backend name string to pass to the circuit executor.

        Returns:
            tuple[EnergyExpectationResult, MeasurementData]: Tuple containing:

                * ``energy_result``: Energy expectation value and variance for the provided Hamiltonian.
                * ``measurement_data``: Raw measurement counts and metadata used to compute the expectation value.

        """


class EnergyEstimatorFactory(AlgorithmFactory):
    """Factory class for creating EnergyEstimator instances."""

    def algorithm_type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    def default_algorithm_name(self) -> str:
        """Return ``qdk`` as the default algorithm name."""
        return "qdk"
