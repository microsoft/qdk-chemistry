"""QDK/Chemistry energy estimator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
)

__all__: list[str] = ["EnergyEstimator", "EnergyEstimatorFactory"]


class EnergyEstimatorSettings(Settings):
    """Settings for EnergyEstimator algorithms."""

    def __init__(self):
        """Initialize the EnergyEstimatorSettings."""
        super().__init__()
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
            "Circuit executor used to run quantum circuits for energy estimation.",
        )


class EnergyEstimator(Algorithm):
    """Abstract base class for energy estimator algorithms."""

    def __init__(self):
        """Initialize the EnergyEstimator."""
        super().__init__()
        self._settings = EnergyEstimatorSettings()

    def type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    @abstractmethod
    def _run_impl(
        self,
        circuit: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        total_shots: int,
        noise_model: QuantumErrorProfile | None = None,
        device_backend_name: str | None = None,
        pre_transpilation_passes: list[str] | None = None,
        post_transpilation_passes: list[str] | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of the Hamiltonian.

        The circuit executor used to run quantum circuits is configured via
        the ``circuit_executor`` setting (an ``AlgorithmRef``).

        Args:
            circuit: Circuit.
            qubit_hamiltonian: ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across the observable terms.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            device_backend_name: Optional device backend name string to pass to the circuit executor.
            pre_transpilation_passes: Optional list of pass names to apply before transpilation.
            post_transpilation_passes: Optional list of pass names to apply after transpilation.

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
