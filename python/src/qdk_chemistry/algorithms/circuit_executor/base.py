"""QDK/Chemistry circuit executor abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, CircuitExecutorData

__all__: list[str] = []


class CircuitExecutor(Algorithm):
    """Abstract base class for circuit executor algorithms."""

    def __init__(self):
        """Initialize the CircuitExecutor with default settings."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    @abstractmethod
    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
    ) -> CircuitExecutorData:
        """Prepare a quantum circuit that encodes the given wavefunction.

        Args:
            circuit (Circuit): The circuit that prepares the initial state.
            shots (int): The number of shots to execute the circuit.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """


class CircuitExecutorFactory(AlgorithmFactory):
    """Factory class for creating CircuitExecutor instances."""

    def __init__(self):
        """Initialize the CircuitExecutorFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as circuit_executor."""
        return "circuit_executor"

    def default_algorithm_name(self) -> str:
        """Return the qdk_full_state_simulator as default algorithm name."""
        return "qdk_full_state_simulator"
