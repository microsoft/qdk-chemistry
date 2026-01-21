"""QDK/Chemistry circuit executor abstractions and utilities.

This module defines the abstract base class for circuit executor algorithms
that execute quantum circuits and return bitstring measurement results.
The module provides an abstract interface, allowing custom circuit strategies,
backends (simulator or quantum hardware), and noise models.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile

__all__: list[str] = ["CircuitExecutor", "CircuitExecutorFactory"]


class CircuitExecutor(Algorithm):
    """Abstract base class for circuit executor algorithms."""

    def __init__(self):
        """Initialize the CircuitExecutor with default settings."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as circuit_executor."""
        return "circuit_executor"

    @abstractmethod
    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit and get measurement results.

        Args:
            circuit: The circuit that prepares the initial state.
            shots: The number of shots to execute the circuit.
            noise: Optional noise profile to apply during execution.

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
