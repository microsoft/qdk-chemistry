"""QDK/Chemistry controlled time evolution unitary circuit mapper abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.time_evolution.controlled_time_evolution import ControlledUnitary

__all__: list[str] = ["ControlledCircuitMapper", "ControlledCircuitMapperFactory"]


class ControlledCircuitMapper(Algorithm):
    """Base class for controlled circuit mapper in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the ControlledCircuitMapper."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, controlled_unitary: ControlledUnitary, *args, **kwargs) -> Circuit:
        """Construct a Circuit representing the controlled unitary for the given ControlledUnitary.

        Args:
            controlled_unitary: The controlled unitary.
            *args: Positional arguments, where the first argument is expected to be the
                controlled unitary.
            **kwargs: Additional keyword arguments for concrete implementation.

        Returns:
            Circuit: A Circuit representing the controlled unitary for the given ControlledUnitary.

        """


class ControlledCircuitMapperFactory(AlgorithmFactory):
    """Factory class for creating ControlledCircuitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def default_algorithm_name(self) -> str:
        """Return pauli_sequence as the default algorithm name."""
        return "pauli_sequence"
