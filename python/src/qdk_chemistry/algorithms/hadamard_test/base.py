"""QDK/Chemistry Hadamard test circuit generator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit

__all__: list[str] = ["HadamardTestFactory", "HadamardTestGenerator"]


class HadamardTestGenerator(Algorithm):
    """Abstract base class for Hadamard test generators."""

    def __init__(self):
        """Initialize a Hadamard test generator."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_generator."""
        return "hadamard_test_generator"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: str = "X",
    ) -> Circuit:
        r"""Run the Hadamard test algorithm for a given state and controlled unitary.

        Args:
            state_preparation: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit. Supported values are ``"X"``, ``"Y"``, and ``"Z"``.

        Returns:
            Circuit representing the Hadamard test workflow for the selected backend.

        """


class HadamardTestFactory(AlgorithmFactory):
    """Factory class for creating Hadamard test generator instances."""

    def __init__(self):
        """Initialize the HadamardTestFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_generator."""
        return "hadamard_test_generator"

    def default_algorithm_name(self) -> str:
        """Return the hadamard_test as default algorithm name."""
        return "hadamard_test"
