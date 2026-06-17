"""QDK/Chemistry Hadamard test circuit builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hadamard_test.base import HadamardTestBasis
from qdk_chemistry.data import Circuit

__all__: list[str] = [
    "HadamardTestCircuitBuilder",
    "HadamardTestCircuitBuilderFactory",
]


class HadamardTestCircuitBuilder(Algorithm):
    """Abstract base class for Hadamard test circuit builders.

    A circuit builder turns a prepared state and a controlled evolution circuit
    into a single backend-specific Hadamard test circuit, separating the
    backend-dependent circuit construction from the backend-agnostic
    orchestration performed by ``HadamardTest``.
    """

    def __init__(self):
        """Initialize the Hadamard test circuit builder."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_circuit_builder."""
        return "hadamard_test_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
        num_ancilla_qubits: int = 0,
    ) -> Circuit:
        r"""Build the Hadamard test circuit for a given state and controlled unitary.

        Currently, the function only accepts the controlled unitary circuit whose index of ancilla qubit is 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X``, ``HadamardTestBasis.Y``, or
              ``HadamardTestBasis.Z``).
            num_ancilla_qubits: Number of ancilla qubits needed by the controlled evolution (0 if none).

        Returns:
            Circuit representing the Hadamard test workflow for the selected backend.

        """


class HadamardTestCircuitBuilderFactory(AlgorithmFactory):
    """Factory class for creating Hadamard test circuit builder instances."""

    def __init__(self):
        """Initialize the HadamardTestCircuitBuilderFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_circuit_builder."""
        return "hadamard_test_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return 'qdk_circuit_builder' as the default algorithm name."""
        return "qdk_circuit_builder"
