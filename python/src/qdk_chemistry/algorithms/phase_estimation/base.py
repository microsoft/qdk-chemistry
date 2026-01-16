"""QDK/Chemistry phase estimation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.circuit_executor import CircuitExecutor
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import (
    ControlledEvolutionCircuitMapper,
)
from qdk_chemistry.data import (
    Circuit,
    ControlledTimeEvolutionUnitary,
    QpeResult,
    QubitHamiltonian,
    TimeEvolutionUnitary,
)

__all__: list[str] = []


class PhaseEstimation(Algorithm):
    """Abstract base class for phase estimation algorithms."""

    def __init__(self):
        """Initialize the PhaseEstimation with default settings."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        circuit_executor: CircuitExecutor,
    ) -> QpeResult:
        """Prepare a quantum circuit that encodes the given wavefunction.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            circuit_executor: The executor to run quantum circuits.

        Returns:
            A QpeResult object containing the results of the phase estimation.

        """

    def _create_time_evolution(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, evolution_builder: TimeEvolutionBuilder
    ) -> TimeEvolutionUnitary:
        """Create the time evolution circuit for the given Hamiltonian and power.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            evolution_builder: The time evolution builder to use.
            time: The evolution time.

        Returns:
            The time evolution unitary circuit.

        """
        return evolution_builder.run(qubit_hamiltonian, time)

    def _create_ctrl_time_evol_circuit(
        self,
        controlled_evolution: ControlledTimeEvolutionUnitary,
        power: int,
        circuit_mapper: ControlledEvolutionCircuitMapper,
    ) -> Circuit:
        """Create the controlled time evolution circuit for the given Hamiltonian and power.

        Args:
            controlled_evolution: The controlled time evolution unitary.
            power: The power to which the controlled unitary should be raised.
            circuit_mapper: The controlled evolution circuit mapper to use.

        Returns:
            The controlled time evolution circuit.

        """
        # Create a new instance of the mapper to avoid setting lock
        circuit_mapper.settings().update("power", power)  # Update the power setting
        # Avoid lock settings
        return circuit_mapper._run_impl(controlled_evolution=controlled_evolution)  # noqa: SLF001

    @staticmethod
    def _validate_state_prep_qubits(state_preparation: Circuit, qubit_hamiltonian: QubitHamiltonian) -> None:
        """Ensure ``state_preparation`` matches the Hamiltonian system size."""
        if state_preparation.num_qubits != qubit_hamiltonian.num_qubits:
            raise ValueError(
                "state_preparation must prepare the same number of system qubits as the Hamiltonian "
                f"(expected {qubit_hamiltonian.num_qubits}, received {state_preparation.num_qubits}).",
            )


class PhaseEstimationFactory(AlgorithmFactory):
    """Factory class for creating PhaseEstimation instances."""

    def __init__(self):
        """Initialize the PhaseEstimationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    def default_algorithm_name(self) -> str:
        """Return the iterative as default algorithm name."""
        return "iterative"
