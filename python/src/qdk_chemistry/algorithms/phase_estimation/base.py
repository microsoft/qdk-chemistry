"""QDK/Chemistry phase estimation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.time_evolution.builder.base import UnitaryBuilder
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import (
    ControlledCircuitMapper,
)
from qdk_chemistry.data import (
    Circuit,
    ControlledUnitary,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
    UnitaryRepresentation,
)

__all__: list[str] = ["PhaseEstimation", "PhaseEstimationFactory", "PhaseEstimationSettings"]


class PhaseEstimationSettings(Settings):
    """Settings for the Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Phase Estimation.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")


class PhaseEstimation(Algorithm):
    """Abstract base class for phase estimation algorithms."""

    def __init__(self, num_bits: int = -1):
        """Initialize the PhaseEstimation with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.

        """
        super().__init__()
        self._settings = PhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: UnitaryBuilder,
        circuit_mapper: ControlledCircuitMapper,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        r"""Run the phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        This method implements the quantum phase estimation procedure:
        1. The state preparation circuit initializes the system in the desired quantum state.
        2. The evolution_builder constructs a unitary from the qubit Hamiltonian.
        3. The circuit_mapper transforms the unitary into controlled-U operations,
           where the control qubits are ancilla qubits used for phase readout.
        4. The circuit_executor runs the resulting quantum circuits on the target backend.
        5. Measurement results are processed to extract the eigenvalue phase estimates.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            evolution_builder: Builder that constructs unitaries from the Hamiltonian.
            circuit_mapper: Maps controlled unitaries to circuit operations.
            circuit_executor: The executor to run quantum circuits on a backend or simulator.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the estimated phases and associated metadata.

        """

    def _create_unitary(
        self, qubit_hamiltonian: QubitHamiltonian, unitary_builder: UnitaryBuilder, **model_kwargs
    ) -> UnitaryRepresentation:
        """Create the unitary for the given Hamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            unitary_builder: The unitary builder to use.
            **model_kwargs: Model-specific keyword arguments (e.g. time for time evolution).

        Returns:
            The unitary representation.

        """
        return unitary_builder.run(qubit_hamiltonian, **model_kwargs)

    # Keep old name as alias for backward compat
    def _create_time_evolution(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, evolution_builder: UnitaryBuilder
    ) -> UnitaryRepresentation:
        """Create the time evolution circuit for the given Hamiltonian and power.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            time: The evolution time.
            evolution_builder: The time evolution builder to use.

        Returns:
            The time evolution unitary circuit.

        """
        return evolution_builder.run(qubit_hamiltonian, time)

    def _create_controlled_circuit(
        self,
        controlled_unitary: ControlledUnitary,
        power: int,
        circuit_mapper: ControlledCircuitMapper,
    ) -> Circuit:
        """Create the controlled circuit for the given controlled unitary and power.

        Args:
            controlled_unitary: The controlled unitary.
            power: The power to which the controlled unitary should be raised.
            circuit_mapper: The controlled circuit mapper to use.

        Returns:
            The controlled circuit.

        """
        circuit_mapper.settings().update("power", power)
        return circuit_mapper._run_impl(controlled_evolution=controlled_unitary)  # noqa: SLF001

    # Keep old name as alias for backward compat
    def _create_ctrl_time_evol_circuit(
        self,
        controlled_evolution: ControlledUnitary,
        power: int,
        circuit_mapper: ControlledCircuitMapper,
    ) -> Circuit:
        """Create the controlled time evolution circuit for the given Hamiltonian and power.

        Args:
            controlled_evolution: The controlled time evolution unitary.
            power: The power to which the controlled unitary should be raised.
            circuit_mapper: The controlled evolution circuit mapper to use.

        Returns:
            The controlled time evolution circuit.

        """
        circuit_mapper.settings().update("power", power)
        return circuit_mapper._run_impl(controlled_evolution=controlled_evolution)  # noqa: SLF001


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
