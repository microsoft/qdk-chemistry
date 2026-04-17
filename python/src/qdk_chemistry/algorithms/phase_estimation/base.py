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
            evolution_time: Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``,
                defaults to 0.0; user needs to set a valid value. This setting is only applicable to
                time evolution-based unitary builders such as Trotter.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")
        self._set_default(
            "evolution_time",
            "float",
            0.0,
            "Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``.",
        )


class PhaseEstimation(Algorithm):
    """Abstract base class for phase estimation algorithms."""

    def __init__(self, num_bits: int = -1, evolution_time: float = 0.0):
        """Initialize the PhaseEstimation with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            evolution_time: Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``,
                defaults to 0.0; user needs to set a valid value.

        """
        super().__init__()
        self._settings = PhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("evolution_time", evolution_time)

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        unitary_builder: UnitaryBuilder,
        circuit_mapper: ControlledCircuitMapper,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        r"""Run the phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        This method implements the quantum phase estimation procedure:
        1. The state preparation circuit initializes the system in the desired quantum state.
        2. The unitary_builder constructs a unitary from the qubit Hamiltonian.
        3. The circuit_mapper transforms the unitary into controlled-U operations,
           where the control qubits are ancilla qubits used for phase readout.
        4. The circuit_executor runs the resulting quantum circuits on the target backend.
        5. Measurement results are processed to extract the eigenvalue phase estimates.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            unitary_builder: Builder that constructs unitaries from the Hamiltonian.
            circuit_mapper: Maps controlled unitaries to circuit operations.
            circuit_executor: The executor to run quantum circuits on a backend or simulator.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the estimated phases and associated metadata.

        """

    def _create_unitary(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, unitary_builder: UnitaryBuilder
    ) -> UnitaryRepresentation:
        """Create the time evolution circuit for the given Hamiltonian and power.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            time: The evolution time.
            unitary_builder: The time evolution builder to use.

        Returns:
            The time evolution unitary circuit.

        """
        return unitary_builder.run(qubit_hamiltonian, time)

    def _create_controlled_circuit(
        self,
        controlled_unitary: ControlledUnitary,
        power: int,
        circuit_mapper: ControlledCircuitMapper,
    ) -> Circuit:
        """Create the controlled time evolution circuit for the given Hamiltonian and power.

        Args:
            controlled_unitary: The controlled time evolution unitary.
            power: The power to which the controlled unitary should be raised.
            circuit_mapper: The controlled evolution circuit mapper to use.

        Returns:
            The controlled time evolution circuit.

        """
        # Create a new instance of the mapper to avoid setting lock
        circuit_mapper.settings().update("power", power)  # Update the power setting
        # Avoid lock settings
        return circuit_mapper._run_impl(controlled_unitary=controlled_unitary)  # noqa: SLF001


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
