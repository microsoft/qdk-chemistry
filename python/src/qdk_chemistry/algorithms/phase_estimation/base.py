"""QDK/Chemistry phase estimation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    ControlledTimeEvolutionUnitary,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
    TimeEvolutionUnitary,
)

__all__: list[str] = ["PhaseEstimation", "PhaseEstimationFactory", "PhaseEstimationSettings"]


class PhaseEstimationSettings(Settings):
    """Settings for the Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Phase Estimation.

        Includes nested algorithm references for the evolution builder,
        circuit mapper, and circuit executor.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")
        self._set_default(
            "evolution_time",
            "float",
            0.0,
            "Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``.",
        )
        self._set_default(
            "evolution_builder",
            "algorithm_ref",
            AlgorithmRef("time_evolution_builder", "trotter"),
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_evolution_circuit_mapper", "pauli_sequence"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
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
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        r"""Run the phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        This method implements the quantum phase estimation procedure:
        1. The state preparation circuit initializes the system in the desired quantum state.
        2. The evolution_builder constructs a time evolution unitary :math:`U = \exp(-iHt)` from the qubit Hamiltonian.
        3. The circuit_mapper transforms the time evolution unitary into controlled-U operations,
           where the control qubits are ancilla qubits used for phase readout.
        4. The circuit_executor runs the resulting quantum circuits on the target backend.
        5. Measurement results are processed to extract the eigenvalue phase estimates.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the estimated phases and associated metadata.

        """

    def _create_time_evolution(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Create the time evolution circuit for the given Hamiltonian and power.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            time: The evolution time.

        Returns:
            The time evolution unitary circuit.

        """
        evolution_builder = self._create_nested("evolution_builder")
        return evolution_builder.run(qubit_hamiltonian, time)

    def _create_ctrl_time_evol_circuit(
        self,
        controlled_evolution: ControlledTimeEvolutionUnitary,
        power: int,
    ) -> Circuit:
        """Create the controlled time evolution circuit for the given Hamiltonian and power.

        Args:
            controlled_evolution: The controlled time evolution unitary.
            power: The power to which the controlled unitary should be raised.

        Returns:
            The controlled time evolution circuit.

        """
        circuit_mapper = self._create_nested("circuit_mapper")
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
