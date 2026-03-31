"""Hamiltonian evolution + observable measurement implementation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.energy_estimator.energy_estimator import EnergyEstimator
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.circuit_mapper.base import EvolutionCircuitMapper
from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    TimeEvolutionUnitary,
)
from qdk_chemistry.utils import Logger

from .base import MeasureSimulation, MeasureSimulationSettings

__all__: list[str] = ["EvolveAndMeasure", "EvolveAndMeasureSettings"]


class EvolveAndMeasureSettings(MeasureSimulationSettings):
    """Settings for the EvolveAndMeasure algorithm."""

    def __init__(self):
        """Initialize the settings for EvolveAndMeasure."""
        super().__init__()


class EvolveAndMeasure(MeasureSimulation):
    """Evolve under a Hamiltonian and measure a target observable."""

    def __init__(self):
        """Initialize EvolveAndMeasure with the given settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = EvolveAndMeasureSettings()

    def _run_impl(
        self,
        qubit_hamiltonians: list[QubitHamiltonian],
        times: list[float],
        observables: list[QubitHamiltonian],
        *,
        state_prep: Circuit | None = None,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: EvolutionCircuitMapper,
        shots: int = 1000,
        circuit_executor: CircuitExecutor,
        energy_estimator: EnergyEstimator,
        noise: QuantumErrorProfile | None = None,
        basis_gates: list[str] | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run evolve-and-measure simulation.

        Args:
            qubit_hamiltonians: List of Hamiltonians used to build time evolution.
            times: Monotonically-increasing list of times to evolve under the Hamiltonians.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Optional circuit that prepares the initial state before time evolution.
            evolution_builder: Time-evolution builder.
            circuit_mapper: Mapper for time-evolution unitary to circuit.
            shots: Number of shots to use for measurement.
            circuit_executor: Circuit executor backend.
            energy_estimator: Energy estimator algorithm.
            noise: Optional noise profile.
            basis_gates: Optional list of basis gates to transpile the circuit into before execution.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        Raises:
            ValueError: If ``qubit_hamiltonians`` is empty.

        """
        if not qubit_hamiltonians:
            raise ValueError("qubit_hamiltonians must not be empty.")
        if not times:
            raise ValueError("times must not be empty.")
        if len(qubit_hamiltonians) != len(times):
            raise ValueError("qubit_hamiltonians and times must have the same length.")

        evolution = self._create_time_evolution(qubit_hamiltonians[0], times[0], evolution_builder)

        for i in range(1, len(qubit_hamiltonians)):
            qubit_hamiltonian = qubit_hamiltonians[i]
            time = times[i]
            delta_t = time - times[i - 1]

            evolution = TimeEvolutionUnitary(
                evolution.get_container().combine(
                    self._create_time_evolution(qubit_hamiltonian, delta_t, evolution_builder).get_container(),
                    evolution_builder.settings().get("weight_threshold"),
                )
            )

        evolution_circuit = self._map_time_evolution_to_circuit(evolution, circuit_mapper)

        if state_prep is not None:
            evolution_circuit = self._prepend_state_prep_circuit(state_prep, evolution_circuit)

        # Transpile to basis gates
        if basis_gates is not None:
            evolution_circuit = self._transpile_to_basis_gates(evolution_circuit, basis_gates)

        measurements = []
        for observable in observables:
            measurements.append(
                self._measure_observable(
                    circuit=evolution_circuit,
                    shots=shots,
                    observable=observable,
                    circuit_executor=circuit_executor,
                    energy_estimator=energy_estimator,
                    noise=noise,
                )
            )
        return measurements

    def name(self) -> str:
        """Return the algorithm name used for registry."""
        return "evolve_and_measure"
