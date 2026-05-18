"""Hamiltonian evolution + observable measurement implementation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    TimeDependentQubitHamiltonian,
)
from qdk_chemistry.utils import Logger

from .base import MeasureSimulation, MeasureSimulationSettings

__all__: list[str] = ["EvolveAndMeasure", "EvolveAndMeasureSettings"]


class EvolveAndMeasureSettings(MeasureSimulationSettings):
    """Settings for the EvolveAndMeasure algorithm."""

    def __init__(self):
        """Initialize the settings for EvolveAndMeasure."""
        super().__init__()
        self._set_default(
            "total_time",
            "float",
            1.0,
            "Total evolution time.",
        )


class EvolveAndMeasure(MeasureSimulation):
    """Evolve under a Hamiltonian and measure a target observable."""

    def __init__(self):
        """Initialize EvolveAndMeasure with the given settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = EvolveAndMeasureSettings()
        self._evolution_circuit: Circuit | None = None

    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        observables: list[QubitHamiltonian],
        state_prep: Circuit,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run evolve-and-measure simulation.

        The evolution builder, circuit mapper, circuit executor, and energy
        estimator are resolved from the algorithm's settings via
        ``AlgorithmRef``.

        Args:
            hamiltonian: Time-dependent Hamiltonian specifying the evolution schedule.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Circuit that prepares the initial state before time evolution.
            noise: Optional noise profile.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        """
        for observable in observables:
            if observable.num_qubits != hamiltonian.num_qubits:
                raise ValueError("All observables must have the same number of qubits as the Hamiltonian.")

        self._evolution_circuit = self._build_evolution_circuit(hamiltonian, state_prep)

        measurements = []
        for observable in observables:
            measurements.append(
                self._measure_observable(
                    circuit=self._evolution_circuit,
                    observable=observable,
                    noise=noise,
                )
            )
        return measurements

    def _build_evolution_circuit(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        state_prep: Circuit,
    ) -> Circuit:
        """Construct the combined evolution circuit.

        The total evolution time is read from settings and passed to the
        time evolution builder, which handles time-stepping internally.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            state_prep: Circuit that prepares the initial state before time evolution.

        Returns:
            The combined evolution circuit.

        """
        total_time: float = self._settings.get("total_time")
        evolution = self._create_time_evolution(hamiltonian, total_time)
        circuit = self._map_time_evolution_to_circuit(evolution)
        return self._prepend_state_prep_circuit(state_prep, circuit, hamiltonian.num_qubits)

    def get_circuit(self) -> Circuit:
        """Get the evolution circuit generated during algorithm execution.

        Returns:
            The evolution circuit.

        Raises:
            ValueError: If no evolution circuit has been generated.

        """
        if self._evolution_circuit is not None:
            return self._evolution_circuit
        raise ValueError("No evolution circuit has been generated. Please run the algorithm first.")

    def name(self) -> str:
        """Return ``evolve_and_measure`` as the algorithm name."""
        return "evolve_and_measure"
