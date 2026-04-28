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

    _evolution_circuit: Circuit | None = None

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
        shots: int = 1000,
        noise: QuantumErrorProfile | None = None,
        device_backend_name: str | None = None,
        pre_transpilation_passes: list[str] | None = None,
        post_transpilation_passes: list[str] | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run evolve-and-measure simulation.

        The evolution builder, circuit mapper, circuit executor, and energy
        estimator are resolved from the algorithm's settings via
        ``AlgorithmRef``.

        Args:
            qubit_hamiltonians: List of Hamiltonians used to build time evolution.
            times: Monotonically-increasing list of times to evolve under the Hamiltonians.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Optional circuit that prepares the initial state before time evolution.
            shots: Number of shots to use for measurement.
            noise: Optional noise profile.
            device_backend_name: Optional device backend name string to pass to the circuit executor.
            pre_transpilation_passes: Optional list of passes to apply before transpilation.
            post_transpilation_passes: Optional list of passes to apply after transpilation.

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
        if times != sorted(times):
            raise ValueError("times must be monotonically increasing.")

        # Ensure all Hamiltonians and observables have the same number of qubits.
        reference_num_qubits = qubit_hamiltonians[0].num_qubits
        for hamiltonian in qubit_hamiltonians[1:]:
            if hamiltonian.num_qubits != reference_num_qubits:
                raise ValueError("All Hamiltonians and observables must have the same number of qubits.")
        for observable in observables:
            if observable.num_qubits != reference_num_qubits:
                raise ValueError("All Hamiltonians and observables must have the same number of qubits.")
        self._evolution_circuit = self._build_evolution_circuit(
            qubit_hamiltonians=qubit_hamiltonians,
            times=times,
            state_prep=state_prep,
        )

        measurements = []
        for observable in observables:
            measurements.append(
                self._measure_observable(
                    circuit=self._evolution_circuit,
                    shots=shots,
                    observable=observable,
                    noise=noise,
                    device_backend_name=device_backend_name,
                    pre_transpilation_passes=pre_transpilation_passes,
                    post_transpilation_passes=post_transpilation_passes,
                )
            )
        return measurements

    def _build_evolution_circuit(
        self,
        qubit_hamiltonians: list[QubitHamiltonian],
        times: list[float],
        *,
        state_prep: Circuit | None = None,
    ) -> Circuit:
        """Construct the combined evolution circuit.

        The evolution builder and circuit mapper are resolved from
        the algorithm's settings via ``AlgorithmRef``.

        Args:
            qubit_hamiltonians: List of Hamiltonians used to build time evolution.
            times: Monotonically-increasing list of times to evolve under the Hamiltonians.
            state_prep: Optional circuit that prepares the initial state before time evolution.

        Returns:
            The combined evolution circuit.

        """
        evolution = self._create_time_evolution(qubit_hamiltonians[0], times[0])

        for i in range(1, len(qubit_hamiltonians)):
            delta_t = times[i] - times[i - 1]
            evolution = TimeEvolutionUnitary(
                evolution.get_container().combine(
                    self._create_time_evolution(qubit_hamiltonians[i], delta_t).get_container(),
                )
            )

        circuit = self._map_time_evolution_to_circuit(evolution)

        if state_prep is not None:
            circuit = self._prepend_state_prep_circuit(state_prep, circuit, qubit_hamiltonians[0].num_qubits)

        self._evolution_circuit = circuit
        return circuit

    def get_circuit(self) -> Circuit | None:
        """Get the evolution circuit used in the simulation."""
        return self._evolution_circuit

    def name(self) -> str:
        """Return ``evolve_and_measure`` as the algorithm name."""
        return "evolve_and_measure"
