"""Hamiltonian evolution + observable measurement implementation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    TimeDependentQubitHamiltonian,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
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
        self._set_default(
            "dt",
            "float",
            0.0,
            "Time step for time-dependent evolution. Each step is passed to the builder",
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

        Args:
            hamiltonian: Time-dependent Hamiltonian.
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

        The interval ``[0, total_time]`` is divided into steps of size
        ``dt``.  At each step the Hamiltonian is evaluated at the midpoint
        and the builder is called with ``dt`` as the evolution time.  The
        resulting per-step unitaries are combined via
        :meth:`PauliProductFormulaContainer.combine`, which merges
        adjacent identical Pauli terms at step boundaries.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            state_prep: Circuit that prepares the initial state.

        Returns:
            The combined state-prep + evolution circuit.

        """
        total_time: float = self._settings.get("total_time")
        evolution = self._build_time_dependent_evolution(hamiltonian, total_time)
        circuit = self._map_time_evolution_to_circuit(evolution)
        return self._prepend_state_prep_circuit(state_prep, circuit, hamiltonian.num_qubits)

    def _build_time_dependent_evolution(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        total_time: float,
    ) -> UnitaryRepresentation:
        """Build the combined unitary for a time-dependent Hamiltonian.

        The interval ``[0, total_time]`` is divided into steps of size
        ``dt`` (from settings).  At each step the Hamiltonian is evaluated
        at the midpoint and the Trotter builder is called with ``dt`` as
        the evolution time.  Per-step containers are combined via
        :meth:`PauliProductFormulaContainer.combine`, which merges
        adjacent identical Pauli terms at step boundaries.

        Args:
            hamiltonian: The time-dependent qubit Hamiltonian.
            total_time: Total evolution time.

        Returns:
            Combined ``UnitaryRepresentation`` for the full evolution.

        """
        dt: float = self._settings.get("dt")
        if dt <= 0.0:
            dt = total_time
        if dt > total_time:
            raise ValueError(f"dt ({dt}) must not exceed total_time ({total_time}).")
        num_steps = max(1, round(total_time / dt))
        dt = total_time / num_steps

        combined_container: PauliProductFormulaContainer | None = None
        for i in range(num_steps):
            t_mid = (i + 0.5) * dt
            h_snapshot = hamiltonian.evaluate(t_mid)
            step_evolution = self._create_time_step_evolution(h_snapshot, dt)
            step_container = step_evolution.get_container()
            if not isinstance(step_container, PauliProductFormulaContainer):
                raise TypeError(f"Expected PauliProductFormulaContainer, got {type(step_container).__name__}.")
            if combined_container is None:
                combined_container = step_container
            else:
                combined_container = combined_container.combine(step_container)

        return UnitaryRepresentation(container=combined_container)

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
