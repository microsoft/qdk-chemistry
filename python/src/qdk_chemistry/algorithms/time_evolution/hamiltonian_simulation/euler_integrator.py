r"""Euler integrator for time-dependent Hamiltonian simulation.

This module implements a Hamiltonian simulation algorithm that uses an
:class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.euler_builder.EulerEvolutionCircuitBuilder`
to construct the evolution circuit and then executes it to measure
observable expectation values.

The circuit builder handles all time-stepping, propagation, Trotterization,
and circuit mapping.  This class adds circuit execution and observable
measurement on top.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from qdk_chemistry.utils import Logger

from .base import HamiltonianSimulation, HamiltonianSimulationSettings

if TYPE_CHECKING:
    from qdk_chemistry.data import (
        Circuit,
        EnergyExpectationResult,
        MeasurementData,
        QuantumErrorProfile,
        QubitHamiltonian,
        TimeDependentQubitHamiltonian,
    )

__all__: list[str] = ["EulerIntegrator", "EulerIntegratorSettings"]


class EulerIntegratorSettings(HamiltonianSimulationSettings):
    """Settings for the Euler integrator."""

    def __init__(self):
        """Initialize the settings for EulerIntegrator."""
        super().__init__()


class EulerIntegrator(HamiltonianSimulation):
    r"""Euler integrator for time-dependent Hamiltonian simulation.

    Delegates circuit construction to the configured
    ``evolution_circuit_builder`` (default: ``EulerEvolutionCircuitBuilder``),
    then executes the circuit and measures observables.

    For resource estimation (without execution), use the
    ``evolution_circuit_builder`` directly via
    ``create("evolution_circuit_builder", "euler", ...)``.

    """

    def __init__(self):
        """Initialize EulerIntegrator."""
        Logger.trace_entering()
        super().__init__()
        self._settings = EulerIntegratorSettings()

    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        observables: list[QubitHamiltonian],
        state_prep: Circuit,
        shots: int = 1000,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run the Euler-integrated Hamiltonian simulation.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Circuit that prepares the initial state before time evolution.
            shots: Number of measurement shots per observable. Defaults to 1000.
            noise: Optional noise profile.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        """
        for observable in observables:
            if observable.num_qubits != hamiltonian.num_qubits:
                raise ValueError("All observables must have the same number of qubits as the Hamiltonian.")

        circuit = self._build_evolution_circuit(hamiltonian, state_prep)

        measurements = []
        for observable in observables:
            measurements.append(
                self._measure_observable(
                    circuit=circuit,
                    observable=observable,
                    shots=shots,
                    noise=noise,
                )
            )
        return measurements

    def name(self) -> str:
        """Return ``euler_integrator`` as the algorithm name."""
        return "euler_integrator"
