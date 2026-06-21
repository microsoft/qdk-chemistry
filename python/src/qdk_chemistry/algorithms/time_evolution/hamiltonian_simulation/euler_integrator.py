r"""Euler integrator for time-dependent Hamiltonian simulation.

This module implements an Euler integrator for solving the time-dependent
Schrodinger equation i*dU/dt = H(t)*U on a quantum computer.

The integrator divides [0, T] into ``floor(T / dt)`` steps of size dt.  If T
is not an exact multiple of dt, a final cleanup step of size ``T mod dt`` is
appended.  At each step
a :class:`~qdk_chemistry.algorithms.propagator.base.Propagator` evaluates
the effective Hamiltonian for the interval, which is then Trotterized into a
quantum circuit.  Per-step unitaries are combined via
UnitaryContainer.combine, which merges adjacent identical Pauli
terms at step boundaries.

The default propagator (``magnus``) computes the Magnus-expanded
Hamiltonian over each interval, giving second-order global accuracy for
smooth drives.  Other propagators can be substituted via the
``propagator`` setting.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.propagator.base import Propagator
from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    TimeDependentQubitHamiltonian,
    UnitaryRepresentation,
)
from qdk_chemistry.utils import Logger

from .base import HamiltonianSimulation, HamiltonianSimulationSettings

if TYPE_CHECKING:
    from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer

__all__: list[str] = ["EulerIntegrator", "EulerIntegratorSettings"]


class EulerIntegratorSettings(HamiltonianSimulationSettings):
    """Settings for the Euler integrator."""

    def __init__(self):
        """Initialize the settings for EulerIntegrator."""
        super().__init__()


class EulerIntegrator(HamiltonianSimulation):
    r"""Euler integrator for time-dependent Hamiltonian simulation.

    Solves :math:`i\partial_t U = H(t)\,U` by dividing :math:`[0, T]` into
    steps of size ``dt``.  At each step a
    :class:`~qdk_chemistry.algorithms.propagator.base.Propagator` computes
    the effective Hamiltonian for the interval, which is Trotterized into
    a quantum circuit.

    The default ``magnus`` propagator integrates the drive function
    over each interval, giving :math:`O(\Delta t^2)` global accuracy for
    smooth drives.

    """

    def __init__(self):
        """Initialize EulerIntegrator."""
        Logger.trace_entering()
        super().__init__()
        self._settings = EulerIntegratorSettings()
        self._evolution_circuit: Circuit | None = None

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

        self._evolution_circuit = self._build_evolution_circuit(hamiltonian, state_prep)

        measurements = []
        for observable in observables:
            measurements.append(
                self._measure_observable(
                    circuit=self._evolution_circuit,
                    observable=observable,
                    shots=shots,
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
        ``dt``.  At each step the propagator evaluates the effective
        Hamiltonian for the interval and the builder Trotterizes it.
        The resulting per-step unitaries are combined via
        :meth:`UnitaryContainer.combine`, which merges
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
        r"""Build the combined unitary via Euler steps.

        Divides :math:`[0, T]` into steps of size ``dt``.  The number of
        full steps is ``floor(T / dt)``.  If ``T`` is not an exact multiple
        of ``dt``, a final cleanup step of size ``T mod dt`` is appended.
        Raises ``ValueError`` if ``dt > T``.

        Args:
            hamiltonian: The time-dependent qubit Hamiltonian.
            total_time: Total evolution time.

        Returns:
            Combined ``UnitaryRepresentation`` for the full evolution.

        """
        dt: float = self._settings.get("dt")
        if total_time == 0.0:
            raise ValueError("total_time must be nonzero.")
        if dt == 0.0:
            raise ValueError(f"dt ({dt}) must be nonzero.")
        if dt / total_time > 1:
            raise ValueError(f"dt ({dt}) must match not exceed total_time ({total_time}).")
        if dt / total_time < 0:
            raise ValueError(f"dt ({dt}) must match the sign of total_time ({total_time}).")

        num_full_steps = int(total_time / dt)
        residual = total_time - num_full_steps * dt

        combined_container: UnitaryContainer | None = None
        propagator = self._create_nested("propagator")
        if not isinstance(propagator, Propagator):
            raise TypeError(f"propagator must be a Propagator, got {type(propagator).__name__}.")
        for i in range(num_full_steps):
            t_start = i * dt
            t_end = (i + 1) * dt
            h_snapshot = propagator.run(hamiltonian, t_start, t_end)
            step_evolution = self._create_time_step_evolution(h_snapshot, dt)
            time_step_container = step_evolution.get_container()
            if combined_container is None:
                combined_container = time_step_container
            else:
                combined_container = combined_container.combine(time_step_container)

        if residual > 0.0:
            t_start = num_full_steps * dt
            t_end = total_time
            h_snapshot = propagator.run(hamiltonian, t_start, t_end)
            step_evolution = self._create_time_step_evolution(h_snapshot, residual)
            time_step_container = step_evolution.get_container()
            if combined_container is None:
                combined_container = time_step_container
            else:
                combined_container = combined_container.combine(time_step_container)

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
        """Return ``euler_integrator`` as the algorithm name."""
        return "euler_integrator"
