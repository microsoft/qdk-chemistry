r"""Euler evolution circuit builder.

Builds a state-preparation + time-evolution circuit by dividing
:math:`[0, T]` into Euler steps of size ``dt``, applying a propagator
at each step to compute the effective Hamiltonian, Trotterizing, and
mapping to a quantum circuit.  No circuit execution is performed.

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

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.propagator.base import Propagator
from qdk_chemistry.data import (
    Circuit,
    QubitHamiltonian,
    TimeDependentQubitHamiltonian,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import EvolutionCircuitBuilder, EvolutionCircuitBuilderSettings

if TYPE_CHECKING:
    from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer

__all__: list[str] = ["EulerEvolutionCircuitBuilder", "EulerEvolutionCircuitBuilderSettings"]


class EulerEvolutionCircuitBuilderSettings(EvolutionCircuitBuilderSettings):
    """Settings for the Euler evolution circuit builder."""

    def __init__(self):
        """Initialize the settings for EulerEvolutionCircuitBuilder."""
        super().__init__()


class EulerEvolutionCircuitBuilder(EvolutionCircuitBuilder):
    r"""Euler-step evolution circuit builder.

    Divides :math:`[0, T]` into steps of size ``dt``.  At each step, the
    configured propagator evaluates the effective Hamiltonian, the evolution
    builder Trotterizes it, and the circuit mapper compiles it to QIR.
    The resulting per-step circuits are combined via
    :meth:`~qdk_chemistry.data.unitary_representation.containers.base.UnitaryContainer.combine`.

    The output is a single :class:`~qdk_chemistry.data.circuit.Circuit` (state-prep + evolution) that
    can be passed to ``circuit.get_qre_application()`` for resource
    estimation.

    """

    def __init__(self):
        """Initialize EulerEvolutionCircuitBuilder."""
        Logger.trace_entering()
        super().__init__()
        self._settings = EulerEvolutionCircuitBuilderSettings()

    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        state_prep: Circuit,
    ) -> Circuit:
        """Build the evolution circuit.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            state_prep: Circuit that prepares the initial state.

        Returns:
            The combined state-prep + evolution circuit.

        """
        total_time: float = self._settings.get("total_time")
        evolution = self._build_time_dependent_evolution(hamiltonian, total_time)
        circuit = self._map_to_circuit(evolution)
        return self._prepend_state_prep(state_prep, circuit, hamiltonian.num_qubits)

    def _build_time_dependent_evolution(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        total_time: float,
    ) -> UnitaryRepresentation:
        r"""Build the combined unitary via Euler steps.

        Divides :math:`[0, T]` into steps of size ``dt``.  The number of
        full steps is ``floor(T / dt)``.  If ``T`` is not an exact multiple
        of ``dt``, a final cleanup step of size ``T mod dt`` is appended.

        Args:
            hamiltonian: The time-dependent qubit Hamiltonian.
            total_time: Total evolution time.

        Returns:
            Combined ``UnitaryRepresentation`` for the full evolution.

        Raises:
            ValueError: If ``dt`` or ``total_time`` are invalid.
            TypeError: If the propagator is not a ``Propagator``.

        """
        dt: float = self._settings.get("dt")
        if total_time == 0.0:
            raise ValueError("total_time must be nonzero.")
        if dt == 0.0:
            raise ValueError(f"dt ({dt}) must be nonzero.")
        if dt / total_time > 1:
            raise ValueError(f"dt ({dt}) must not exceed total_time ({total_time}).")
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

    def _create_time_step_evolution(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
    ) -> UnitaryRepresentation:
        """Create the time-evolution unitary for one step.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian for this time step.
            time: Duration of the time step.

        Returns:
            The unitary representation for this step.

        Raises:
            TypeError: If the evolution builder is not a ``TimeEvolutionBuilder``.

        """
        evolution_builder = self._create_nested("evolution_builder")
        if not isinstance(evolution_builder, TimeEvolutionBuilder):
            raise TypeError(
                f"evolution_builder must be a TimeEvolutionBuilder, got {type(evolution_builder).__name__}."
            )
        evolution_builder.settings().set("time", time)
        return evolution_builder.run(qubit_hamiltonian)

    def _map_to_circuit(self, evolution: UnitaryRepresentation) -> Circuit:
        """Map a time-evolution unitary into an executable circuit.

        Args:
            evolution: The unitary representation to compile.

        Returns:
            The compiled quantum circuit.

        """
        circuit_mapper = self._create_nested("circuit_mapper")
        return circuit_mapper.run(evolution)

    def _prepend_state_prep(self, state_prep: Circuit, circuit: Circuit, num_qubits: int) -> Circuit:
        """Compose state-preparation and evolution circuits.

        Args:
            state_prep: Circuit that prepares the initial state.
            circuit: The evolution circuit to prepend state-prep to.
            num_qubits: Number of qubits in the system.

        Returns:
            The combined state-prep + evolution circuit.

        Raises:
            RuntimeError: If either circuit lacks a Q# operation handle.
            ValueError: If the circuits have incompatible encodings.

        """
        state_prep_op = state_prep._qsharp_op  # noqa: SLF001
        circuit_op = circuit._qsharp_op  # noqa: SLF001
        if state_prep_op is None or circuit_op is None:
            raise RuntimeError("State-preparation circuit composition requires Q# operations on both circuits.")

        if state_prep.encoding is not None and circuit.encoding is not None and state_prep.encoding != circuit.encoding:
            raise ValueError(
                "State-preparation circuit and evolution circuit use different encodings "
                f"('{state_prep.encoding}' and '{circuit.encoding}')."
            )

        target_indices = list(range(num_qubits))
        combined_encoding = circuit.encoding if circuit.encoding is not None else state_prep.encoding
        sequential_parameters = {
            "first": state_prep_op,
            "second": circuit_op,
            "targets": target_indices,
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.CircuitComposition.MakeSequentialCircuit,
                parameter=sequential_parameters,
            ),
            qsharp_op=QSHARP_UTILS.CircuitComposition.MakeSequentialOp(state_prep_op, circuit_op),
            encoding=combined_encoding,
        )

    def name(self) -> str:
        """Return ``euler`` as the algorithm name."""
        return "euler"
