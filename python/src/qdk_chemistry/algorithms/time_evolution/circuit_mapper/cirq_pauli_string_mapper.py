"""QDK/Chemistry Cirq PauliStringPhasor evolution circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    PauliProductFormulaContainer,
)

from .base import EvolutionCircuitMapper

__all__: list[str] = ["CirqPauliStringMapper", "CirqPauliStringMapperSettings"]


class CirqPauliStringMapperSettings(Settings):
    """Settings for CirqPauliStringMapper."""

    def __init__(self):
        """Initialize CirqPauliStringMapperSettings with default values."""
        super().__init__()


class CirqPauliStringMapper(EvolutionCircuitMapper):
    r"""Evolution circuit mapper that produces a Cirq circuit using ``PauliStringPhasor`` operations.

    Each :class:`~qdk_chemistry.data.time_evolution.containers.pauli_product_formula.ExponentiatedPauliTerm`
    :math:`e^{-i\theta P}` is mapped to a ``cirq.PauliStringPhasor`` with
    ``exponent_neg = 2\theta / \pi``, which implements :math:`e^{-i\theta P}` up to global phase.
    When ``step_reps > 1`` the single-step circuit is wrapped in a ``cirq.CircuitOperation``.

    Notes:
        * Requires the ``cirq-core`` package.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self):
        """Initialize the CirqPauliStringMapper."""
        super().__init__()
        self._settings = CirqPauliStringMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "cirq_pauli_string"

    def type_name(self) -> str:
        """Return evolution_circuit_mapper as the algorithm type name."""
        return "evolution_circuit_mapper"

    def _run_impl(self, evolution: TimeEvolutionUnitary) -> Circuit:
        r"""Construct a Cirq circuit implementing the time evolution unitary.

        Args:
            evolution: The time evolution unitary to convert to a circuit.

        Returns:
            A :class:`~qdk_chemistry.data.circuit.Circuit` whose Cirq representation is the time-evolution unitary.

        Raises:
            ImportError: If ``cirq-core`` is not installed.
            ValueError: If the time evolution unitary container type is not supported.

        """
        try:
            import cirq  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("The 'cirq-core' package is required for CirqPauliStringMapper.") from exc

        unitary_container = evolution.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {evolution.get_container_type()} container type is not supported. "
                "CirqPauliStringMapper only supports PauliProductFormula container."
            )

        pauli_gate = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
        qubits = cirq.LineQubit.range(unitary_container.num_qubits)

        moments: list[cirq.Moment] = []
        for term in unitary_container.step_terms:
            if not term.pauli_term:
                continue
            pauli_string = cirq.PauliString({qubits[idx]: pauli_gate[op] for idx, op in term.pauli_term.items()})
            phasor = cirq.PauliStringPhasor(
                pauli_string,
                exponent_neg=2 * term.angle / math.pi,
                exponent_pos=0,
            )
            moments.append(cirq.Moment(phasor))

        step_circuit = cirq.FrozenCircuit(moments)
        if unitary_container.step_reps <= 1:
            cirq_circuit = step_circuit
        else:
            cirq_circuit = cirq.Circuit(
                cirq.CircuitOperation(step_circuit, repetitions=unitary_container.step_reps)
            ).freeze()

        return Circuit(cirq=cirq_circuit)
