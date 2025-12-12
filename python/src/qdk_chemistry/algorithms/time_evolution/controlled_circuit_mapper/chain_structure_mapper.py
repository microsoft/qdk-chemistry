"""QDK/Chemistry chain structure controlled evolution circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, qasm3

from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.data.time_evolution.controlled_time_evolution import ControlledTimeEvolutionUnitary
from qdk_chemistry.utils import Logger

from .base import ControlledEvolutionCircuitMapper


class ChainStructureMapperSettings(Settings):
    """Settings for ChainStructureMapper.

    Attributes:
        power (int): The power of the controlled unitary to be constructed.

    """

    def __init__(self):
        """Initialize ChainStructureMapperSettings with default values."""
        super().__init__()
        self._set_default("power", "int", 1)


class ChainStructureMapper(ControlledEvolutionCircuitMapper):
    """Chain structure controlled evolution circuit mapper implementation."""

    def __init__(self, power: int = 1):
        """Initialize the ChainStructureMapper."""
        super().__init__()
        self._settings = ChainStructureMapperSettings()
        self._settings.set("power", power)

    def _run_impl(
        self, controlled_evolution: ControlledTimeEvolutionUnitary, system_indices: Sequence[int] | None = None
    ) -> Circuit:
        """Construct a Circuit representing the controlled unitary for the given ControlledTimeEvolutionUnitary.

        Args:
            controlled_evolution (ControlledTimeEvolutionUnitary): The controlled time evolution unitary.
            system_indices (Sequence[int] | None): The system qubit indices. If None, assumes system qubits are all
                qubits except the control qubit.

        Returns:
            Circuit: A Circuit representing the controlled unitary for the given ControlledTimeEvolutionUnitary.

        """
        if not isinstance(controlled_evolution.time_evolution_unitary._container, PauliProductFormulaContainer):  # noqa: SLF001
            raise ValueError(
                f"The {controlled_evolution.get_unitary_container_type()} container type is not supported. "
                "ChainStructureMapper only supports PauliProductFormula container for time evolution unitary."
            )

        num_system_qubits = controlled_evolution.get_num_system_qubits()
        total_qubits = num_system_qubits + len(controlled_evolution.control_index)

        if system_indices is None:
            system_indices = [i for i in range(total_qubits) if i != controlled_evolution.control_index]

        circuit = QuantumCircuit(total_qubits)
        append_controlled_time_evolution(
            circuit, controlled_evolution, system_indices=system_indices, power=self._settings.get("power")
        )

        qasm_str = qasm3.dumps(circuit)
        return Circuit(qasm=qasm_str)


def _append_controlled_pauli_rotation(
    circuit: QuantumCircuit,
    control_qubit: int,
    system_qubits: Sequence[int],
    term: ExponentiatedPauliTerm,
) -> QuantumCircuit:
    """Append a controlled ``exp(-i angle * P)`` to ``circuit``.

    Args:
        circuit: Quantum circuit receiving the controlled rotation.
        control_qubit: Index of the ancilla qubit providing the control.
        system_qubits: Ordered collection of system qubit indices.
        term: Pauli term describing the rotation axis.

    Returns:
        The quantum circuit with the controlled rotation appended.

    """
    Logger.trace_entering()
    if not term.pauli_term:
        # Identity contribution results in a controlled phase on the ancilla.
        circuit.p(-term.angle, control_qubit)
        return circuit

    involved_indices = sorted(term.pauli_term.keys())
    involved_qubits = [system_qubits[i] for i in involved_indices]

    # Basis-change into Z
    for idx, qubit in zip(involved_indices, involved_qubits, strict=True):
        pauli = term.pauli_term[idx]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)

    target = involved_qubits[-1]
    for qubit in involved_qubits[:-1]:
        circuit.cx(qubit, target)

    circuit.crz(2 * term.angle, control_qubit, target)

    for qubit in reversed(involved_qubits[:-1]):
        circuit.cx(qubit, target)

    for idx, qubit in reversed(list(zip(involved_indices, involved_qubits, strict=True))):
        pauli = term.pauli_term[idx]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.h(qubit)
            circuit.s(qubit)

    return circuit


def append_controlled_time_evolution(
    circuit: QuantumCircuit,
    controlled_evolution: ControlledTimeEvolutionUnitary,
    *,
    system_indices: Sequence[int],
    power: int = 1,
) -> None:
    """Append the controlled unitary ``(exp(-i H time))**power``.

    Args:
        circuit: Circuit being extended.
        controlled_evolution: The controlled time evolution unitary.
        system_indices: The system qubit indices.
            If None, assumes system qubits are all qubits except the control qubit.
        power: Number of repeated applications (``U`` raised to ``power``).

    Raises:
        ValueError: If the container type is not supported.
        ValueError: If ``power`` is less than 1.

    """
    Logger.trace_entering()
    if not isinstance(controlled_evolution.time_evolution_unitary._container, PauliProductFormulaContainer):  # noqa: SLF001
        raise ValueError(
            f"The {controlled_evolution.get_unitary_container_type()} container type is not supported. "
            "Only supports PauliProductFormula container for time evolution unitary."
        )

    if power < 1:
        raise ValueError("power must be at least 1 for controlled time evolution.")

    control_qubit = controlled_evolution.control_bits

    for _ in range(power):
        for _ in range(controlled_evolution.time_evolution_unitary._container.step_reps):  # noqa: SLF001
            for term in controlled_evolution.time_evolution_unitary._container.step_terms:  # noqa: SLF001
                if np.isclose(term.angle, 0.0):
                    continue
                _append_controlled_pauli_rotation(
                    circuit,
                    control_qubit,
                    system_indices,
                    term,
                )
