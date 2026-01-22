"""QDK/Chemistry sequence structure controlled evolution circuit mapper."""

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

__all__: list[str] = ["PauliSequenceMapper", "PauliSequenceMapperSettings"]


class PauliSequenceMapperSettings(Settings):
    """Settings for PauliSequenceMapper."""

    def __init__(self):
        """Initialize PauliSequenceMapperSettings with default values.

        Attributes:
            power: The power of the controlled unitary to be constructed. It controls
                how many times the controlled evolution operator :math:`U` is repeated.

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power of the controlled unitary to be constructed.")


class PauliSequenceMapper(ControlledEvolutionCircuitMapper):
    r"""Controlled evolution circuit mapper using Pauli product formula term sequences.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a controlled version of :math:`U(t)` using the following pattern:

    1. Each Pauli operator :math:`P_j` is basis-rotated into the :math:`Z` basis.
    2. Qubits involved in :math:`P_j` are entangled into a sequence using CNOT gates.
    3. A controlled :math:`R_z` rotation implements
        :math:`e^{-i\,\theta_j\,P_j} \;\rightarrow\; \text{CRZ}(2 \theta_j)`.
    4. The basis rotations and entangling operations are uncomputed.

    The process repeats for all terms in :math:`U_{\mathrm{step}}`, for :math:`r` step repetitions,
    and for the specified power.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self, power: int = 1):
        """Initialize the PauliSequenceMapper.

        Args:
            power: The power of the controlled unitary to be constructed. It controls
                how many times the controlled evolution operator :math:`U` is repeated.

        """
        super().__init__()
        self._settings = PauliSequenceMapperSettings()
        self._settings.set("power", power)

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_evolution_circuit_mapper as the algorithm type name."""
        return "controlled_evolution_circuit_mapper"

    def _run_impl(
        self, controlled_evolution: ControlledTimeEvolutionUnitary, target_indices: Sequence[int] | None = None
    ) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled time evolution unitary.

        Args:
            controlled_evolution: The controlled time evolution unitary containing the Hamiltonian
            and evolution parameters.
            target_indices: Indices of the target qubits in the circuit. If None, defaults to all
            qubits except the control qubits at controlled_evolution.control_indices.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U^{\text{power}}`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the time evolution unitary container type is not supported.
            ValueError: If multiple control qubits are provided.

        """
        unitary_container = controlled_evolution.time_evolution_unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {controlled_evolution.get_unitary_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for time evolution unitary."
            )

        if len(controlled_evolution.control_indices) != 1:
            raise ValueError("PauliSequenceMapper currently only supports a single control qubit.")

        total_qubits = controlled_evolution.get_num_total_qubits()

        if target_indices is None:
            target_indices = [i for i in range(total_qubits) if i not in controlled_evolution.control_indices]

        circuit = QuantumCircuit(total_qubits)

        append_controlled_time_evolution(
            circuit,
            exponential_terms=unitary_container.step_terms,
            reps=unitary_container.step_reps,
            control_qubit=controlled_evolution.control_indices[0],
            target_qubits=target_indices,
            power=self._settings.get("power"),
        )

        qasm_str = qasm3.dumps(circuit)
        return Circuit(qasm=qasm_str)


def _append_controlled_pauli_rotation(
    circuit: QuantumCircuit,
    control_qubit: int,
    target_qubits: Sequence[int],
    term: ExponentiatedPauliTerm,
) -> QuantumCircuit:
    """Append a controlled ``exp(-i angle * P)`` to ``circuit``.

    Args:
        circuit: Quantum circuit receiving the controlled rotation.
        control_qubit: Index of the ancilla qubit providing the control.
        target_qubits: Ordered collection of target qubit indices.
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
    involved_qubits = [target_qubits[i] for i in involved_indices]

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
    *,
    exponential_terms: list[ExponentiatedPauliTerm],
    reps: int,
    control_qubit: int,
    target_qubits: Sequence[int],
    power: int = 1,
) -> None:
    """Append the controlled unitary ``(exp(-i H time))**power``.

    Args:
        circuit: Circuit being extended.
        exponential_terms: List of exponentiated Pauli terms describing the rotation axes.
        reps: Number of repetitions for each step.
        control_qubit: Index of the control qubit.
        target_qubits: The target qubit indices.
            If None, assumes target qubits are all qubits except the control qubit.
        power: Number of repeated applications (``U`` raised to ``power``).

    Raises:
        ValueError: If ``power`` is less than 1.

    """
    Logger.trace_entering()

    if power < 1:
        raise ValueError("power must be at least 1 for controlled time evolution.")

    # Create a new circuit for the controlled time evolution
    num_qubits = len(target_qubits) + 1
    power_evolution_circuit = QuantumCircuit(num_qubits, name=f"ctrl_time_evol_power_{power}")

    ctrl_evol_circuit = QuantumCircuit(num_qubits, name="ctrl_time_evol")
    for _ in range(reps):
        for term in exponential_terms:
            if np.isclose(term.angle, 0.0):
                continue
            _append_controlled_pauli_rotation(
                ctrl_evol_circuit,
                control_qubit,
                target_qubits,
                term,
            )
    # Convert to gate and repeat for the specified power
    ctrl_evol_gate = ctrl_evol_circuit.to_gate()
    for _ in range(power):
        power_evolution_circuit.append(ctrl_evol_gate, list(range(num_qubits)))

    # Convert to gate and append to original circuit
    circuit.append(power_evolution_circuit.to_gate(), list(range(num_qubits)))
