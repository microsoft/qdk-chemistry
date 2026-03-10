"""Reusable real-observable measurement helpers.

This module provides backend-agnostic utilities for measuring
``Re(<psi|U|psi>)`` using a Hadamard test where:

- ``|psi>`` is prepared by ``state_preparation``
- ``U`` is supplied as a prebuilt controlled evolution circuit

The helper is intentionally independent from specific phase-estimation
algorithms so it can be reused by ODMD, iterative phase estimation,
or other workflows that need overlap measurements.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, QuantumErrorProfile
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["measure_observable"]


def _create_circuit_from_qiskit(
    state_preparation: Circuit,
    num_system_qubits: int,
    ctrl_time_evol_unitary_circuit: Circuit,
) -> Circuit:
    """Build a Qiskit Hadamard-test circuit for measuring the real overlap.

    Args:
        state_preparation: Circuit that prepares the trial state on system qubits.
        num_system_qubits: Number of qubits in the system register.
        ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.

    Returns:
        Circuit encoded as QASM for backend execution.

    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3

    # Build the base circuit with registers.
    ancilla = QuantumRegister(1, "ancilla")
    system_target = QuantumRegister(num_system_qubits, "system")
    classical = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(ancilla, system_target, classical)

    # Apply state preparation.
    state_prep_qc = state_preparation.get_qiskit_circuit()
    circuit.append(state_prep_qc.to_gate(), system_target)

    # Prepare ancilla and apply controlled time evolution.
    control = ancilla[0]
    target_qubits = list(system_target)
    circuit.h(control)

    ctrl_evol_qc = ctrl_time_evol_unitary_circuit.get_qiskit_circuit()
    circuit.append(ctrl_evol_qc.to_gate(), [control, *target_qubits])

    # Final Hadamard and measurement for the real part.
    circuit.h(control)
    circuit.measure(control, classical[0])

    Logger.info("Completed qiskit circuit for real observable measurement.")
    return Circuit(qasm=qasm3.dumps(circuit))


def _create_circuit_from_qsharp_op(
    state_preparation: Circuit,
    num_system_qubits: int,
    ctrl_time_evol_unitary_circuit: Circuit,
) -> Circuit:
    """Build a Q# Hadamard-test circuit for measuring the real overlap.

    Args:
        state_preparation: Circuit that prepares the trial state on system qubits.
        num_system_qubits: Number of qubits in the system register.
        ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.

    Returns:
        Circuit containing compiled Q# and QIR representations.

    """
    state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
    ctrl_evol_op = ctrl_time_evol_unitary_circuit._qsharp_op  # noqa: SLF001
    hadamard_test_qsc = qsharp.circuit(
        QSHARP_UTILS.DynamicModeDecomposition.MakeODMDCircuit,
        state_prep_op,
        ctrl_evol_op,
        0,
        [1 + i for i in range(num_system_qubits)],
    )
    hadamard_test_qir = qsharp.compile(
        QSHARP_UTILS.DynamicModeDecomposition.MakeODMDCircuit,
        state_prep_op,
        ctrl_evol_op,
        0,
        [1 + i for i in range(num_system_qubits)],
    )

    Logger.info("Completed qsharp circuit for real observable measurement.")
    return Circuit(qsharp=hadamard_test_qsc, qir=hadamard_test_qir)


def measure_observable(
    state_preparation: Circuit,
    num_system_qubits: int,
    ctrl_time_evol_unitary_circuit: Circuit,
    circuit_executor: CircuitExecutor,
    shots: int,
    noise: QuantumErrorProfile | None = None,
) -> float:
    """Measure ``Re(<psi|U|psi>)`` using a backend-supported Hadamard test.

    Args:
        state_preparation: Circuit that prepares the trial state ``|psi>``.
        num_system_qubits: Number of system qubits acted on by ``state_preparation`` and ``U``.
        ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing ``U``.
        circuit_executor: Backend executor used to run the generated measurement circuit.
        shots: Number of shots used for the expectation estimate.
        noise: Optional noise profile for noisy simulation.

    Returns:
        Real-valued overlap estimate computed from ancilla measurement counts.

    Raises:
        RuntimeError: If neither Q# nor Qiskit representations are available.

    """
    if state_preparation._qsharp_op and ctrl_time_evol_unitary_circuit._qsharp_op:  # noqa: SLF001
        circuit = _create_circuit_from_qsharp_op(
            state_preparation=state_preparation,
            num_system_qubits=num_system_qubits,
            ctrl_time_evol_unitary_circuit=ctrl_time_evol_unitary_circuit,
        )
    elif state_preparation.get_qiskit_circuit() and ctrl_time_evol_unitary_circuit.get_qiskit_circuit():
        circuit = _create_circuit_from_qiskit(
            state_preparation=state_preparation,
            num_system_qubits=num_system_qubits,
            ctrl_time_evol_unitary_circuit=ctrl_time_evol_unitary_circuit,
        )
    else:
        raise RuntimeError("Failed to measure observable: Q# operations or Qiskit dependencies are not available.")

    executor_data = circuit_executor.run(circuit, shots=shots, noise=noise)
    bitstring_result = executor_data.bitstring_counts
    Logger.info(
        "Measured real observable from Hadamard test, "
        f"{bitstring_result.get('0', 0)} zeros, {bitstring_result.get('1', 0)} ones"
    )
    observable_value = (bitstring_result.get("0", 0) - bitstring_result.get("1", 0)) / shots
    Logger.info(f"Measured observable value {observable_value}")
    return observable_value
