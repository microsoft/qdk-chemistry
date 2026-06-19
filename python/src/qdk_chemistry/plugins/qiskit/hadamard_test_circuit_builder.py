"""Qiskit Hadamard test circuit builder implementation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hadamard_test.base import HadamardTestBasis
from qdk_chemistry.algorithms.hadamard_test.circuit_builder.base import HadamardTestCircuitBuilder
from qdk_chemistry.data import AlgorithmRef, Circuit, UnitaryRepresentation
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitHadamardTestCircuitBuilder"]


class QiskitHadamardTestCircuitBuilder(HadamardTestCircuitBuilder):
    """Hadamard test circuit builder based on the Qiskit framework."""

    def __init__(
        self,
        controlled_circuit_mapper: AlgorithmRef | None = None,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ):
        """Initialize QiskitHadamardTestCircuitBuilder.

        Args:
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X``, ``HadamardTestBasis.Y``, or
              ``HadamardTestBasis.Z``).

        """
        Logger.trace_entering()
        super().__init__(
            controlled_circuit_mapper=controlled_circuit_mapper,
            test_basis=test_basis,
        )

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Qiskit backend.

        The target unitary is mapped into a controlled evolution circuit internally; the
        resulting controlled unitary circuit must place its control qubit at index 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).

        Returns:
            Circuit containing the OpenQASM3 representation of the Qiskit Hadamard test circuit.

        Raises:
            ModuleNotFoundError: If Qiskit is not installed.
            ValueError: If input circuits are incompatible with the expected qubit layout.

        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3  # noqa: PLC0415
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Qiskit is required to use QiskitHadamardTestCircuitBuilder. "
                "Install qiskit or use QdkHadamardTestCircuitBuilder."
            ) from err

        test_basis = HadamardTestBasis(self._settings.get("test_basis"))
        num_system_qubits = unitary.get_num_qubits()
        ctrl_time_evol_unitary_circuit = self._create_controlled_circuit(unitary)

        # Build the base circuit with registers.
        control = QuantumRegister(1, "control")
        system_target = QuantumRegister(num_system_qubits, "system")
        registers = [control, system_target]
        classical = ClassicalRegister(1, "c")
        registers.append(classical)
        circuit = QuantumCircuit(*registers)

        # Apply state preparation.
        try:
            state_prep_qc = state_preparation_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError(
                "Input state_preparation_circuit cannot be used for QiskitHadamardTestCircuitBuilder."
            ) from err
        if state_prep_qc.num_qubits != num_system_qubits:
            raise ValueError(
                "Input state_preparation_circuit has incompatible width for "
                f"QiskitHadamardTestCircuitBuilder: expected {num_system_qubits} "
                f"qubits, got {state_prep_qc.num_qubits}."
            )
        circuit.append(state_prep_qc.to_gate(), system_target)

        # Prepare control and apply controlled time evolution.
        control_qubit = control[0]
        target_qubits = list(system_target)

        circuit.h(control_qubit)

        try:
            ctrl_evol_qc = ctrl_time_evol_unitary_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError(
                "Input ctrl_time_evol_unitary_circuit cannot be used for QiskitHadamardTestCircuitBuilder."
            ) from err
        expected_ctrl_evol_qubits = 1 + num_system_qubits
        if ctrl_evol_qc.num_qubits != expected_ctrl_evol_qubits:
            raise ValueError(
                "Input ctrl_time_evol_unitary_circuit has incompatible width for "
                f"QiskitHadamardTestCircuitBuilder: expected {expected_ctrl_evol_qubits} "
                f"qubits, got {ctrl_evol_qc.num_qubits}."
            )
        circuit.append(ctrl_evol_qc.to_gate(), [control_qubit, *target_qubits])

        # Final basis rotation and measurement on the control qubit.
        if test_basis is HadamardTestBasis.X:
            circuit.h(control_qubit)
        elif test_basis is HadamardTestBasis.Y:
            circuit.sdg(control_qubit)
            circuit.h(control_qubit)
        elif test_basis is HadamardTestBasis.Z:
            pass
        else:
            raise ValueError(f"Unsupported test basis: {test_basis}.")
        circuit.measure(control_qubit, classical[0])

        Logger.debug(f"Completed qiskit circuit for measurement on {test_basis.value} basis.")
        return Circuit(qasm=qasm3.dumps(circuit))

    def name(self) -> str:
        """Return the name of the QiskitHadamardTestCircuitBuilder algorithm."""
        return "qiskit"
