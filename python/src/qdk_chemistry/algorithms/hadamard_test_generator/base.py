"""QDK/Chemistry Hadamard test circuit generator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from abc import abstractmethod
from enum import Enum
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import ControlledEvolutionCircuitMapper
from qdk_chemistry.data import Circuit, CircuitExecutorData, ControlledTimeEvolutionUnitary, TimeEvolutionUnitary

__all__: list[str] = [
    "HadamardTest",
    "HadamardTestBasis",
    "HadamardTestFactory",
    "basis_to_qsharp_pauli",
]


class HadamardTestBasis(Enum):
    """Measurement bases supported by the Hadamard test control qubit."""

    X = "X"
    Y = "Y"

    def __str__(self) -> str:
        """Return the string label ("X" or "Y") for this basis."""
        return str(self.value)


def basis_to_qsharp_pauli(basis: HadamardTestBasis) -> Any:
    """Map a ``HadamardTestBasis`` to ``qsharp.Pauli`` for Q# interop."""
    try:
        from qdk import qsharp as _qsharp  # noqa: PLC0415
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "qdk.qsharp is required to convert Hadamard test bases into qsharp.Pauli values."
        ) from err

    return getattr(_qsharp.Pauli, basis.value)


class HadamardTest(Algorithm):
    """Abstract base class for Hadamard test generators."""

    def __init__(self):
        """Initialize a Hadamard test generator."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test."""
        return "hadamard_test"

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        num_system_qubits: int,
        time_evolution_unitary: TimeEvolutionUnitary,
        mapper_type: str = "pauli_sequence",
        unitary_power: int = 1,
        simulator_type: str = "qdk_full_state_simulator",
        shots: int = 1,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ) -> CircuitExecutorData:
        r"""Run the Hadamard test by building and executing a backend-specific circuit.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            time_evolution_unitary: Time evolution unitary :math:`\exp(-i H \Delta t)`.
            mapper_type: Algorithm name for controlled evolution circuit mapper.
            unitary_power: Power :math:`n` used to form the controlled unitary :math:`U^n`.
            simulator_type: Algorithm name for the circuit executor.
            shots: Number of shots to execute the circuit.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        Returns:
            CircuitExecutorData returned directly by the given simulator.

        """
        if not isinstance(test_basis, HadamardTestBasis):
            raise TypeError("test_basis must be an instance of HadamardTestBasis.")
        if not isinstance(time_evolution_unitary, TimeEvolutionUnitary):
            raise TypeError("time_evolution_unitary must be an instance of TimeEvolutionUnitary.")
        if not isinstance(mapper_type, str) or not mapper_type:
            raise TypeError("mapper_type must be a non-empty string.")
        if not isinstance(unitary_power, int):
            raise TypeError("unitary_power must be an integer.")
        if unitary_power < 1:
            raise ValueError("unitary_power must be a positive integer.")
        if not isinstance(simulator_type, str) or not simulator_type:
            raise TypeError("simulator_type must be a non-empty string.")
        if not isinstance(shots, int):
            raise TypeError("shots must be an integer.")
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        controlled_evolution = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=time_evolution_unitary,
            control_indices=[0],
        )

        try:
            mapper = create("controlled_evolution_circuit_mapper", mapper_type)
        except KeyError as err:
            raise ValueError(f"Unknown controlled evolution circuit mapper type: {mapper_type}.") from err
        if not isinstance(mapper, ControlledEvolutionCircuitMapper):
            raise TypeError("mapper_type did not resolve to a ControlledEvolutionCircuitMapper.")
        mapper.settings().update("power", unitary_power)
        ctrl_time_evol_unitary_circuit = mapper.run(controlled_evolution=controlled_evolution)

        circuit = self._build_hadamard_test_circuit(
            state_preparation_circuit,
            num_system_qubits,
            ctrl_time_evol_unitary_circuit,
            test_basis,
        )

        try:
            simulator = create("circuit_executor", simulator_type)
        except KeyError as err:
            raise ValueError(f"Unknown simulator type: {simulator_type}.") from err
        if not isinstance(simulator, CircuitExecutor):
            raise TypeError("simulator_type did not resolve to a CircuitExecutor.")

        return simulator.run(circuit, shots=shots)

    @abstractmethod
    def _build_hadamard_test_circuit(
        self,
        state_preparation_circuit: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ) -> Circuit:
        r"""Build the Hadamard test circuit for a given state and controlled unitary.

        Currently, the function only accepts the controlled unitary circuit whose index of ancilla qubit is 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        Returns:
            Circuit representing the Hadamard test workflow for the selected backend.

        """


class HadamardTestFactory(AlgorithmFactory):
    """Factory class for creating Hadamard test generator instances."""

    def __init__(self):
        """Initialize the HadamardTestFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as hadamard_test."""
        return "hadamard_test"

    def default_algorithm_name(self) -> str:
        """Return 'qdk_hadamard_test' as the default algorithm name."""
        return "qdk_hadamard_test"
