"""QDK/Chemistry Hadamard test circuit generator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from enum import Enum
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    CircuitExecutorData,
    ControlledUnitary,
    Settings,
    UnitaryRepresentation,
)

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
    Z = "Z"

    def __str__(self) -> str:
        """Return the string label ("X", "Y", or "Z") for this basis."""
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


class HadamardTestSettings(Settings):
    """Settings for the Hadamard test algorithm."""

    def __init__(self):
        """Initialize the settings for the Hadamard test.

        Includes nested algorithm references for the circuit builder,
        the controlled circuit mapper and the circuit executor.

        """
        super().__init__()
        self._set_default(
            "circuit_builder",
            "algorithm_ref",
            AlgorithmRef("hadamard_test_circuit_builder", "qdk_circuit_builder"),
        )
        self._set_default(
            "controlled_circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
        )


class HadamardTest(Algorithm):
    """Hadamard test generator.

    Orchestrates the backend-agnostic Hadamard test workflow: it validates the
    inputs, builds the controlled evolution circuit via the nested
    ``controlled_circuit_mapper``, delegates the backend-specific circuit
    construction to the nested ``hadamard_test_circuit_builder``, and executes
    the resulting circuit with the nested ``circuit_executor``.
    """

    def __init__(self):
        """Initialize a Hadamard test generator."""
        super().__init__()
        self._settings = HadamardTestSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test."""
        return "hadamard_test"

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
        shots: int,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
        num_ancilla_qubits: int = 0,
    ) -> CircuitExecutorData:
        r"""Run the Hadamard test by building and executing a backend-specific circuit.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).
            shots: Number of shots to execute the circuit.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X``, ``HadamardTestBasis.Y``, or
              ``HadamardTestBasis.Z``).
            num_ancilla_qubits: Number of ancilla qubits needed by the controlled evolution (0 if none).

        Returns:
            CircuitExecutorData returned directly by the given simulator.

        """
        if not isinstance(test_basis, HadamardTestBasis):
            raise TypeError("test_basis must be an instance of HadamardTestBasis.")
        if not isinstance(unitary, UnitaryRepresentation):
            raise TypeError("unitary must be an instance of UnitaryRepresentation.")
        num_system_qubits = unitary.get_num_qubits()
        if not isinstance(shots, int):
            raise TypeError("shots must be an integer.")
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        controlled_evolution = ControlledUnitary(
            unitary=unitary,
            control_indices=[0],
        )

        mapper = self._create_nested("controlled_circuit_mapper")

        ctrl_time_evol_unitary_circuit = mapper.run(controlled_unitary=controlled_evolution)

        circuit_builder = self._create_nested("circuit_builder")

        circuit = circuit_builder.run(
            state_preparation_circuit,
            num_system_qubits,
            ctrl_time_evol_unitary_circuit,
            test_basis,
            num_ancilla_qubits,
        )

        circuit_executor = self._create_nested("circuit_executor")

        return circuit_executor.run(circuit, shots=shots)

    def name(self) -> str:
        """Return the name of the Hadamard test algorithm."""
        return "qdk_hadamard_test"


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
