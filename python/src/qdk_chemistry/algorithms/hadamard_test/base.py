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

        Includes nested algorithm references for the circuit builder and the
        circuit executor, the measurement basis for the control qubit, and the
        number of ancilla qubits.

        """
        super().__init__()
        self._set_default(
            "circuit_builder",
            "algorithm_ref",
            AlgorithmRef("hadamard_test_circuit_builder", "qdk"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
        )
        self._set_default(
            "test_basis",
            "string",
            HadamardTestBasis.X.value,
            "Measurement basis for the control qubit ('X', 'Y', or 'Z').",
            [basis.value for basis in HadamardTestBasis],
        )
        self._set_default(
            "num_ancilla_qubits",
            "int",
            0,
            "Number of ancilla qubits needed by the controlled evolution (0 if none).",
        )


class HadamardTest(Algorithm):
    """Hadamard test generator.

    Orchestrates the backend-agnostic Hadamard test workflow: it validates the
    inputs, delegates the backend-specific circuit construction (including
    mapping the target unitary into a controlled evolution circuit) to the
    nested ``hadamard_test_circuit_builder``, and executes the resulting circuit
    with the nested ``circuit_executor``.
    """

    def __init__(
        self,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
        num_ancilla_qubits: int = 0,
    ):
        """Initialize a Hadamard test generator.

        Args:
            test_basis: Measurement basis for the control qubit.
            num_ancilla_qubits: Number of ancilla qubits needed by the controlled evolution.

        """
        super().__init__()
        self._settings = HadamardTestSettings()
        self._settings.set("test_basis", test_basis.value)
        if num_ancilla_qubits < 0:
            raise ValueError("num_ancilla_qubits must be a non-negative integer.")
        self._settings.set("num_ancilla_qubits", num_ancilla_qubits)

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test."""
        return "hadamard_test"

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
        shots: int,
    ) -> CircuitExecutorData:
        r"""Run the Hadamard test by building and executing a backend-specific circuit.

        The measurement basis and ancilla count are read from this algorithm's settings
        (``test_basis`` and ``num_ancilla_qubits``).

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).
            shots: Number of shots to execute the circuit.

        Returns:
            CircuitExecutorData returned directly by the given simulator.

        """
        if not isinstance(state_preparation_circuit, Circuit):
            raise TypeError("state_preparation_circuit must be an instance of Circuit.")
        if not isinstance(unitary, UnitaryRepresentation):
            raise TypeError("unitary must be an instance of UnitaryRepresentation.")
        if not isinstance(shots, int):
            raise TypeError("shots must be an integer.")
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        test_basis = HadamardTestBasis(self._settings.get("test_basis"))
        num_ancilla_qubits = self._settings.get("num_ancilla_qubits")
        if num_ancilla_qubits < 0:
            raise ValueError("num_ancilla_qubits must be a non-negative integer.")

        circuit_builder = self._create_nested("circuit_builder")
        circuit_builder.settings().set("test_basis", test_basis.value)
        circuit_builder.settings().set("num_ancilla_qubits", num_ancilla_qubits)

        circuit = circuit_builder.run(
            state_preparation_circuit,
            unitary,
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
