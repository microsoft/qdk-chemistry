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

        Includes optional nested algorithm references for the circuit executor,
        controlled circuit mapper, and the measurement basis for the control
        qubit.

        """
        super().__init__()
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
        self._set_default(
            "test_basis",
            "string",
            HadamardTestBasis.X.value,
            "Measurement basis for the control qubit ('X', 'Y', or 'Z').",
            [basis.value for basis in HadamardTestBasis],
        )


class HadamardTest(Algorithm):
    """Hadamard test generator.

    Orchestrates the backend-agnostic Hadamard test workflow: it validates the
    inputs, delegates circuit construction (including mapping the target
    unitary into a controlled evolution circuit) to
    ``hadamard_test_circuit_builder`` (currently fixed to ``qdk``), and
    executes the resulting circuit with the nested ``circuit_executor``.
    """

    def __init__(
        self,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
        circuit_executor: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize a Hadamard test generator.

        Args:
            test_basis: Measurement basis for the control qubit.
            circuit_executor: Optional algorithm reference for circuit execution.
            controlled_circuit_mapper: Optional algorithm reference for controlled circuit mapping.

        """
        super().__init__()
        self._settings = HadamardTestSettings()
        self._settings.set("test_basis", test_basis.value)
        if circuit_executor is not None:
            if not isinstance(circuit_executor, AlgorithmRef):
                raise TypeError("circuit_executor must be an instance of AlgorithmRef.")
            self._settings.set("circuit_executor", circuit_executor)
        if controlled_circuit_mapper is not None:
            if not isinstance(controlled_circuit_mapper, AlgorithmRef):
                raise TypeError("controlled_circuit_mapper must be an instance of AlgorithmRef.")
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

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

        The measurement basis is read from this algorithm's settings (``test_basis``).

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

        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        circuit_builder = create("hadamard_test_circuit_builder", "qdk")
        circuit_builder.settings().set("test_basis", test_basis.value)
        controlled_circuit_mapper = self._settings.get("controlled_circuit_mapper")
        circuit_builder.settings().set("controlled_circuit_mapper", controlled_circuit_mapper)

        circuit = circuit_builder.run(
            state_preparation_circuit,
            unitary,
        )

        circuit_executor_ref = self._settings.get("circuit_executor")
        if circuit_executor_ref.algorithm_type != "circuit_executor":
            raise ValueError(
                "circuit_executor must reference algorithm type 'circuit_executor', "
                f"got '{circuit_executor_ref.algorithm_type}'."
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
