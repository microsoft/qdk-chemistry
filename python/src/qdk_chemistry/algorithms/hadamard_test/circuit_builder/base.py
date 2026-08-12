"""QDK/Chemistry Hadamard test circuit builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hadamard_test.hadamard_test import HadamardTestBasis
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Settings,
    UnitaryRepresentation,
)

__all__: list[str] = [
    "HadamardTestCircuitBuilder",
    "HadamardTestCircuitBuilderFactory",
    "HadamardTestCircuitBuilderSettings",
]


class HadamardTestCircuitBuilderSettings(Settings):
    """Settings for the Hadamard test circuit builder algorithm."""

    def __init__(self):
        """Initialize the settings for the Hadamard test circuit builder."""
        super().__init__()
        self._set_default(
            "controlled_circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        self._set_default(
            "test_basis",
            "string",
            HadamardTestBasis.X.value,
            "Measurement basis for the control qubit ('X' or 'Y').",
            [basis.value for basis in HadamardTestBasis],
        )


class HadamardTestCircuitBuilder(Algorithm):
    """Abstract base class for Hadamard test circuit builders.

    A circuit builder turns a prepared state and a target unitary into a single
    backend-specific Hadamard test circuit. It owns the controlled circuit
    mapper used to synthesize the controlled evolution.
    """

    def __init__(
        self,
        controlled_circuit_mapper: AlgorithmRef | None = None,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ):
        """Initialize the Hadamard test circuit builder.

        Args:
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        """
        super().__init__()
        self._settings = HadamardTestCircuitBuilderSettings()
        if controlled_circuit_mapper is not None:
            if not isinstance(controlled_circuit_mapper, AlgorithmRef):
                raise TypeError("controlled_circuit_mapper must be an instance of AlgorithmRef.")
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

        if isinstance(test_basis, HadamardTestBasis):
            test_basis_value = test_basis.value
        elif isinstance(test_basis, str):
            try:
                test_basis_value = HadamardTestBasis(test_basis).value
            except ValueError as err:
                raise ValueError("test_basis must be one of {'X', 'Y'}.") from err
        else:
            raise TypeError("test_basis must be an instance of HadamardTestBasis or a string.")

        self._settings.set("test_basis", test_basis_value)

    def type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_circuit_builder."""
        return "hadamard_test_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
    ) -> Circuit:
        r"""Build the Hadamard test circuit for a given state and target unitary.

        The unitary is mapped into a controlled evolution circuit internally via
        :meth:`_create_controlled_circuit`.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).

        Returns:
            Circuit representing the Hadamard test workflow for the selected backend.

        """

    def _create_controlled_circuit(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Map a target unitary into a controlled evolution circuit.

        Map ``unitary`` controlled on the Hadamard test control qubit (index 0)
        and runs the nested ``controlled_circuit_mapper`` to synthesize the controlled-:math:`U` circuit.

        Args:
            unitary: Unitary representation :math:`U` to map into a controlled circuit.

        Returns:
            The controlled circuit implementing controlled-:math:`U`.

        """
        circuit_mapper_ref = self._settings.get("controlled_circuit_mapper")
        if circuit_mapper_ref.algorithm_type != "controlled_circuit_mapper":
            raise ValueError(
                "controlled_circuit_mapper must reference algorithm type 'controlled_circuit_mapper', "
                f"got '{circuit_mapper_ref.algorithm_type}'."
            )

        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit_mapper.settings().update("control_indices", [0])
        return circuit_mapper.run(unitary=unitary)


class HadamardTestCircuitBuilderFactory(AlgorithmFactory):
    """Factory class for creating Hadamard test circuit builder instances."""

    def __init__(self):
        """Initialize the HadamardTestCircuitBuilderFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as hadamard_test_circuit_builder."""
        return "hadamard_test_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return 'qdk' as the default algorithm name."""
        return "qdk"
