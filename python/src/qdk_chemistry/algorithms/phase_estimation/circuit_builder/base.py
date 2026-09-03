"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from collections.abc import Sequence

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    Settings,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    PauliProductFormulaContainer,
)

__all__: list[str] = [
    "IterativeQpeCircuitBuilder",
    "QpeCircuitBuilder",
    "QpeCircuitBuilderFactory",
    "QpeCircuitBuilderSettings",
    "StandardQpeCircuitBuilder",
]


class QpeCircuitBuilderSettings(Settings):
    """Settings for the Phase Estimation Builder algorithm."""

    def __init__(self):
        """Initialize the settings for the Phase Estimation Builder.

        Includes nested algorithm references for the evolution builder
        and the circuit mapper used to construct phase estimation circuits.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        )
        self._set_default(
            "controlled_circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )


class QpeCircuitBuilder(Algorithm):
    """Abstract base class for phase estimation circuit builders."""

    def __init__(
        self,
        num_bits: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize the QpeCircuitBuilder with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.

        """
        super().__init__()
        self._settings = QpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if controlled_circuit_mapper is not None:
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

    def type_name(self) -> str:
        """Return the algorithm type name as qpe_circuit_builder."""
        return "qpe_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Build phase estimation circuits.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build circuits.

        Returns:
            A list of quantum circuits for phase estimation.

        """

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: QubitOperator,
        power: int,
    ) -> tuple[Circuit, int]:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_ancilla_qubits) where circuit implements
            controlled-:math:`U^{\\text{power}}` and num_ancilla_qubits is the number
            of ancilla qubits used by the unitary beyond the system qubits.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        num_ancilla_qubits = unitary_rep.get_num_qubits() - qubit_hamiltonian.num_qubits
        return self._map_controlled_circuit(unitary_rep), num_ancilla_qubits

    def _map_controlled_circuit(self, unitary_rep: UnitaryRepresentation) -> Circuit:
        """Map a unitary representation to a circuit controlled on qubit 0.

        Args:
            unitary_rep: The unitary representation to map.

        Returns:
            The controlled circuit.

        """
        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit_mapper.settings().update("control_indices", [0])
        return circuit_mapper.run(unitary_rep)

    def _create_controlled_circuits(
        self,
        qubit_hamiltonian: QubitOperator,
        powers: Sequence[int],
    ) -> tuple[list[Circuit], int]:
        r"""Create controlled-:math:`U^{p}` circuits for every power *p* in *powers*.

        Under the ``"repeat"`` power strategy the power never reaches the product
        formula itself, it only multiplies the repetition count, so the decomposition
        is built once and re-mapped per power instead of being rebuilt for every rung
        of the ladder. Any other strategy falls back to one full build per power.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            powers: The powers to which the unitary should be raised.

        Returns:
            A tuple of (circuits, num_ancilla_qubits) with one circuit per entry of
            *powers*, in the same order.

        """
        unitary_builder = self._create_nested("unitary_builder")
        settings = unitary_builder.settings()

        if settings.has("power_strategy") and settings.get("power_strategy") == "repeat":
            settings.update("power", 1)
            unitary_rep = unitary_builder.run(qubit_hamiltonian)
            container = unitary_rep.get_container()
            if isinstance(container, PauliProductFormulaContainer):
                num_ancilla_qubits = unitary_rep.get_num_qubits() - qubit_hamiltonian.num_qubits
                circuits = [
                    self._map_controlled_circuit(
                        UnitaryRepresentation(
                            container=PauliProductFormulaContainer(
                                step_terms=container.step_terms,
                                step_reps=container.step_reps * power,
                                num_qubits=container.num_qubits,
                                scale=container.scale,
                            )
                        )
                    )
                    for power in powers
                ]
                return circuits, num_ancilla_qubits

        circuits = []
        num_ancilla_qubits = 0
        for power in powers:
            circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, power)
            circuits.append(circuit)
        return circuits, num_ancilla_qubits


class QpeCircuitBuilderFactory(AlgorithmFactory):
    """Factory class for creating QpeCircuitBuilder instances."""

    def __init__(self):
        """Initialize the QpeCircuitBuilderFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as qpe_circuit_builder."""
        return "qpe_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return qdk_iterative as default algorithm name."""
        return "qdk_iterative"


class IterativeQpeCircuitBuilder(QpeCircuitBuilder):
    """Abstract base class for iterative phase estimation circuit builders.

    Serves as a type-checking abstraction for implementations of the iterative
    (Kitaev-style) quantum phase estimation algorithm.

    """


class StandardQpeCircuitBuilder(QpeCircuitBuilder):
    """Abstract base class for standard (QFT-based) phase estimation circuit builders.

    Serves as a type-checking abstraction for implementations of the standard
    (non-iterative) quantum phase estimation algorithm using QFT.

    """
