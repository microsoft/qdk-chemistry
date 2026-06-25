"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    ControlledUnitary,
    FactorizedHamiltonianContainer,
    QubitHamiltonian,
    Settings,
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
        qubit_hamiltonian: QubitHamiltonian | FactorizedHamiltonianContainer,
    ) -> list[Circuit]:
        """Build phase estimation circuits.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian or FactorizedHamiltonianContainer
                for which to build circuits.

        Returns:
            A list of quantum circuits for phase estimation.

        """

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: QubitHamiltonian | FactorizedHamiltonianContainer,
        power: int,
    ) -> tuple[Circuit, int, Circuit | None]:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian (or FactorizedHamiltonianContainer) to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_ancilla_qubits, ancilla_prep_op) where circuit implements
            controlled-:math:`U^{\\text{power}}`, num_ancilla_qubits is the number
            of ancilla qubits used by the unitary beyond the system qubits, and
            ancilla_prep_op is a Q# callable to initialize the ancillas (no-op if not needed).

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit = circuit_mapper.run(controlled_unitary=controlled_unitary)

        # Use mapper's num_ancillary_qubits if available (e.g. SOSSAMapper computes
        # alias-sampling-aware register sizes); fall back to container num_qubits.
        if hasattr(circuit_mapper, "num_ancillary_qubits"):
            container = controlled_unitary.unitary.get_container()
            num_ancilla_qubits = circuit_mapper.num_ancillary_qubits(container)
        else:
            num_system_qubits = qubit_hamiltonian.num_qubits
            num_ancilla_qubits = unitary_rep.get_num_qubits() - num_system_qubits

        # Get ancilla prep circuit from mapper if available (e.g. phase gradient init).
        if hasattr(circuit_mapper, "get_ancilla_prep_op"):
            ancilla_prep_circuit = circuit_mapper.get_ancilla_prep_op()
        else:
            ancilla_prep_circuit = None

        return circuit, num_ancilla_qubits, ancilla_prep_circuit


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
