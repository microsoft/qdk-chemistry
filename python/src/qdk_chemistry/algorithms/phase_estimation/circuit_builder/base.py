"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from contextlib import contextmanager

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
        self._shared_base_unitary: UnitaryRepresentation | None = None
        self._share_base_unitary = False
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
        unitary_rep = self._powered_unitary(qubit_hamiltonian, power)
        num_ancilla_qubits = unitary_rep.get_num_qubits() - qubit_hamiltonian.num_qubits
        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit_mapper.settings().update("control_indices", [0])
        circuit = circuit_mapper.run(unitary_rep)
        return circuit, num_ancilla_qubits

    @contextmanager
    def _shared_unitary_scope(self):
        """Let the controlled builds inside the block share one base-power unitary."""
        previous_flag, previous_rep = self._share_base_unitary, self._shared_base_unitary
        self._share_base_unitary, self._shared_base_unitary = True, None
        try:
            yield
        finally:
            self._share_base_unitary, self._shared_base_unitary = previous_flag, previous_rep

    def _powered_unitary(self, qubit_hamiltonian: QubitOperator, power: int) -> UnitaryRepresentation:
        r"""Return :math:`U^{\\text{power}}`, reusing a shared base step where that is exact.

        A ``"repeat"`` power strategy leaves the evolution time, and therefore every
        Pauli angle, untouched; the power enters only as a step-repetition count. Inside
        a :meth:`_shared_unitary_scope` the decomposition is then built once and each
        power is served by rescaling ``step_reps``, which avoids repeating identical
        work for every phase-estimation bit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            The unitary representation for the requested power.

        """
        unitary_builder = self._create_nested("unitary_builder")
        settings = unitary_builder.settings()
        repeats_fixed_step = settings.has("power_strategy") and settings.get("power_strategy") == "repeat"

        if not self._share_base_unitary or not repeats_fixed_step:
            settings.update("power", power)
            return unitary_builder.run(qubit_hamiltonian)

        if self._shared_base_unitary is None:
            settings.update("power", 1)
            base_unitary = unitary_builder.run(qubit_hamiltonian)
            if not isinstance(base_unitary.get_container(), PauliProductFormulaContainer):
                self._share_base_unitary = False
                settings.update("power", power)
                return unitary_builder.run(qubit_hamiltonian)
            self._shared_base_unitary = base_unitary

        if power == 1:
            return self._shared_base_unitary

        container = self._shared_base_unitary.get_container()
        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=container.step_terms,
                step_reps=container.step_reps * power,
                num_qubits=container.num_qubits,
                scale=container.scale,
                conjugating_terms=container.conjugating_terms,
            )
        )

    @staticmethod
    def _validate_state_prep_width(state_preparation: Circuit, num_qubits_passed: int) -> None:
        """Check that the state preparation fits the register phase estimation hands it.

        Args:
            state_preparation: The state preparation circuit.
            num_qubits_passed: Width of the register phase estimation applies it to.

        Raises:
            ValueError: If the state preparation acts on more qubits than it is given.

        """
        width = state_preparation.num_qubits
        if width is not None and width > num_qubits_passed:
            raise ValueError(
                f"State preparation acts on {width} qubits but phase estimation applies it to "
                f"{num_qubits_passed}. Choose a state preparation that fits the system register."
            )


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
