"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from abc import abstractmethod
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    HamiltonianDescription,
    QubitOperator,
    Settings,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSABlockEncodingContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

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

    #: Whether the builder prepares the phase gradient its controlled unitaries declare and shares it among them.
    _shares_phase_gradient: bool = False

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
        qubit_hamiltonian: HamiltonianDescription,
    ) -> list[Circuit]:
        """Build phase estimation circuits.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The Hamiltonian description the unitary builder takes.

        Returns:
            A list of quantum circuits for phase estimation.

        """

    def _build_controlled_unitary(
        self,
        qubit_hamiltonian: HamiltonianDescription,
        power: int,
    ) -> tuple[Circuit, int, int]:
        r"""Build the controlled unitary once, with the widths of its registers.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The Hamiltonian description the unitary builder takes.
            power: The power to which the unitary should be raised.

        Returns:
            The controlled circuit, work width, and system width; work excludes the system, the control,
            and the shared phase gradient that phase estimation appends itself.

        Raises:
            ValueError: If the builder does not support the unitary representation, the callable width is
                too small, or the circuit requests a shared phase gradient that this builder does not initialize.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        self._validate_unitary(unitary_rep)
        num_system_qubits = self._num_system_qubits(qubit_hamiltonian, unitary_rep)
        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit_mapper.settings().update("control_indices", [0])
        circuit = circuit_mapper.run(unitary_rep)
        width = circuit.num_qubits
        if width is None:
            warnings.warn(
                f"Controlled circuit mapper '{circuit_mapper.name()}' does not declare num_qubits. "
                "Falling back to the unitary representation's width is deprecated; declare the full "
                "callable width, including its control and working qubits.",
                DeprecationWarning,
                stacklevel=2,
            )
            width = unitary_rep.get_num_qubits() + 1
        num_gradient_qubits = circuit.metadata.num_phase_gradient_ancillas
        if num_gradient_qubits and not self._shares_phase_gradient:
            raise ValueError("This QPE builder does not initialize shared phase gradient ancillas.")
        minimum_width = num_system_qubits + 1 + num_gradient_qubits
        if width < minimum_width:
            raise ValueError(
                f"The controlled circuit declares {width} qubits, but requires at least "
                f"{minimum_width} including its control and phase gradient."
            )
        return circuit, width - minimum_width, num_system_qubits

    def _validate_unitary(self, unitary_rep: UnitaryRepresentation) -> None:
        """Reject a unitary representation this builder cannot estimate.

        Args:
            unitary_rep: The unitary representation the controlled circuit is built from.

        """

    def _controlled_circuit_and_widths(
        self,
        qubit_hamiltonian: HamiltonianDescription,
        power: int,
    ) -> tuple[Circuit, int, int]:
        r"""Build controlled-:math:`U^{\\text{power}}` with the widths of its registers.

        A :class:`~qdk_chemistry.data.QubitOperator` goes through :meth:`_create_controlled_circuit`,
        so subclasses that override it keep working; other descriptions use :meth:`_build_controlled_unitary`.

        Args:
            qubit_hamiltonian: The Hamiltonian description the unitary builder takes.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_ancilla_qubits, num_system_qubits).

        """
        if isinstance(qubit_hamiltonian, QubitOperator):
            circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, power)
            return circuit, num_ancilla_qubits, qubit_hamiltonian.num_qubits
        return self._build_controlled_unitary(qubit_hamiltonian, power)

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: HamiltonianDescription,
        power: int,
    ) -> tuple[Circuit, int]:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Args:
            qubit_hamiltonian: The Hamiltonian description the unitary builder takes.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_ancilla_qubits) where circuit implements
            controlled-:math:`U^{\\text{power}}` and num_ancilla_qubits is the number
            of ancilla qubits used by the unitary beyond the system qubits.

        """
        circuit, num_ancilla_qubits, _ = self._build_controlled_unitary(qubit_hamiltonian, power)
        return circuit, num_ancilla_qubits

    @staticmethod
    def _num_system_qubits(qubit_hamiltonian: HamiltonianDescription, unitary_rep: UnitaryRepresentation) -> int:
        """Return the width of the system register the unitary acts on.

        Args:
            qubit_hamiltonian: The Hamiltonian description the unitary builder takes.
            unitary_rep: The unitary built from ``qubit_hamiltonian``.

        Returns:
            The number of system qubits.

        """
        if isinstance(qubit_hamiltonian, QubitOperator):
            return qubit_hamiltonian.num_qubits
        return unitary_rep.get_num_system_qubits()

    @staticmethod
    def _shared_register(state_preparation: Circuit, unitary_circuits: list[Circuit]) -> tuple[Any, int, bool, bool]:
        """Resolve the phase gradient register the state preparation and the unitaries request through their metadata.

        Args:
            state_preparation: The state preparation circuit.
            unitary_circuits: The unitaries phase estimation queries, which share one register layout.

        Returns:
            The register preparation, its width, and whether the state preparation and the unitaries consume it.

        Raises:
            ValueError: If the unitaries disagree on the register, or the state preparation asks for a different width.

        """
        unitary_widths = {circuit.metadata.num_phase_gradient_ancillas for circuit in unitary_circuits}
        if len(unitary_widths) > 1:
            raise ValueError(
                "The unitaries of one phase estimation must all request the same phase gradient register. "
                f"Got widths {sorted(unitary_widths)}."
            )
        unitary_width = unitary_widths.pop() if unitary_widths else 0
        state_prep_width = state_preparation.metadata.num_phase_gradient_ancillas
        if unitary_width and state_prep_width and unitary_width != state_prep_width:
            raise ValueError(
                f"State preparation expects {state_prep_width} phase gradient ancillas but the unitaries "
                f"expect {unitary_width}."
            )
        width = max(unitary_width, state_prep_width)
        if not width:
            return QSHARP_UTILS.PrepSelPrep.NoOpPrepare, 0, False, False
        return QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, width, bool(state_prep_width), bool(unitary_width)

    @staticmethod
    def _validate_state_prep_width(state_preparation: Circuit, num_system_qubits: int) -> None:
        """Check that the state preparation fits the register phase estimation hands it.

        Args:
            state_preparation: The state preparation circuit.
            num_system_qubits: Width of the system register phase estimation applies it to.

        Raises:
            ValueError: If the state preparation acts on more system qubits than it is given, or, when it consumes
                the shared phase gradient that phase estimation appends after the system register, on fewer.

        """
        width = state_preparation.num_qubits
        if width is None:
            return
        num_gradient_qubits = state_preparation.metadata.num_phase_gradient_ancillas
        num_qubits_passed = num_system_qubits + num_gradient_qubits
        if width > num_qubits_passed:
            raise ValueError(
                f"State preparation acts on {width} qubits but phase estimation applies it to "
                f"{num_qubits_passed}. Choose a state preparation that fits the system register."
            )
        if num_gradient_qubits and width != num_qubits_passed:
            raise ValueError(
                f"State preparation acts on {width - num_gradient_qubits} system qubits but phase estimation has "
                f"{num_system_qubits}. A state preparation that consumes the phase gradient must cover the whole "
                "system register, because phase estimation appends the gradient after it."
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

    def _validate_unitary(self, unitary_rep: UnitaryRepresentation) -> None:
        r"""Reject the SOSSA walk, whose eigenstates defeat the phase feedback.

        A system eigenstate is an equal mix of the walk eigenvectors at :math:`\pm\phi`, and every
        sample re-prepares that mix. Feedback that corrects one branch mis-corrects the other, so
        a bit is right with probability :math:`(1 + \cos^2\Phi)/2` for feedback angle :math:`\Phi`,
        which approaches 1/2 for general :math:`\phi`.

        Args:
            unitary_rep: The unitary representation the controlled circuit is built from.

        Raises:
            ValueError: If ``unitary_rep`` holds a SOSSA block encoding.

        """
        if isinstance(unitary_rep.get_container(), SOSSABlockEncodingContainer):
            raise ValueError(
                "Iterative phase estimation does not support the SOSSA block encoding: its phase feedback "
                "cannot separate the walk eigenvalues at +phi and -phi. Use qdk_standard or qdk_unary."
            )


class StandardQpeCircuitBuilder(QpeCircuitBuilder):
    """Abstract base class for standard (QFT-based) phase estimation circuit builders.

    Serves as a type-checking abstraction for implementations of the standard
    (non-iterative) quantum phase estimation algorithm using QFT.

    """
