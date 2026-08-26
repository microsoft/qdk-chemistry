"""QDK/Chemistry PREPARE-SELECT-PREPARE circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk import qsharp

from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer, Select
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = ["PSPMapper", "PSPMapperSettings"]


class PSPMapperSettings(Settings):
    """Settings for the PSPMapper.

    Attributes:
        prepare: Algorithm reference for the PREPARE oracle state preparation.
            Defaults to ``DensePureStatePreparation``.

    """

    def __init__(self):
        """Initialize the settings for PSPMapper."""
        super().__init__()
        self._set_default(
            "prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "dense_pure_state"),
            "Algorithm for the PREPARE oracle. ",
        )


class PSPMapper(CircuitMapper):
    r"""Circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    Composes a block encoding from:

    1. **PREPARE** — amplitude-loading into the ancilla register, resolved via
       the ``prepare`` setting.  Defaults to ``DensePureStatePreparation``.
    2. **SELECT** — Pauli SELECT oracle applied on the system register,
       constructed directly from the block-encoding container's SELECT data.

    The two callables are stitched together by the Q# ``PrepSelPrep`` operation:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    When the input is an :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`,
    the block encoding is additionally wrapped with the reflection operator to form a quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    """

    def __init__(self):
        """Initialize the PSPMapper."""
        super().__init__()
        self._settings = PSPMapperSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: ``"prepare_select_prepare"``.

        """
        return "prepare_select_prepare"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: ``"circuit_mapper"``.

        """
        return "circuit_mapper"

    @staticmethod
    def _build_pauli_select_op(select: Select):
        """Build the Pauli SELECT Q# operation from a Select data object.

        Converts each controlled operation's Pauli string into Q# ``Pauli`` enums
        and packages them with sign phases into a ``PauliSelectParams`` struct.

        Args:
            select: The SELECT oracle data object containing controlled operations,
                phases, and qubit layout.

        Returns:
            A Q# callable implementing the Pauli SELECT oracle.

        """
        pauli_terms: list[list[qsharp.Pauli]] = []
        control_states: list[int] = []
        for op in select.controlled_operations:
            base_paulis = [qsharp.Pauli.I] * select.num_target_qubits
            for i, pauli_char in enumerate(reversed(op.operation)):
                if pauli_char != "I":
                    base_paulis[i] = getattr(qsharp.Pauli, pauli_char)
            pauli_terms.append(base_paulis)
            control_states.append(op.ctrl_state)
        phases = [int(s) for s in select.phases]
        select_params = QSHARP_UTILS.Select.PauliSelectParams(
            pauliTerms=pauli_terms, signs=phases, controlStates=control_states
        )
        return QSHARP_UTILS.Select.MakeSelectOp(select_params)

    def resolve_lcu(self, container: UnitaryContainer) -> tuple[LCUContainer, bool]:
        """Unwrap a container into its LCU data and whether it is a quantum walk.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The LCU data and whether the container is a quantum walk.

        Raises:
            ValueError: If the container is neither an LCU nor an LCU walk.

        """
        if isinstance(container, LCUWalkContainer):
            return container.block_encoding, True
        if isinstance(container, LCUContainer):
            return container, False
        raise ValueError(
            f"Container type '{type(container).__name__}' is not supported. "
            "PSPMapper requires LCUContainer or LCUWalkContainer."
        )

    def build_select_ops(self, container: UnitaryContainer) -> tuple[Any, int]:
        """Return the SELECT oracle and the width of the system register it targets.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The Q# SELECT callable and the number of system qubits.

        """
        lcu, _ = self.resolve_lcu(container)
        return self._build_pauli_select_op(lcu.select), lcu.select.num_target_qubits

    def build_prep_ops(self, container: UnitaryContainer) -> tuple[Any, int, int]:
        """Return the PREPARE oracle and the widths of the ancilla it acts on.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The Q# PREPARE callable, the index width SELECT controls on, and the block
            ancilla width.

        Raises:
            ValueError: If the PREPARE circuit carries no Q# operation, does not declare the
                width it acts on, declares one too narrow for the index SELECT controls on, or
                expects shared ancilla this mapper does not supply.

        """
        lcu, _ = self.resolve_lcu(container)
        if lcu.prepare is None:
            return QSHARP_UTILS.PrepSelPrep.NoOpPrepare, 0, 0

        prepare_algorithm = self._create_nested("prepare")
        prepare_circuit = prepare_algorithm.run(lcu.prepare)
        prepare_op = prepare_circuit._qsharp_op  # noqa: SLF001
        if prepare_op is None:
            raise ValueError("The PREPARE circuit has no Q# operation to embed in the block encoding.")

        num_block_ancillas = prepare_circuit.num_qubits
        num_select_qubits = lcu.num_prepare_ancillas
        if num_block_ancillas is None:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' does not declare num_qubits, so the "
                "block ancilla register cannot be sized."
            )
        if num_block_ancillas <= 0:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' declares num_qubits={num_block_ancillas}, "
                "but a PREPARE oracle must act on at least one qubit."
            )
        if prepare_circuit.num_phase_gradient_ancillas:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' expects "
                f"{prepare_circuit.num_phase_gradient_ancillas} phase gradient ancilla, "
                "which PSPMapper does not supply."
            )
        if num_block_ancillas < num_select_qubits:
            raise ValueError(
                f"The PREPARE circuit acts on {num_block_ancillas} qubits, but the LCU decomposition indexes "
                f"{num_select_qubits} of them. SELECT would control on qubits PREPARE does not own."
            )
        return prepare_op, num_select_qubits, num_block_ancillas

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct the block-encoding circuit on the flat ``[system | ancilla]`` register.

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the block encoding.

        """
        container = unitary.get_container()
        _, use_quantum_walk = self.resolve_lcu(container)
        select_op, num_system_qubits = self.build_select_ops(container)
        prepare_op, num_select_qubits, num_block_ancillas = self.build_prep_ops(container)

        qsharp_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(
            prepare_op, select_op, num_system_qubits, num_select_qubits
        )
        if use_quantum_walk:
            reflection_op = QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(num_system_qubits, num_block_ancillas)
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeWalkOp(qsharp_op, reflection_op)

        if container.power != 1:
            qsharp_op = QSHARP_UTILS.CircuitComposition.MakeRepeatedOp(
                "PSPWalk" if use_quantum_walk else "PrepSelPrep",
                qsharp_op,
                container.power,
            )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
            parameter={
                "prepareOp": prepare_op,
                "selectOp": select_op,
                "numSystemQubits": num_system_qubits,
                "numSelectQubits": num_select_qubits,
                "numBlockAncillaQubits": num_block_ancillas,
                "power": container.power,
                "useWalk": use_quantum_walk,
            },
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=num_system_qubits + num_block_ancillas,
        )
