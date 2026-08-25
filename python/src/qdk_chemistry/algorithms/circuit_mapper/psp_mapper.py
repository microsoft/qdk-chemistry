"""QDK/Chemistry PREPARE-SELECT-PREPARE circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
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

__all__: list[str] = ["PSPMapper", "PSPMapperSettings", "PrepareRegisters"]


@dataclass(frozen=True)
class PrepareRegisters:
    r"""How a block encoding lays out the registers its PREPARE oracle needs.

    Assembled from the widths the state preparation reports. For simple state
    preparations all three collapse to the same index register, but a PREPARE oracle may
    leave scratch entangled with the index it produces, and may read a resource the whole
    circuit shares rather than re-create it per call.

    Attributes:
        num_select_qubits: Width of the index SELECT controls on. Only these qubits carry
            index information.
        num_block_ancillas: Total width PREPARE owns, index plus entangled scratch.
            ``PREPARE``\ :sup:`†` returns all of it to :math:`|0\rangle`, and a
            qubitization walk reflects about exactly this register.
        num_shared_ancillas: Width of ancilla prepared once for the whole circuit and left
            in a non-zero state between uses, such as a phase gradient. Deliberately
            excluded from the walk reflection.

    """

    num_select_qubits: int
    num_block_ancillas: int
    num_shared_ancillas: int = 0


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

    def build_prepare_select_ops(self, container: UnitaryContainer) -> tuple[Any, Any, int, PrepareRegisters]:
        """Return the PREPARE and SELECT Q# oracles, the system register size, and the layout.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The PREPARE Q# callable, the SELECT Q# callable, the system register size, and
            the registers PREPARE expects.

        Raises:
            ValueError: If the PREPARE oracle's index width disagrees with the number of
                ancilla the LCU decomposition sized itself for.

        """
        lcu, _ = self.resolve_lcu(container)
        if lcu.prepare is None:
            return (
                QSHARP_UTILS.PrepSelPrep.NoOpPrepare,
                self._build_pauli_select_op(lcu.select),
                lcu.select.num_target_qubits,
                PrepareRegisters(num_select_qubits=0, num_block_ancillas=0),
            )

        prepare = self._create_nested("prepare")
        num_select_qubits = prepare.num_system_qubits(lcu.prepare)
        if num_select_qubits != lcu.num_prepare_ancillas:
            raise ValueError(
                f"PREPARE oracle indexes {num_select_qubits} qubits but the LCU "
                f"decomposition has {lcu.num_prepare_ancillas} prepare ancilla. SELECT would "
                "control on the wrong register. This happens when the state preparation "
                "indexes coefficients by list position while the decomposition indexes them "
                "by determinant bit pattern."
            )
        registers = PrepareRegisters(
            num_select_qubits=num_select_qubits,
            num_block_ancillas=num_select_qubits + prepare.num_entangled_ancillas(lcu.prepare),
            num_shared_ancillas=prepare.num_phase_gradient_ancillas(lcu.prepare),
        )
        return (
            prepare.prepare_oracle(lcu.prepare),
            self._build_pauli_select_op(lcu.select),
            lcu.select.num_target_qubits,
            registers,
        )

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
        prepare_op, select_op, num_system, registers = self.build_prepare_select_ops(container)

        qsharp_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(
            prepare_op, select_op, num_system, registers.num_select_qubits
        )
        if use_quantum_walk:
            reflection_op = QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(num_system, registers.num_block_ancillas)
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeWalkOp(qsharp_op, reflection_op)

        if container.power != 1:
            qsharp_op = QSHARP_UTILS.CircuitComposition.MakeRepeatedOp(
                "PSPWalk" if use_quantum_walk else "PrepSelPrep",
                qsharp_op,
                container.power,
            )

        # Outermost, so the one gradient preparation amortizes over every repetition.
        if registers.num_shared_ancillas > 0:
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeWithSharedPhaseGradientOp(
                qsharp_op, num_system + registers.num_block_ancillas
            )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
            parameter={
                "prepareOp": prepare_op,
                "selectOp": select_op,
                "numSystemQubits": num_system,
                "numSelectQubits": registers.num_select_qubits,
                "numBlockAncillaQubits": registers.num_block_ancillas,
                "numSharedQubits": registers.num_shared_ancillas,
                "power": container.power,
                "useWalk": use_quantum_walk,
            },
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=num_system + registers.num_block_ancillas + registers.num_shared_ancillas,
        )
