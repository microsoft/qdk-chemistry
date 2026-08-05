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
            "Algorithm for the PREPARE oracle state preparation. ",
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

    def resolve_lcu(self, container: Any) -> tuple[LCUContainer, bool]:
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

    def build_prepare_select_ops(self, container: Any) -> tuple[Any, Any, int]:
        """Return the PREPARE and SELECT Q# oracles and the system register size.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The PREPARE Q# callable, the SELECT Q# callable, and the system register size.

        """
        lcu, _ = self.resolve_lcu(container)
        if lcu.prepare is not None:
            prepare_op = self._create_nested("prepare").run(lcu.prepare)._qsharp_op  # noqa: SLF001
        else:
            prepare_op = QSHARP_UTILS.PrepSelPrep.NoOpPrepare
        return prepare_op, self._build_pauli_select_op(lcu.select), lcu.select.num_target_qubits

    def block_encoding_op(self, container: Any):
        """Return one application of the block encoding, without reflection or power.

        Args:
            container: The container held by the unitary representation.

        Returns:
            A Q# callable applying ``B[H]`` once to the flat ``[system | ancilla]`` register.

        """
        prepare_op, select_op, num_system = self.build_prepare_select_ops(container)
        return QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(prepare_op, select_op, num_system)

    def reflection_op(self, container: Any):
        """Return the reflection a qubitization walk pairs the block encoding with.

        Args:
            container: The container held by the unitary representation.

        Returns:
            A Q# callable reflecting about the all-zero state of the ancilla register.

        """
        lcu, _ = self.resolve_lcu(container)
        return QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(lcu.select.num_target_qubits)

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the block encoding needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register.

        """
        lcu, _ = self.resolve_lcu(container)
        return lcu.num_prepare_ancillas

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
        lcu, use_quantum_walk = self.resolve_lcu(container)

        qsharp_op = self.block_encoding_op(container)
        if use_quantum_walk:
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeWalkOp(qsharp_op, self.reflection_op(container))

        if container.power != 1:
            # Repetition branches on the estimator cache, which strips Adj/Ctl, so it is only
            # wrapped on when it repeats. Callers composing further need those functors.
            qsharp_op = QSHARP_UTILS.CircuitComposition.MakeRepeatedOp("PSPMapper", qsharp_op, container.power)

        prepare_op, select_op, num_system = self.build_prepare_select_ops(container)
        qsharp_factory = QsharpFactoryData(
            # QIR generation only resolves entry-point callables one level deep, so the oracles
            # are handed over unstitched and Q# composes them; see ``MakePrepSelPrepCircuit``.
            program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
            parameter={
                "prepareOp": prepare_op,
                "selectOp": select_op,
                "numSystemQubits": num_system,
                "numAncillaQubits": lcu.num_prepare_ancillas,
                "power": container.power,
                "useWalk": use_quantum_walk,
            },
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)
