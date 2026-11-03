"""QDK/Chemistry PREPARE-SELECT-PREPARE circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from contextlib import suppress
from typing import Any

from qdk import qsharp

from qdk_chemistry.data import AlgorithmRef, SettingNotFound, Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = ["PSPMapper", "PSPMapperSettings"]

#: SELECT constructions available through the ``select_method`` setting.
SELECT_METHODS = ("multi_controlled", "unary_iteration")


class PSPMapperSettings(Settings):
    """Settings for the PSPMapper.

    Attributes:
        prepare: Algorithm reference for the PREPARE oracle state preparation.
            Defaults to alias sampling.
        select_method: ``"unary_iteration"`` (default, :cite:`Babbush2018`) or ``"multi_controlled"``.

    """

    def __init__(self):
        """Initialize the settings for PSPMapper."""
        super().__init__()
        self._set_default(
            "prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "alias_sampling"),
            "Algorithm for the PREPARE oracle. A PREPARE that rotates through a phase gradient (e.g. 'qrom') "
            "uses one shared register its caller prepares.",
        )
        self._set_default(
            "select_method",
            "string",
            "unary_iteration",
            "SELECT construction: 'multi_controlled' or 'unary_iteration'.",
            list(SELECT_METHODS),
        )


class PSPMapper(CircuitMapper):
    r"""Circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    Composes a block encoding from:

    1. **PREPARE** — amplitude-loading into the ancilla register, resolved via
       the ``prepare`` setting.  Defaults to alias sampling.
    2. **SELECT** — Pauli SELECT oracle applied on the system register,
       constructed directly from the block-encoding container's SELECT data.

    The two callables are stitched together by the Q# ``PrepSelPrep`` operation:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    When the input is an :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`,
    the block encoding is additionally wrapped with the reflection operator to form a quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    Note:
        With an alias or QROM PREPARE, ``Circuit.get_qir()`` and ``Circuit.to_json()`` currently
        fail inside the Q# compiler (its measurement-based ``Adjoint Std.TableLookup.Select``);
        ``estimate()`` and simulation work. Use ``dense_pure_state`` when QIR export is needed.

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

        Each controlled operation's Pauli string becomes a list of Q# ``Pauli`` enums,
        packaged with the sign phases into a ``PauliSelectParams`` struct.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The Q# SELECT callable and the number of system qubits.

        """
        lcu, _ = self.resolve_lcu(container)
        select = lcu.select
        pauli_terms: list[list[qsharp.Pauli]] = []
        control_states: list[int] = []
        for op in select.controlled_operations:
            base_paulis = [qsharp.Pauli.I] * select.num_target_qubits
            for i, pauli_char in enumerate(reversed(op.operation)):
                if pauli_char != "I":
                    base_paulis[i] = getattr(qsharp.Pauli, pauli_char)
            pauli_terms.append(base_paulis)
            control_states.append(op.ctrl_state)
        select_params = QSHARP_UTILS.Select.PauliSelectParams(
            pauliTerms=pauli_terms, signs=[int(s) for s in select.phases], controlStates=control_states
        )
        if self._settings.get("select_method") == "unary_iteration":
            return QSHARP_UTILS.Select.MakeUnarySelectOp(select_params), select.num_target_qubits
        return QSHARP_UTILS.Select.MakeSelectOp(select_params), select.num_target_qubits

    def build_prep_ops(self, container: UnitaryContainer) -> tuple[Any, int, int, int]:
        """Return the PREPARE oracle and the widths of the ancilla it acts on.

        A PREPARE that can rotate through a caller-owned phase gradient is configured to, so
        PREPARE and its adjoint share one register its caller prepares once.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The Q# PREPARE callable, the index width SELECT controls on, the block ancilla
            width the walk reflects about, and the phase gradient width after it.

        Raises:
            ValueError: If the PREPARE circuit carries no Q# operation, does not declare the
                width it acts on, or declares one too narrow for the index SELECT controls on.

        """
        lcu, _ = self.resolve_lcu(container)
        if lcu.prepare is None:
            return QSHARP_UTILS.PrepSelPrep.NoOpPrepare, 0, 0, 0

        prepare_algorithm = self._create_nested("prepare")
        with suppress(SettingNotFound):
            prepare_algorithm.settings().set("allocate_phase_gradient", False)
        prepare_circuit = prepare_algorithm.run(lcu.prepare)
        prepare_op = prepare_circuit._qsharp_op  # noqa: SLF001
        if prepare_op is None:
            raise ValueError("The PREPARE circuit has no Q# operation to embed in the block encoding.")

        num_prepare_qubits = prepare_circuit.num_qubits
        num_select_qubits = lcu.num_prepare_ancillas
        if num_prepare_qubits is None:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' does not declare num_qubits, so the "
                "block ancilla register cannot be sized."
            )
        num_gradient_qubits = prepare_circuit.metadata.num_phase_gradient_ancillas
        num_block_ancillas = num_prepare_qubits - num_gradient_qubits
        if num_block_ancillas <= 0:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' declares num_qubits={num_prepare_qubits}, "
                "but a PREPARE oracle must act on at least one qubit besides its phase gradient."
            )
        if num_block_ancillas < num_select_qubits:
            raise ValueError(
                f"The PREPARE circuit acts on {num_block_ancillas} qubits, but the LCU decomposition indexes "
                f"{num_select_qubits} of them. SELECT would control on qubits PREPARE does not own."
            )
        return prepare_op, num_select_qubits, num_block_ancillas, num_gradient_qubits

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct the block-encoding circuit on the flat ``[system | ancilla]`` register.

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the block encoding on
            ``[system | block ancilla | phase gradient]``, declaring the phase gradient its
            caller must prepare.

        """
        container = unitary.get_container()
        _, use_quantum_walk = self.resolve_lcu(container)
        select_op, num_system_qubits = self.build_select_ops(container)
        prepare_op, num_select_qubits, num_block_ancillas, num_gradient_qubits = self.build_prep_ops(container)

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
                "numPhaseGradientQubits": num_gradient_qubits,
                "power": container.power,
                "useWalk": use_quantum_walk,
            },
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=num_system_qubits + num_block_ancillas + num_gradient_qubits,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_qubits),
        )
