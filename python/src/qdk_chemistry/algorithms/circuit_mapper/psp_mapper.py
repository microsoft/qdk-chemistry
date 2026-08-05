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

    No reflection is applied, so a
    :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`
    maps to its underlying block encoding rather than to the walk
    :math:`W = (2|0\rangle\langle 0| - I) \cdot B[H]`. Whoever schedules the walk owns the
    reflection: :class:`~qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_psp_mapper.ControlledPSPMapper`
    materializes it for standard phase estimation, while unary-iteration phase estimation
    interleaves its own so it can omit exactly one.

    """

    def __init__(self):
        """Initialize the PSPMapper."""
        super().__init__()
        self._settings = PSPMapperSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"prepare_select_prepare"``.

        """
        return "prepare_select_prepare"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"circuit_mapper"``.

        """
        return "circuit_mapper"

    def resolve_lcu(self, container: Any) -> tuple[LCUContainer, int, bool]:
        """Unwrap a container into its LCU data, power, and whether it is a quantum walk.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The LCU data, the requested power, and whether the container is a quantum walk.

        Raises:
            ValueError: If the container is neither an LCU nor an LCU walk.

        """
        if isinstance(container, LCUWalkContainer):
            return container.block_encoding, container.power, True
        if isinstance(container, LCUContainer):
            return container, container.power, False
        raise ValueError(
            f"Container type '{type(container).__name__}' is not supported. "
            "PSPMapper requires LCUContainer or LCUWalkContainer."
        )

    def _run_impl(self, evolution: UnitaryRepresentation) -> Circuit:
        r"""Construct the block-encoding circuit on the flat ``[system | ancilla]`` register.

        Args:
            evolution: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the block encoding.

        """
        container = evolution.get_container()
        lcu, power, _ = self.resolve_lcu(container)
        prepare_op, select_op, num_system = self.build_prepare_select_ops(container)
        num_ancilla = lcu.num_prepare_ancillas

        psp_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system,
            "numAncillaQubits": num_ancilla,
            "power": power,
        }

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
            parameter=psp_parameters,
        )
        qsharp_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(
            prepare_op, select_op, num_system, num_ancilla, power
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the walk needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register, which is also the register the walk
            reflection acts on.

        """
        lcu, _, _ = self.resolve_lcu(container)
        return lcu.num_prepare_ancillas

    def build_prepare_select_ops(self, container: Any) -> tuple[Any, Any, int]:
        """Expose the PREPARE and SELECT oracles so callers can compose their own schedule.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The PREPARE Q# callable, the SELECT Q# callable, and the system register size.

        """
        lcu, _, _ = self.resolve_lcu(container)
        return self._build_prepare_op(lcu), self._build_pauli_select_op(lcu.select), lcu.select.num_target_qubits

    def _build_prepare_op(self, lcu: LCUContainer):
        """Build the PREPARE Q# operation from an LCU container.

        For the 0-ancilla case the wavefunction has 0 modes, producing a no-op.

        Args:
            lcu: The LCU container holding the prepare wavefunction.

        Returns:
            A Q# callable implementing the PREPARE oracle.

        """
        if lcu.prepare is not None:
            prepare_algorithm = self._create_nested("prepare")
            prepare_circuit = prepare_algorithm.run(lcu.prepare)
            return prepare_circuit._qsharp_op  # noqa: SLF001
        return QSHARP_UTILS.PrepSelPrep.NoOpPrepare

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
