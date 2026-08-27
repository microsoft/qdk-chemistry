"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_mapper.psp_mapper import PSPMapper
from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledPSPMapper", "ControlledPSPMapperSettings"]


class ControlledPSPMapperSettings(ControlledCircuitMapperSettings):
    """Settings for the ControlledPSPMapper.

    Attributes:
        prepare: Algorithm reference for the PREPARE oracle state preparation.
            Defaults to ``DensePureStatePreparation``.

    """

    def __init__(self):
        """Initialize the settings for ControlledPSPMapper."""
        super().__init__()
        self._set_default(
            "prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "dense_pure_state"),
            "Algorithm for the PREPARE oracle state preparation. ",
        )


class ControlledPSPMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    A wrapper over :class:`~qdk_chemistry.algorithms.circuit_mapper.psp_mapper.PSPMapper`, which
    owns the PREPARE and SELECT oracles:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    When the input is an :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`,
    the block encoding is additionally wrapped with the reflection operator to form a quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    That walk is a self-inverse block encoding plus a reflection, which is the shape
    unary-iteration phase estimation schedules, so this mapper can also drive it via
    :meth:`build_walk_op`.

    """

    def __init__(self):
        """Initialize the ControlledPSPMapper."""
        super().__init__()
        self._settings = ControlledPSPMapperSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"prepare_select_prepare"``.

        """
        return "prepare_select_prepare"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"controlled_circuit_mapper"``.

        """
        return "controlled_circuit_mapper"

    def _block_mapper(self) -> PSPMapper:
        """Build the uncontrolled PSP mapper to delegate the block encoding to.

        Returns:
            A :class:`PSPMapper` carrying this mapper's ``prepare`` setting.

        """
        mapper = PSPMapper()
        mapper.settings().set("prepare", self._settings.get("prepare"))
        return mapper

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled block-encoding circuit.

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        Raises:
            ValueError: If the control qubit is not a single qubit at index 0.

        """
        control_indices = self._get_control_indices()
        if len(control_indices) != 1 or control_indices[0] != 0:
            raise ValueError("ControlledPSPMapper currently only supports a single control qubit at index 0.")

        block_mapper = self._block_mapper()
        container = unitary.get_container()
        _, use_quantum_walk = block_mapper.resolve_lcu(container)
        select_op, num_system_qubits = block_mapper.build_select_ops(container)
        prepare_op, num_select_qubits, num_block_ancillas = block_mapper.build_prep_ops(container)

        step_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(
            prepare_op, select_op, num_system_qubits, num_select_qubits
        )
        if use_quantum_walk:
            reflection_op = QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(num_system_qubits, num_block_ancillas)
            step_op = QSHARP_UTILS.PrepSelPrep.MakeWalkOp(step_op, reflection_op)

        controlled_op = QSHARP_UTILS.CircuitComposition.MakeControlledOp(step_op)
        repeated_op = QSHARP_UTILS.CircuitComposition.MakeRepeatedOp(
            "ControlledPSPWalk" if use_quantum_walk else "ControlledPrepSelPrep",
            controlled_op,
            container.power,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepCircuit,
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
            qsharp_op=QSHARP_UTILS.CircuitComposition.MakeSingleControlOp(repeated_op),
        )
