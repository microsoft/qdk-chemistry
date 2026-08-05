"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

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
    owns the PREPARE and SELECT oracles and the block encoding

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    The circuit is assembled from the generic combinators in
    ``QDKChemistry.Utils.CircuitComposition``: the block encoding is paired with its reflection
    into a walk when the container is an
    :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`,

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    then controlled, then repeated to the container's power.

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
        """Build the uncontrolled PSP mapper this one delegates the block encoding to.

        Returns:
            A :class:`PSPMapper` carrying this mapper's ``prepare`` setting.

        """
        mapper = PSPMapper()
        mapper.settings().set("prepare", self._settings.get("prepare"))
        return mapper

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the block encoding needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register.

        """
        return self._block_mapper().num_ancillary_qubits(container)

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled block-encoding circuit.

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        Raises:
            ValueError: If more than one control qubit is requested.

        """
        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledPSPMapper currently only supports a single control qubit.")

        block_mapper = self._block_mapper()
        container = unitary.get_container()
        lcu, use_quantum_walk = block_mapper.resolve_lcu(container)

        step_op = block_mapper.block_encoding_op(container)
        if use_quantum_walk:
            step_op = QSHARP_UTILS.PrepSelPrep.MakeWalkOp(step_op, block_mapper.reflection_op(container))

        controlled_op = QSHARP_UTILS.CircuitComposition.MakeControlledOp(step_op)
        repeated_op = QSHARP_UTILS.CircuitComposition.MakeRepeatedOp(
            "ControlledPSPWalk" if use_quantum_walk else "ControlledPrepSelPrep",
            controlled_op,
            container.power,
        )

        prepare_op, select_op, num_system = block_mapper.build_prepare_select_ops(container)
        qsharp_factory = QsharpFactoryData(
            # QIR generation only resolves entry-point callables one level deep, so the oracles
            # are handed over unstitched and Q# composes them; see ``MakePrepSelPrepCircuit``.
            program=QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepCircuit,
            parameter={
                "prepareOp": prepare_op,
                "selectOp": select_op,
                "numSystemQubits": num_system,
                "numAncillaQubits": lcu.num_prepare_ancillas,
                "power": container.power,
                "useWalk": use_quantum_walk,
            },
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            # Phase estimation takes a single control qubit rather than a control register.
            qsharp_op=QSHARP_UTILS.CircuitComposition.MakeSingleControlOp(repeated_op),
        )
