"""QDK/Chemistry controlled SOSSA quantum walk mapper :cite:`Low2025`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_mapper.sossa_mapper import SOSSAMapper, _sossa_mapper_setting_defaults
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSABlockEncodingContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledSOSSAMapper", "ControlledSOSSAMapperSettings"]


class ControlledSOSSAMapperSettings(ControlledCircuitMapperSettings):
    """Settings for the ControlledSOSSAMapper.

    Adds the options of :class:`~qdk_chemistry.algorithms.circuit_mapper.sossa_mapper.SOSSAMapperSettings`,
    which choose the PREPARE and SELECT oracles of the walk's block encoding.

    """

    def __init__(self):
        """Initialize the settings for ControlledSOSSAMapper."""
        super().__init__()
        for default in _sossa_mapper_setting_defaults():
            self._set_default(*default)


class ControlledSOSSAMapper(ControlledCircuitMapper):
    r"""Controlled SOSSA quantum walk :cite:`Low2025` for standard phase estimation.

    Emits controlled-:math:`W^{\text{power}}` for the walk

    .. math::

        W = \mathrm{Ref}_a \cdot B, \qquad B = U^\dagger \cdot \mathrm{Ref}_B \cdot U,

    whose block encoding :math:`B` is the one
    :class:`~qdk_chemistry.algorithms.circuit_mapper.sossa_mapper.SOSSAMapper` builds. Only the two
    reflections are controlled. With the control off :math:`U^\dagger U = I`, so :math:`U` runs
    unconditionally and its PREPARE oracles need no controlled form.

    The returned operation takes the control qubit and the ``[system | ancillas | phase gradient]``
    target register. The circuit declares the control plus that target as ``num_qubits`` and the
    gradient suffix in its metadata, so phase estimation prepares the gradient once and shares it.
    Iterative builders reject the walk; see
    :class:`~qdk_chemistry.data.unitary_representation.containers.sossa.SOSSABlockEncodingContainer`.

    """

    def __init__(self):
        """Initialize the ControlledSOSSAMapper."""
        super().__init__()
        self._settings = ControlledSOSSAMapperSettings()

    def name(self) -> str:
        """Return ``sossa`` as the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return ``controlled_circuit_mapper`` as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _block_mapper(self) -> SOSSAMapper:
        """Build the uncontrolled SOSSA mapper that owns the oracles of the block encoding.

        Returns:
            A :class:`SOSSAMapper` carrying this mapper's oracle settings.

        """
        mapper = SOSSAMapper()
        for key in mapper.settings().keys():  # noqa: SIM118
            mapper.settings().set(key, self._settings.get(key))
        return mapper

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct controlled-:math:`W^{\text{power}}`.

        Args:
            unitary: The unitary representation containing the SOSSA block encoding.

        Returns:
            Circuit: The controlled walk, declaring its width including the control and the phase
            gradient qubits its caller must prepare.

        Raises:
            ValueError: If the container is not a :class:`SOSSABlockEncodingContainer`, the control is
                not a single qubit at index 0, ``target_indices`` is set, or the power is not positive.

        """
        container = unitary.get_container()
        if not isinstance(container, SOSSABlockEncodingContainer):
            raise ValueError(f"The {unitary.get_container_type()} container type is not supported.")
        control_indices = self._get_control_indices()
        if len(control_indices) != 1 or control_indices[0] != 0:
            raise ValueError("ControlledSOSSAMapper only supports a single control qubit at index 0.")
        if self._settings.get("target_indices"):
            raise ValueError(
                "ControlledSOSSAMapper acts on a fixed [system | ancillas | phase gradient] target register "
                "and does not support target_indices."
            )
        if container.power < 1:
            raise ValueError(f"The controlled SOSSA walk requires a positive power, got {container.power}.")

        ops, regs, register_layout = self._block_mapper().build_block_encoding_ops(container)
        walk_data = {key: value for key, value in container.to_json().items() if key != "power"}
        cache_name = f"ControlledSOSSAWalk_{self.hash(walk_data)}"
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkCircuit,
                parameter={**ops, "layout": register_layout, "cacheName": cache_name, "power": container.power},
            ),
            qsharp_op=QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkOp(
                ops["outerPrepareOp"],
                ops["freeRiderOp"],
                ops["innerPrepareOp"],
                ops["selectOp"],
                register_layout,
                cache_name,
                container.power,
            ),
            num_qubits=1 + regs["num_system_qubits"] + regs["num_ancilla_qubits"],
            metadata=CircuitMetadata(num_phase_gradient_ancillas=regs["num_phase_gradient_qubits"]),
        )
