"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledPSPMapper", "ControlledPSPMapperSettings"]


class ControlledPSPMapperSettings(ControlledCircuitMapperSettings):
    """Settings for the ControlledPSPMapper.

    Attributes:
        circuit_mapper: Algorithm reference for the uncontrolled block-encoding mapper that
            builds the PREPARE and SELECT oracles. Defaults to ``PSPMapper``.

    """

    def __init__(self):
        """Initialize the settings for ControlledPSPMapper."""
        super().__init__()
        self.set("circuit_mapper", AlgorithmRef("circuit_mapper", "prepare_select_prepare"))


class ControlledPSPMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    A thin wrapper over :class:`~qdk_chemistry.algorithms.circuit_mapper.psp_mapper.PSPMapper`,
    which owns the block encoding

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    and the reflection it pairs with. This class only adds the control register, letting the Q#
    ``Controlled`` functor do the work — that is what keeps ``PrepSelPrep``'s rule that PREPARE
    and its inverse stay unconditional in one place. For an
    :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`
    it composes the block encoding with that reflection into the quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

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

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the block encoding needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register, which is also the register the walk
            reflection acts on.

        """
        return self._create_nested("circuit_mapper").num_ancillary_qubits(container)

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

        block_mapper = self._create_nested("circuit_mapper")
        container = unitary.get_container()
        lcu, power, use_quantum_walk = block_mapper.resolve_lcu(container)
        block_encoding = block_mapper.run(unitary)._qsharp_op  # noqa: SLF001
        num_qubits = lcu.select.num_target_qubits + lcu.num_prepare_ancillas

        if use_quantum_walk:
            reflection = block_mapper.reflection_op(container)
            parameters = {
                "blockEncoding": block_encoding,
                "applyReflection": reflection,
                "numQubits": num_qubits,
                "power": power,
            }
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledWalkCircuit
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeControlledWalkOp(block_encoding, reflection, power)
        else:
            parameters = {"blockEncoding": block_encoding, "numQubits": num_qubits}
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledBlockEncodingCircuit
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeControlledBlockEncodingOp(block_encoding)

        qsharp_factory = QsharpFactoryData(program=make_circuit, parameter=parameters)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)
