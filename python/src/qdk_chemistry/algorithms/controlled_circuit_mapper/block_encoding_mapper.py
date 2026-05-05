"""QDK/Chemistry LCU controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer, Prepare, Select
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["BlockEncodingMapper"]


class BlockEncodingMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper using the PREPARE-SELECT-PREPARE† pattern.

    Given an block encoding of a Hamiltonian :math:`H = \sum_j \alpha_j P_j`, this mapper
    constructs a controlled version of the LCU block encoding:

    .. math::

        \text{C-LCU} = \text{C-PREPARE}^\dagger \cdot \text{C-SELECT} \cdot \text{C-PREPARE}

    where the entire block is controlled on the QPE ancilla qubit.

    The PREPARE oracle encodes :math:`\sqrt{|\alpha_j| / \lambda}` into an ancilla register.
    The SELECT oracle applies the Pauli string :math:`P_j` controlled on ancilla state
    :math:`|j\rangle`.

    When the LCU container has ``quantum_walk=True``, the block encoding is wrapped with
    a quantum walk operator :math:`W = R \\cdot B[H]` where :math:`R = 2|0\\rangle\\langle 0| - I`
    is a reflection on the ancilla register. This should be used with **QPE** to extract
    eigenvalues from the walk operator's spectrum.

    When ``quantum_walk=False``, the plain block encoding is used. This should be used
    with a **Hadamard test** to extract expectation values.

    Notes:
        * Currently supports only single-control-qubit scenarios.

    """

    def __init__(self):
        """Initialize the BlockEncodingMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name."""
        return "block_encoding"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled LCU block encoding.

        If the container has ``quantum_walk=True``, wraps the block encoding with the
        quantum walk operator :math:`W = R \cdot B[H]`. Use QPE to extract eigenvalues
        from the walk operator spectrum.

        If ``quantum_walk=False``, produces the plain block encoding circuit.
        Use a Hadamard test to extract expectation values.

        Args:
            controlled_unitary: The controlled unitary containing the LCU decomposition.

        Returns:
            Circuit: A quantum circuit implementing the controlled LCU operation.

        Raises:
            ValueError: If the unitary container type is not BlockEncodingContainer.
            ValueError: If multiple control qubits are provided.

        """
        unitary_container = controlled_unitary.unitary.get_container()
        if not isinstance(unitary_container, BlockEncodingContainer):
            raise ValueError(
                f"The {controlled_unitary.get_unitary_container_type()} container type is not supported. "
                "BlockEncodingMapper only supports BlockEncodingContainer."
            )

        if len(controlled_unitary.control_indices) != 1:
            raise ValueError("BlockEncodingMapper currently only supports a single control qubit.")

        num_system_qubits = unitary_container.num_system_qubits
        num_select_qubits = unitary_container.num_select_qubits
        power = getattr(unitary_container, "power", 1)
        prepare_op = self._create_prepare_op(unitary_container.prepare)
        select_op = self._create_select_op(unitary_container.select, num_system_qubits)

        lcu_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system_qubits,
            "numSelectQubits": num_select_qubits,
            "power": power,
        }

        if unitary_container.reflect is not None:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.LCU.MakeLCUQuantumWalkCircuit,
                parameter=lcu_parameters,
            )
            lcu_op = QSHARP_UTILS.LCU.MakeLCUQuantumWalkOp(
                prepare_op, select_op, num_system_qubits, num_select_qubits, power
            )
        else:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.LCU.MakeLCUCircuit,
                parameter=lcu_parameters,
            )
            lcu_op = QSHARP_UTILS.LCU.MakeLCUOp(prepare_op, select_op, num_system_qubits, num_select_qubits, power)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=lcu_op)

    @staticmethod
    def _create_prepare_op(prepare: Prepare):
        """Create the PREPARE oracle callable from Prepare sub-object."""
        if prepare.method == "lcu":
            amplitudes = [float(a) for a in prepare.statevector]
            prepare_params = QSHARP_UTILS.LCU.DefaultPrepareParams(amplitudes=amplitudes)
            return QSHARP_UTILS.LCU.MakePrepareOp(prepare_params)
        raise NotImplementedError(f"Unsupported PREPARE method: {prepare.method}")

    @staticmethod
    def _create_select_op(select: Select, num_system_qubits: int):
        """Create the SELECT oracle callable from Select sub-object."""
        if select.method == "lcu":
            pauli_terms: list[list[qsharp.Pauli]] = []
            for op in select.controlled_operations:
                base_paulis = [qsharp.Pauli.I] * num_system_qubits
                for i, pauli_char in enumerate(op.operation):
                    if pauli_char != "I":
                        base_paulis[op.target_qubits[i]] = getattr(qsharp.Pauli, pauli_char)
                pauli_terms.append(base_paulis)
            signs = [int(s) for s in select.signs]
            select_params = QSHARP_UTILS.LCU.DefaultSelectParams(pauliTerms=pauli_terms, signs=signs)
            return QSHARP_UTILS.LCU.MakeSelectOp(select_params)
        raise NotImplementedError(f"Unsupported SELECT method: {select.method}")
