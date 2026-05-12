"""QDK/Chemistry PREPARE-SELECT controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["PrepSelPrepMapper", "PrepSelPrepSettings"]


class PrepSelPrepSettings(Settings):
    """Settings for the PrepSelPrepMapper.

    Attributes:
        state_prep: Algorithm reference for the PREPARE oracle state preparation.
            Defaults to ``DensePureStatePreparation``.
        select_mapper: Algorithm reference for the SELECT oracle mapper.
            Defaults to ``MultiControlledSelectMapper``.

    """

    def __init__(self):
        """Initialize the settings for PrepSelPrepMapper."""
        super().__init__()
        self._set_default(
            "state_prep",
            "algorithm_ref",
            AlgorithmRef("state_prep", "dense_pure_state"),
        )
        self._set_default(
            "select_mapper",
            "algorithm_ref",
            AlgorithmRef("select_mapper", "multi_controlled_select"),
        )


class PrepSelPrepMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    Composes a controlled block encoding from two independent sub-algorithms:

    1. **PREPARE** — amplitude-loading into the ancilla register, resolved via
       the ``state_prep`` setting.  Defaults to ``DensePureStatePreparation``.
    2. **SELECT** — multi-controlled unitary application on the system register,
       resolved via the ``select_mapper`` setting.

    The two callables are stitched together by the Q# ``BlockEncoding`` operation:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    When the container has ``quantum_walk=True``, the block encoding is wrapped with
    a quantum walk operator:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    The quantum walk variant is used with **QPE** to extract eigenvalues,
    while the plain block encoding is used with a **Hadamard test** for
    expectation values.

    """

    def __init__(self):
        """Initialize the PrepSelPrepMapper."""
        super().__init__()
        self._settings = PrepSelPrepSettings()

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

    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        r"""Construct a controlled block-encoding circuit.

        The method proceeds in three stages:

        1. **PREPARE** — delegates to the nested ``state_prep`` algorithm
           to build a Q# callable that loads amplitudes into the ancilla register.
        2. **SELECT** — delegates to the nested ``select_mapper`` algorithm
           to build a Q# callable that applies controlled unitaries.
        3. **Compose** — stitches controlled PREPARE-SELECT-PREPARE into either a plain block
           encoding or a quantum walk step (when ``quantum_walk=True``), via the
           Q# ``PrepSelPrep`` / ``QuantumWalkStep`` operations.

        Args:
            controlled_unitary: The controlled unitary containing the block-encoding
                decomposition (PREPARE and SELECT data).

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        """
        unitary_container = controlled_unitary.unitary.get_container()
        if not isinstance(unitary_container, BlockEncodingContainer):
            raise ValueError(
                f"The {controlled_unitary.get_unitary_container_type()} container type is not supported. "
                "PrepSelPrepMapper only supports BlockEncodingContainer."
            )

        if len(controlled_unitary.control_indices) != 1:
            raise ValueError("PrepSelPrepMapper currently only supports a single control qubit.")

        power: int = unitary_container.power
        prepare = unitary_container.prepare
        select = unitary_container.select

        # 1. Create PREPARE circuit via the state-preparation algorithm.
        prepare_algorithm = self._create_nested("state_prep")
        reversed_qubits = list(reversed(prepare.prepare_qubits))
        prepare_circuit = prepare_algorithm.prepare_from_statevector(
            prepare.statevector, prepare.num_prepare_qubits, reversed_qubits
        )
        prepare_op = prepare_circuit._qsharp_op  # noqa: SLF001

        # 2. Create SELECT circuit via the select-mapper algorithm.
        select_mapper = self._create_nested("select_mapper")
        select_circuit = select_mapper.run(select)
        select_op = select_circuit._qsharp_op  # noqa: SLF001

        # 3. Compose into a controlled PREPARE-SELECT-PREPARE (optionally with quantum walk).
        num_system = select.num_target_qubits
        num_ancilla = select.num_prepare_qubits

        psp_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system,
            "numAncillaQubits": num_ancilla,
            "power": power,
        }

        if unitary_container.quantum_walk:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.PrepSelPrep.MakeQuantumWalkCircuit,
                parameter=psp_parameters,
            )
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeQuantumWalkOp(
                prepare_op, select_op, num_system, num_ancilla, power
            )
        else:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
                parameter=psp_parameters,
            )
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(
                prepare_op, select_op, num_system, num_ancilla, power
            )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)
