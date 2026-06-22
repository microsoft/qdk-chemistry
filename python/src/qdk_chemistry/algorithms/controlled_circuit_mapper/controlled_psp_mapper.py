"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer, Select
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledPSPMapper", "ControlledPSPMapperSettings"]


class ControlledPSPMapperSettings(Settings):
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
        )


class ControlledPSPMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    Composes a controlled block encoding from:

    1. **PREPARE** — amplitude-loading into the ancilla register, resolved via
       the ``prepare`` setting.  Defaults to ``DensePureStatePreparation``.
    2. **SELECT** — Pauli SELECT oracle applied on the system register,
       constructed directly from the block-encoding container's SELECT data.

    The two callables are stitched together by the Q# ``PrepSelPrep`` operation:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    When the container has ``quantum_walk=True``, the block encoding is wrapped with
    a quantum walk operator:

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

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled block-encoding circuit.

        The method proceeds in three stages:

        1. **PREPARE** — delegates to the nested ``state_prep`` algorithm
           to build a Q# callable that loads amplitudes into the ancilla register.
        2. **SELECT** — builds the Pauli SELECT oracle directly from the
           block-encoding container's SELECT data.
        3. **Compose** — stitches controlled PREPARE-SELECT-PREPARE into either a plain block
           encoding or a quantum walk step (when ``quantum_walk=True``), via the
           Q# ``PrepSelPrep`` / ``QuantumWalkStep`` operations.

        Args:
            unitary: The unitary representation containing the block-encoding
                decomposition (PREPARE and SELECT data). Control and target
                indices are read from settings.

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, BlockEncodingContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "ControlledPSPMapper only supports BlockEncodingContainer."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledPSPMapper currently only supports a single control qubit.")

        power = unitary_container.power
        prepare_wavefunction = unitary_container.prepare
        select = unitary_container.select

        # 1. Create PREPARE circuit via the state-preparation algorithm.
        #    For the 0-ancilla case the wavefunction has 0 modes, producing a
        #    no-op circuit.
        if prepare_wavefunction is not None:
            prepare_algorithm = self._create_nested("prepare")
            prepare_circuit = prepare_algorithm.run(prepare_wavefunction)
            prepare_op = prepare_circuit._qsharp_op  # noqa: SLF001
        else:
            prepare_op = QSHARP_UTILS.PrepSelPrep.NoOpPrepare

        # 2. Create SELECT circuit directly (Pauli SELECT oracle).
        select_op = self._build_pauli_select_op(select)

        # 3. Compose into a controlled PREPARE-SELECT-PREPARE (optionally with quantum walk).
        num_system = select.num_target_qubits
        num_ancilla = unitary_container.num_prepare_ancillas

        psp_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system,
            "numAncillaQubits": num_ancilla,
            "power": power,
        }

        if unitary_container.quantum_walk:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.PrepSelPrep.MakeControlledQuantumWalkCircuit,
                parameter=psp_parameters,
            )
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeControlledQuantumWalkOp(
                prepare_op, select_op, num_system, num_ancilla, power
            )
        else:
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepCircuit,
                parameter=psp_parameters,
            )
            qsharp_op = QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepOp(
                prepare_op, select_op, num_system, num_ancilla, power
            )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

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
