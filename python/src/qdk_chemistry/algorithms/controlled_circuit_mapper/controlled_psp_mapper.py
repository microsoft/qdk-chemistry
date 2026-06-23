"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer, Select
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
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

    When the input is an :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`,
    the block encoding is additionally wrapped with the reflection operator to form a quantum walk:

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

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        """
        container = unitary.get_container()

        # Resolve container type → LCU data + dispatch flag
        if isinstance(container, LCUWalkContainer):
            lcu = container.block_encoding
            power = container.power
            use_quantum_walk = True
        elif isinstance(container, LCUContainer):
            lcu = container
            power = container.power
            use_quantum_walk = False
        else:
            raise ValueError(
                f"Container type '{unitary.get_container_type()}' is not supported. "
                "ControlledPSPMapper requires LCUContainer or LCUWalkContainer."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledPSPMapper currently only supports a single control qubit.")

        # 1. PREPARE — build state-preparation oracle
        prepare_op = self._build_prepare_op(lcu)

        # 2. SELECT — build Pauli SELECT oracle
        select_op = self._build_pauli_select_op(lcu.select)

        # 3. Compose controlled circuit
        num_system = lcu.select.num_target_qubits
        num_ancilla = lcu.num_prepare_ancillas

        if use_quantum_walk:
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledPSPWalkCircuit
            make_op = QSHARP_UTILS.PrepSelPrep.MakeControlledPSPWalkOp
        else:
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepCircuit
            make_op = QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepOp

        psp_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system,
            "numAncillaQubits": num_ancilla,
            "power": power,
        }

        qsharp_factory = QsharpFactoryData(program=make_circuit, parameter=psp_parameters)
        qsharp_op = make_op(prepare_op, select_op, num_system, num_ancilla, power)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

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
