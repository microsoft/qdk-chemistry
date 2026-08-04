"""QDK/Chemistry PREPARE-SELECT-PREPARE controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

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
            "Algorithm for the PREPARE oracle state preparation. ",
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
    the block encoding is additionally wrapped with the reflection operator to form a
    quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    That walk is self-inverse up to the reflection, which is what lets this mapper drive
    unary-iteration phase estimation.

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

    def _resolve_lcu(self, container: Any) -> tuple[LCUContainer, int, bool]:
        """Unwrap a container into its LCU data, power, and whether to apply the reflection.

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
            "ControlledPSPMapper requires LCUContainer or LCUWalkContainer."
        )

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled block-encoding circuit.

        Args:
            unitary: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the controlled block encoding.

        """
        lcu, power, use_quantum_walk = self._resolve_lcu(unitary.get_container())

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

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the walk needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register, which is also the register the walk
            reflection acts on.

        """
        lcu, _, _ = self._resolve_lcu(container)
        return lcu.num_prepare_ancillas

    def build_walk_op(
        self,
        unitary: UnitaryRepresentation,
        num_queries: int,
        use_unary_iteration: bool = True,
    ) -> Any:
        """Build a PSP walk callable acting on (control register, system + ancilla register).

        When ``use_unary_iteration`` is ``True`` the control register is the phase register and
        the generic signed-power schedule applies ``num_queries`` block encodings while
        skipping the one reflection its address selects, so branch ``t`` realizes
        ``W^(num_queries - 2t)``. Otherwise the control register holds a single qubit and the
        controlled walk is repeated ``num_queries`` times.

        Args:
            unitary: The unitary representation containing the LCU block encoding.
            num_queries: Number of block encodings to apply.
            use_unary_iteration: Whether the control register is a phase register iterated over.

        Returns:
            A Q# callable accepting the control register and the combined system/ancilla register.

        Raises:
            ValueError: If ``num_queries`` is not positive, or the walk has no ancilla to
                reflect about.

        """
        if num_queries <= 0:
            raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")

        lcu, _, _ = self._resolve_lcu(unitary.get_container())
        if lcu.num_prepare_ancillas == 0:
            raise ValueError(
                "A signed-power schedule needs a non-empty ancilla register to reflect about, "
                "but this block encoding has none."
            )

        return QSHARP_UTILS.UnaryPhaseEstimation.MakePSPWalkOp(
            self._build_prepare_op(lcu),
            self._build_pauli_select_op(lcu.select),
            lcu.select.num_target_qubits,
            num_queries,
            use_unary_iteration,
        )

    def get_ancilla_prep_op(self) -> Any:
        """Return the Q# ancilla preparation op used by external algorithms like QPE.

        A PSP walk needs its ancillas in the all-zero state, which is how phase estimation
        already allocates them, so no preparation is required.

        Returns:
            A Q# callable that leaves the ancilla register untouched.

        """
        return QSHARP_UTILS.StatePreparation.MakeNoOpAncillaPrep()

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
