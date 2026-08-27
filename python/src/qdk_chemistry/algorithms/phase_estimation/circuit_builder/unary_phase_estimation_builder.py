"""Unary-iteration phase estimation circuit builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any, NamedTuple

import numpy as np

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAWalkContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = [
    "QdkUnaryQpeCircuitBuilder",
    "QdkUnaryQpeCircuitBuilderSettings",
    "cosine_window_state",
]


class _WalkSchedule(NamedTuple):
    """The walk-specific half of the arguments to a unary QPE Q# entry point.

    The two supported walks disagree on how the query schedule is assembled, so each
    supplies its own entry point and the operands that go with it. Everything else about
    the circuit - the window state, the state preparation, the register arithmetic - is
    common and stays in the caller.

    Attributes:
        program: The Q# entry point that consumes ``walk_ops``.
        walk_ops: The callables naming the schedule, ordered as the entry point takes them.
        prepare_shared_op: Initializes the shared ancillas once around the schedule.
        num_ancilla_qubits: Block-encoding ancillas the walk reflects about.
        num_shared_ancillas: Persistent ancillas sitting past them.
        trailing_flags: Entry-point specific flags appended after the common arguments.

    """

    program: Any
    walk_ops: dict[str, Any]
    prepare_shared_op: Any
    num_ancilla_qubits: int
    num_shared_ancillas: int
    trailing_flags: dict[str, Any]


def _prepare_shared_op(num_shared_ancillas: int) -> Any:
    """Return the op initializing the shared ancilla register, or a no-op if it is empty."""
    if num_shared_ancillas:
        return QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState
    return QSHARP_UTILS.PrepSelPrep.NoOpPrepare


def cosine_window_state(num_queries: int) -> list[float]:
    r"""Return the phase-register amplitudes, zero-padded to a whole number of qubits.

    The cosine window is the Heisenberg-limited control state of
    :cite:`Babbush2018` (Eq. 17), :math:`\psi_t \propto \sin(\pi (t + 1) / (p + 2))`
    over the :math:`p + 1` reflection slots.

    Args:
        num_queries: Number of walk blocks; the window spans ``num_queries + 1`` slots.

    Returns:
        Real amplitudes normalized to unit norm, zero-padded to a whole number of qubits.

    """
    num_queries_int = int(num_queries)
    if num_queries_int <= 0 or num_queries_int != num_queries:
        raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")

    dimension = 1 << num_queries_int.bit_length()
    num_states = num_queries_int + 1
    amplitudes = np.sin(np.pi * (np.arange(num_states) + 1) / (num_states + 1))
    padded = np.zeros(dimension)
    padded[:num_states] = amplitudes / np.linalg.norm(amplitudes)
    return padded.tolist()


class QdkUnaryQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the unary-iteration phase estimation circuit builder."""

    def __init__(self) -> None:
        """Initialize the unary-iteration QPE builder settings."""
        super().__init__()
        self._set_default(
            "num_queries",
            "int",
            -1,
            "Number of walk queries I. The Heisenberg-limited setting for a target "
            "phase-estimation energy error epsilon is I = ceil(pi * lambda / (2 * epsilon)), "
            "where lambda is the block-encoding 1-norm; see Lee2021 Eq. (45). "
            "Doesn't need to be a power of two.",
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "prepare_select_prepare"),
            "Mapper producing the uncontrolled block encoding. It must lay the block encoding "
            "out as [system | ancilla].",
        )
        self.set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True))


class QdkUnaryQpeCircuitBuilder(QpeCircuitBuilder):
    r"""Phase estimation circuit builder driven by unary iteration.

    Standard QPE applies controlled :math:`U^{2^k}` once per phase qubit and therefore
    consumes a power-of-two number of walk queries. This builder instead emits a single
    chain of ``num_queries`` self-inverse walk blocks and uses unary iteration over the
    phase register to select which interleaved reflection is omitted, so any positive
    query count is supported.

    The phase register is prepared in a cosine window state to suppress the spectral leakage
    of the truncated schedule. The unitary builder must set ``quantum_walk=True`` for the
    post-processing formula.

    References:
        * :cite:`Babbush2018` — cosine-window control state.
        * :cite:`Lee2021` — non-power-of-two query schedule.

    """

    def __init__(
        self,
        num_queries: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        circuit_mapper: AlgorithmRef | None = None,
    ) -> None:
        """Initialize the unary-iteration QPE circuit builder.

        Args:
            num_queries: Number of queries to the block encoding. Default to -1; user needs to set a valid value.
            unitary_builder: Optional algorithm reference for the unitary builder.
            circuit_mapper: Optional algorithm reference for the block-encoding circuit mapper.

        """
        Logger.trace_entering()
        super().__init__(num_bits=-1)
        self._settings = QdkUnaryQpeCircuitBuilderSettings()
        self._settings.set("num_queries", num_queries)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if circuit_mapper is not None:
            self._settings.set("circuit_mapper", circuit_mapper)

    def resolve_num_queries(self) -> tuple[int, int]:
        """Return the query count to apply and the phase-register size addressing it.

        Returns:
            The number of walk blocks the schedule applies, and the number of phase qubits
            needed to address its ``num_queries + 1`` reflection slots.

        Raises:
            ValueError: If the configured ``num_queries`` is not a positive integer.

        """
        num_queries = int(self._settings.get("num_queries"))
        if num_queries <= 0:
            raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")
        if num_queries > 1 and num_queries & (num_queries - 1) == 0:
            Logger.info(
                f"num_queries={num_queries} spends {num_queries.bit_length()} phase qubits to address "
                f"{num_queries + 1} reflection slots, one more than num_queries={num_queries - 1} needs. "
                "Standard QPE (qpe_circuit_builder qdk_standard) fills its register exactly."
            )
        return num_queries, num_queries.bit_length()

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Build the unary-iteration QPE circuit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.

        Returns:
            A single-element list containing the unary-iteration QPE circuit.

        Raises:
            RuntimeError: If the state preparation circuit has no Q# operation.
            ValueError: If the unitary representation is not a supported quantum walk, if the
                mapper does not declare its register width, if the block encoding has no ancilla
                register for the walk to reflect about, or if the state preparation and block
                encoding disagree on the shared ancilla count.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()

        num_queries, num_phase_qubits = self.resolve_num_queries()
        configured_num_bits = self._settings.get("num_bits")
        if configured_num_bits > 0 and configured_num_bits != num_phase_qubits:
            Logger.warn(
                f"num_bits={configured_num_bits} is ignored; num_queries={num_queries} needs {num_phase_qubits}."
            )

        num_system_qubits = qubit_hamiltonian.num_qubits
        state_prep_shared = state_preparation.metadata.num_phase_gradient_ancillas

        if isinstance(container, SOSSAWalkContainer):
            schedule = self._sossa_schedule(unitary_rep, container, num_queries, state_prep_shared)
        elif isinstance(container, LCUWalkContainer):
            schedule = self._lcu_schedule(container, num_system_qubits, state_prep_shared)
        else:
            raise ValueError(f"Requires a LCU or SOSSA walk unitary representation, got '{type(container).__name__}'.")

        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise RuntimeError("State preparation has no Q# operation.")

        phase_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=list(range(num_phase_qubits - 1, -1, -1)),
            stateVector=cosine_window_state(num_queries),
            expansionOps=[],
            numQubits=num_phase_qubits,
        )
        # Q# arguments are applied positionally, so this insertion order is the entry
        # point's signature order, not decoration.
        parameters = {
            "statePrep": state_prep_op,
            **schedule.walk_ops,
            "phaseQubitPrep": QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(phase_prep_params),
            "prepareSharedOp": schedule.prepare_shared_op,
            "numQueries": num_queries,
            "numSystemQubits": num_system_qubits,
            "numAncillas": schedule.num_ancilla_qubits,
            "numSharedAncillas": schedule.num_shared_ancillas,
            "statePrepUsesShared": bool(state_prep_shared),
            **schedule.trailing_flags,
        }
        circuit = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=schedule.program,
                parameter=parameters,
            ),
            num_qubits=(
                num_phase_qubits + num_system_qubits + schedule.num_ancilla_qubits + schedule.num_shared_ancillas
            ),
        )
        return [circuit]

    def _lcu_schedule(
        self,
        container: LCUWalkContainer,
        num_system_qubits: int,
        state_prep_shared: int,
    ) -> _WalkSchedule:
        """Assemble the schedule for an LCU walk out of a block encoding and a reflection.

        Args:
            container: The LCU walk container to map.
            num_system_qubits: Width of the system register.
            state_prep_shared: Shared ancillas the state preparation expects.

        Returns:
            The walk-specific arguments for ``MakeUnaryQPECircuit``.

        Raises:
            ValueError: If the mapper does not report its register width, if no ancilla
                register is left to reflect about, or if the state preparation and block
                encoding disagree on the shared ancilla count.

        """
        container_power = getattr(container, "power", 1)
        block_encoding_container = container.block_encoding
        if container_power != 1 or block_encoding_container.power != 1:
            Logger.warn(f"The unitary representation's power {container_power} is ignored.")
            block_encoding_container = LCUContainer(
                prepare=block_encoding_container.prepare, select=block_encoding_container.select, power=1
            )

        mapper = self._create_nested("circuit_mapper")
        block_encoding = mapper.run(UnitaryRepresentation(container=block_encoding_container))
        num_qubits = block_encoding.num_qubits
        if num_qubits is None:
            raise ValueError(f"Circuit mapper '{type(mapper).__name__}' did not report num_qubits.")

        block_encoding_shared = block_encoding.metadata.num_phase_gradient_ancillas
        if block_encoding_shared and state_prep_shared and block_encoding_shared != state_prep_shared:
            raise ValueError(
                f"State preparation expects {state_prep_shared} phase gradient ancilla but the "
                f"block encoding expects {block_encoding_shared}."
            )
        num_shared_ancillas = max(block_encoding_shared, state_prep_shared)

        num_ancilla_qubits = num_qubits - num_system_qubits - block_encoding_shared
        if num_ancilla_qubits <= 0:
            raise ValueError(f"Requires a non-empty ancilla register to reflect about, got {num_ancilla_qubits}.")

        return _WalkSchedule(
            program=QSHARP_UTILS.UnaryPhaseEstimation.MakeUnaryQPECircuit,
            walk_ops={
                "applyBlockEncoding": block_encoding._qsharp_op,  # noqa: SLF001
                "applyReflection": QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(
                    num_system_qubits, num_ancilla_qubits
                ),
            },
            prepare_shared_op=_prepare_shared_op(num_shared_ancillas),
            num_ancilla_qubits=num_ancilla_qubits,
            num_shared_ancillas=num_shared_ancillas,
            trailing_flags={"blockEncodingUsesShared": bool(block_encoding_shared)},
        )

    def _sossa_schedule(
        self,
        unitary_rep: UnitaryRepresentation,
        container: SOSSAWalkContainer,
        num_queries: int,
        state_prep_shared: int,
    ) -> _WalkSchedule:
        """Take the assembled schedule from the SOSSA mapper.

        A SOSSA walk fuses its reflection into the walk step, so it cannot be split into
        the block encoding and separately controlled reflection that ``MakeUnaryQPECircuit``
        composes. The mapper hands back the whole signed power schedule instead, which is
        what ``MakeUnaryQPECircuitFromSchedule`` takes.

        Args:
            unitary_rep: The unitary representation carrying the SOSSA walk.
            container: The SOSSA walk container describing the register layout.
            num_queries: Number of walk blocks the schedule applies.
            state_prep_shared: Shared ancillas the state preparation expects.

        Returns:
            The walk-specific arguments for ``MakeUnaryQPECircuitFromSchedule``.

        Raises:
            ValueError: If the configured mapper cannot build a SOSSA walk, if no ancilla
                register is left to reflect about, or if the state preparation disagrees
                with the walk on the shared ancilla count.

        """
        mapper = self._create_nested("controlled_circuit_mapper")
        # Each capability is probed as a literal so a rename of the mapper method is caught by
        # test_mapper_capability_literals; a loop over a tuple of names hides them from that scan.
        missing = [
            name
            for name, provided in (
                ("build_walk_op", hasattr(mapper, "build_walk_op")),
                ("num_ancilla_qubits", hasattr(mapper, "num_ancilla_qubits")),
                ("num_shared_ancilla_qubits", hasattr(mapper, "num_shared_ancilla_qubits")),
            )
            if not provided
        ]
        if missing:
            raise ValueError(
                f"A SOSSA walk needs a controlled_circuit_mapper exposing {missing}; "
                f"'{type(mapper).__name__}' does not. Set controlled_circuit_mapper to 'sossa'."
            )

        # The schedule owns the whole [system | ancillas | shared] layout it was built for,
        # so an unequal count from state preparation would shift its register slicing rather
        # than merely over-allocate.
        num_shared_ancillas = mapper.num_shared_ancilla_qubits(container)
        if state_prep_shared and state_prep_shared != num_shared_ancillas:
            raise ValueError(
                f"State preparation expects {state_prep_shared} phase gradient ancilla but the "
                f"SOSSA walk expects {num_shared_ancillas}."
            )

        num_ancilla_qubits = mapper.num_ancilla_qubits(container) - num_shared_ancillas
        if num_ancilla_qubits <= 0:
            raise ValueError(f"Requires a non-empty ancilla register to reflect about, got {num_ancilla_qubits}.")

        return _WalkSchedule(
            program=QSHARP_UTILS.UnaryPhaseEstimation.MakeUnaryQPECircuitFromSchedule,
            walk_ops={"applySchedule": mapper.build_walk_op(unitary_rep, num_queries, use_unary_iteration=True)},
            prepare_shared_op=_prepare_shared_op(num_shared_ancillas),
            num_ancilla_qubits=num_ancilla_qubits,
            num_shared_ancillas=num_shared_ancillas,
            trailing_flags={},
        )

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_unary"
