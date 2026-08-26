"""Unary-iteration phase estimation circuit builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = [
    "QdkUnaryQpeCircuitBuilder",
    "QdkUnaryQpeCircuitBuilderSettings",
    "cosine_window_state",
]


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
            ValueError: If the unitary representation is not a quantum walk, if the mapper does
                not declare its register width, if the block encoding has no ancilla register
                for the walk to reflect about, or if the state preparation and block encoding
                disagree on the shared ancilla count.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()
        if not isinstance(container, LCUWalkContainer):
            raise ValueError(f"Requires a LCU walk unitary representation, got '{type(container).__name__}'.")

        container_power = getattr(container, "power", 1)
        block_encoding_container = container.block_encoding
        if container_power != 1 or block_encoding_container.power != 1:
            Logger.warn(f"The unitary representation's power {container_power} is ignored.")
            block_encoding_container = LCUContainer(
                prepare=block_encoding_container.prepare, select=block_encoding_container.select, power=1
            )

        num_queries, num_phase_qubits = self.resolve_num_queries()
        configured_num_bits = self._settings.get("num_bits")
        if configured_num_bits > 0 and configured_num_bits != num_phase_qubits:
            Logger.warn(
                f"num_bits={configured_num_bits} is ignored; num_queries={num_queries} needs {num_phase_qubits}."
            )

        mapper = self._create_nested("circuit_mapper")
        block_encoding = mapper.run(UnitaryRepresentation(container=block_encoding_container))
        block_encoding_op = block_encoding._qsharp_op  # noqa: SLF001
        num_qubits = block_encoding.num_qubits
        if num_qubits is None:
            raise ValueError(f"Circuit mapper '{type(mapper).__name__}' did not report num_qubits.")
        num_system_qubits = qubit_hamiltonian.num_qubits

        block_encoding_shared = block_encoding.metadata.num_phase_gradient_ancillas
        state_prep_shared = state_preparation.metadata.num_phase_gradient_ancillas
        if block_encoding_shared and state_prep_shared and block_encoding_shared != state_prep_shared:
            raise ValueError(
                f"State preparation expects {state_prep_shared} phase gradient ancilla but the "
                f"block encoding expects {block_encoding_shared}."
            )
        num_phase_gradient_ancillas = max(block_encoding_shared, state_prep_shared)
        prepare_shared_op = (
            QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState
            if num_phase_gradient_ancillas
            else QSHARP_UTILS.PrepSelPrep.NoOpPrepare
        )
        num_ancilla_qubits = num_qubits - num_system_qubits - block_encoding_shared
        if num_ancilla_qubits <= 0:
            raise ValueError(f"Requires a non-empty ancilla register to reflect about, got {num_ancilla_qubits}.")

        apply_reflection = QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(num_system_qubits, num_ancilla_qubits)
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise RuntimeError("State preparation has no Q# operation.")

        phase_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=list(range(num_phase_qubits - 1, -1, -1)),
            stateVector=cosine_window_state(num_queries),
            expansionOps=[],
            numQubits=num_phase_qubits,
        )
        parameters = {
            "statePrep": state_prep_op,
            "applyBlockEncoding": block_encoding_op,
            "applyReflection": apply_reflection,
            "phaseQubitPrep": QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(phase_prep_params),
            "prepareSharedOp": prepare_shared_op,
            "numQueries": num_queries,
            "numSystemQubits": num_system_qubits,
            "numAncillas": num_ancilla_qubits,
            "numSharedAncillas": num_phase_gradient_ancillas,
            "statePrepUsesShared": bool(state_prep_shared),
            "blockEncodingUsesShared": bool(block_encoding_shared),
        }
        circuit = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.UnaryPhaseEstimation.MakeUnaryQPECircuit,
                parameter=parameters,
            ),
            num_qubits=num_phase_qubits + num_system_qubits + num_ancilla_qubits + num_phase_gradient_ancillas,
        )
        return [circuit]

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_unary"
