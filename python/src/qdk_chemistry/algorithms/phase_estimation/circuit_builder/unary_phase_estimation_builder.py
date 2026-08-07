"""Unary-iteration phase estimation circuit builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = [
    "QdkUnaryQpeCircuitBuilder",
    "QdkUnaryQpeCircuitBuilderSettings",
]


def num_phase_bits(num_queries: int) -> int:
    """Return the phase-register size addressing ``num_queries + 1``.

    Args:
        num_queries: Number of walk blocks applied by the schedule.

    Returns:
        The number of phase qubits.

    Raises:
        ValueError: If ``num_queries`` is not positive.

    """
    if num_queries <= 0:
        raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")
    return int(num_queries).bit_length()


def cosine_window_state(num_queries: int) -> list[float]:
    r"""Return the phase-register amplitudes, zero-padded to a whole number of qubits.

    The cosine window is the Heisenberg-limited control state of
    :cite:`Babbush2018` (Eq. 17), :math:`\psi_t \propto \sin(\pi (t + 1) / (p + 2))`
    over the :math:`p + 1` reflection slots. It is the optimal single-lobe
    window for phase estimation and needs no special functions.

    Args:
        num_queries: Number of walk blocks; the window spans ``num_queries + 1`` slots.

    Returns:
        Real amplitudes normalized to unit norm, zero-padded to a whole number of qubits.

    """
    dimension = 1 << num_phase_bits(num_queries)
    num_states = num_queries + 1
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
            "Number of walk blocks the signed-power schedule applies. Sizing it as "
            "ceil(pi * lambda / (2 * epsilon)) targets an energy error epsilon for a block encoding "
            "of normalization lambda (Lee et al. 2021, Eq. 45); unlike standard QPE it need not be "
            "a power of two.",
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "prepare_select_prepare"),
            "Mapper producing the uncontrolled block encoding the schedule applies.",
        )
        self.set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True))


class QdkUnaryQpeCircuitBuilder(QpeCircuitBuilder):
    r"""Phase estimation circuit builder driven by unary iteration over a signed-power schedule.

    Standard QPE applies controlled :math:`U^{2^k}` once per phase qubit and therefore
    consumes a power-of-two number of walk queries. This builder instead emits a single
    chain of ``num_queries`` self-inverse walk blocks and uses unary iteration over the
    phase register to select which interleaved reflection is omitted, so any positive
    query count is supported.

    The phase register is prepared in a cosine window state to suppress the spectral leakage
    of the truncated schedule. It only supports block-encoded Hamiltonians with a unitary builder.

    The phase register addresses one of :math:`p + 1` reflection slots, where :math:`p`
    is the query count. Omitting slot :math:`t` realizes :math:`W^{p-2t}` while applying
    exactly :math:`p` walk blocks, so :math:`p` need not be a power of two.

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

    def resolve_num_queries(self, unitary_rep: UnitaryRepresentation) -> int:
        """Return the configured query count, ignoring any power carried by the container.

        The schedule applies exactly ``num_queries`` walk blocks, so the count is the query
        complexity of the estimate. Choosing ``ceil(pi * lambda / (2 * epsilon))`` targets an
        energy error ``epsilon`` for a block encoding of normalization ``lambda``
        (:cite:`Lee2021`, Eq. 45).

        Args:
            unitary_rep: The unitary representation the schedule will be built from.

        Returns:
            The number of walk blocks the schedule applies.

        Raises:
            ValueError: If the configured ``num_queries`` is not a positive integer.

        """
        container_power = getattr(unitary_rep.get_container(), "power", 1)
        if container_power != 1:
            Logger.warn(f"The unitary representation carries power {container_power}, which is ignored.")

        num_queries = self._settings.get("num_queries")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")
        return int(num_queries)

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
            ValueError: If the unitary representation is not a quantum walk, if the configured
                circuit mapper does not expose the block-encoding API, or if the block encoding
                has no ancilla register for the walk to reflect about.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()
        if not isinstance(container, LCUWalkContainer):
            raise ValueError(f"Requires a block encoding unitary representation, got '{type(container).__name__}'.")

        num_queries = self.resolve_num_queries(unitary_rep)
        num_bits = num_phase_bits(num_queries)
        configured_num_bits = self._settings.get("num_bits")
        if configured_num_bits > 0:
            Logger.info(
                f"num_bits={configured_num_bits} is ignored; the phase register is sized to {num_bits} "
                f"bits so it can address the {num_queries + 1} reflection slots of num_queries={num_queries}."
            )

        mapper = self._create_nested("circuit_mapper")
        # The schedule interleaves the reflections itself so it can omit the one the phase
        # register addresses, so it needs the block encoding and the reflection separately.
        # run() only hands them back already composed into a walk, which cannot be split again.
        missing = [
            name for name in ("num_ancilla_qubits", "block_encoding_op", "reflection_op") if not hasattr(mapper, name)
        ]
        if missing:
            raise ValueError(
                f"Circuit mapper '{type(mapper).__name__}' does not expose {', '.join(missing)}. "
                "Unary QPE needs a block-encoding mapper such as 'prepare_select_prepare'."
            )

        num_ancilla_qubits = mapper.num_ancilla_qubits(container)
        if num_ancilla_qubits == 0:
            raise ValueError("Requires a non-empty ancilla register to reflect about.")

        block_encoding_op = mapper.block_encoding_op(container)
        apply_reflection = mapper.reflection_op(container)

        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise RuntimeError("State preparation has no Q# operation.")

        # rowMap is reversed so the window state is big-endian, matching ApplyQFT.
        phase_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=list(range(num_bits - 1, -1, -1)),
            stateVector=cosine_window_state(num_queries),
            expansionOps=[],
            numQubits=num_bits,
        )
        parameters = {
            "statePrep": state_prep_op,
            "applyBlockEncoding": block_encoding_op,
            "applyReflection": apply_reflection,
            "phaseQubitPrep": QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(phase_prep_params),
            "numQueries": num_queries,
            "ancillas": list(range(num_bits)),
            "systems": [index + num_bits for index in range(qubit_hamiltonian.num_qubits)],
            "numAncillas": num_ancilla_qubits,
        }
        circuit = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.UnaryPhaseEstimation.MakeUnaryQPECircuit,
                parameter=parameters,
            )
        )
        Logger.info(f"Built unary QPE circuit with {num_queries} queries and {num_bits} phase qubits.")
        return [circuit]

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_unary"
