"""Unary-iteration phase estimation circuit builder for arbitrary query counts.

Standard QPE applies controlled :math:`U^{2^k}` once per phase qubit and therefore
consumes a power-of-two number of walk queries. This builder instead emits a single
chain of ``num_queries`` self-inverse walk blocks and uses unary iteration over the
phase register to select which interleaved reflection is omitted, so any positive
query count is supported.

The phase register is prepared in a cosine window state to suppress the spectral leakage of the truncated schedule.
It only support block-encoded Hamiltonians unitary builder.

References:
    * :cite:`Babbush2018` — Heisenberg-limited phase estimation with a
      cosine-window control state.
    * :cite:`Lee2021` — tensor hypercontraction; non-power-of-two query schedule
      and its extra Toffoli cost.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = [
    "QdkUnaryQpeCircuitBuilder",
    "QdkUnaryQpeCircuitBuilderSettings",
    "cosine_window_state",
    "num_phase_bits",
]


def num_phase_bits(num_queries: int) -> int:
    """Return the phase-register size addressing ``num_queries + 1`` reflection slots.

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
    Babbush et al. (2018), :math:`\psi_t \propto \sin(\pi (t + 1) / (p + 2))`
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
            "Number of walk blocks to apply; ignored when the unitary representation carries a power.",
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

    The phase register addresses one of :math:`p + 1` reflection slots, where :math:`p`
    is the query count. Omitting slot :math:`t` realizes :math:`W^{p-2t}` while applying
    exactly :math:`p` walk blocks, so :math:`p` need not be a power of two.

    The block encoding is the generic LCU built in quantum-walk mode, mapped to an
    uncontrolled circuit by :class:`~qdk_chemistry.algorithms.circuit_mapper.psp_mapper.PSPMapper`.
    The mapper never materializes the walk: this builder interleaves the reflections itself so
    it can omit exactly one, which is what makes a non-power-of-two query count possible.

    """

    def __init__(
        self,
        num_queries: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        circuit_mapper: AlgorithmRef | None = None,
    ) -> None:
        """Initialize the unary-iteration QPE circuit builder.

        Args:
            num_queries: Number of walk blocks; used when the unitary representation has no power.
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
        """Return the query count, preferring the power carried by the unitary representation.

        Args:
            unitary_rep: The unitary representation produced by the nested unitary builder.

        Returns:
            The number of walk blocks to apply.

        Raises:
            ValueError: If neither the unitary representation nor the settings supply a positive count.

        """
        container_power = getattr(unitary_rep.get_container(), "power", 1)
        if container_power > 1:
            return int(container_power)
        num_queries = self._settings.get("num_queries")
        if num_queries <= 0:
            raise ValueError(
                "num_queries must be a positive integer, either from the unitary representation's "
                f"power or from the builder settings. Got {num_queries}."
            )
        return int(num_queries)

    def phase_fraction_from_measurement(self, measured_phase_fraction: float, phase_band: str) -> float:
        """Resolve the doubled measured phase within the requested half-band.

        The schedule only realizes even offsets from the query count, so the measurement is
        invariant under a sign flip of the eigenvalue and the band has to be supplied.

        Args:
            measured_phase_fraction: The raw fraction read from the phase register.
            phase_band: ``"lower"`` for a non-negative eigenvalue, ``"upper"`` for a non-positive one.

        Returns:
            The walk phase fraction in the requested half-band.

        """
        doubled_phase = measured_phase_fraction % 1.0
        folded_phase = min(doubled_phase, (-doubled_phase) % 1.0) / 2.0
        if phase_band == "lower":
            return folded_phase
        return 0.5 - folded_phase

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
            ValueError: If the block encoding has no ancilla register for the walk to reflect about.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        num_queries = self.resolve_num_queries(unitary_rep)
        num_bits = num_phase_bits(num_queries)

        mapper = self._create_nested("circuit_mapper")
        container = unitary_rep.get_container()
        num_ancilla_qubits = mapper.num_ancillary_qubits(container)
        if num_ancilla_qubits == 0:
            raise ValueError(
                "A signed-power schedule needs a non-empty ancilla register to reflect about, "
                "but this block encoding has none."
            )

        # The mapper returns the block encoding alone; the schedule below owns the reflections,
        # so the walk is never materialized as a controlled operation. Power is forced to 1
        # because the schedule applies the block once per slot.
        unitary_builder.settings().update("power", 1)
        block_encoding_op = mapper.run(unitary_builder.run(qubit_hamiltonian))._qsharp_op  # noqa: SLF001
        num_system_qubits = qubit_hamiltonian.num_qubits
        reflection_register_of = QSHARP_UTILS.PrepSelPrep.MakeTrailingAncillaSelector(num_system_qubits)

        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise RuntimeError("Failed to create unary QPE circuit: state preparation has no Q# operation.")

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
            "reflectionRegisterOf": reflection_register_of,
            "numQueries": num_queries,
            "ancillas": list(range(num_bits)),
            "systems": [index + num_bits for index in range(qubit_hamiltonian.num_qubits)],
            "phaseQubitPrep": QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(phase_prep_params),
            "numAncillas": num_ancilla_qubits,
            "ancillaPrep": QSHARP_UTILS.StatePreparation.MakeNoOpAncillaPrep(),
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
