r"""Amplitude-amplified standard phase estimation.

Composes the standard (QFT-based) QPE circuit builder with the generic
amplitude-amplification loop.  The QPE circuit built in ``coherent`` mode is the
state preparation :math:`U_\\psi`; the good subspace is the set of branches whose
phase register falls inside an accepted energy window *and* whose block-encoding
signal ancillas are all zero.

This is the poor-overlap workflow: a guiding state with small overlap on the
target eigenvector makes the accepted fraction of shots small, and amplification
buys that fraction back at a cost of :math:`2k+1` coherent QPE preparations per
attempt.  Amplification changes *how often* the window is accepted; it does not
change what is accepted, and it cannot repair a badly chosen window or
insufficient phase resolution.

References:
    L. Lin, *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.amplitude_amplification import schedule
from qdk_chemistry.algorithms.amplitude_amplification.base import (
    AmplitudeAmplification,
    AmplitudeAmplificationSettings,
)
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    StandardQpeCircuitBuilder,
    coherent_qpe_measured_indices,
    split_coherent_qpe_bitstring,
)
from qdk_chemistry.data import (
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplifiedPhaseEstimation",
    "AmplifiedPhaseEstimationSettings",
]


class AmplifiedPhaseEstimationSettings(AmplitudeAmplificationSettings):
    """Settings for amplitude-amplified standard phase estimation."""

    def __init__(self):
        """Initialize the settings for amplified phase estimation."""
        super().__init__()


class AmplifiedPhaseEstimation(AmplitudeAmplification):
    """Amplitude-amplified standard (QFT-based) phase estimation.

    Example:
        >>> from qdk_chemistry.algorithms import create  # doctest: +SKIP
        >>> qpe = create("amplitude_amplification", "qdk_amplified_qpe")  # doctest: +SKIP
        >>> qpe.settings().set("max_overlap", 0.05)  # doctest: +SKIP
        >>> qpe.settings().set("min_energy", -1.1)  # doctest: +SKIP
        >>> qpe.settings().set("max_energy", -0.9)  # doctest: +SKIP
        >>> result = qpe.run(state_preparation, hamiltonian)  # doctest: +SKIP

    """

    def __init__(self, shots: int = 100):
        """Initialize amplified standard phase estimation.

        Args:
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = AmplifiedPhaseEstimationSettings()
        self._settings.set("shots", shots)

    def name(self) -> str:
        """Return the algorithm name as qdk_amplified_qpe."""
        return "qdk_amplified_qpe"

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run amplitude-amplified standard phase estimation.

        Args:
            state_preparation: The circuit that prepares the initial (guiding) state.
            qubit_hamiltonian: The qubit Hamiltonian whose eigenvalue is estimated.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult built from the dominant *accepted* bitstring, with the
            acceptance statistics in :attr:`~qdk_chemistry.data.QpeResult.metadata`.

        Raises:
            TypeError: If the nested QPE circuit builder is not a standard builder.
            RuntimeError: If no shot was accepted.

        """
        Logger.trace_entering()
        circuit_builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(circuit_builder, StandardQpeCircuitBuilder):
            raise TypeError(
                f"Expected qpe_circuit_builder to be an instance of StandardQpeCircuitBuilder, "
                f"but got {type(circuit_builder)} instead."
            )
        circuit_builder.settings().update("coherent", True)
        num_bits = circuit_builder.settings().get("num_bits")

        # Resolve the encoding before building, so the accepted window can be
        # expressed in energy units for any container type.
        unitary_builder = circuit_builder._create_nested("unitary_builder")  # noqa: SLF001
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()
        num_system_qubits = qubit_hamiltonian.num_qubits
        num_ancilla_qubits = unitary_rep.get_num_qubits() - num_system_qubits

        accepted_indices = self._resolve_accepted_indices(num_bits, container.eigenvalue_from_phase)
        rounds = self.resolve_rounds()

        coherent_circuit = circuit_builder.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )[0]
        circuit = self._build_amplified_circuit(
            coherent_circuit=coherent_circuit,
            num_bits=num_bits,
            num_system_qubits=num_system_qubits,
            num_ancilla_qubits=num_ancilla_qubits,
            accepted_indices=accepted_indices,
            rounds=rounds,
        )

        shots = self._settings.get("shots")
        circuit_executor = self._create_nested("circuit_executor")
        execution_data = circuit_executor.run(circuit, shots=shots, noise=noise)

        accepted_counts = self._accepted_counts(execution_data.bitstring_counts, num_bits, accepted_indices)
        if not accepted_counts:
            raise RuntimeError(
                f"No shot landed in the accepted window {sorted(accepted_indices)} after {rounds} "
                f"amplification round(s) over {shots} shot(s). Widen the window, increase the "
                "shot count, or revisit the round policy."
            )
        accepted_shots = sum(accepted_counts.values())

        dominant_bitstring = max(accepted_counts, key=accepted_counts.get)
        raw_phase = int(dominant_bitstring, 2) / (2**num_bits)

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=raw_phase,
            eigenvalue_from_phase=container.eigenvalue_from_phase,
            bitstring_msb_first=dominant_bitstring,
            metadata=self._acceptance_metadata(
                rounds=rounds,
                accepted_indices=accepted_indices,
                accepted_shots=accepted_shots,
                shots=shots,
            ),
        )

    def _resolve_accepted_indices(self, num_bits: int, eigenvalue_from_phase) -> list[int]:
        """Resolve which phase-register indices count as good.

        An explicit ``accepted_phase_indices`` list wins. Otherwise the window is
        derived from ``[min_energy, max_energy]`` using the encoding's own
        phase-to-eigenvalue map, which keeps the window meaningful for Trotter
        and qubitization alike.

        Args:
            num_bits: Number of phase qubits.
            eigenvalue_from_phase: The container's phase-fraction-to-eigenvalue map.

        Returns:
            The sorted list of accepted phase-register indices.

        Raises:
            ValueError: If neither an index list nor a usable energy window was given,
                if an index is out of range, or if the window contains no bin.

        """
        explicit = [int(index) for index in self._settings.get("accepted_phase_indices")]
        if explicit:
            if any(index < 0 or index >= 2**num_bits for index in explicit):
                raise ValueError(
                    f"accepted_phase_indices must lie in [0, {2**num_bits}) for a {num_bits}-bit phase register."
                )
            return sorted(set(explicit))

        min_energy = float(self._settings.get("min_energy"))
        max_energy = float(self._settings.get("max_energy"))
        if max_energy <= min_energy:
            raise ValueError(
                "Set accepted_phase_indices, or an energy window with max_energy > min_energy, "
                "to define the subspace to amplify."
            )

        accepted = [
            index
            for index in range(2**num_bits)
            if min_energy <= eigenvalue_from_phase(index / (2**num_bits)) <= max_energy
        ]
        if not accepted:
            raise ValueError(
                f"The energy window [{min_energy}, {max_energy}] contains no phase bin at "
                f"{num_bits}-bit resolution. Widen the window or add phase bits."
            )
        Logger.info(f"Amplifying {len(accepted)} of {2**num_bits} phase bins.")
        return accepted

    def _build_amplified_circuit(
        self,
        *,
        coherent_circuit: Circuit,
        num_bits: int,
        num_system_qubits: int,
        num_ancilla_qubits: int,
        accepted_indices: list[int],
        rounds: int,
    ) -> Circuit:
        """Wrap the coherent QPE circuit in the amplification loop.

        Args:
            coherent_circuit: The measurement-free QPE circuit acting as the preparation.
            num_bits: Number of phase qubits.
            num_system_qubits: Number of system qubits.
            num_ancilla_qubits: Number of block-encoding signal ancillas.
            accepted_indices: Phase-register indices that count as good.
            rounds: Number of amplification rounds (or phase-sequence length for fixed point).

        Returns:
            The circuit to execute.

        """
        amplification = QSHARP_UTILS.AmplitudeAmplification
        preparation = coherent_circuit._qsharp_op  # noqa: SLF001
        signal_ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
        marker = amplification.MakeQpeAcceptanceMarkerOp(num_bits, signal_ancilla_indices, accepted_indices)
        num_qubits = num_bits + num_system_qubits + num_ancilla_qubits
        measured_indices = coherent_qpe_measured_indices(num_bits, num_system_qubits, num_ancilla_qubits)

        if str(self._settings.get("round_policy")) == "fixed_point" and int(self._settings.get("rounds")) < 0:
            mark_phases, state_phases = schedule.fixed_point_phases(rounds, float(self._settings.get("tolerance")))
            parameters = {
                "preparation": preparation,
                "markingOracle": marker,
                "markPhases": mark_phases,
                "statePhases": state_phases,
                "numQubits": num_qubits,
                "measuredIndices": measured_indices,
            }
            program = amplification.MakeFixedPointAmplifiedCircuit
        else:
            parameters = {
                "preparation": preparation,
                "markingOracle": marker,
                "rounds": rounds,
                "numQubits": num_qubits,
                "measuredIndices": measured_indices,
            }
            program = amplification.MakeAmplifiedCircuit

        return Circuit(qsharp_factory=QsharpFactoryData(program=program, parameter=parameters))

    @staticmethod
    def _accepted_counts(
        bitstring_counts: dict[str, int],
        num_bits: int,
        accepted_indices: list[int],
    ) -> dict[str, int]:
        """Keep only the shots that landed in the good subspace.

        A shot is accepted when its phase index is in the window *and* every
        block-encoding signal ancilla measured zero; a nonzero ancilla means the
        block encoding did not project onto the signal block, so the phase
        register carries no eigenvalue information for that branch.

        Args:
            bitstring_counts: Counts keyed by executor bitstring.
            num_bits: Number of phase qubits.
            accepted_indices: Phase-register indices that count as good.

        Returns:
            Counts keyed by the phase bits alone, restricted to accepted shots.

        """
        accepted_set = set(accepted_indices)
        counts: dict[str, int] = {}
        for bitstring, count in bitstring_counts.items():
            phase_bits, ancilla_bits = split_coherent_qpe_bitstring(bitstring, num_bits)
            if any(bit != "0" for bit in ancilla_bits):
                continue
            if int(phase_bits, 2) not in accepted_set:
                continue
            counts[phase_bits] = counts.get(phase_bits, 0) + count
        return counts

    def _acceptance_metadata(
        self,
        *,
        rounds: int,
        accepted_indices: list[int],
        accepted_shots: int,
        shots: int,
    ) -> dict[str, object]:
        """Assemble the acceptance statistics reported alongside the phase.

        Args:
            rounds: Number of amplification rounds actually run.
            accepted_indices: Phase-register indices that count as good.
            accepted_shots: Number of shots that landed in the good subspace.
            shots: Total number of shots executed.

        Returns:
            A metadata dictionary for the QpeResult.

        """
        min_overlap = float(self._settings.get("min_overlap"))
        max_overlap = float(self._settings.get("max_overlap"))
        metadata: dict[str, object] = {
            "amplification_rounds": rounds,
            "round_policy": str(self._settings.get("round_policy")),
            "accepted_phase_indices": list(accepted_indices),
            "accepted_shots": accepted_shots,
            "total_shots": shots,
            "acceptance_probability": accepted_shots / shots if shots else 0.0,
            # One attempt costs 2k+1 coherent preparations of the QPE circuit.
            "preparations_per_shot": 2 * rounds + 1,
        }
        if 0.0 < min_overlap <= 1.0:
            metadata["predicted_acceptance_at_min_overlap"] = schedule.success_probability(min_overlap, rounds)
        if 0.0 < max_overlap <= 1.0:
            metadata["predicted_acceptance_at_max_overlap"] = schedule.success_probability(max_overlap, rounds)
        return metadata
