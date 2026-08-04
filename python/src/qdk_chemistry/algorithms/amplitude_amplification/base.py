r"""QDK/Chemistry amplitude amplification.

Amplitude amplification boosts the probability that a prepared state is found in
a *good* subspace.  It is built from two independent halves:

* a **preparation** :math:`U_\psi`, which defines the state being amplified and
  the register whose measurement decides success, and
* a **marking oracle**, which defines what counts as good.

This module keeps those halves separate.  The preparation is supplied through
the ``reflect_to_good_space`` setting as a nested algorithm reference, so the
amplification loop itself knows nothing about phase estimation: it only needs a
coherent (measurement-free, adjointable) circuit to reflect about.  Today that
slot is filled by a :doc:`QPE circuit builder </user/comprehensive/algorithms/qpe_circuit_builder>`
and the good subspace is an energy window on its phase register, which is the
poor-overlap workflow this algorithm exists for: a guiding state with small
overlap on the target eigenvector makes the accepted fraction of shots small,
and amplification buys that fraction back at a cost of :math:`2k+1` coherent
preparations per attempt.

Amplification changes *how often* the window is accepted; it does not change
what is accepted, and it cannot repair a badly chosen window or insufficient
phase resolution.

References:
    L. Lin, *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import split_coherent_qpe_bitstring
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
    Settings,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
]


class AmplitudeAmplificationSettings(Settings):
    r"""Settings for amplitude amplification.

    The round-count settings answer the central practical question: how many
    rounds can be run when the overlap of the guiding state is only known
    approximately?

    Plain amplitude amplification rotates by :math:`(2k+1)\vartheta` with
    :math:`\vartheta = \arcsin\sqrt{a}`, so its acceptance probability
    :math:`\sin^2((2k+1)\vartheta)` *falls back to zero* once the rotation runs
    past :math:`\pi/2`.  Guessing ``k`` from an uncertain overlap therefore
    risks overshooting, and an overshoot is indistinguishable from a small
    overlap in the measured counts -- it fails silently.

    The round count is consequently derived from the Yoder-Low-Chuang
    fixed-point schedule, which replaces that sinusoid by a Chebyshev plateau:
    acceptance climbs monotonically and then stays above
    :math:`1 - \text{tolerance}^2` for *every* overlap at or above
    ``min_overlap``.  Only a **lower** bound is required, which is the bound a
    classical overlap estimate actually provides, and no overshoot is possible.
    The guarantee costs roughly twice the queries of a perfectly-tuned plain
    schedule.

    Set ``rounds`` explicitly to bypass the schedule and run that many plain
    Grover iterates instead.

    """

    def __init__(self):
        """Initialize the settings for amplitude amplification."""
        super().__init__()
        self._set_default(
            "reflect_to_good_space",
            "algorithm_ref",
            AlgorithmRef("qpe_circuit_builder", "qdk_standard"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        self._set_default("shots", "int", 100, "The number of shots to execute the circuit.")
        self._set_default(
            "rounds",
            "int",
            -1,
            "Explicit number of plain Grover iterates. Negative derives a fixed-point schedule instead.",
        )
        self._set_default(
            "min_overlap",
            "double",
            0.0,
            "Lower bound on the probability that the prepared state lands in the good subspace.",
        )
        self._set_default(
            "tolerance",
            "double",
            0.1,
            "Fixed-point amplification tolerance; success is guaranteed to exceed 1 - tolerance^2.",
        )
        self._set_default(
            "accepted_phase_indices",
            "vector<int>",
            [],
            "Phase-register indices that count as good. Empty means derive them from the energy window.",
        )
        self._set_default(
            "min_energy",
            "double",
            0.0,
            "Lower edge of the accepted energy window, used when accepted_phase_indices is empty.",
        )
        self._set_default(
            "max_energy",
            "double",
            0.0,
            "Upper edge of the accepted energy window. Must exceed min_energy to be used.",
        )


class AmplitudeAmplification(Algorithm):
    r"""Amplitude amplification around a nested coherent circuit.

    The circuit supplied through ``reflect_to_good_space`` is built with
    ``measurement='none'`` to obtain the measurement-free, adjointable
    preparation :math:`U_\psi` that the amplification loop reflects about.  The
    good subspace is the set of branches whose phase register falls inside the
    accepted window *and* whose block-encoding signal ancillas are all zero.

    Example:
        >>> from qdk_chemistry.algorithms import create  # doctest: +SKIP
        >>> from qdk_chemistry.data import AlgorithmRef  # doctest: +SKIP
        >>> aa = create("amplitude_amplification")  # doctest: +SKIP
        >>> aa.settings().update(
        ...     "reflect_to_good_space",
        ...     AlgorithmRef("qpe_circuit_builder", "qdk_standard", num_bits=8),
        ... )  # doctest: +SKIP
        >>> aa.settings().update("min_overlap", 0.05)  # doctest: +SKIP
        >>> aa.settings().update("min_energy", -1.1)  # doctest: +SKIP
        >>> aa.settings().update("max_energy", -0.9)  # doctest: +SKIP
        >>> result = aa.run(state_preparation, hamiltonian)  # doctest: +SKIP

    """

    def __init__(self, shots: int = 100):
        """Initialize amplitude amplification.

        Args:
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = AmplitudeAmplificationSettings()
        self._settings.set("shots", shots)

    def type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

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
        """Amplify the good subspace of a prepared state and measure the outcome.

        Args:
            state_preparation: The circuit that prepares the initial (guiding) state.
            qubit_hamiltonian: The qubit Hamiltonian defining the unitary to amplify against.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult built from the dominant *accepted* bitstring.

        Raises:
            RuntimeError: If no shot was accepted.

        """
        Logger.trace_entering()
        circuit_builder = self._create_nested("reflect_to_good_space")
        # Builders that do not expose the setting are still accepted; they just
        # have to return a measurement-free circuit on their own.
        if circuit_builder.settings().has("measurement"):
            circuit_builder.settings().update("measurement", "none")
        num_bits = int(circuit_builder.settings().get("num_bits"))

        # Resolve the encoding before building, so the accepted window can be
        # expressed in energy units for any container type.
        unitary_builder = circuit_builder._create_nested("unitary_builder")  # noqa: SLF001
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()
        num_system_qubits = qubit_hamiltonian.num_qubits
        num_ancilla_qubits = unitary_rep.get_num_qubits() - num_system_qubits

        accepted_indices = self._resolve_accepted_indices(num_bits, container.eigenvalue_from_phase)
        rounds = self.resolve_rounds()

        preparation = self._coherent_preparation(
            circuit_builder=circuit_builder,
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        circuit = self._build_amplified_circuit(
            preparation=preparation,
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
                "shot count, or lower 'min_overlap'."
            )

        dominant_bitstring = max(accepted_counts, key=accepted_counts.get)
        raw_phase = int(dominant_bitstring, 2) / (2**num_bits)

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=raw_phase,
            eigenvalue_from_phase=container.eigenvalue_from_phase,
            bitstring_msb_first=dominant_bitstring,
        )

    def resolve_rounds(self) -> int:
        """Resolve the number of amplification rounds from the settings.

        An explicit non-negative ``rounds`` always wins and selects that many
        plain Grover iterates. Otherwise the count is derived from the
        Yoder-Low-Chuang fixed-point schedule, which needs only ``min_overlap``
        and ``tolerance`` and cannot overshoot.

        Returns:
            The number of amplification rounds to run.

        Raises:
            ValueError: If no explicit ``rounds`` was given and ``min_overlap``
                is not a usable lower bound.

        """
        rounds = int(self._settings.get("rounds"))
        if rounds >= 0:
            return rounds

        min_overlap = float(self._settings.get("min_overlap"))
        if min_overlap <= 0.0:
            raise ValueError(
                "Deriving a round count needs a positive 'min_overlap' lower bound on the overlap "
                "of the guiding state with the good subspace. Set 'min_overlap', or set 'rounds' "
                "explicitly to run a fixed number of plain Grover iterates."
            )
        return self.fixed_point_rounds(min_overlap, float(self._settings.get("tolerance")))

    @staticmethod
    def _validate_overlap(overlap: float, name: str = "overlap") -> None:
        """Raise if ``overlap`` is not a usable squared overlap.

        Args:
            overlap: The value to check.
            name: Name used in the error message.

        Raises:
            ValueError: If ``overlap`` is not finite or does not lie in ``(0, 1]``.

        """
        if not math.isfinite(overlap) or not 0.0 < overlap <= 1.0:
            raise ValueError(f"{name} must be finite and lie in (0, 1]. Got {overlap}.")

    @staticmethod
    def _validate_schedule_rounds(rounds: int) -> None:
        """Raise if ``rounds`` is not a valid round count.

        Args:
            rounds: The number of amplification rounds.

        Raises:
            ValueError: If ``rounds`` is negative.

        """
        if rounds < 0:
            raise ValueError(f"rounds must be nonnegative. Got {rounds}.")

    @classmethod
    def _rotation_angle(cls, overlap: float) -> float:
        r"""Return the half-rotation angle :math:`\vartheta = \arcsin\sqrt{a}`.

        Args:
            overlap: The squared overlap ``a`` of the prepared state with the good
                subspace, in ``(0, 1]``.

        Returns:
            The angle in radians, in ``(0, pi/2]``.

        """
        cls._validate_overlap(overlap)
        return math.asin(math.sqrt(overlap))

    @classmethod
    def success_probability(cls, overlap: float, rounds: int) -> float:
        r"""Return the acceptance probability :math:`\sin^2((2k+1)\vartheta)`.

        After ``k`` rounds the prepared state has rotated by :math:`(2k+1)\vartheta`
        inside the two-dimensional invariant subspace, so the probability of
        landing in the good subspace is :math:`\sin^2((2k+1)\vartheta)` with
        :math:`\vartheta = \arcsin\sqrt{a}`.

        Args:
            overlap: The squared overlap of the prepared state with the good subspace.
            rounds: The number of amplification rounds ``k``.

        Returns:
            The probability of measuring the good subspace after ``rounds`` rounds.

        """
        cls._validate_schedule_rounds(rounds)
        angle = cls._rotation_angle(overlap)
        return math.sin((2 * rounds + 1) * angle) ** 2

    @classmethod
    def fixed_point_rounds(cls, min_overlap: float, tolerance: float) -> int:
        r"""Return the iterate count for fixed-point amplification.

        The Yoder-Low-Chuang schedule reaches acceptance probability at least
        :math:`1 - \delta^2` for *every* overlap at or above ``min_overlap`` once
        the number of queries satisfies
        :math:`L \ge \log(2/\delta)/\sqrt{a_{\min}}`. This returns the smallest
        ``l`` with ``L = 2l + 1`` meeting that bound.

        Args:
            min_overlap: Lower bound on the squared overlap.
            tolerance: The failure amplitude ``delta`` in ``(0, 1)``; the acceptance
                probability is at least ``1 - delta ** 2``.

        Returns:
            The number of iterates ``l``, so that ``2 * l + 1`` queries are used.

        Raises:
            ValueError: If ``tolerance`` is not in ``(0, 1)``.

        """
        cls._validate_overlap(min_overlap, "min_overlap")
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")
        queries = math.log(2.0 / tolerance) / math.sqrt(min_overlap)
        return max(1, math.ceil((math.ceil(queries) - 1) / 2.0))

    @staticmethod
    def fixed_point_phases(rounds: int, tolerance: float) -> tuple[list[float], list[float]]:
        r"""Return the Yoder-Low-Chuang phase sequence for fixed-point amplification.

        With ``L = 2 * rounds + 1`` queries and

        .. math::

            \gamma^{-1} = T_{1/L}(1/\delta)
                        = \cosh\!\big(L^{-1}\operatorname{arccosh}(1/\delta)\big),

        the state-reflection phases are

        .. math::

            \beta_j = 2\operatorname{arccot}\!\big(\tan(2\pi j/L)\sqrt{1-\gamma^2}\big),
            \qquad j = 1,\dots,l ,

        and the mark phases are the same list reversed,
        :math:`\alpha_j = \beta_{l+1-j}`. Both reflections are taken in the
        ``I - (1 - e^{i\varphi}) P`` convention used by the Q# implementation, with
        the mark applied before the state reflection, which is why no sign flip
        appears between the two sequences.

        The resulting acceptance probability climbs monotonically up to
        :math:`a = 1 - T_{1/L}(1/\delta)^{-2}` and from there stays inside
        :math:`[1-\delta^2, 1]` for every larger overlap -- a plateau, not a global
        maximum. There is no first maximum to run past, so overshoot is impossible
        above the design threshold.

        Args:
            rounds: The number of iterates ``l``; ``2 * l + 1`` queries are used.
            tolerance: The failure amplitude ``delta`` in ``(0, 1)``.

        Returns:
            The mark phases ``alpha`` and the state phases ``beta``, both of length
            ``rounds`` and ordered by iterate.

        Raises:
            ValueError: If ``rounds`` is not positive or ``tolerance`` is not in
                ``(0, 1)``.

        """
        if rounds < 1:
            raise ValueError(f"rounds must be positive. Got {rounds}.")
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")

        queries = 2 * rounds + 1
        gamma = 1.0 / math.cosh(math.acosh(1.0 / tolerance) / queries)
        scale = math.sqrt(max(0.0, 1.0 - gamma * gamma))

        # arccot with range (0, pi), so that the phases are continuous in j.
        state_phases = [
            2.0 * math.atan2(1.0, math.tan(2.0 * math.pi * j / queries) * scale) for j in range(1, rounds + 1)
        ]
        mark_phases = list(reversed(state_phases))
        return mark_phases, state_phases

    @staticmethod
    def _coherent_preparation(
        *,
        circuit_builder: Algorithm,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ):
        r"""Build the preparation the amplification loop reflects about.

        The nested algorithm is asked for a measurement-free circuit and must
        answer with one that carries an adjointable Q# operation; reflecting about
        :math:`U_\psi` requires applying its adjoint.

        Args:
            circuit_builder: The nested algorithm from ``reflect_to_good_space``.
            state_preparation: The circuit that prepares the initial (guiding) state.
            qubit_hamiltonian: The qubit Hamiltonian defining the unitary.

        Returns:
            The adjointable Q# operation implementing the preparation.

        Raises:
            TypeError: If the nested algorithm did not honor ``measurement="none"``.

        """
        circuit = circuit_builder.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )[0]
        operation = circuit._qsharp_op  # noqa: SLF001
        if operation is None:
            raise TypeError(
                f"The '{circuit_builder.name()}' algorithm in 'reflect_to_good_space' did not produce a "
                "coherent circuit. Amplitude amplification reflects about the prepared state, so the "
                "nested algorithm must honor measurement='none' and return a measurement-free, "
                "adjointable Q# circuit. Use the qdk_standard QPE circuit builder."
            )
        return operation

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
        preparation,
        num_bits: int,
        num_system_qubits: int,
        num_ancilla_qubits: int,
        accepted_indices: list[int],
        rounds: int,
    ) -> Circuit:
        """Wrap the coherent preparation in the amplification loop.

        Args:
            preparation: The adjointable Q# operation acting as the preparation.
            num_bits: Number of phase qubits.
            num_system_qubits: Number of system qubits.
            num_ancilla_qubits: Number of block-encoding signal ancillas.
            accepted_indices: Phase-register indices that count as good.
            rounds: Number of amplification rounds (or phase-sequence length for fixed point).

        Returns:
            The circuit to execute.

        """
        amplification = QSHARP_UTILS.AmplitudeAmplification
        signal_ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
        marker = amplification.MakeQpeAcceptanceMarkerOp(num_bits, signal_ancilla_indices, accepted_indices)
        num_qubits = num_bits + num_system_qubits + num_ancilla_qubits
        # The executor reverses the Q# results, so emitting the signal ancillas
        # reversed and ahead of the phase indices makes the key read phase
        # register most-significant-bit first, then the ancillas in order.
        ancilla_indices = list(range(num_bits + num_system_qubits, num_qubits))
        measured_indices = list(reversed(ancilla_indices)) + list(range(num_bits))

        if int(self._settings.get("rounds")) < 0:
            mark_phases, state_phases = self.fixed_point_phases(rounds, float(self._settings.get("tolerance")))
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


class AmplitudeAmplificationFactory(AlgorithmFactory):
    """Factory class for creating AmplitudeAmplification instances."""

    def __init__(self):
        """Initialize the AmplitudeAmplificationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    def default_algorithm_name(self) -> str:
        """Return qdk_amplified_qpe as the default algorithm name."""
        return "qdk_amplified_qpe"
