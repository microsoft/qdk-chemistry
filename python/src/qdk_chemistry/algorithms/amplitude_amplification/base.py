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
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    coherent_qpe_measured_indices,
    split_coherent_qpe_bitstring,
)
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
    "ROUND_POLICIES",
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
]

ROUND_POLICIES: list[str] = ["fixed", "safe", "robust", "optimal", "fixed_point"]
"""Supported strategies for choosing the number of amplification rounds."""


class AmplitudeAmplificationSettings(Settings):
    r"""Settings for amplitude amplification.

    The round-count settings are the answer to the central practical question:
    how many rounds can be run when the overlap of the guiding state is only
    known approximately?  After :math:`k` rounds the acceptance probability is
    :math:`\sin^2((2k+1)\theta)` with :math:`\theta = \arcsin\sqrt{a}`, so
    running too many rounds *rotates past* the maximum and destroys acceptance.
    Overshoot is therefore controlled by an **upper** bound on the overlap; a
    lower bound is the dangerous input.

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
            "Explicit number of amplification rounds. Negative means derive it from round_policy.",
        )
        self._set_default(
            "round_policy",
            "string",
            "safe",
            "How to derive the round count when rounds is negative.",
            ROUND_POLICIES,
        )
        self._set_default(
            "min_overlap",
            "double",
            0.0,
            "Lower bound on the probability that the prepared state lands in the good subspace.",
        )
        self._set_default(
            "max_overlap",
            "double",
            1.0,
            "Upper bound on that probability. This is what prevents overshoot.",
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

    The circuit supplied through ``reflect_to_good_space`` is run in ``coherent``
    mode to obtain the measurement-free, adjointable preparation :math:`U_\psi`
    that the amplification loop reflects about.  The good subspace is the set of
    branches whose phase register falls inside the accepted window *and* whose
    block-encoding signal ancillas are all zero.

    Example:
        >>> from qdk_chemistry.algorithms import create  # doctest: +SKIP
        >>> from qdk_chemistry.data import AlgorithmRef  # doctest: +SKIP
        >>> aa = create("amplitude_amplification")  # doctest: +SKIP
        >>> aa.settings().update(
        ...     "reflect_to_good_space",
        ...     AlgorithmRef("qpe_circuit_builder", "qdk_standard", num_bits=8),
        ... )  # doctest: +SKIP
        >>> aa.settings().update("max_overlap", 0.05)  # doctest: +SKIP
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
            A QpeResult built from the dominant *accepted* bitstring, with the
            acceptance statistics in :attr:`~qdk_chemistry.data.QpeResult.metadata`.

        Raises:
            RuntimeError: If no shot was accepted.

        """
        Logger.trace_entering()
        circuit_builder = self._create_nested("reflect_to_good_space")
        circuit_builder.settings().update("coherent", True)
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

    def resolve_rounds(self) -> int:
        """Resolve the number of amplification rounds from the settings.

        An explicit non-negative ``rounds`` always wins. Otherwise the count is
        derived from ``round_policy``:

        * ``safe`` -- the largest count that cannot overshoot for any overlap up
          to ``max_overlap``. Acceptance is then monotonically increasing in the
          true overlap.
        * ``robust`` -- the count maximizing the worst case over
          ``[min_overlap, max_overlap]``.
        * ``optimal`` -- the count that maximizes acceptance at ``min_overlap``.
          This is the textbook choice and the one that overshoots when the true
          overlap turns out to be larger than assumed.
        * ``fixed_point`` -- the count needed for the Yoder-Low-Chuang phase
          sequence to reach ``1 - tolerance^2`` for every overlap at or above
          ``min_overlap``.
        * ``fixed`` -- requires an explicit ``rounds``.

        Returns:
            The number of amplification rounds to run.

        Raises:
            ValueError: If the policy is ``fixed`` but no explicit round count was given.

        """
        rounds = int(self._settings.get("rounds"))
        if rounds >= 0:
            return rounds

        policy = str(self._settings.get("round_policy"))
        if policy == "fixed":
            raise ValueError("round_policy is 'fixed' but no non-negative 'rounds' setting was provided.")

        min_overlap = float(self._settings.get("min_overlap"))
        max_overlap = float(self._settings.get("max_overlap"))
        if policy == "safe":
            return self.safe_rounds(max_overlap)
        if policy == "robust":
            return self.robust_rounds(min_overlap, max_overlap)
        if policy == "optimal":
            return self.optimal_rounds(min_overlap)
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

    @classmethod
    def _validate_interval(cls, min_overlap: float, max_overlap: float) -> None:
        """Raise if ``[min_overlap, max_overlap]`` is not a usable overlap interval.

        Args:
            min_overlap: Lower bound on the squared overlap.
            max_overlap: Upper bound on the squared overlap.

        Raises:
            ValueError: If either bound is invalid or the interval is empty.

        """
        cls._validate_overlap(min_overlap, "min_overlap")
        cls._validate_overlap(max_overlap, "max_overlap")
        if min_overlap > max_overlap:
            raise ValueError(f"min_overlap must not exceed max_overlap. Got {min_overlap} > {max_overlap}.")

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
    def optimal_rounds(cls, overlap: float) -> int:
        r"""Return the round count closest to the first maximum, for a *known* overlap.

        This is :math:`\mathrm{round}\big(\pi/(4\vartheta) - 1/2\big)`, the integer
        nearest the continuous optimum :math:`k^* = \pi/(4\vartheta) - 1/2` where
        :math:`(2k+1)\vartheta = \pi/2`.

        Using this with a merely *estimated* overlap is the classic way to
        overshoot: if the true overlap is larger than assumed the rotation runs
        past the maximum. Prefer :meth:`safe_rounds` or :meth:`robust_rounds` when
        the overlap is uncertain.

        Args:
            overlap: The squared overlap of the prepared state with the good subspace.

        Returns:
            The nonnegative round count nearest the first acceptance maximum.

        """
        angle = cls._rotation_angle(overlap)
        return max(0, round(math.pi / (4.0 * angle) - 0.5))

    @classmethod
    def safe_rounds(cls, max_overlap: float) -> int:
        r"""Return the largest round count that cannot overshoot.

        Returns :math:`\lfloor \pi/(4\vartheta_{\max}) - 1/2 \rfloor`, where
        :math:`\vartheta_{\max} = \arcsin\sqrt{a_{\max}}`. For this ``k`` every
        overlap ``a <= max_overlap`` satisfies :math:`(2k+1)\vartheta \le \pi/2`,
        so the acceptance probability is a monotonically increasing function of the
        true overlap: being luckier than expected can only help.

        Only the *upper* bound matters here. A lower bound tells you how well the
        schedule will do, not whether it is safe.

        Args:
            max_overlap: Upper bound on the squared overlap.

        Returns:
            The largest nonnegative round count with no overshoot risk.

        """
        cls._validate_overlap(max_overlap, "max_overlap")
        angle = cls._rotation_angle(max_overlap)
        # Floor with a small tolerance so that an overlap sitting exactly on a
        # boundary (a_max = 1/4, where pi/(4*theta) - 1/2 evaluates to
        # 0.9999999999999998) is not pushed down a round by floating-point error.
        return max(0, math.floor(math.pi / (4.0 * angle) - 0.5 + 1e-12))

    @classmethod
    def _worst_case_success_probability(cls, rounds: int, min_overlap: float, max_overlap: float) -> float:
        r"""Return the guaranteed acceptance probability over an overlap interval.

        The acceptance probability :math:`\sin^2((2k+1)\vartheta)` has interior
        minima only where :math:`(2k+1)\vartheta` is an integer multiple of
        :math:`\pi`. The worst case over the interval is therefore exactly zero
        when the rotation sweeps through such a multiple, and otherwise attained at
        one of the two endpoints. No search is needed.

        Args:
            rounds: The number of amplification rounds ``k``.
            min_overlap: Lower bound on the squared overlap.
            max_overlap: Upper bound on the squared overlap.

        Returns:
            The smallest acceptance probability consistent with the interval.

        """
        cls._validate_schedule_rounds(rounds)
        cls._validate_interval(min_overlap, max_overlap)

        factor = 2 * rounds + 1
        low = factor * cls._rotation_angle(min_overlap)
        high = factor * cls._rotation_angle(max_overlap)

        # A multiple of pi inside the swept arc drives acceptance to zero.
        if math.floor(high / math.pi) > math.floor(low / math.pi) or low % math.pi == 0.0:
            return 0.0
        return min(math.sin(low) ** 2, math.sin(high) ** 2)

    @classmethod
    def robust_rounds(cls, min_overlap: float, max_overlap: float) -> int:
        """Return the round count maximizing the guaranteed acceptance probability.

        This is the minimax choice over the overlap interval. It is never worse
        than :meth:`safe_rounds` and is usually equal to it; it can be larger when
        the interval is narrow enough that a mild, bounded overshoot at the top of
        the interval buys more at the bottom than it costs at the top.

        Ties are broken toward the smallest round count, since rounds are queries.

        Args:
            min_overlap: Lower bound on the squared overlap.
            max_overlap: Upper bound on the squared overlap.

        Returns:
            The nonnegative round count with the best worst-case acceptance
            probability.

        """
        cls._validate_interval(min_overlap, max_overlap)

        # Beyond the optimum for the smallest admissible overlap, every candidate
        # has already swept past the first maximum for the whole interval.
        upper = cls.optimal_rounds(min_overlap)
        best_rounds = 0
        best_probability = cls._worst_case_success_probability(0, min_overlap, max_overlap)
        for candidate in range(1, upper + 1):
            probability = cls._worst_case_success_probability(candidate, min_overlap, max_overlap)
            if probability > best_probability:
                best_rounds = candidate
                best_probability = probability
        return best_rounds

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

        The nested algorithm is asked for a coherent circuit and must answer with
        one that carries an adjointable Q# operation; reflecting about
        :math:`U_\psi` requires applying its adjoint.

        Args:
            circuit_builder: The nested algorithm from ``reflect_to_good_space``.
            state_preparation: The circuit that prepares the initial (guiding) state.
            qubit_hamiltonian: The qubit Hamiltonian defining the unitary.

        Returns:
            The adjointable Q# operation implementing the preparation.

        Raises:
            TypeError: If the nested algorithm did not honor ``coherent`` mode.

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
                "nested algorithm must honor the 'coherent' setting and return a measurement-free, "
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
        measured_indices = coherent_qpe_measured_indices(num_bits, num_system_qubits, num_ancilla_qubits)

        if str(self._settings.get("round_policy")) == "fixed_point" and int(self._settings.get("rounds")) < 0:
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
            # One attempt costs 2k+1 coherent preparations of the nested circuit.
            "preparations_per_shot": 2 * rounds + 1,
        }
        if 0.0 < min_overlap <= 1.0:
            metadata["predicted_acceptance_at_min_overlap"] = self.success_probability(min_overlap, rounds)
        if 0.0 < max_overlap <= 1.0:
            metadata["predicted_acceptance_at_max_overlap"] = self.success_probability(max_overlap, rounds)
        return metadata


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
