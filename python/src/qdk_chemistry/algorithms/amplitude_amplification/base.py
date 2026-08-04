r"""QDK/Chemistry amplitude amplification abstractions.

Amplitude amplification boosts the probability that a prepared state is found in
a *good* subspace.  The two halves are deliberately independent:

* a **preparation** :math:`U_\\psi`, which defines the state being amplified, and
* a **marking oracle**, which defines what counts as good.

Swapping the preparation for a coherent phase-estimation circuit and the marker
for an energy-window test turns the same machinery into amplified phase
estimation, which is what
:class:`~qdk_chemistry.algorithms.amplitude_amplification.amplified_phase_estimation.AmplifiedPhaseEstimation`
does.

References:
    L. Lin, *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.amplitude_amplification import schedule
from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
    Settings,
)

__all__: list[str] = [
    "ROUND_POLICIES",
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
]

ROUND_POLICIES: list[str] = ["fixed", "safe", "robust", "optimal", "fixed_point"]
"""Supported strategies for choosing the number of amplification rounds."""


class AmplitudeAmplificationSettings(Settings):
    r"""Settings shared by all amplitude amplification algorithms.

    The round-count settings are the answer to the central practical question:
    how many rounds can be run when the overlap of the guiding state is only
    known approximately?  After :math:`k` rounds the acceptance probability is
    :math:`\\sin^2((2k+1)\\theta)` with :math:`\\theta = \\arcsin\\sqrt{a}`, so
    running too many rounds *rotates past* the maximum and destroys acceptance.
    Overshoot is therefore controlled by an **upper** bound on the overlap; a
    lower bound is the dangerous input.

    """

    def __init__(self):
        """Initialize the settings for amplitude amplification."""
        super().__init__()
        self._set_default(
            "qpe_circuit_builder",
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
    """Abstract base class for amplitude amplification algorithms."""

    def __init__(self):
        """Initialize the AmplitudeAmplification with default settings."""
        super().__init__()
        self._settings = AmplitudeAmplificationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    @abstractmethod
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
            A QpeResult whose metadata carries the acceptance statistics.

        """

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
            return schedule.safe_rounds(max_overlap)
        if policy == "robust":
            return schedule.robust_rounds(min_overlap, max_overlap)
        if policy == "optimal":
            return schedule.optimal_rounds(min_overlap)
        return schedule.fixed_point_rounds(min_overlap, float(self._settings.get("tolerance")))


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
