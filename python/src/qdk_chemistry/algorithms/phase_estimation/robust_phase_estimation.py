r"""Robust (randomized) phase estimation driver.

This module implements ``RobustPhaseEstimation``, a phase-estimation algorithm
suited to *randomized* time-evolution channels (qDRIFT / partially randomized),
where the textbook iterative-QPE trick of coherently powering a single unitary
draw is invalid. Instead it samples the Hadamard-test signal
:math:`g(t) = \langle\psi|e^{-iHt}|\psi\rangle` at a geometric ladder of
evolution times, averaging shots into a complex signal per round, and refines
the eigenphase with a robust angle-consistency update.

References:
    Günther, J., Witteveen, F., et al. (2025). Phase estimation with partially
    randomized time evolution. PRX Quantum 7, 020332. arXiv:2503.05647.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.rpe import (
    energy_from_rpe_angle,
    expectation_from_counts,
    num_rounds,
    qdrift_phase_to_energy,
    qdrift_schedule,
    rpe_angle_update,
    wrap_to_principal,
)

from .base import PhaseEstimation

__all__: list[str] = ["RobustPhaseEstimation", "RobustPhaseEstimationSettings"]


class RobustPhaseEstimationSettings(Settings):
    """Settings for the Robust Phase Estimation algorithm."""

    def __init__(self):
        """Initialize settings with nested algorithm references and scalar parameters."""
        super().__init__()
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
            "Time-evolution builder used to realize U(t); sized per round.",
        )
        self._set_default(
            "hadamard_test",
            "algorithm_ref",
            AlgorithmRef("hadamard_test", "qdk"),
            "Hadamard test used to sample the real/imaginary parts of the signal.",
        )
        self._set_default(
            "target_accuracy",
            "double",
            1e-3,
            "Target absolute accuracy epsilon on the energy; sets the round count.",
        )
        self._set_default(
            "base_time",
            "double",
            0.0,
            "Base evolution time tau (round-0 time). 0.0 selects pi/(2*lambda) automatically.",
        )
        self._set_default(
            "unitary_accuracy_fraction",
            "double",
            0.5,
            "Fraction f of the total target_accuracy budget assigned to the unitary builder: "
            "epsilon_unitary = f * target_accuracy and epsilon_rpe = (1 - f) * target_accuracy. "
            "Use a value in [0, 1); 0.5 splits the budget evenly. Ignored for pure qDRIFT, "
            "which auto-sets f = 0 so the whole budget sizes the RPE ladder.",
        )
        self._set_default(
            "energy_correction",
            "string",
            "auto",
            "Phase-to-energy map: 'auto' (qDRIFT -> tangent de-biasing, otherwise linear), "
            "'linear', or 'qdrift_tangent'.",
            ["auto", "linear", "qdrift_tangent"],
        )
        self._set_default(
            "seed",
            "int",
            -1,
            "Random seed for the evolution builder. Use -1 for non-deterministic sampling.",
        )


class RobustPhaseEstimation(PhaseEstimation):
    """Robust phase estimation for randomized time-evolution channels."""

    def __init__(
        self,
        target_accuracy: float = 1e-3,
        base_time: float = 0.0,
        unitary_accuracy_fraction: float = 0.5,
        energy_correction: str = "auto",
        seed: int = -1,
    ):
        """Initialize RobustPhaseEstimation.

        Args:
            target_accuracy: Total target absolute accuracy on the energy. It is
                split into a unitary-builder budget and an RPE budget via
                ``unitary_accuracy_fraction``.
            base_time: Base evolution time ``tau``. ``0.0`` selects ``pi/(2*lambda)``.
            unitary_accuracy_fraction: Fraction ``f`` of ``target_accuracy`` given
                to the unitary builder (``epsilon_unitary = f * target_accuracy``,
                ``epsilon_rpe = (1 - f) * target_accuracy``). Use a value in ``[0, 1)``.
                Ignored for pure qDRIFT, which auto-sets ``f = 0`` (the qDRIFT
                builder is sized by its sample schedule and de-biased by the
                tangent map, so the whole budget sizes the RPE ladder).
            energy_correction: Phase-to-energy map: ``"auto"`` (qDRIFT -> tangent,
                otherwise linear), ``"linear"``, or ``"qdrift_tangent"``.
            seed: Random seed for the evolution builder (``-1`` for non-deterministic).

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = RobustPhaseEstimationSettings()
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("base_time", base_time)
        self._settings.set("unitary_accuracy_fraction", unitary_accuracy_fraction)
        self._settings.set("energy_correction", energy_correction)
        self._settings.set("seed", seed)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run robust phase estimation and return the recovered energy.

        Args:
            state_preparation: Circuit preparing the trial state on the system qubits.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy is estimated.
            noise: Noise model. Not supported in this MVP and ignored if provided.

        Returns:
            QpeResult: Result carrying the resolved energy.

        """
        Logger.trace_entering()
        if noise is not None:
            Logger.warning("RobustPhaseEstimation does not support noise yet; ignoring the noise model.")

        epsilon_total = self._settings.get("target_accuracy")
        seed = self._settings.get("seed")

        category = self._classify_builder()
        correction = self._select_correction(category)

        fraction = min(max(float(self._settings.get("unitary_accuracy_fraction")), 0.0), 1.0)
        if category == "qdrift":
            # Pure qDRIFT ignores the unitary-accuracy budget: its depth comes
            # from the per-round num_samples schedule and its systematic bias is
            # removed by the tangent de-biasing map, not by an accuracy target.
            # Route the whole budget to the RPE ladder instead of stranding a
            # share of it on a builder that will not spend it.
            fraction = 0.0
        epsilon_unitary = fraction * epsilon_total
        epsilon_rpe = (1.0 - fraction) * epsilon_total
        if epsilon_rpe <= 0.0:
            # Degenerate split (fraction == 1): keep the RPE ladder finite.
            epsilon_rpe = epsilon_total

        lambda_norm = float(np.sum(np.abs(np.asarray(qubit_hamiltonian.coefficients, dtype=float))))
        base_time = self._settings.get("base_time")
        if base_time <= 0.0:
            base_time = float(np.pi / (2.0 * lambda_norm)) if lambda_norm > 0.0 else 1.0

        total = num_rounds(lambda_norm, epsilon_rpe)
        Logger.info(
            f"RobustPhaseEstimation: lambda={lambda_norm:.6g}, base_time={base_time:.6g}, "
            f"rounds={total + 1}, builder={category}, correction={correction}, "
            f"eps_rpe={epsilon_rpe:.3g}, eps_unitary={epsilon_unitary:.3g}."
        )

        theta = 0.0
        final_samples = 1
        randomized = category in ("qdrift", "partial_randomized")
        for m in range(total + 1):
            shots, samples = qdrift_schedule(total, m)
            evolution_time = (2**m) * base_time
            final_samples = samples

            if randomized:
                # Faithful estimator: each Hadamard repetition uses a freshly
                # drawn circuit, so the shot-average estimates the expected
                # signal <psi|E_C[U_C]|psi> rather than a single frozen draw.
                real_part, imag_part = self._sample_signal_randomized(
                    state_preparation,
                    qubit_hamiltonian,
                    evolution_time,
                    samples,
                    shots,
                    seed,
                    m,
                    category,
                    epsilon_unitary,
                )
            else:
                # Deterministic builder: the circuit is fixed, so one build
                # measured `shots` times only samples measurement noise.
                unitary = self._build_unitary(
                    qubit_hamiltonian, evolution_time, samples, seed, m, category, epsilon_unitary
                )
                real_part = self._sample_signal(state_preparation, unitary, shots, "X")
                imag_part = self._sample_signal(state_preparation, unitary, shots, "Y")

            measured_phase = float(np.angle(complex(real_part, imag_part)))
            theta = rpe_angle_update(theta, measured_phase, m)
            Logger.debug(f"Round {m}: shots={shots}, samples={samples}, phi={measured_phase:.6f}, theta={theta:.6f}.")

        energy = self._resolve_energy(theta, base_time, total, lambda_norm, final_samples, correction=correction)
        metadata = {
            "lambda": lambda_norm,
            "base_time": base_time,
            "num_rounds": total + 1,
            "target_accuracy": float(epsilon_total),
            "epsilon_rpe": float(epsilon_rpe),
            "epsilon_unitary": float(epsilon_unitary),
            "unitary_accuracy_fraction": float(fraction),
            "unitary_builder": category,
            "energy_correction": correction,
        }
        return QpeResult.from_energy(
            method=self.name(),
            energy=energy,
            evolution_time=base_time,
            metadata=metadata,
        )

    def _classify_builder(self) -> str:
        """Classify the configured unitary builder by its sample-count settings.

        Returns ``"partial_randomized"`` when the builder exposes
        ``num_random_samples`` (the partially randomized family), ``"qdrift"``
        when it exposes ``num_samples`` (pure qDRIFT), and otherwise
        ``"deterministic_or_exact"`` (Trotter / Zassenhaus).
        """
        settings = self._create_nested("unitary_builder").settings()
        if settings.has("num_random_samples"):
            return "partial_randomized"
        if settings.has("num_samples"):
            return "qdrift"
        return "deterministic_or_exact"

    def _select_correction(self, category: str) -> str:
        """Resolve the phase-to-energy correction mode for the builder category.

        ``"auto"`` maps pure qDRIFT to the tangent de-biasing and every other
        builder family (partially randomized, deterministic) to the linear map;
        an explicit ``"linear"`` / ``"qdrift_tangent"`` overrides the inference.
        """
        mode = self._settings.get("energy_correction")
        if mode != "auto":
            return mode
        return "qdrift_tangent" if category == "qdrift" else "linear"

    def _build_unitary(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        evolution_time: float,
        samples: int,
        seed: int,
        round_index: int,
        category: str,
        epsilon_unitary: float,
        *,
        explicit_seed: bool = False,
    ):
        """Create and size the time-evolution unitary for one round.

        qDRIFT keeps the explicit per-round sample schedule (paired with the
        tangent energy map). Accuracy-aware builders (partially randomized or
        deterministic) instead receive the per-round unitary accuracy budget
        ``epsilon_unitary`` and self-size their internal resolution.

        When ``explicit_seed`` is set the caller has already derived a per-draw
        seed and ``seed`` is used verbatim; otherwise it is offset by
        ``round_index`` so successive rounds draw independently.
        """
        builder = self._create_nested("unitary_builder")
        settings = builder.settings()
        settings.set("time", evolution_time)
        if category == "qdrift":
            if settings.has("num_samples"):
                settings.set("num_samples", int(samples))
        elif settings.has("target_accuracy"):
            settings.set("target_accuracy", float(epsilon_unitary))
        if seed >= 0 and settings.has("seed"):
            settings.set("seed", int(seed) if explicit_seed else int(seed) + round_index)
        return builder.run(qubit_hamiltonian)

    def _sample_signal(self, state_preparation: Circuit, unitary, shots: int, test_basis: str) -> float:
        """Run one Hadamard test in the given basis and reduce counts to an expectation."""
        hadamard_test = self._create_nested("hadamard_test")
        hadamard_test.settings().set("test_basis", test_basis)
        executor_data = hadamard_test.run(state_preparation, unitary, shots)
        return expectation_from_counts(executor_data.bitstring_counts)

    def _sample_signal_randomized(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        evolution_time: float,
        samples: int,
        shots: int,
        seed: int,
        round_index: int,
        category: str,
        epsilon_unitary: float,
    ) -> tuple[float, float]:
        r"""Estimate the round signal by averaging over independent fresh draws.

        For randomized builders (qDRIFT / partially randomized) the faithful
        quantity is the expected signal
        :math:`\langle\psi|\mathbb{E}_C[U_C(t)]|\psi\rangle`, not any single
        realization. Each of the ``shots`` Hadamard repetitions therefore uses a
        freshly sampled circuit; the same draw supplies the X (real) and Y
        (imaginary) sample so the complex signal stays consistent, and the
        per-draw expectations are averaged. This is the qDRIFT sampling contract
        of Günther et al. (2025), App. B.2 (``shots`` independent draws of depth
        ``samples``).
        """
        real_accumulator = 0.0
        imag_accumulator = 0.0
        for draw_index in range(shots):
            draw_seed = self._derive_seed(seed, round_index, draw_index) if seed >= 0 else -1
            unitary = self._build_unitary(
                qubit_hamiltonian,
                evolution_time,
                samples,
                draw_seed,
                round_index,
                category,
                epsilon_unitary,
                explicit_seed=True,
            )
            real_accumulator += self._sample_signal(state_preparation, unitary, 1, "X")
            imag_accumulator += self._sample_signal(state_preparation, unitary, 1, "Y")
        inverse_shots = 1.0 / float(shots)
        return real_accumulator * inverse_shots, imag_accumulator * inverse_shots

    @staticmethod
    def _derive_seed(seed: int, round_index: int, draw_index: int) -> int:
        """Derive an independent, reproducible builder seed for one draw.

        Mixes ``(seed, round_index, draw_index)`` through ``SeedSequence`` so that
        every draw in every round is statistically independent yet fully
        reproducible for a fixed top-level ``seed``.
        """
        sequence = np.random.SeedSequence([int(seed), int(round_index), int(draw_index)])
        return int(sequence.generate_state(1)[0])

    @staticmethod
    def _resolve_energy(
        theta: float,
        base_time: float,
        total_rounds: int,
        lambda_norm: float,
        final_samples: int,
        *,
        correction: str,
    ) -> float:
        """Map the recovered per-base-time phase to an energy."""
        if correction != "qdrift_tangent":
            return energy_from_rpe_angle(theta, base_time)
        # Apply the qDRIFT tangent de-biasing at the final (largest-time) round,
        # using the unwrapped phase consistent with the principal per-base phase.
        principal = wrap_to_principal(theta)
        final_time = (2**total_rounds) * base_time
        final_phase = principal * (2**total_rounds)
        return qdrift_phase_to_energy(final_phase, final_time, lambda_norm, final_samples)

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "qdk_robust"
