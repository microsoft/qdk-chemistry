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
            "randomized",
            "bool",
            True,
            "If True, apply the qDRIFT tangent de-biasing to the recovered phase.",
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
        randomized: bool = True,
        seed: int = -1,
    ):
        """Initialize RobustPhaseEstimation.

        Args:
            target_accuracy: Target absolute accuracy on the energy.
            base_time: Base evolution time ``tau``. ``0.0`` selects ``pi/(2*lambda)``.
            randomized: Whether to apply the qDRIFT tangent de-biasing.
            seed: Random seed for the evolution builder (``-1`` for non-deterministic).

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = RobustPhaseEstimationSettings()
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("base_time", base_time)
        self._settings.set("randomized", randomized)
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

        epsilon = self._settings.get("target_accuracy")
        randomized = self._settings.get("randomized")
        seed = self._settings.get("seed")

        lambda_norm = float(np.sum(np.abs(np.asarray(qubit_hamiltonian.coefficients, dtype=float))))
        base_time = self._settings.get("base_time")
        if base_time <= 0.0:
            base_time = float(np.pi / (2.0 * lambda_norm)) if lambda_norm > 0.0 else 1.0

        total = num_rounds(lambda_norm, epsilon)
        Logger.info(f"RobustPhaseEstimation: lambda={lambda_norm:.6g}, base_time={base_time:.6g}, rounds={total + 1}.")

        theta = 0.0
        final_samples = 1
        for m in range(total + 1):
            shots, samples = qdrift_schedule(total, m)
            evolution_time = (2**m) * base_time
            final_samples = samples

            unitary = self._build_unitary(qubit_hamiltonian, evolution_time, samples, seed, m)
            real_part = self._sample_signal(state_preparation, unitary, shots, "X")
            imag_part = self._sample_signal(state_preparation, unitary, shots, "Y")

            measured_phase = float(np.angle(complex(real_part, imag_part)))
            theta = rpe_angle_update(theta, measured_phase, m)
            Logger.debug(f"Round {m}: shots={shots}, samples={samples}, phi={measured_phase:.6f}, theta={theta:.6f}.")

        energy = self._resolve_energy(theta, base_time, total, lambda_norm, final_samples, randomized=randomized)
        metadata = {
            "lambda": lambda_norm,
            "base_time": base_time,
            "num_rounds": total + 1,
            "randomized": bool(randomized),
        }
        return QpeResult.from_energy(
            method=self.name(),
            energy=energy,
            evolution_time=base_time,
            metadata=metadata,
        )

    def _build_unitary(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        evolution_time: float,
        samples: int,
        seed: int,
        round_index: int,
    ):
        """Create and size the time-evolution unitary for one round."""
        builder = self._create_nested("unitary_builder")
        builder.settings().set("time", evolution_time)
        if builder.settings().has("num_samples"):
            builder.settings().set("num_samples", int(samples))
        if seed >= 0 and builder.settings().has("seed"):
            # Offset per round so successive rounds draw independent samples.
            builder.settings().set("seed", int(seed) + round_index)
        return builder.run(qubit_hamiltonian)

    def _sample_signal(self, state_preparation: Circuit, unitary, shots: int, test_basis: str) -> float:
        """Run one Hadamard test in the given basis and reduce counts to an expectation."""
        hadamard_test = self._create_nested("hadamard_test")
        hadamard_test.settings().set("test_basis", test_basis)
        executor_data = hadamard_test.run(state_preparation, unitary, shots)
        return expectation_from_counts(executor_data.bitstring_counts)

    @staticmethod
    def _resolve_energy(
        theta: float,
        base_time: float,
        total_rounds: int,
        lambda_norm: float,
        final_samples: int,
        *,
        randomized: bool,
    ) -> float:
        """Map the recovered per-base-time phase to an energy."""
        if not randomized:
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
