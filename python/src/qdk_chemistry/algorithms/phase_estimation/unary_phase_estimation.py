r"""Unary-iteration phase estimation with an arbitrary number of walk queries.

This module implements phase estimation whose query schedule is driven by
unary iteration over the phase register rather than by controlled powers of the walk
operator. A single chain of ``num_queries`` self-inverse blocks is applied, and the
branch that omits reflection slot :math:`t` realizes :math:`W^{p-2t}`, so the total
query count need not be a power of two.

Because every branch phase is doubled relative to the walk phase, the measured
fraction :math:`y` satisfies :math:`y = \pm 2\varphi \bmod 1`. The conjugate bins
are merged before the winner is chosen. Doubling also makes the histogram exactly
invariant under :math:`E \to -E`, so the sign cannot be recovered from the counts;
the ``phase_band`` setting supplies it.

References:
    * :cite:`Berry2024`, Appendix D.
    * :cite:`Lee2021` — tensor hypercontraction; prescription for a
      non-power-of-two number of queries.
    * :cite:`Babbush2018` — Heisenberg-limited phase estimation with a
      sine-window control state, where each block applies :math:`W` or
      :math:`W^\dagger` and hence doubles the phase.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Callable

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
)
from qdk_chemistry.utils import Logger

from .base import PhaseEstimation, PhaseEstimationSettings
from .circuit_builder.unary_phase_estimation_builder import QdkUnaryQpeCircuitBuilder, num_phase_bits

__all__: list[str] = [
    "UnaryPhaseEstimation",
    "UnaryPhaseEstimationSettings",
]

PHASE_BANDS: tuple[str, ...] = ("lower", "upper")


def _select_dominant_decoded_phase(
    counts: dict[str, int],
    num_bits: int,
    decoder: Callable[[float], float],
) -> tuple[float, str, float]:
    """Aggregate raw bitstrings by decoded phase before selecting the winner.

    Args:
        counts: Measured bitstring counts, most-significant bit first.
        num_bits: Size of the phase register.
        decoder: Maps a measured fraction to the walk phase fraction.

    Returns:
        A tuple of (decoded phase fraction, representative bitstring, its raw measured fraction).

    """
    decoded_counts: dict[float, int] = {}
    representatives: dict[float, tuple[str, float, int]] = {}
    for bitstring, count in counts.items():
        measured_phase = int(bitstring, 2) / (2**num_bits)
        decoded_phase = decoder(measured_phase)
        decoded_counts[decoded_phase] = decoded_counts.get(decoded_phase, 0) + count
        representative = representatives.get(decoded_phase)
        if representative is None or count > representative[2]:
            representatives[decoded_phase] = (bitstring, measured_phase, count)

    phase_fraction = max(decoded_counts, key=decoded_counts.__getitem__)
    dominant_bitstring, measured_phase, _ = representatives[phase_fraction]
    return phase_fraction, dominant_bitstring, measured_phase


class UnaryPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the unary-iteration phase estimation algorithm."""

    def __init__(self) -> None:
        """Initialize the settings for unary-iteration phase estimation."""
        super().__init__()
        self._set_default(
            "shots",
            "int",
            100,
            "The number of shots to execute the circuit.",
        )
        self._set_default(
            "phase_band",
            "string",
            "upper",
            "Half-band used to resolve the doubled measured phase; picks the eigenvalue sign.",
            limit=list(PHASE_BANDS),
        )
        # The inherited key already exists, and set_default is a no-op for those.
        self.set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary"))


class UnaryPhaseEstimation(PhaseEstimation):
    """Phase estimation using unary iteration over an arbitrary-length query schedule."""

    def __init__(self, shots: int = 100, phase_band: str = "upper") -> None:
        """Initialize the unary-iteration phase estimation routine.

        Args:
            shots: The number of shots to execute the circuit.
            phase_band: ``"lower"`` for a non-negative eigenvalue, ``"upper"`` for a non-positive one.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = UnaryPhaseEstimationSettings()
        self._settings.set("shots", shots)
        self._settings.set("phase_band", phase_band)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run unary-iteration phase estimation for the given state preparation and Hamiltonian.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the results of the phase estimation.

        Raises:
            TypeError: If the configured circuit builder is not a unary-iteration builder.

        """
        Logger.trace_entering()
        circuit_executor = self._create_nested("circuit_executor")
        circuit_builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(circuit_builder, QdkUnaryQpeCircuitBuilder):
            raise TypeError(
                f"Expected qpe_circuit_builder to be an instance of QdkUnaryQpeCircuitBuilder, "
                f"but got {type(circuit_builder)} instead."
            )

        # Resolve container before running the circuit
        unitary_builder = circuit_builder._create_nested("unitary_builder")  # noqa: SLF001
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()

        num_bits = num_phase_bits(circuit_builder.resolve_num_queries(unitary_rep))
        circuits = circuit_builder.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        execution_data = circuit_executor.run(circuits[0], shots=self._settings.get("shots"), noise=noise)
        counts = execution_data.bitstring_counts

        phase_band = self._settings.get("phase_band")
        phase_fraction, dominant_bitstring, measured_phase = _select_dominant_decoded_phase(
            counts,
            num_bits,
            lambda measured: circuit_builder.phase_fraction_from_measurement(measured, phase_band),
        )

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            eigenvalue_from_phase=container.eigenvalue_from_phase,
            bits_msb_first=dominant_bitstring,
            metadata={"measured_phase_fraction": measured_phase},
        )

    def name(self) -> str:
        """Return the algorithm name as qdk_unary."""
        return "qdk_unary"
