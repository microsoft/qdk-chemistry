r"""Unary-iteration phase estimation with a number of walk queries."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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


def _post_process_phase_estimation(
    counts: dict[str, int],
    num_bits: int,
    use_positive_sign: bool = False,
) -> tuple[float, str, float]:
    r"""Reduce measured shot counts to a walk phase fraction.

    Every branch phase is doubled relative to the walk phase, so a measured bin
    :math:`y` satisfies :math:`y = \pm 2\varphi \bmod 1`. The conjugate bins :math:`y`
    and :math:`1 - y` therefore describe the same eigenvalue and are summed before the
    winner is chosen. What comes out is an ordinary QPE phase fraction, converted to an
    energy by the unitary representation's ``eigenvalue_from_phase`` exactly as standard
    QPE does, which for a walk operator is :math:`E = \lambda \cos(2\pi\varphi)`.

    Args:
        counts: Measured bitstring counts, most-significant bit first.
        num_bits: Size of the phase register.
        use_positive_sign: ``True`` selects the non-negative eigenvalue branch,
            ``False`` (the default) the non-positive one, as wanted for a ground state.

    Returns:
        A tuple of (decoded phase fraction, representative bitstring, its raw measured fraction).

    """
    decoded_counts: dict[float, int] = {}
    representatives: dict[float, tuple[str, float, int]] = {}
    for bitstring, count in counts.items():
        measured_phase = int(bitstring, 2) / (2**num_bits)
        doubled_phase = measured_phase % 1.0
        folded_phase = min(doubled_phase, (-doubled_phase) % 1.0) / 2.0
        decoded_phase = folded_phase if use_positive_sign else 0.5 - folded_phase
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
            "use_positive_sign",
            "bool",
            False,
            "Whether the doubled measured phase resolves to a non-negative eigenvalue rather than a non-positive one.",
        )
        self.set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary"))


class UnaryPhaseEstimation(PhaseEstimation):
    """Phase estimation using unary iteration over an arbitrary-length query schedule."""

    def __init__(self, shots: int = 100, use_positive_sign: bool = False) -> None:
        """Initialize the unary-iteration phase estimation routine.

        Args:
            shots: The number of shots to execute the circuit.
            use_positive_sign: ``True`` selects the non-negative eigenvalue branch,
                ``False`` (the default) the non-positive one, as wanted for a ground state.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = UnaryPhaseEstimationSettings()
        self._settings.set("shots", shots)
        self._settings.set("use_positive_sign", use_positive_sign)

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

        phase_fraction, dominant_bitstring, measured_phase = _post_process_phase_estimation(
            counts,
            num_bits,
            self._settings.get("use_positive_sign"),
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
