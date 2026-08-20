r"""Unary-iteration phase estimation with a number of walk queries."""

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
from .circuit_builder.unary_phase_estimation_builder import QdkUnaryQpeCircuitBuilder

__all__: list[str] = [
    "UnaryPhaseEstimation",
    "UnaryPhaseEstimationSettings",
]

# Unary iteration threads a callable through a recursive Q# operation, which cannot be
# defunctionalized, so the circuit never lowers to QIR. Only the sparse-state simulator
# interprets Q# directly and can run it.
_SUPPORTED_CIRCUIT_EXECUTOR = "qdk_sparse_state_simulator"


def _post_process_phase_estimation(
    counts: dict[str, int],
    num_bits: int,
    method: str,
    resolve_positive_branch: bool,
    eigenvalue_from_phase: Callable[[float], float],
) -> QpeResult:
    r"""Process the measured results from unary-iteration phase estimation into a QpeResult.

    Every branch phase is doubled relative to the walk phase, so a measured bin
    :math:`y` satisfies :math:`y = \pm 2\varphi \bmod 1`, where :math:`\varphi` is the walk
    phase fraction of :math:`E = \lambda \cos(2\pi\varphi)`. Since :math:`\varphi` and
    :math:`1/2 - \varphi` are observationally identical and map to :math:`E` and :math:`-E`,
    the two eigenvalue signs cannot be distinguished, and ``resolve_positive_branch`` supplies the
    missing information:

    * ``False`` (the default) returns :math:`\varphi \in [1/4, 1/2]`, hence
      :math:`\cos(2\pi\varphi) \le 0` and :math:`E \le 0` for every input.
    * ``True`` returns :math:`\varphi \in [0, 1/4]`, hence :math:`E \ge 0`.

    Args:
        counts: Measured bitstring counts, most-significant bit first.
        num_bits: Size of the phase register.
        method: Phase estimation algorithm label recorded on the result.
        resolve_positive_branch: ``True`` selects the non-negative eigenvalue branch,
            ``False`` the non-positive one, as wanted for a ground state.
        eigenvalue_from_phase: A callable mapping a walk phase fraction to a Hamiltonian eigenvalue.

    Returns:
        A :class:`~qdk_chemistry.data.QpeResult` whose ``phase_fraction`` is the measured bin,
        ``canonical_phase_fraction`` is the decoded walk phase, and ``branching``
        holds both sign candidates.

    """
    num_bins = 2**num_bits
    canonical_counts: dict[float, int] = {}
    for bitstring, count in counts.items():
        measured = int(bitstring, 2) / num_bins
        folded = min(measured, (-measured) % 1.0) / 2.0
        canonical = folded if resolve_positive_branch else 0.5 - folded
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + count

    # Ties are broken toward the smaller phase fraction so that equal counts decode
    # to the same phase regardless of the order the shots arrive in.
    canonical_phase_fraction = max(canonical_counts, key=lambda phase: (canonical_counts[phase], -phase))
    raw_energy = eigenvalue_from_phase(canonical_phase_fraction)
    mirror_energy = eigenvalue_from_phase(0.5 - canonical_phase_fraction)

    phase_fraction = 2.0 * min(canonical_phase_fraction, 0.5 - canonical_phase_fraction)
    bitstring_msb_first = format(round(phase_fraction * num_bins), f"0{num_bits}b")

    return QpeResult.from_phase_fraction(
        method=method,
        phase_fraction=phase_fraction,
        eigenvalue_from_phase=eigenvalue_from_phase,
        canonical_phase_fraction=canonical_phase_fraction,
        branching=tuple(sorted((raw_energy, mirror_energy))),
        resolved_energy=raw_energy,
        bits_msb_first=tuple(int(bit) for bit in bitstring_msb_first),
        bitstring_msb_first=bitstring_msb_first,
    )


class UnaryPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the unary-iteration phase estimation algorithm."""

    def __init__(self) -> None:
        """Initialize the settings for unary-iteration phase estimation."""
        super().__init__()
        self._set_default(
            "shots",
            "int",
            3,
            "The number of shots to execute the circuit.",
        )
        self._set_default(
            "resolve_positive_branch",
            "bool",
            False,
            "Whether the doubled measured phase resolves to a positive eigenvalue rather than a negative one.",
        )
        self.set("qpe_circuit_builder", AlgorithmRef("qpe_circuit_builder", "qdk_unary"))


class UnaryPhaseEstimation(PhaseEstimation):
    """Phase estimation using unary iteration over an arbitrary-length query schedule."""

    def __init__(self, shots: int = 3, resolve_positive_branch: bool = False) -> None:
        """Initialize the unary-iteration phase estimation routine.

        Args:
            shots: The number of shots to execute the circuit.
            resolve_positive_branch: ``True`` selects the non-negative eigenvalue branch,
                ``False`` (the default) the non-positive one, as wanted for a ground state.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = UnaryPhaseEstimationSettings()
        self._settings.set("shots", shots)
        self._settings.set("resolve_positive_branch", resolve_positive_branch)

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
            TypeError: If the configured circuit builder is not a unary-iteration builder,
                or if the configured circuit executor is not the sparse-state simulator.

        """
        Logger.trace_entering()
        circuit_executor = self._create_nested("circuit_executor")
        if circuit_executor.name() != _SUPPORTED_CIRCUIT_EXECUTOR:
            raise TypeError(
                f"Unary-iteration phase estimation only supports the '{_SUPPORTED_CIRCUIT_EXECUTOR}' "
                f"circuit executor, but got '{circuit_executor.name()}' instead."
            )
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

        _, num_bits = circuit_builder.resolve_num_queries()
        circuits = circuit_builder.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        execution_data = circuit_executor.run(circuits[0], shots=self._settings.get("shots"), noise=noise)
        counts = execution_data.bitstring_counts

        return _post_process_phase_estimation(
            counts,
            num_bits,
            method=self.name(),
            resolve_positive_branch=self._settings.get("resolve_positive_branch"),
            eigenvalue_from_phase=container.eigenvalue_from_phase,
        )

    def name(self) -> str:
        """Return the algorithm name as qdk_unary."""
        return "qdk_unary"
