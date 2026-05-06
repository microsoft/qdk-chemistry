"""Standard (QFT-based) phase estimation implementation via qiskit.

This module implements the standard quantum phase estimation algorithm using the qiskit synthesis
of the inverse Quantum Fourier Transform (QFT), which measures all phase bits in parallel using
multiple ancilla qubits.

References:
    Nielsen, M. A., & Chuang, I. L. (2010). :cite:`Nielsen-Chuang2010-QPE`

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.phase_estimation.base import PhaseEstimation, PhaseEstimationSettings
from qdk_chemistry.data import (
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.plugins.qiskit.standard_phase_estimation_builder import QiskitStandardPhaseEstimationBuilder
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitStandardPhaseEstimation", "QiskitStandardPhaseEstimationSettings"]


class QiskitStandardPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Qiskit Standard Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Qiskit Standard Phase Estimation.

        Args:
            qft_do_swaps: Whether to include the final swap layer in the inverse QFT.
            shots: The number of shots to execute the circuit.

        """
        super().__init__()
        self._set_default(
            "qft_do_swaps",
            "bool",
            True,
            "Whether to include the final swap layer in the inverse QFT.",
        )
        self._set_default(
            "shots",
            "int",
            3,
            "The number of shots to execute the circuit.",
        )


class QiskitStandardPhaseEstimation(PhaseEstimation):
    """Standard QFT-based (non-iterative) phase estimation."""

    def __init__(self, num_bits: int = -1, qft_do_swaps: bool = True, shots: int = 3):
        """Initialize the Qiskit standard phase estimation routine.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            qft_do_swaps: Whether to include the final swap layer in the inverse QFT.
                Defaults to ``True`` so that the measured bit string is
                ordered from most-significant to least-significant bit.
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QiskitStandardPhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("qft_do_swaps", qft_do_swaps)
        self._settings.set("shots", shots)

    def _create_builder(self) -> QiskitStandardPhaseEstimationBuilder:
        """Create a StandardPhaseEstimationBuilder with settings propagated from this algorithm.

        Returns:
            A StandardPhaseEstimationBuilder instance configured with matching
            unitary_builder, circuit_mapper, num_bits, and qft_do_swaps settings.

        """
        builder = QiskitStandardPhaseEstimationBuilder(
            num_bits=self.settings().get("num_bits"),
            qft_do_swaps=self.settings().get("qft_do_swaps"),
        )
        builder.settings().update("unitary_builder", self.settings().get("unitary_builder"))
        builder.settings().update("circuit_mapper", self.settings().get("circuit_mapper"))
        return builder

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the standard phase estimation algorithm given the state preparation and qubit Hamiltonian.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the results of the phase estimation.

        """
        Logger.trace_entering()
        circuit_executor = self._create_nested("circuit_executor")
        builder = self._create_builder()
        circuit = builder.build_circuit(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        shots = self._settings.get("shots")
        execution_data = circuit_executor.run(circuit, shots=shots, noise=noise)
        counts = execution_data.bitstring_counts

        dominant_bitstring = max(counts, key=counts.get)
        raw_phase = int(dominant_bitstring, 2) / (2 ** self._settings.get("num_bits"))

        unitary_builder = self._create_nested("unitary_builder")
        if isinstance(unitary_builder, TimeEvolutionBuilder):
            evolution_time = unitary_builder.settings().get("time")
            return QpeResult.from_phase_fraction(
                method=self.name(),
                phase_fraction=raw_phase,
                evolution_time=evolution_time,
                bits_msb_first=dominant_bitstring,
            )
        raise NotImplementedError(
            "QPE result construction currently only supports post-processing from time evolution. "
            f"Got {type(unitary_builder)} instead."
        )

    def name(self) -> str:
        """Return the algorithm name as qiskit_standard."""
        return "qiskit_standard"
