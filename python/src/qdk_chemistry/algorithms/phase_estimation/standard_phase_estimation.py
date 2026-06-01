"""Standard (QFT-based) phase estimation implementation.

This module implements the standard quantum phase estimation algorithm using the
inverse Quantum Fourier Transform (QFT), which measures all phase bits in parallel using multiple ancilla qubits.

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
from qdk_chemistry.utils import Logger

from .circuit_builder.base import StandardQpeCircuitBuilder

__all__: list[str] = [
    "StandardPhaseEstimation",
    "StandardPhaseEstimationSettings",
]


class StandardPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Standard Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Standard Phase Estimation.

        Args:
            shots: The number of shots to execute the circuit.

        """
        super().__init__()
        self._set_default(
            "shots",
            "int",
            3,
            "The number of shots to execute the circuit.",
        )


class StandardPhaseEstimation(PhaseEstimation):
    """Standard QFT-based (non-iterative) phase estimation."""

    def __init__(self, num_bits: int = -1, shots: int = 3):
        """Initialize the standard phase estimation routine.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = StandardPhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("shots", shots)

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
        builder = self._create_nested("circuit_builder")
        if not isinstance(builder, StandardQpeCircuitBuilder):
            raise TypeError(
                f"Expected qpe_circuit_builder to be an instance of StandardQpeCircuitBuilder, "
                f"but got {type(builder)} instead."
            )
        num_bits = builder.settings().get("num_bits")
        circuits = builder.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        circuit = circuits[0]
        shots = self._settings.get("shots")
        execution_data = circuit_executor.run(circuit, shots=shots, noise=noise)
        counts = execution_data.bitstring_counts

        dominant_bitstring = max(counts, key=counts.get)
        raw_phase = int(dominant_bitstring, 2) / (2**num_bits)

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
        """Return the algorithm name as standard."""
        return "standard"
