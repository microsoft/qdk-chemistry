"""Traditional (QFT-based) phase estimation implementation.

This module implements the standard quantum phase estimation algorithm using the
Quantum Fourier Transform (QFT), which measures all phase bits in parallel using
multiple ancilla qubits.

References:
    Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and Quantum
    Information" (10th Anniversary Edition), Ch. 5.2.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.synthesis.qft.qft_decompose_full import synth_qft_full

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.phase_estimation.base import PhaseEstimation, PhaseEstimationAlgorithm
from qdk_chemistry.utils.time_evolution import (
    PauliEvolutionTerm,
    append_controlled_time_evolution,
    extract_terms_from_hamiltonian,
)

_LOGGER = logging.getLogger(__name__)

__all__: list[str] = []


class TraditionalPhaseEstimation(PhaseEstimation):
    """Standard QFT-based (non-iterative) phase estimation."""

    algorithm = PhaseEstimationAlgorithm.TRADITIONAL

    def __init__(self, hamiltonian: QubitHamiltonian, evolution_time: float, *, qft_do_swaps: bool = True):
        """Initialize the traditional phase estimation routine.

        Args:
            hamiltonian: Target Hamiltonian.
            evolution_time: Time parameter ``t`` for ``U = exp(-i H t)``.
            qft_do_swaps: Whether to include the final swap layer in the inverse

                QFT. Defaults to ``True`` so that the measured bit string is
                ordered from most-significant to least-significant bit.

        """
        super().__init__(hamiltonian, evolution_time)
        self._terms: list[PauliEvolutionTerm] = extract_terms_from_hamiltonian(hamiltonian)
        self._qft_do_swaps = qft_do_swaps
        _LOGGER.debug(
            "Initialized %s with %d evolution terms, evolution time %.6f, qft_do_swaps=%s.",
            self.__class__.__name__,
            len(self._terms),
            evolution_time,
            qft_do_swaps,
        )

    def create_circuit(
        self,
        state_prep: QuantumCircuit,
        *,
        num_bits: int,
        measurement_register: ClassicalRegister | None = None,
        include_measurement: bool = True,
    ) -> QuantumCircuit:
        """Build the traditional QPE circuit."""
        if num_bits <= 0:
            raise ValueError("num_bits must be a positive integer.")

        self._validate_state_prep_qubits(state_prep)

        ancilla = QuantumRegister(num_bits, "ancilla")
        system = QuantumRegister(state_prep.num_qubits, "system")

        classical: ClassicalRegister | None = None
        if include_measurement:
            classical = measurement_register or ClassicalRegister(num_bits, "c")
            qc = QuantumCircuit(ancilla, system, classical)
        else:
            qc = QuantumCircuit(ancilla, system)
            if measurement_register is not None:
                qc.add_register(measurement_register)

        _LOGGER.debug(
            "Creating traditional QPE circuit with %d ancilla qubits and measurement=%s.",
            num_bits,
            include_measurement,
        )

        qc.compose(state_prep, qubits=system, inplace=True)
        qc.barrier(label="state_prep")

        for idx in range(num_bits):
            qc.h(ancilla[idx])

        for ancilla_idx in range(num_bits):
            power = 2**ancilla_idx
            append_controlled_time_evolution(
                qc,
                ancilla[ancilla_idx],
                system,
                self._terms,
                time=self.evolution_time,
                power=power,
            )

        inverse_qft = synth_qft_full(num_bits, do_swaps=self._qft_do_swaps, inverse=True)
        qc.barrier(label="controlled-U")
        qc.compose(inverse_qft, qubits=ancilla, inplace=True)

        if include_measurement and classical is not None:
            qc.barrier(label="iqft")
            qc.measure(ancilla, classical)

        _LOGGER.debug(
            "Completed traditional QPE circuit with %d qubits, include_measurement=%s.",
            qc.num_qubits,
            include_measurement,
        )

        return qc
