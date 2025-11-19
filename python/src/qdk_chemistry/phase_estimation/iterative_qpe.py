"""Iterative phase estimation implementation.

This module implements the Kitaev-style iterative quantum phase estimation (IQPE)
algorithm, which measures phase bits sequentially from most-significant to least-significant
using a single ancilla qubit and adaptive feedback corrections.

References:
    Kitaev, A. (1995). "Quantum measurements and the Abelian Stabilizer Problem."
    arXiv:quant-ph/9511026. https://arxiv.org/abs/quant-ph/9511026

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.phase_estimation.base import PhaseEstimation, PhaseEstimationAlgorithm
from qdk_chemistry.utils.phase import (
    accumulated_phase_from_bits,
    iterative_phase_feedback_update,
    phase_fraction_from_feedback,
)
from qdk_chemistry.utils.time_evolution import (
    PauliEvolutionTerm,
    append_controlled_time_evolution,
    extract_terms_from_hamiltonian,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IterativePhaseEstimationIteration:
    """Container describing a single iteration circuit."""

    circuit: QuantumCircuit
    iteration: int
    total_iterations: int
    power: int
    phase_correction: float


class IterativePhaseEstimation(PhaseEstimation):
    """Most-significant-bit-first iterative phase estimation (Kitaev style)."""

    algorithm = PhaseEstimationAlgorithm.ITERATIVE

    def __init__(self, hamiltonian: QubitHamiltonian, evolution_time: float):
        """Configure iterative QPE for ``hamiltonian`` evolved for ``evolution_time``.

        Args:
            hamiltonian: The target Hamiltonian whose eigenvalues will be estimated.
            evolution_time: Time parameter ``t`` used in the time-evolution unitary
                ``U = exp(-i H t)``.

        """
        super().__init__(hamiltonian, evolution_time)
        self._terms: list[PauliEvolutionTerm] = extract_terms_from_hamiltonian(hamiltonian)
        _LOGGER.debug(
            "Initialized %s with %d evolution terms and evolution time %.6f.",
            self.__class__.__name__,
            len(self._terms),
            evolution_time,
        )

    def create_iteration_circuit(
        self,
        state_prep: QuantumCircuit,
        *,
        iteration: int,
        total_iterations: int,
        phase_correction: float = 0.0,
        measurement_register: ClassicalRegister | None = None,
        iteration_name: str | None = None,
    ) -> QuantumCircuit:
        """Construct a single IQPE iteration circuit.

        Args:
            state_prep: Trial-state preparation circuit that prepares the initial
                quantum state on the system qubits.
            iteration: Current iteration index (0-based), where 0 corresponds to the
                most-significant bit.
            total_iterations: Total number of phase bits to measure across all iterations.
            phase_correction: Feedback phase angle to apply before controlled evolution.
                Defaults to 0.0 for the first iteration.
            measurement_register: Optional classical register for storing the measurement
                result. If None, a new register named ``c{iteration}`` is created.
            iteration_name: Optional custom name for the circuit. If None, no specific
                name is assigned.

        Returns:
            A quantum circuit implementing one IQPE iteration with an ancilla qubit,
            system qubits, and classical measurement register.

        """
        _validate_iteration_inputs(iteration, total_iterations)
        self._validate_state_prep_qubits(state_prep)

        ancilla = QuantumRegister(1, "ancilla")
        system = QuantumRegister(state_prep.num_qubits, "system")
        classical = measurement_register or ClassicalRegister(1, f"c{iteration}")
        circuit = QuantumCircuit(ancilla, system, classical, name=iteration_name)

        circuit.compose(state_prep, qubits=system, inplace=True)
        circuit.barrier(label="state_prep")

        control = ancilla[0]
        system_qubits = list(system)

        _LOGGER.debug(
            "Creating IQPE iteration %d/%d with phase correction %.6f.",
            iteration + 1,
            total_iterations,
            phase_correction,
        )

        circuit.h(control)
        if phase_correction:
            circuit.rz(phase_correction, control)

        power = 2 ** (total_iterations - iteration - 1)
        append_controlled_time_evolution(
            circuit,
            control,
            system_qubits,
            self._terms,
            time=self.evolution_time,
            power=power,
        )

        circuit.h(control)
        circuit.measure(control, classical[0])

        _LOGGER.debug(
            "Completed IQPE iteration %d/%d producing circuit with %d qubits and %d classical bits.",
            iteration + 1,
            total_iterations,
            circuit.num_qubits,
            circuit.num_clbits,
        )

        return circuit

    def create_iteration(
        self,
        state_prep: QuantumCircuit,
        *,
        iteration: int,
        total_iterations: int,
        phase_correction: float = 0.0,
        measurement_register: ClassicalRegister | None = None,
        iteration_name: str | None = None,
    ) -> IterativePhaseEstimationIteration:
        """Build an iteration and return contextual metadata.

        Args:
            state_prep: Trial-state preparation circuit that prepares the initial
                quantum state on the system qubits.
            iteration: Current iteration index (0-based), where 0 corresponds to the
                most-significant bit.
            total_iterations: Total number of phase bits to measure across all iterations.
            phase_correction: Feedback phase angle to apply before controlled evolution.
                Defaults to 0.0 for the first iteration.
            measurement_register: Optional classical register for storing the measurement
                result. If None, a new register named ``c{iteration}`` is created.
            iteration_name: Optional custom name for the circuit. If None, no specific
                name is assigned.

        Returns:
            An IterativePhaseEstimationIteration object containing the circuit and
            metadata including iteration number, total iterations, power of evolution,
            and phase correction.

        """
        circuit = self.create_iteration_circuit(
            state_prep,
            iteration=iteration,
            total_iterations=total_iterations,
            phase_correction=phase_correction,
            measurement_register=measurement_register,
            iteration_name=iteration_name,
        )

        power = 2 ** (total_iterations - iteration - 1)
        return IterativePhaseEstimationIteration(
            circuit=circuit,
            iteration=iteration,
            total_iterations=total_iterations,
            power=power,
            phase_correction=phase_correction,
        )

    def create_iterations(
        self,
        state_prep: QuantumCircuit,
        *,
        num_bits: int,
        phase_corrections: Sequence[float] | None = None,
        measurement_registers: Sequence[ClassicalRegister] | None = None,
        iteration_names: Sequence[str | None] | None = None,
    ) -> list[IterativePhaseEstimationIteration]:
        """Generate ``num_bits`` iteration circuits with optional phase feedback.

        Args:
            state_prep: Trial-state preparation circuit that prepares the initial
                quantum state on the system qubits.
            num_bits: Total number of IQPE iterations to generate. Must be a positive
                integer.
            phase_corrections: Optional list of feedback phases Φ(k+1) to apply
                before each iteration. Must have length equal to ``num_bits`` if provided.
                Defaults to zeros if not specified.
            measurement_registers: Optional collection of classical registers,
                one per iteration. Must have length equal to ``num_bits`` if provided.
            iteration_names: Optional custom names for the per-iteration circuits.
                Must have length equal to ``num_bits`` if provided.

        Returns:
            A list of IterativePhaseEstimationIteration objects, one for each iteration,
            containing circuits and metadata.

        Raises:
            ValueError: If ``num_bits`` is not positive, or if the length of
                ``phase_corrections``, ``measurement_registers``, or ``iteration_names``
                does not match ``num_bits``.

        """
        if num_bits <= 0:
            raise ValueError("num_bits must be a positive integer.")

        if phase_corrections is None:
            phase_corrections = [0.0] * num_bits
        elif len(phase_corrections) != num_bits:
            raise ValueError("phase_corrections must have length equal to num_bits.")

        if measurement_registers is not None and len(measurement_registers) != num_bits:
            raise ValueError("measurement_registers must have length equal to num_bits when provided.")

        if iteration_names is not None and len(iteration_names) != num_bits:
            raise ValueError("iteration_names must have length equal to num_bits when provided.")

        iterations: list[IterativePhaseEstimationIteration] = []
        for idx in range(num_bits):
            measurement_register = None if measurement_registers is None else measurement_registers[idx]
            iteration_name = None if iteration_names is None else iteration_names[idx]

            _LOGGER.debug("Assembling IQPE iteration object %d/%d.", idx + 1, num_bits)

            iteration_circuit = self.create_iteration(
                state_prep,
                iteration=idx,
                total_iterations=num_bits,
                phase_correction=phase_corrections[idx],
                measurement_register=measurement_register,
                iteration_name=iteration_name,
            )

            iterations.append(iteration_circuit)

        return iterations

    @staticmethod
    def update_phase_feedback(current_phase: float, measured_bit: int) -> float:
        """Update the feedback angle after measuring an iteration.

        Args:
            current_phase: The accumulated phase feedback from previous iterations.
            measured_bit: The measurement result (0 or 1) from the current iteration.

        Returns:
            The updated phase feedback to be used in the next iteration.

        """
        return iterative_phase_feedback_update(current_phase, measured_bit)

    @staticmethod
    def phase_fraction_from_feedback(phase_feedback: float) -> float:
        """Compute the phase fraction ``φ`` from the final feedback phase.

        Args:
            phase_feedback: The accumulated phase feedback after all iterations.

        Returns:
            The phase fraction φ in the range [0, 1), representing the fractional
            part of the eigenphase.

        """
        return phase_fraction_from_feedback(phase_feedback)

    @staticmethod
    def phase_feedback_from_bits(bits_msb_first: Sequence[int]) -> float:
        """Convenience helper that wraps :func:`accumulated_phase_from_bits`.

        Args:
            bits_msb_first: Sequence of measured bits ordered from most-significant
                to least-significant.

        Returns:
            The accumulated phase feedback computed from the bit sequence.

        """
        return accumulated_phase_from_bits(bits_msb_first)


def _validate_iteration_inputs(iteration: int, total_iterations: int) -> None:
    """Validate iteration parameters for IQPE circuit construction.

    Args:
        iteration: The current iteration index (0-based).
        total_iterations: The total number of iterations.

    Raises:
        ValueError: If ``total_iterations`` is not positive, or if ``iteration``
            is outside the valid range [0, total_iterations - 1].

    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be a positive integer.")
    if iteration < 0 or iteration >= total_iterations:
        raise ValueError(
            f"iteration index {iteration} is outside the valid range [0, {total_iterations - 1}].",
        )
