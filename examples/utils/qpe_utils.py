"""Utility functions for QPE."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import Counter
from pathlib import Path

import numpy as np
from qdk.openqasm import compile
from qdk.simulation import run_qir
from qdk_chemistry.algorithms import IterativePhaseEstimation
from qdk_chemistry.data import (
    QpeResult,
    QubitHamiltonian,
    SciWavefunctionContainer,
    Wavefunction,
)
from qdk_chemistry.utils import Logger
from qiskit import QuantumCircuit, qasm3


def prepare_2_dets_trial_state(
    wf: Wavefunction, rotation_angle: float = np.pi / 12
) -> tuple[Wavefunction, float]:
    """Scan rotation angles for 2-determinant wavefunction.

        psi(theta) = cos(theta)*|D1> + sin(theta)*|D2|

    Args:
        wf: Original wavefunction (used to extract determinants)
        rotation_angle: Rotation angle (in radians)

    Returns:
        wavefunction: Wavefunction object for the given rotation angle
        fidelity: Fidelity with respect to the exact wavefunction

    """
    dets = wf.get_top_determinants(max_determinants=2)
    orbitals = wf.get_orbitals()

    c1_new = np.cos(round(rotation_angle, 4))
    c2_new = np.sin(round(rotation_angle, 4))

    # Only include terms with non-zero coefficients
    coeffs_new = []
    dets_new = []

    for coeff, det in zip([c1_new, c2_new], dets):
        if not np.isclose(coeff, 0.0):
            coeffs_new.append(coeff)
            dets_new.append(det)

    # Convert to numpy arrays and normalize
    coeffs_new = np.array(coeffs_new, dtype=float)
    coeffs_new /= np.linalg.norm(coeffs_new)

    # Construct trial wavefunction
    rotated_wf = Wavefunction(SciWavefunctionContainer(coeffs_new, dets_new, orbitals))

    # Fidelity with original reference wf
    coeffs_wf = np.array(list(dets.values()))
    fidelity = np.abs(np.vdot(coeffs_new, coeffs_wf)) ** 2

    return rotated_wf, fidelity


def run_single_trial_iqpe(
    qubit_hamiltonian: QubitHamiltonian,
    state_prep: QuantumCircuit,
    time: float,
    precision: int,
    shots: int,
    trial_seed: int,
    reference_energy: float,
    output_dir: str | None = None,
) -> QpeResult:
    """Helper function to run a single IQPE trial with the given seed.

    Args:
        qubit_hamiltonian: Qubit Hamiltonian for the system
        state_prep: Qiskit circuit for state preparation
        time: Evolution time for IQPE
        precision: Number of bits of precision
        shots: Shots per iteration
        trial_seed: Random seed for simulator
        reference_energy: Reference energy for phase estimation
        output_dir: Directory to save the result JSON file (optional)
    Returns:
        `QpeResult` for a single IQPE trial

    """
    Logger.trace_entering()
    iqpe = IterativePhaseEstimation(qubit_hamiltonian, time)
    phase_feedback = 0.0
    bits: list[int] = []

    Logger.info(f"Running IQPE trial with seed {trial_seed}")

    for iteration in range(precision):
        Logger.info(f"Iteration {iteration + 1}/{precision}")
        iter_info = iqpe.create_iteration(
            state_prep,
            iteration=iteration,
            total_iterations=precision,
            phase_correction=phase_feedback,
        )
        compiled = iter_info.circuit
        Logger.info("Circuit generated")
        circuit_qasm = qasm3.dumps(compiled)
        Logger.info("Circuit is dumped to qasm")
        qir = compile(circuit_qasm)
        Logger.info("Circuit is compiled to qir")
        result = run_qir(qir, shots=shots, seed=trial_seed, type="cpu")
        Logger.info(f"Measurement results obtained: {result}")
        flat_bitstring = [str(x[0]) for x in result]
        counts = dict(Counter(flat_bitstring))
        measured_bit = 0 if counts.get("Zero", 0) >= counts.get("One", 0) else 1

        bits.append(measured_bit)
        phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

    phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)
    result = QpeResult.from_phase_fraction(
        method=IterativePhaseEstimation.algorithm,
        phase_fraction=phase_fraction,
        evolution_time=time,
        bits_msb_first=bits,
        reference_energy=reference_energy,
    )
    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True)
        result.to_json_file(f"{output_dir}/iqpe_result_{trial_seed}.qpe_result.json")
    return result


def run_iqpe(
    qubit_hamiltonian: QubitHamiltonian,
    state_prep: QuantumCircuit,
    *,
    time: float,
    precision: int,
    shots: int,
    seed: int,
    reference_energy: float,
    trials: int,
    output_dir: str | None = None,
) -> list[QpeResult]:
    """Run multiple IQPE trials with different seeds.

    Args:
        qubit_hamiltonian: Qubit Hamiltonian for the system
        state_prep: Qiskit circuit for state preparation
        time: Evolution time for IQPE
        precision: Number of bits of precision
        shots: Shots per iteration
        seed: Base random seed for simulator
        reference_energy: Reference energy for phase estimation
        trials: Number of trials to run
        output_dir: Directory to save the result JSON files (optional)

    Returns:
        List of `QpeResult` for each trial

    """
    results = []
    for trial in range(trials):
        trial_seed = seed + trial  # Different seed per trial
        result = run_single_trial_iqpe(
            qubit_hamiltonian,
            state_prep,
            time,
            precision,
            shots,
            trial_seed,
            reference_energy,
            output_dir=output_dir,
        )
        results.append(result)
    return results
