"""Utility functions for QPE."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import QubitHamiltonian, SciWavefunctionContainer, Wavefunction


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


def compute_evolution_time(
    qubit_hamiltonian: QubitHamiltonian,
    num_bits: int,
    solve_hamiltonian: bool = True,
    target_energy_precision: float = 1e-3,
    initial_time_guess: float | None = None,
) -> float:
    """Compute an evolution time for iterative QPE to achieve a target energy precision.

    Starts from a base time of pi / ||H|| (Schatten norm of qubit Hamiltonian), or an optional initial_time_guess.
    If solve_hamiltonian is True, obtains a reference energy from qubit Hamiltonian via a sparse matrix solver
    and refines the time to reduce phase discretization error given num_bits of precision.
    If False, returns the base time without refinement.

    Args:
        qubit_hamiltonian: Qubit Hamiltonian.
        num_bits: Number of precision bits used in iterative QPE.
        solve_hamiltonian: Whether to solve for a reference energy to refine the time. Defaults to True.
        target_energy_precision: Desired energy precision in Hartree. Defaults to 1e-3.
        initial_time_guess: Optional initial evolution time guess.

    Returns:
        Computed evolution time.

    """
    # Compute base evolution time from Hamiltonian norm or use provided guess
    bound_time = np.pi / qubit_hamiltonian.schatten_norm
    base_time = (
        min(initial_time_guess, bound_time)
        if initial_time_guess is not None
        else bound_time
    )

    if not solve_hamiltonian:
        return base_time

    # Use the reference energy from the qubit Hamiltonian
    solver = create("qubit_hamiltonian_solver", "qdk_sparse_matrix_solver")
    reference_energy, _ = solver.run(qubit_hamiltonian)

    # Compute the expected phase from the reference energy
    expected_phase = (base_time * reference_energy) / (2 * np.pi) % 1

    # Discretize to the nearest representable phase with given precision bits
    bit_phase = round(expected_phase * 2**num_bits) / 2**num_bits

    # Compute the energy error from phase discretization
    if abs(base_time) < np.finfo(type(base_time)).eps:
        raise ValueError(
            f"Cannot compute discretization energy error: base_time {base_time}is too close to zero."
        )
    discretization_energy_error = (2 * np.pi * (bit_phase - expected_phase)) / base_time

    # Shift the energy error to achieve the target precision
    shifted_energy = discretization_energy_error - target_energy_precision

    # Compute the adjusted evolution time, guarding against division by zero
    denominator = reference_energy + target_energy_precision
    if abs(denominator) < np.finfo(type(denominator)).eps:
        raise ValueError(
            "Cannot compute adjusted evolution time: reference_energy + "
            f"target_energy_precision is too close to zero "
            f"(reference_energy={reference_energy}, "
            f"target_energy_precision={target_energy_precision})."
        )

    proposed_time = base_time + shifted_energy * base_time / denominator
    return proposed_time
