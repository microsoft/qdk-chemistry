"""Wavefunction utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Hamiltonian, SciWavefunctionContainer, Wavefunction
from qdk_chemistry.utils import Logger

__all__ = [
    "calculate_sparse_wavefunction",
    "get_active_determinants_info",
]


def get_active_determinants_info(wavefunction: Wavefunction, max_determinants: int | None = None) -> str:
    """Generate a string representation of the CI coefficients and configurations.

    Args:
        wavefunction: The Wavefunction object.
        max_determinants: Maximum number of determinants to include in the summary. If None, include all determinants.

    Returns:
        A formatted string listing CI coefficients and their corresponding configurations.

    """
    Logger.trace_entering()
    info_str = ""
    info_str += f"Stored wavefunction with {wavefunction.size()} determinants\n"
    info_str += "Determinants:\n"
    orbitals = wavefunction.get_orbitals()
    num_orbital_chars = 0
    if orbitals.has_active_space():
        alpha_indices = orbitals.get_active_space_indices()[0]
        num_orbital_chars = len(alpha_indices)

    for det, coeff in wavefunction.get_top_determinants(max_determinants=max_determinants).items():
        det_string = det.to_string()
        if num_orbital_chars:
            det_string = det_string[:num_orbital_chars]
        if isinstance(coeff, complex):
            coeff_repr = f"{coeff.real:.8f}" if abs(coeff.imag) <= 1.0e-12 else f"{coeff.real:.8f} + {coeff.imag:.8f}i"
        else:
            coeff_repr = f"{coeff:.8f}"
        info_str += f"  {det_string}: {coeff_repr}\n"
    return info_str


def calculate_sparse_wavefunction(
    reference_wavefunction: Wavefunction,
    hamiltonian: Hamiltonian,
    reference_energy: float,
    energy_tolerance: float,
    max_determinants: int,
    pmc_calculator: str = "macis_pmc",
) -> tuple[float, Wavefunction]:
    """Screen the wavefunction down to a sparse-CI subset based on energy tolerance w.r.t. a reference energy.

    Args:
        reference_wavefunction: The initial wavefunction to be screened.
        hamiltonian: The Hamiltonian corresponding to the wavefunction.
        reference_energy: The reference energy to compare against.
        energy_tolerance: The acceptable energy difference from the reference energy.
        max_determinants: The maximum number of determinants to consider.
        pmc_calculator: The PMC calculator to use for energy projection.

    Returns:
        Energy value and a Wavefunction object representing the sparse-CI wavefunction.

    """
    Logger.trace_entering()
    ranked = reference_wavefunction.get_top_determinants(max_determinants=max_determinants)
    if not ranked:
        Logger.warn("No determinants found; returning an empty wavefunction.")
        return Wavefunction(SciWavefunctionContainer(np.array([]), [], reference_wavefunction.get_orbitals()))

    projector = create("projected_multi_configuration_calculator", pmc_calculator)

    best_energy = None
    best_wavefunction = None
    best_count = 0
    diff = 0.0
    found = False
    count = 1
    projected_energy = 0.0
    projected_wavefunction: Wavefunction = None

    ranked_items = list(ranked.items())
    for count in range(1, len(ranked_items) + 1):
        leading_det_subset = [det for det, _ in ranked_items[:count]]
        projected_energy, projected_wavefunction = projector.run(hamiltonian, list(leading_det_subset))
        diff = abs(float(projected_energy) - float(reference_energy))
        if diff <= energy_tolerance:
            found = True
            break

    best_energy = projected_energy
    best_wavefunction = projected_wavefunction
    best_count = count

    if count == len(ranked_items) and not found:
        Logger.warn(
            f"Sparse CI tolerance not reached with {best_count} determinants; returning the full truncated set."
        )

    Logger.info(f"Sparse CI finder ({best_count} dets) = {best_energy:.8f} Hartree (ΔE = {diff * 1000.0:.4f} mHartree)")
    determinants = list(best_wavefunction.get_active_determinants())
    coeffs = [best_wavefunction.get_coefficient(det) for det in determinants]
    sci_container = SciWavefunctionContainer(
        np.array(coeffs),
        determinants,
        best_wavefunction.get_orbitals(),
    )
    return best_energy, Wavefunction(sci_container)
