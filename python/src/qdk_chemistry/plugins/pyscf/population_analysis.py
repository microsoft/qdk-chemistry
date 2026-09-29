"""PySCF-based population analysis for qdk_chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from pyscf.scf.hf import mulliken_pop

from qdk_chemistry.algorithms import PopulationAnalyzer
from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data._spin_channels import spin_channel_indices
from qdk_chemistry.data.symmetry import axes
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf
from qdk_chemistry.utils import Logger

__all__ = ["PyscfPopulationAnalysisSettings", "PyscfPopulationAnalyzer"]


def _embed_active_one_rdm(
    active_one_rdm: np.ndarray,
    active_indices: list[int],
    inactive_indices: list[int],
    n_orbitals: int,
    inactive_occupation: float,
) -> np.ndarray:
    active_one_rdm = np.asarray(active_one_rdm)
    expected_shape = (len(active_indices), len(active_indices))
    if active_one_rdm.shape != expected_shape:
        raise ValueError("PySCF population analysis requires 1-RDM dimensions to match the active space.")

    all_indices = active_indices + inactive_indices
    if any(index < 0 or index >= n_orbitals for index in all_indices):
        raise ValueError("PySCF population analysis encountered an invalid orbital index.")

    one_rdm = np.zeros((n_orbitals, n_orbitals), dtype=active_one_rdm.dtype)
    one_rdm[np.ix_(active_indices, active_indices)] = active_one_rdm
    one_rdm[inactive_indices, inactive_indices] = inactive_occupation
    return one_rdm


def _density_from_wavefunction(wavefunction: Wavefunction) -> np.ndarray:
    orbitals = wavefunction.get_orbitals()
    n_orbitals = orbitals.get_num_molecular_orbitals()

    if orbitals.is_unrestricted():
        if not wavefunction.has_one_rdm_spin_dependent():
            raise ValueError(
                "PySCF population analysis requires spin-dependent active-space 1-RDM blocks for unrestricted orbitals."
            )
        active_alpha, active_beta = wavefunction.get_active_one_rdm_spin_dependent()
        active_indices = orbitals.active_indices()
        inactive_indices = orbitals.inactive_indices()
        one_rdm_alpha = _embed_active_one_rdm(
            active_alpha,
            spin_channel_indices(active_indices, axes.alpha()),
            spin_channel_indices(inactive_indices, axes.alpha()),
            n_orbitals,
            1.0,
        )
        one_rdm_beta = _embed_active_one_rdm(
            active_beta,
            spin_channel_indices(active_indices, axes.beta()),
            spin_channel_indices(inactive_indices, axes.beta()),
            n_orbitals,
            1.0,
        )
        density_alpha, density_beta = orbitals.calculate_ao_density_matrix_from_rdm(one_rdm_alpha, one_rdm_beta)
        return np.asarray(density_alpha) + np.asarray(density_beta)

    if not wavefunction.has_one_rdm_spin_traced():
        raise ValueError("PySCF population analysis requires a spin-traced active-space 1-RDM.")
    one_rdm = _embed_active_one_rdm(
        wavefunction.get_active_one_rdm_spin_traced(),
        spin_channel_indices(orbitals.active_indices(), axes.alpha()),
        spin_channel_indices(orbitals.inactive_indices(), axes.alpha()),
        n_orbitals,
        2.0,
    )
    return np.asarray(orbitals.calculate_ao_density_matrix_from_rdm(one_rdm))


class PyscfPopulationAnalysisSettings(Settings):
    """Settings for PySCF population analysis."""

    def __init__(self):
        """Initialize PySCF population-analysis settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("method", "string", "mulliken", "Population-analysis method", ["mulliken"])


class PyscfPopulationAnalyzer(PopulationAnalyzer):
    """PySCF implementation of Mulliken electron-population analysis."""

    def __init__(self):
        """Initialize the PySCF population analyzer."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfPopulationAnalysisSettings()

    def _run_impl(
        self,
        wavefunction: Wavefunction,
    ) -> list[float]:
        """Compute electron populations using PySCF Mulliken analysis."""
        Logger.trace_entering()
        method = self._settings.get("method").lower()
        if method != "mulliken":
            raise ValueError(f"Unsupported PySCF population-analysis method: {method}")

        return self._populations_from_wavefunction(wavefunction)

    def _populations_from_wavefunction(self, wavefunction: Wavefunction) -> list[float]:
        orbitals = wavefunction.get_orbitals()
        if orbitals is None:
            raise ValueError("PySCF population analysis requires a wavefunction with orbitals.")
        if not orbitals.has_basis_set():
            raise ValueError("PySCF population analysis requires orbitals with an associated basis set.")

        occ_alpha, occ_beta = wavefunction.get_total_orbital_occupations()
        mean_field = orbitals_to_scf(
            orbitals,
            np.asarray(occ_alpha, dtype=float),
            np.asarray(occ_beta, dtype=float),
        )
        density = _density_from_wavefunction(wavefunction)
        ao_populations, _ = mulliken_pop(mean_field.mol, density, s=orbitals.get_overlap_matrix(), verbose=0)
        ao_slices = mean_field.mol.aoslice_by_atom()
        return [float(np.sum(ao_populations[start:stop])) for _, _, start, stop in ao_slices]

    def name(self) -> str:
        """Return the analyzer name."""
        Logger.trace_entering()
        return "pyscf"
