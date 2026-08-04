"""PySCF-based population analysis for qdk_chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from pyscf.scf.hf import mulliken_pop

from qdk_chemistry.algorithms import PopulationAnalyzer, ScfSolver
from qdk_chemistry.data import AlgorithmRef, Settings, Wavefunction
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf
from qdk_chemistry.utils import Logger

__all__ = ["PyscfPopulationAnalysisSettings", "PyscfPopulationAnalyzer"]


class PyscfPopulationAnalysisSettings(Settings):
    """Settings for PySCF population analysis."""

    def __init__(self):
        """Initialize PySCF population-analysis settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("method", "string", "mulliken", "Population-analysis method", ["mulliken"])
        self._set_default(
            "scf_solver",
            "algorithm_ref",
            AlgorithmRef("scf_solver", "pyscf"),
        )


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
        charge: int,
        spin_multiplicity: int,
        n_inactive_orbitals: int,
    ) -> list[float]:
        """Compute electron populations using PySCF Mulliken analysis."""
        Logger.trace_entering()
        del charge, spin_multiplicity, n_inactive_orbitals
        method = self._settings.get("method").lower()
        if method != "mulliken":
            raise ValueError(f"Unsupported PySCF population-analysis method: {method}")

        scf_solver: ScfSolver = self._create_nested("scf_solver")
        return self._populations_from_wavefunction(wavefunction, scf_solver)

    def _populations_from_wavefunction(self, wavefunction: Wavefunction, scf_solver: ScfSolver) -> list[float]:
        orbitals = wavefunction.get_orbitals()
        if orbitals is None:
            raise ValueError("PySCF population analysis requires a wavefunction with orbitals.")
        if not orbitals.has_basis_set():
            raise ValueError("PySCF population analysis requires orbitals with an associated basis set.")

        occ_alpha, occ_beta = wavefunction.get_total_orbital_occupations()
        scf_settings = scf_solver.settings()
        mean_field = orbitals_to_scf(
            orbitals,
            np.asarray(occ_alpha, dtype=float),
            np.asarray(occ_beta, dtype=float),
            scf_settings.get("scf_type"),
            scf_settings.get("method"),
        )
        density = np.asarray(mean_field.make_rdm1())
        if density.ndim == 3 and density.shape[0] == 2:
            density = density.sum(axis=0)
        ao_populations, _ = mulliken_pop(mean_field.mol, density, s=mean_field.get_ovlp(), verbose=0)
        ao_slices = mean_field.mol.aoslice_by_atom()
        return [float(np.sum(ao_populations[start:stop])) for _, _, start, stop in ao_slices]

    def name(self) -> str:
        """Return the analyzer name."""
        Logger.trace_entering()
        return "pyscf"
