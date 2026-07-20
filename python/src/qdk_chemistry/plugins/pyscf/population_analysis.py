"""PySCF-based population analysis for qdk_chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import PopulationAnalyzer
from qdk_chemistry.data import Settings, Structure, Wavefunction
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf
from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver
from qdk_chemistry.utils import Logger

__all__ = ["PyscfPopulationAnalysisSettings", "PyscfPopulationAnalyzer"]


class PyscfPopulationAnalysisSettings(Settings):
    """Settings for PySCF population analysis."""

    def __init__(self):
        """Initialize PySCF population-analysis settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("method", "string", "mulliken", "Population-analysis method", ["mulliken"])
        self._set_default("basis_set", "string", "def2-svp", "Basis set used for structure inputs")
        self._set_default("scf_method", "string", "hf", "SCF electronic-structure method")
        self._set_default("scf_type", "string", "auto", "SCF reference type", ["auto", "restricted", "unrestricted"])
        self._set_default("convergence_threshold", "double", 1e-7, "SCF convergence threshold")
        self._set_default("max_iterations", "int", 50, "Maximum SCF iterations", (1, 1000))
        self._set_default("xc_grid", "int", 3, "Density functional integration grid level", list(range(10)))


class PyscfPopulationAnalyzer(PopulationAnalyzer):
    """PySCF implementation of Mulliken electron-population analysis."""

    def __init__(self):
        """Initialize the PySCF population analyzer."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfPopulationAnalysisSettings()

    def _run_impl(
        self,
        input_data: Structure | Wavefunction,
        charge: int,
        spin_multiplicity: int,
        n_inactive_orbitals: int,
    ) -> list[float]:
        """Compute electron populations using PySCF Mulliken analysis."""
        Logger.trace_entering()
        del n_inactive_orbitals
        method = self._settings.get("method").lower()
        if method != "mulliken":
            raise ValueError(f"Unsupported PySCF population-analysis method: {method}")

        if isinstance(input_data, Structure):
            solver = self._create_scf_solver()
            _, wavefunction = solver.run(
                input_data,
                charge,
                spin_multiplicity,
                self._settings.get("basis_set"),
            )
            return self._populations_from_wavefunction(wavefunction)

        if isinstance(input_data, Wavefunction):
            return self._populations_from_wavefunction(input_data)

        raise TypeError("PySCF population analysis requires a Structure or Wavefunction input.")

    def _create_scf_solver(self) -> PyscfScfSolver:
        solver = PyscfScfSolver()
        solver_settings = solver.settings()
        solver_settings.set("method", self._settings.get("scf_method"))
        solver_settings.set("scf_type", self._settings.get("scf_type"))
        solver_settings.set("convergence_threshold", self._settings.get("convergence_threshold"))
        solver_settings.set("max_iterations", self._settings.get("max_iterations"))
        solver_settings.set("xc_grid", self._settings.get("xc_grid"))
        return solver

    def _populations_from_wavefunction(self, wavefunction: Wavefunction) -> list[float]:
        orbitals = wavefunction.get_orbitals()
        if orbitals is None:
            raise ValueError("PySCF population analysis requires a wavefunction with orbitals.")

        occ_alpha, occ_beta = wavefunction.get_total_orbital_occupations()
        mean_field = orbitals_to_scf(
            orbitals,
            np.asarray(occ_alpha, dtype=float),
            np.asarray(occ_beta, dtype=float),
            self._settings.get("scf_type"),
            self._settings.get("scf_method"),
        )
        mean_field.verbose = 0
        density = mean_field.make_rdm1()
        ao_populations, _ = mean_field.mulliken_pop(mean_field.mol, density, s=mean_field.get_ovlp())
        ao_slices = mean_field.mol.aoslice_by_atom()
        return [float(np.sum(ao_populations[start:stop])) for _, _, start, stop in ao_slices]

    def name(self) -> str:
        """Return the analyzer name."""
        Logger.trace_entering()
        return "pyscf"

    def aliases(self) -> list[str]:
        """Return accepted analyzer aliases."""
        return ["pyscf", "pyscf_mulliken"]
