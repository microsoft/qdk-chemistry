# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem CCSD(T) dynamical-correlation calculator.

Implements the :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator`
interface to run ExaChem's CCSD(T) method as an external MPI process and return
the perturbative-triples total energy.

Unlike DUCC (which downfolds to an active-space Hamiltonian), CCSD(T) is a
full-space single-point method that yields a total energy. Following the PySCF
coupled-cluster contract, this calculator takes an :class:`~qdk_chemistry.data.Ansatz`,
extracts its molecular reference (geometry, basis, and MO coefficients), and runs
ExaChem in ``noscf`` mode so ExaChem regenerates its own integrals from the
supplied orbitals.

Both restricted (RHF) and unrestricted (UHF) references are supported. ExaChem
requires an unrestricted reference whenever the multiplicity exceeds one;
restricted-open-shell (ROHF) references are not supported by ExaChem's CC module.

References:
    - Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989).

"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.algorithms import DynamicalCorrelationCalculator
from qdk_chemistry.data import Settings
from qdk_chemistry.plugins.exachem.cli import CcsdtInputConfig, ExachemResult, run_exachem
from qdk_chemistry.plugins.exachem.conversion import parse_ccsdt_energy
from qdk_chemistry.plugins.exachem.scf_export import export_scf_files

logger = logging.getLogger(__name__)


class ExachemCcsdtSettings(Settings):
    """Settings for the ExaChem CCSD(T) calculator.

    Attributes:
        mpi_ranks (int): Number of MPI processes (default: 1).
        exachem_binary (str): Path to the ExaChem binary, or empty for auto-detect (default: ``""``).
        work_dir (str): Working directory, or empty for a temp dir (default: ``""``).
        timeout (int): Subprocess timeout in seconds (default: 3600).
        ccsd_threshold (float): CCSD convergence threshold (default: 1e-6).
        cd_diagtol (float): Cholesky decomposition diagonal tolerance (default: 1e-5).
        freeze_core (int): Number of frozen core orbitals (default: 0).
        freeze_virtual (int): Number of frozen virtual orbitals (default: 0).

    """

    def __init__(self):
        """Initialize the settings with default values."""
        super().__init__()
        self._set_default("mpi_ranks", "int", 1)
        self._set_default("exachem_binary", "string", "")
        self._set_default("work_dir", "string", "")
        self._set_default("timeout", "int", 3600)
        self._set_default("ccsd_threshold", "double", 1e-6)
        self._set_default("cd_diagtol", "double", 1e-5)
        self._set_default("freeze_core", "int", 0)
        self._set_default("freeze_virtual", "int", 0)


class ExachemCcsdtCalculator(DynamicalCorrelationCalculator):
    """CCSD(T) total-energy calculator via ExaChem CLI.

    Runs ExaChem's CCSD(T) implementation as an external MPI process, skipping
    ExaChem's internal SCF by providing the Ansatz's pre-computed MO coefficients
    and density matrices. ExaChem performs Cholesky decomposition → CCSD →
    perturbative (T) on the supplied orbitals and reports the CCSD(T) total energy.

    The calculator follows the same ``run(ansatz)`` contract as the PySCF
    coupled-cluster calculator, returning a ``(total_energy, wavefunction, None)``
    tuple. ExaChem does not export coupled-cluster amplitudes, so the returned
    wavefunction is the input reference wavefunction unchanged; only the energy
    carries the CCSD(T) result.

    Because ExaChem regenerates its integrals from a Gaussian basis, the Ansatz
    must be backed by a molecular :class:`~qdk_chemistry.data.BasisSet` (it cannot
    operate on arbitrary model Hamiltonians).
    """

    def __init__(self):
        """Initialize the calculator with default settings."""
        super().__init__()
        self._settings = ExachemCcsdtSettings()

    def name(self) -> str:
        """Return the name of this calculator implementation."""
        return "exachem_ccsd_t"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["exachem_ccsd_t", "exachem_ccsdt", "ccsd_t"]

    def _run_impl(self, ansatz):
        """Run ExaChem CCSD(T) on the Ansatz's molecular reference.

        Args:
            ansatz: The :class:`~qdk_chemistry.data.Ansatz` whose orbitals and
                wavefunction define the molecular reference.

        Returns:
            A tuple ``(total_energy, wavefunction, None)`` where ``total_energy``
            is the CCSD(T) total energy in Hartree and ``wavefunction`` is the
            input reference wavefunction (ExaChem does not export amplitudes).

        Raises:
            ValueError: If the Ansatz is not backed by a molecular basis set.
            ExachemNotFoundError: If ExaChem or the MPI launcher is not found.
            ExachemRunError: If ExaChem fails.
            RuntimeError: If the CCSD(T) total energy cannot be parsed.

        """
        s = self._settings
        wavefunction = ansatz.get_wavefunction()
        orbitals = wavefunction.get_orbitals()

        if not orbitals.has_basis_set():
            raise ValueError(
                "ExaChem CCSD(T) requires an Ansatz backed by a molecular basis set; "
                "the provided orbitals have no associated BasisSet."
            )
        basis_set = orbitals.get_basis_set()
        structure = basis_set.get_structure()
        basis_name = basis_set.get_name()

        # Build ExaChem geometry lines in Bohr (qdk stores coordinates in Bohr).
        symbols = structure.get_atomic_symbols()
        coords = np.asarray(structure.get_coordinates())
        atoms = [f"{sym} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}" for sym, xyz in zip(symbols, coords, strict=False)]

        n_alpha, n_beta = wavefunction.get_total_num_electrons()
        multiplicity = (n_alpha - n_beta) + 1
        total_nuclear_charge = round(structure.get_total_nuclear_charge())
        charge = total_nuclear_charge - (n_alpha + n_beta)

        # ExaChem requires an unrestricted reference for any open-shell system.
        is_unrestricted = orbitals.is_unrestricted() or multiplicity > 1
        scf_type = "unrestricted" if is_unrestricted else "restricted"

        alpha_occ, beta_occ = wavefunction.get_total_orbital_occupations()
        mo_coeff_alpha = np.asarray(orbitals.get_coefficients_alpha())

        # Prepare the working directory and SCF restart prefix.
        work = s.get("work_dir") or None
        work_path = Path(work) if work else Path(tempfile.mkdtemp(prefix="exachem_ccsdt_"))
        work_path.mkdir(parents=True, exist_ok=True)

        input_prefix = "ccsdt_input"
        scf_prefix_name = f"{input_prefix}.{basis_name}"
        scf_type_dir = work_path / f"{scf_prefix_name}_files" / scf_type
        scf_dir = scf_type_dir / "scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        scf_files_prefix = scf_dir / scf_prefix_name
        runcontext_prefix = scf_type_dir / scf_prefix_name

        if is_unrestricted:
            # Per-spin densities: ExaChem reads D_alpha and D_beta separately.
            density_alpha, density_beta = orbitals.calculate_ao_density_matrix(alpha_occ, beta_occ)
            mo_coeff_beta = np.asarray(orbitals.get_coefficients_beta())
            export_scf_files(
                files_prefix=scf_files_prefix,
                mo_coeff_alpha=mo_coeff_alpha,
                density_alpha=np.asarray(density_alpha),
                ao_tilesize=30,
                runcontext_prefix=runcontext_prefix,
                basis_set=basis_set,
                mo_coeff_beta=mo_coeff_beta,
                density_beta=np.asarray(density_beta),
            )
        else:
            # Restricted: ExaChem reads the total (alpha+beta) density from .alpha.density.
            total_occ = np.asarray(alpha_occ) + np.asarray(beta_occ)
            density_total = np.asarray(orbitals.calculate_ao_density_matrix(total_occ))
            export_scf_files(
                files_prefix=scf_files_prefix,
                mo_coeff_alpha=mo_coeff_alpha,
                density_alpha=density_total,
                ao_tilesize=30,
                runcontext_prefix=runcontext_prefix,
                basis_set=basis_set,
            )
        logger.info("Exported SCF data for noscf CCSD(T) (%s) to %s", scf_type, scf_dir)

        config = CcsdtInputConfig(
            atoms=atoms,
            basis=basis_name,
            charge=charge,
            multiplicity=multiplicity,
            units="bohr",
            ccsd_threshold=s.get("ccsd_threshold"),
            cd_diagtol=s.get("cd_diagtol"),
            freeze_core=s.get("freeze_core"),
            freeze_virtual=s.get("freeze_virtual"),
            scf_type=scf_type,
            noscf=True,
            input_prefix=input_prefix,
        )

        binary = s.get("exachem_binary") or None
        result: ExachemResult = run_exachem(
            config,
            nprocs=s.get("mpi_ranks"),
            work_dir=work_path,
            exachem_binary=Path(binary) if binary else None,
            timeout=s.get("timeout"),
            scf_files_prefix=scf_files_prefix,
        )

        energies = parse_ccsdt_energy(result.stdout)
        if energies.ccsd_pt_total is None:
            raise RuntimeError(
                "ExaChem completed but the CCSD(T) total energy could not be parsed from stdout. "
                f"Check {result.work_dir} for output."
            )
        total_energy = energies.ccsd_pt_total

        # Clean up temporary files unless a work_dir was explicitly provided.
        if not s.get("work_dir"):
            shutil.rmtree(result.work_dir, ignore_errors=True)
            logger.debug("Cleaned up work directory %s", result.work_dir)

        # ExaChem does not export amplitudes; return the reference wavefunction.
        return total_energy, wavefunction, None
