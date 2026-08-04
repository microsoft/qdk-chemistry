# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem CCSD dynamical-correlation calculator with T-amplitude read-in.

Implements the :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator`
interface to run ExaChem's CCSD method (Coupled Cluster Singles and Doubles) as
an external MPI process, then reads the converged T1/T2 cluster amplitudes back
into a qdk-chemistry :class:`~qdk_chemistry.data.AmplitudeContainer`.

Unlike the CCSD(T) calculator (which only returns a total energy), ExaChem's
plain CCSD writes its converged amplitudes to disk. This calculator enables
ExaChem's ``CC.PRINT.tamplitudes`` option so the T1/T2 tensors are written as
text files (``<prefix>.print_t1amp.txt`` / ``<prefix>.print_t2amp.txt``), parses
them, and returns a wavefunction carrying the amplitudes -- mirroring the PySCF
coupled-cluster calculator's contract.

Both restricted (RHF) and unrestricted (UHF) references are supported. ExaChem
requires an unrestricted reference whenever the multiplicity exceeds one.

References:
    - https://exachem.readthedocs.io/en/latest/user_guide/coupledcluster.html#ccsd

"""

from __future__ import annotations

import glob
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.algorithms import DynamicalCorrelationCalculator
from qdk_chemistry.data import AmplitudeContainer, AmplitudeType, Settings, Wavefunction
from qdk_chemistry.plugins.exachem.cli import CcsdInputConfig, ExachemResult, run_exachem
from qdk_chemistry.plugins.exachem.conversion import (
    parse_ccsd_amplitudes_restricted,
    parse_ccsd_amplitudes_unrestricted,
    parse_ccsdt_energy,
)
from qdk_chemistry.plugins.exachem.scf_export import export_scf_files

logger = logging.getLogger(__name__)

__all__ = ["ExachemCcsdCalculator", "ExachemCcsdSettings"]


class ExachemCcsdSettings(Settings):
    """Settings for the ExaChem CCSD calculator.

    Attributes:
        exachem_binary (str): Path to the ExaChem binary, or empty to find ``ExaChem`` on ``PATH``.
        mpi_ranks (int): Number of MPI processes (default: 1).
        mpi_bind_to (str): Binding policy per rank; empty defers to the launcher (default: ``"core"``).
        work_dir (str): Working directory, or empty for a temp dir (default: ``""``).
        timeout (int): Subprocess timeout in seconds (default: 3600).
        ccsd_threshold (float): CCSD convergence threshold (default: 1e-6).
        cd_diagtol (float): Cholesky decomposition diagonal tolerance (default: 1e-5).
        freeze_core (int): Number of frozen core orbitals (default: 0).
        freeze_virtual (int): Number of frozen virtual orbitals (default: 0).
        store_amplitudes (bool): Read the T1/T2 amplitudes back into the returned wavefunction.

    """

    def __init__(self):
        """Initialize the settings with default values."""
        super().__init__()
        self._set_default(
            "exachem_binary",
            "string",
            "",
            "Full path to the ExaChem binary; empty finds 'ExaChem' on PATH",
        )
        self._set_default("mpi_ranks", "int", 1, "Number of MPI processes to launch ExaChem with")
        self._set_default(
            "mpi_bind_to", "string", "core", "Binding policy for each MPI rank; empty defers to the launcher default"
        )
        self._set_default("work_dir", "string", "", "Working directory for ExaChem input/output; empty uses a temp dir")
        self._set_default("timeout", "int", 3600, "Maximum seconds to wait for ExaChem to finish")
        self._set_default("ccsd_threshold", "double", 1e-6, "CCSD convergence threshold")
        self._set_default("cd_diagtol", "double", 1e-5, "Cholesky decomposition diagonal tolerance")
        self._set_default("freeze_core", "int", 0, "Number of frozen core orbitals")
        self._set_default("freeze_virtual", "int", 0, "Number of frozen virtual orbitals")
        self._set_default(
            "store_amplitudes",
            "bool",
            True,
            "Read the converged T1/T2 amplitudes into the returned wavefunction",
        )


class ExachemCcsdCalculator(DynamicalCorrelationCalculator):
    """CCSD calculator via ExaChem CLI that returns the cluster amplitudes.

    Runs ExaChem's CCSD implementation as an external MPI process on the Ansatz's
    pre-computed MO coefficients (``noscf`` restart), then reads the converged
    T1/T2 amplitudes into an :class:`~qdk_chemistry.data.AmplitudeContainer`.

    Follows the same ``run(ansatz)`` contract as the PySCF coupled-cluster
    calculator, returning ``(total_energy, wavefunction, None)`` where
    ``wavefunction`` carries the CCSD amplitudes.

    The Ansatz must be backed by a molecular :class:`~qdk_chemistry.data.BasisSet`.
    """

    def __init__(self):
        """Initialize the calculator with default settings."""
        super().__init__()
        self._settings = ExachemCcsdSettings()

    def name(self) -> str:
        """Return the name of this calculator implementation."""
        return "exachem_ccsd"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["exachem_ccsd", "exachem_coupled_cluster"]

    def _run_impl(self, ansatz):
        """Run ExaChem CCSD on the Ansatz's molecular reference.

        Args:
            ansatz: The :class:`~qdk_chemistry.data.Ansatz` whose orbitals and wavefunction define the reference.

        Returns:
            A tuple ``(total_energy, wavefunction, None)`` where ``total_energy``
            is the CCSD total energy in Hartree and ``wavefunction`` carries the
            CCSD T1/T2 amplitudes (when ``store_amplitudes`` is enabled).

        Raises:
            ValueError: If the Ansatz is not backed by a molecular basis set.
            ExachemNotFoundError: If ExaChem or the MPI launcher is not found.
            ExachemRunError: If ExaChem fails.
            RuntimeError: If the CCSD energy or amplitude files cannot be found.

        """
        s = self._settings
        wavefunction = ansatz.get_wavefunction()
        orbitals = wavefunction.get_orbitals()

        if not orbitals.has_basis_set():
            raise ValueError(
                "ExaChem CCSD requires an Ansatz backed by a molecular basis set; "
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
        cleanup_work_dir = work is None
        work_path = Path(work) if work else Path(tempfile.mkdtemp(prefix="exachem_ccsd_"))
        work_path.mkdir(parents=True, exist_ok=True)

        try:
            input_prefix = "ccsd_input"
            scf_prefix_name = f"{input_prefix}.{basis_name}"
            scf_type_dir = work_path / f"{scf_prefix_name}_files" / scf_type
            scf_dir = scf_type_dir / "scf"
            scf_dir.mkdir(parents=True, exist_ok=True)
            scf_files_prefix = scf_dir / scf_prefix_name
            runcontext_prefix = scf_type_dir / scf_prefix_name

            # Feed ExaChem qdk-chemistry's own basis (written here and read via
            # LIBINT_DATA_PATH) so both codes share an identical inter-shell order and
            # basis parameters; the AO export then only needs the within-shell p-swap.
            basis_data_dir = work_path / "qdk_libint_basis"

            if is_unrestricted:
                density_alpha, density_beta = orbitals.calculate_ao_density_matrix(alpha_occ, beta_occ)
                mo_coeff_beta = np.asarray(orbitals.get_coefficients_beta())
                export_scf_files(
                    files_prefix=scf_files_prefix,
                    mo_coeff_alpha=mo_coeff_alpha,
                    density_alpha=np.asarray(density_alpha),
                    basis_set=basis_set,
                    basis_name=basis_name,
                    elements=list(symbols),
                    basis_data_dir=basis_data_dir,
                    ao_tilesize=30,
                    runcontext_prefix=runcontext_prefix,
                    mo_coeff_beta=mo_coeff_beta,
                    density_beta=np.asarray(density_beta),
                )
            else:
                total_occ = np.asarray(alpha_occ) + np.asarray(beta_occ)
                density_total = np.asarray(orbitals.calculate_ao_density_matrix(total_occ))
                export_scf_files(
                    files_prefix=scf_files_prefix,
                    mo_coeff_alpha=mo_coeff_alpha,
                    density_alpha=density_total,
                    basis_set=basis_set,
                    basis_name=basis_name,
                    elements=list(symbols),
                    basis_data_dir=basis_data_dir,
                    ao_tilesize=30,
                    runcontext_prefix=runcontext_prefix,
                )
            logger.info("Exported SCF data for noscf CCSD (%s) to %s", scf_type, scf_dir)

            store_amplitudes = bool(s.get("store_amplitudes"))
            freeze_core = int(s.get("freeze_core"))
            freeze_virtual = int(s.get("freeze_virtual"))

            config = CcsdInputConfig(
                atoms=atoms,
                basis=basis_name,
                charge=charge,
                multiplicity=multiplicity,
                units="bohr",
                ccsd_threshold=s.get("ccsd_threshold"),
                cd_diagtol=s.get("cd_diagtol"),
                freeze_core=freeze_core,
                freeze_virtual=freeze_virtual,
                scf_type=scf_type,
                noscf=True,
                write_amplitudes=store_amplitudes,
                input_prefix=input_prefix,
            )

            binary = s.get("exachem_binary") or None
            result: ExachemResult = run_exachem(
                config,
                nprocs=s.get("mpi_ranks"),
                work_dir=work_path,
                exachem_binary=Path(binary) if binary else None,
                mpi_bind_to=s.get("mpi_bind_to"),
                timeout=s.get("timeout"),
                scf_files_prefix=scf_files_prefix,
                libint_data_path=basis_data_dir,
            )

            energies = parse_ccsdt_energy(result.stdout)
            if energies.ccsd_total is None:
                raise RuntimeError(
                    "ExaChem completed but the CCSD total energy could not be parsed from stdout. "
                    f"Check {result.work_dir} for output."
                )
            total_energy = energies.ccsd_total

            # Build the returned wavefunction, optionally carrying the amplitudes.
            cc_container = self._build_amplitude_container(
                result.work_dir,
                orbitals,
                wavefunction,
                n_alpha,
                n_beta,
                orbitals.get_num_molecular_orbitals(),
                is_unrestricted,
                freeze_core,
                freeze_virtual,
                store_amplitudes,
            )
            updated_wavefunction = Wavefunction(cc_container)

            return total_energy, updated_wavefunction, None
        finally:
            # Remove every input/output/basis file written during the run, unless
            # the caller supplied an explicit work_dir (then they own the files).
            if cleanup_work_dir:
                shutil.rmtree(work_path, ignore_errors=True)
                logger.debug("Cleaned up work directory %s", work_path)

    @staticmethod
    def _find_amplitude_files(work_dir: Path) -> tuple[Path, Path]:
        """Locate ExaChem's T1/T2 amplitude text files under ``work_dir``."""
        t1 = glob.glob(str(Path(work_dir) / "**" / "*print_t1amp.txt"), recursive=True)
        t2 = glob.glob(str(Path(work_dir) / "**" / "*print_t2amp.txt"), recursive=True)
        if not t1 or not t2:
            raise RuntimeError(
                "ExaChem CCSD completed but the T-amplitude files "
                "(*.print_t1amp.txt / *.print_t2amp.txt) were not found in "
                f"{work_dir}. Ensure CC.PRINT.tamplitudes was enabled."
            )
        return Path(t1[0]), Path(t2[0])

    def _build_amplitude_container(
        self,
        work_dir,
        orbitals,
        wavefunction,
        n_alpha,
        n_beta,
        nmo,
        is_unrestricted,
        freeze_core,
        freeze_virtual,
        store_amplitudes,
    ):
        """Parse the T amplitudes and build the qdk-chemistry AmplitudeContainer."""
        if not store_amplitudes:
            return AmplitudeContainer(orbitals, wavefunction, AmplitudeType.CoupledCluster, sector="electrons")

        t1_path, t2_path = self._find_amplitude_files(work_dir)

        if is_unrestricted:
            noa = n_alpha - freeze_core
            nob = n_beta - freeze_core
            nva = (nmo - n_alpha) - freeze_virtual
            nvb = (nmo - n_beta) - freeze_virtual
            t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb = parse_ccsd_amplitudes_unrestricted(
                t1_path, t2_path, noa, nob, nva, nvb
            )
            # AmplitudeContainer's unrestricted signature expects the alpha-beta
            # (aabb) block before the same-spin (aaaa, bbbb) blocks.
            return AmplitudeContainer(
                orbitals,
                wavefunction,
                AmplitudeType.CoupledCluster,
                np.reshape(t1_aa, (t1_aa.size, 1)),
                np.reshape(t1_bb, (t1_bb.size, 1)),
                np.reshape(t2_abab, (t2_abab.size, 1)),
                np.reshape(t2_aaaa, (t2_aaaa.size, 1)),
                np.reshape(t2_bbbb, (t2_bbbb.size, 1)),
                sector="electrons",
            )

        nocc = n_alpha - freeze_core
        nvir = (nmo - n_alpha) - freeze_virtual
        t1, t2 = parse_ccsd_amplitudes_restricted(t1_path, t2_path, nocc, nvir)
        return AmplitudeContainer(
            orbitals,
            wavefunction,
            AmplitudeType.CoupledCluster,
            np.reshape(t1, (t1.size, 1)),
            np.reshape(t2, (t2.size, 1)),
            sector="electrons",
        )
