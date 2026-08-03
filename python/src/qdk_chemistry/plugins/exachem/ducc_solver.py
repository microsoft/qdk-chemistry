# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem DUCC effective-Hamiltonian builder.

Implements the :class:`~qdk_chemistry.algorithms.EffectiveHamiltonian` interface
to run ExaChem's Double Unitary Coupled Cluster (DUCC) method as an external MPI
process, parse the resulting FCIDUMP output, and return the downfolded
active-space Hamiltonian.

The DUCC method produces a Hermitian effective Hamiltonian for the active space
that incorporates dynamical correlation from external orbitals through a unitary
coupled-cluster similarity transformation. This is the proper unitary analogue of
the SES-CC downfolding approach.

References:
    - N.P. Bauman et al., J. Chem. Phys. 151, 014107 (2019)
    - K. Kowalski, J. Chem. Phys. 148, 094104 (2018)

"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.algorithms import EffectiveHamiltonian
from qdk_chemistry.data import Settings
from qdk_chemistry.plugins.exachem.cli import DuccInputConfig, ExachemResult, run_exachem
from qdk_chemistry.plugins.exachem.conversion import (
    fcidump_to_hamiltonian,
    parse_ducc_results,
    parse_energy_shift,
    parse_fcidump,
)
from qdk_chemistry.plugins.exachem.scf_export import export_scf_files

logger = logging.getLogger(__name__)


def _active_counts(active_orbitals, nocc: int) -> tuple[int, int]:
    """Count the active occupied and active virtual spatial orbitals.

    Args:
        active_orbitals: Orbitals carrying an active-space designation.
        nocc: Number of occupied spatial orbitals in the reference.

    Returns:
        Tuple ``(nactive_occupied, nactive_virtual)``.

    Raises:
        ValueError: If no active space is designated or either channel is empty.

    """
    if not active_orbitals.has_active_space():
        raise ValueError("ExaChem DUCC requires active_orbitals with a designated active space.")

    # get_active_space_indices() returns (alpha, beta) lists of SPATIAL indices;
    # the closed-shell restriction is enforced by the caller, so alpha suffices.
    alpha_indices, beta_indices = active_orbitals.get_active_space_indices()
    if sorted(alpha_indices) != sorted(beta_indices):
        raise ValueError("ExaChem DUCC requires identical alpha and beta active spaces.")

    spatial = sorted(int(p) for p in alpha_indices)
    nactive_o = sum(1 for p in spatial if p < nocc)
    nactive_v = len(spatial) - nactive_o
    if nactive_o == 0 or nactive_v == 0:
        raise ValueError(
            f"ExaChem DUCC needs both active occupied and active virtual orbitals; "
            f"got {nactive_o} occupied and {nactive_v} virtual."
        )
    return nactive_o, nactive_v


class ExachemDuccSettings(Settings):
    """Settings for the ExaChem DUCC effective-Hamiltonian builder.

    Attributes:
        ducc_level (int): DUCC/BCH truncation level (default: 2).
        mpi_ranks (int): Number of MPI processes (default: 1).
        exachem_binary (str): Path to the ExaChem binary, or empty for auto-detect (default: ``""``).
        work_dir (str): Working directory, or empty for a temp dir (default: ``""``).
        timeout (int): Subprocess timeout in seconds (default: 3600).
        ccsd_threshold (float): CCSD convergence threshold (default: 1e-6).
        cd_diagtol (float): Cholesky decomposition diagonal tolerance (default: 1e-5).

    """

    def __init__(self):
        """Initialize the settings with default values."""
        super().__init__()
        self._set_default("ducc_level", "int", 2)
        self._set_default("mpi_ranks", "int", 1)
        self._set_default("exachem_binary", "string", "")
        self._set_default("work_dir", "string", "")
        self._set_default("timeout", "int", 3600)
        self._set_default("ccsd_threshold", "double", 1e-6)
        self._set_default("cd_diagtol", "double", 1e-5)


class ExachemDuccSolver(EffectiveHamiltonian):
    """DUCC effective-Hamiltonian builder via ExaChem CLI.

    Runs ExaChem's DUCC implementation as an external MPI process, skipping
    ExaChem's internal SCF by exporting the input orbitals' MO coefficients and
    density matrix in ExaChem's serial-IO restart format. ExaChem performs
    Cholesky decomposition -> CCSD -> DUCC on the supplied orbitals, producing a
    downfolded active-space Hamiltonian in FCIDUMP format.

    The geometry, basis set and MO coefficients come from the input Hamiltonian's
    orbitals, the electron counts from the input wavefunction, and the active
    space from the active-space indices of ``active_orbitals``.

    Examples:
        >>> solver = ExachemDuccSolver()
        >>> solver.settings().set("ducc_level", 2)
        >>> effective = solver.run(hamiltonian, ccsd_wavefunction, active_orbitals)

    """

    def __init__(self):
        """Initialize the solver with default settings."""
        super().__init__()
        self._settings = ExachemDuccSettings()

    def name(self) -> str:
        """Return ``"exachem_ducc"``."""
        return "exachem_ducc"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["exachem_ducc", "ducc"]

    def _run_impl(self, hamiltonian, wavefunction, active_orbitals):
        """Run ExaChem DUCC and return the downfolded active-space Hamiltonian.

        Args:
            hamiltonian: Full-space Hamiltonian whose orbitals supply the geometry, basis and MO coefficients.
            wavefunction: Full-space wavefunction supplying the alpha/beta electron counts.
            active_orbitals: Orbitals whose active-space indices designate the active subset.

        Returns:
            The effective active-space :class:`~qdk_chemistry.data.Hamiltonian`.

        Raises:
            ValueError: If the reference is open-shell or the inputs are inconsistent.
            ExachemNotFoundError: If ExaChem or the MPI launcher is not found.
            ExachemRunError: If ExaChem fails.
            RuntimeError: If ExaChem produces no DUCC output files.

        """
        s = self._settings

        n_alpha, n_beta = wavefunction.get_total_num_electrons()
        if n_alpha != n_beta:
            raise ValueError(
                "ExaChem DUCC currently only supports closed-shell (restricted) references. "
                "Open-shell support is not yet implemented in ExaChem's DUCC module."
            )

        orbitals = hamiltonian.get_orbitals()
        if not orbitals.has_basis_set():
            raise ValueError(
                "ExaChem DUCC requires a Hamiltonian backed by a molecular basis set; "
                "the provided orbitals have no associated BasisSet."
            )
        basis_set = orbitals.get_basis_set()
        basis_name = basis_set.get_name()
        structure = basis_set.get_structure()

        # qdk-chemistry stores coordinates in Bohr, so feed ExaChem Bohr directly.
        element_symbols = [element.name for element in structure.get_elements()]
        coordinates = np.asarray(structure.get_coordinates()).reshape(-1, 3)
        atoms = [
            f"{symbol} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}"
            for symbol, xyz in zip(element_symbols, coordinates, strict=True)
        ]

        nactive_o, nactive_v = _active_counts(active_orbitals, n_alpha)

        config = DuccInputConfig(
            atoms=atoms,
            basis=basis_name,
            charge=0,
            multiplicity=1,
            units="bohr",
            nactive_oa=nactive_o,
            nactive_ob=nactive_o,
            nactive_va=nactive_v,
            nactive_vb=nactive_v,
            ducc_level=s.get("ducc_level"),
            ccsd_threshold=s.get("ccsd_threshold"),
            cd_diagtol=s.get("cd_diagtol"),
            scf_type="restricted",
            noscf=True,
        )

        mo_coeff_alpha = np.asarray(orbitals.get_coefficients_alpha())
        # ExaChem expects the TOTAL density (alpha + beta), not alpha-only.
        # For restricted closed-shell: D_total = 2 * D_alpha
        density_alpha = np.asarray(orbitals.calculate_ao_density_matrix(n_alpha, n_beta)[0])
        density_for_export = density_alpha * 2.0

        binary = s.get("exachem_binary") or None
        work = s.get("work_dir") or None
        cleanup_work_dir = work is None
        work_path = Path(work) if work else Path(tempfile.mkdtemp(prefix="exachem_ducc_"))
        work_path.mkdir(parents=True, exist_ok=True)

        try:
            scf_prefix_name = f"ducc_input.{basis_name}"
            scf_type_dir = work_path / f"{scf_prefix_name}_files" / "restricted"
            scf_dir = scf_type_dir / "scf"
            scf_dir.mkdir(parents=True, exist_ok=True)
            scf_files_prefix = scf_dir / scf_prefix_name

            # Run context JSON goes one level up from scf/ (at the scf_type level)
            runcontext_prefix = scf_type_dir / scf_prefix_name

            # Feed ExaChem qdk-chemistry's own basis (written as a Gaussian-94 file
            # here and read via LIBINT_DATA_PATH) so the two codes use an identical
            # inter-shell order and identical basis parameters.  The AO export then
            # only needs the within-shell p-component correction.
            basis_data_dir = work_path / "qdk_libint_basis"

            export_scf_files(
                files_prefix=scf_files_prefix,
                mo_coeff_alpha=mo_coeff_alpha,
                density_alpha=density_for_export,
                ao_tilesize=30,
                runcontext_prefix=runcontext_prefix,
                basis_set=basis_set,
                basis_name=basis_name,
                elements=element_symbols,
                basis_data_dir=basis_data_dir,
            )
            logger.info("Exported SCF data for noscf mode to %s", scf_dir)

            result: ExachemResult = run_exachem(
                config,
                nprocs=s.get("mpi_ranks"),
                work_dir=work_path,
                exachem_binary=Path(binary) if binary else None,
                timeout=s.get("timeout"),
                scf_files_prefix=scf_files_prefix,
                libint_data_path=basis_data_dir,
            )

            # Parse DUCC results (prefer native format, fall back to FCIDUMP)
            if result.ducc_results_path and result.ducc_json_path:
                logger.info("Parsing DUCC results from %s", result.ducc_results_path)
                fcidump = parse_ducc_results(result.ducc_results_path, result.ducc_json_path)
            elif result.fcidump_path:
                logger.info("Parsing FCIDUMP from %s", result.fcidump_path)
                fcidump = parse_fcidump(result.fcidump_path)
            else:
                raise RuntimeError(
                    f"ExaChem completed but no DUCC output files were produced. "
                    f"Check {result.work_dir} for output files."
                )

            # Extract the DUCC energy shift from ExaChem's stdout (see
            # conversion.parse_energy_shift). The shift excludes nuclear repulsion;
            # at ducc_lvl=0 ExaChem prints no "Total Energy Shift" line, so the
            # helper falls back to (Full SCF - Bare Active Space SCF). The correct
            # qdk core_energy is core_energy = energy_shift + V_nuc, so that
            # E_total = E_CI_active(from MACIS) + core_energy.
            total_energy_shift = parse_energy_shift(result.stdout)

            if total_energy_shift is not None:
                core_energy = total_energy_shift + fcidump.nuclear_repulsion
            else:
                core_energy = fcidump.nuclear_repulsion
                logger.warning("Could not extract energy shift from ExaChem stdout; using nuc_rep only")

            return fcidump_to_hamiltonian(
                fcidump,
                atoms=atoms,
                basis=basis_name,
                units="bohr",
                core_energy_override=core_energy,
            )
        finally:
            # Remove every input/output/basis file written during the run, unless
            # the caller supplied an explicit work_dir (then they own the files).
            if cleanup_work_dir:
                shutil.rmtree(work_path, ignore_errors=True)
                logger.debug("Cleaned up work directory %s", work_path)
