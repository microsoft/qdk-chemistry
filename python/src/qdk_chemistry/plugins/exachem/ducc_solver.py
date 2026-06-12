# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""ExaChem DUCC Hamiltonian downfolding solver.

Implements the :class:`~qdk_chemistry.algorithms.base.Algorithm` interface to run
ExaChem's Double Unitary Coupled Cluster (DUCC) method as an external MPI process,
parse the resulting FCIDUMP output, and return the downfolded active-space Hamiltonian.

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
from pathlib import Path

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.plugins.exachem.cli import DuccInputConfig, ExachemResult, run_exachem
from qdk_chemistry.plugins.exachem.conversion import (
    fcidump_to_hamiltonian,
    parse_ducc_results,
    parse_fcidump,
)
from qdk_chemistry.plugins.exachem.scf_export import export_scf_files

logger = logging.getLogger(__name__)


class HamiltonianDownfolderFactory(AlgorithmFactory):
    """Factory for Hamiltonian downfolding algorithms."""

    def algorithm_type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def default_algorithm_name(self) -> str:
        """Return ``"exachem_ducc"``."""
        return "exachem_ducc"


class ExachemDuccSolver(Algorithm):
    """DUCC Hamiltonian downfolding via ExaChem CLI.

    Runs ExaChem's DUCC implementation as an external MPI process, skipping
    ExaChem's internal SCF by providing pre-computed MO coefficients and
    density matrices. ExaChem performs Cholesky decomposition → CCSD → DUCC
    on the supplied orbitals, producing a downfolded active-space Hamiltonian
    in FCIDUMP format.

    Settings:
        atoms (list[str]): Atom coordinate lines, e.g. ``["H 0.0 0.0 0.0", "O 0.0 0.0 1.0"]``.
        basis (str): Gaussian basis set name (default: ``"cc-pvdz"``).
        charge (int): Molecular charge (default: 0).
        multiplicity (int): Spin multiplicity 2S+1 (default: 1).
        units (str): Coordinate units, ``"angstrom"`` or ``"bohr"`` (default: ``"angstrom"``).
        nactive_oa (int): Number of active occupied alpha orbitals (default: 0).
        nactive_ob (int): Number of active occupied beta orbitals (default: 0).
        nactive_va (int): Number of active virtual alpha orbitals (default: 0).
        nactive_vb (int): Number of active virtual beta orbitals (default: 0).
        ducc_level (int): DUCC truncation level (default: 2).
        mpi_ranks (int): Number of MPI processes (default: 1).
        exachem_binary (str): Path to ExaChem binary, or empty for auto-detect (default: ``""``).
        work_dir (str): Working directory, or empty for a temp dir (default: ``""``).
        timeout (int): Subprocess timeout in seconds (default: 3600).
        scf_type (str): SCF type, only ``"restricted"`` is supported (default: ``"restricted"``).
        ccsd_threshold (float): CCSD convergence threshold (default: 1e-6).

    Examples:
        >>> import numpy as np
        >>> solver = ExachemDuccSolver()
        >>> solver.settings().set("atoms", ["H 0.0 0.0 0.0", "H 0.0 0.0 0.74"])
        >>> solver.settings().set("basis", "cc-pvdz")
        >>> solver.settings().set("nactive_oa", 1)
        >>> solver.settings().set("nactive_va", 1)
        >>> result = solver.run(
        ...     mo_coeff_alpha=C_alpha, density_alpha=D_alpha,
        ... )  # returns FcidumpData

    """

    def __init__(self):
        super().__init__()
        s = self._settings
        s._set_default("atoms", "vector<string>", [])
        s._set_default("basis", "string", "cc-pvdz")
        s._set_default("charge", "int", 0)
        s._set_default("multiplicity", "int", 1)
        s._set_default("units", "string", "angstrom")
        s._set_default("nactive_oa", "int", 0)
        s._set_default("nactive_ob", "int", 0)
        s._set_default("nactive_va", "int", 0)
        s._set_default("nactive_vb", "int", 0)
        s._set_default("ducc_level", "int", 2)
        s._set_default("mpi_ranks", "int", 1)
        s._set_default("exachem_binary", "string", "")
        s._set_default("work_dir", "string", "")
        s._set_default("timeout", "int", 3600)
        s._set_default("scf_type", "string", "restricted")
        s._set_default("ccsd_threshold", "double", 1e-6)

    def type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def name(self) -> str:
        """Return ``"exachem_ducc"``."""
        return "exachem_ducc"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["exachem_ducc", "ducc"]

    def _run_impl(self, *args, **kwargs):
        """Run ExaChem DUCC with pre-computed SCF data and return downfolded Hamiltonian.

        Requires pre-computed MO coefficients and density matrices as keyword
        arguments. ExaChem skips its internal SCF and runs
        Cholesky decomposition → CCSD → DUCC on the provided orbitals.

        The molecule geometry (``atoms``) and ``basis`` must still be set in
        Settings so ExaChem can construct the basis set and integrals.

        Keyword Args:
            mo_coeff_alpha (numpy.ndarray): Alpha MO coefficients, shape ``(nbf, nmo)``. Required.
            density_alpha (numpy.ndarray): Alpha AO density matrix, shape ``(nbf, nbf)``. Required.

        Returns:
            :class:`~qdk_chemistry.data.Hamiltonian` containing the downfolded active-space integrals.

        Raises:
            ExachemNotFoundError: If ExaChem or MPI launcher is not found.
            ExachemRunError: If ExaChem fails.
            ValueError: If required arguments are missing.

        """
        s = self._settings
        atoms = s.get("atoms")
        if not atoms:
            raise ValueError("No atoms configured. Set settings 'atoms' to a list of coordinate strings.")

        mo_coeff_alpha = kwargs.get("mo_coeff_alpha")
        density_alpha = kwargs.get("density_alpha")
        if mo_coeff_alpha is None or density_alpha is None:
            raise ValueError("mo_coeff_alpha and density_alpha are required keyword arguments.")

        if s.get("scf_type") != "restricted":
            raise ValueError(
                "ExaChem DUCC currently only supports closed-shell (restricted) calculations. "
                "Open-shell support is not yet implemented in ExaChem's DUCC module."
            )

        config = DuccInputConfig(
            atoms=atoms,
            basis=s.get("basis"),
            charge=s.get("charge"),
            multiplicity=s.get("multiplicity"),
            units=s.get("units"),
            nactive_oa=s.get("nactive_oa"),
            nactive_ob=s.get("nactive_ob"),
            nactive_va=s.get("nactive_va"),
            nactive_vb=s.get("nactive_vb"),
            ducc_level=s.get("ducc_level"),
            ccsd_threshold=s.get("ccsd_threshold"),
            scf_type=s.get("scf_type"),
            noscf=True,
        )

        binary = s.get("exachem_binary") or None
        work = s.get("work_dir") or None
        work_path = Path(work) if work else None

        import tempfile

        if work_path is None:
            work_path = Path(tempfile.mkdtemp(prefix="exachem_ducc_"))
        work_path.mkdir(parents=True, exist_ok=True)

        scf_prefix_name = f"ducc_input.{s.get('basis')}"
        scf_type_dir = work_path / f"{scf_prefix_name}_files" / s.get("scf_type")
        scf_dir = scf_type_dir / "scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        scf_files_prefix = scf_dir / scf_prefix_name

        # Run context JSON goes one level up from scf/ (at the scf_type level)
        runcontext_prefix = scf_type_dir / scf_prefix_name

        # Build basis set for AO reordering
        from qdk_chemistry.data import BasisSet, Element, Structure

        atom_coords = []
        atom_elements = []
        for line in atoms:
            parts = line.split()
            atom_elements.append(getattr(Element, parts[0]))
            atom_coords.append([float(x) for x in parts[1:4]])
        coords_np = np.array(atom_coords)
        if s.get("units").lower() == "angstrom":
            coords_np = coords_np / 0.529177249
        structure = Structure(coords_np, atom_elements)
        basis_set = BasisSet.from_basis_name(s.get("basis"), structure)

        # ExaChem expects the TOTAL density (alpha + beta), not alpha-only.
        # For restricted closed-shell: D_total = 2 * D_alpha
        density_for_export = density_alpha * 2.0

        export_scf_files(
            files_prefix=scf_files_prefix,
            mo_coeff_alpha=mo_coeff_alpha,
            density_alpha=density_for_export,
            ao_tilesize=30,
            runcontext_prefix=runcontext_prefix,
            basis_set=basis_set,
        )
        logger.info("Exported SCF data for noscf mode to %s", scf_dir)

        result: ExachemResult = run_exachem(
            config,
            nprocs=s.get("mpi_ranks"),
            work_dir=work_path,
            exachem_binary=Path(binary) if binary else None,
            timeout=s.get("timeout"),
            scf_files_prefix=scf_files_prefix,
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
                f"ExaChem completed but no DUCC output files were produced. Check {result.work_dir} for output files."
            )

        # Extract the DUCC energy shift from ExaChem's stdout.
        # ExaChem's "Total Energy Shift" is initialized as (E_SCF - V_nuc) then
        # modified by fully contracted DUCC corrections. It excludes nuclear
        # repulsion. The correct core_energy for a qdk Hamiltonian is:
        #   core_energy = Total_Energy_Shift + V_nuc
        # so that E_total = E_CI_active(from MACIS) + core_energy.
        total_energy_shift = None
        for line in result.stdout.split("\n"):
            if "Total Energy Shift" in line:
                try:
                    total_energy_shift = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        if total_energy_shift is not None:
            core_energy = total_energy_shift + fcidump.nuclear_repulsion
        else:
            core_energy = fcidump.nuclear_repulsion
            logger.warning("Could not extract Total Energy Shift from ExaChem stdout; using nuc_rep only")

        # Convert to Hamiltonian
        hamiltonian = fcidump_to_hamiltonian(
            fcidump,
            atoms=list(s.get("atoms")),
            basis=s.get("basis"),
            units=s.get("units"),
            core_energy_override=core_energy,
        )

        # Clean up temporary files unless a work_dir was explicitly provided
        if not s.get("work_dir"):
            import shutil

            shutil.rmtree(result.work_dir, ignore_errors=True)
            logger.debug("Cleaned up work directory %s", result.work_dir)

        return hamiltonian
