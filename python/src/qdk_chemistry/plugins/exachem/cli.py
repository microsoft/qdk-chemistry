# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""CLI utilities for locating and running the ExaChem binary.

ExaChem requires MPI (via TAMM/Global Arrays) and runs as an external process.
This module handles binary discovery, input file generation, and subprocess
management.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class ExachemNotFoundError(RuntimeError):
    """Raised when the ExaChem binary cannot be located."""


class ExachemRunError(RuntimeError):
    """Raised when ExaChem exits with a non-zero return code."""


def find_exachem_binary() -> Path:
    """Locate the ExaChem binary.

    Resolution order:
        1. ``EXACHEM_PATH`` environment variable (full path to binary)
        2. ``ExaChem`` on ``PATH``

    Returns:
        Path to the ExaChem binary.

    Raises:
        ExachemNotFoundError: If the binary cannot be found.

    """
    env_path = os.environ.get("EXACHEM_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise ExachemNotFoundError(f"EXACHEM_PATH={env_path!r} does not point to an executable file.")

    which = shutil.which("ExaChem")
    if which:
        return Path(which)

    raise ExachemNotFoundError(
        "ExaChem binary not found. Install ExaChem and either add it to PATH "
        "or set EXACHEM_PATH to the full binary path."
    )


def find_mpi_launcher() -> list[str]:
    """Find an MPI launcher (``mpirun`` or ``srun``).

    Returns:
        Command prefix list, e.g. ``["mpirun"]`` or ``["srun"]``.

    Raises:
        ExachemNotFoundError: If no MPI launcher is available.

    """
    for launcher in ("mpirun", "srun"):
        if shutil.which(launcher):
            return [launcher]
    raise ExachemNotFoundError("No MPI launcher found (tried mpirun, srun). Install an MPI runtime.")


@dataclass
class DuccInputConfig:
    """Configuration for an ExaChem DUCC calculation.

    Attributes:
        atoms: List of atom lines, e.g. ``["H 0.0 0.0 0.0", "O 0.0 0.0 1.0"]``.
        basis: Gaussian basis set name, e.g. ``"cc-pvdz"``.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        units: Coordinate units (``"angstrom"`` or ``"bohr"``).
        nactive_oa: Number of active occupied alpha orbitals.
        nactive_ob: Number of active occupied beta orbitals.
        nactive_va: Number of active virtual alpha orbitals.
        nactive_vb: Number of active virtual beta orbitals.
        ducc_level: DUCC truncation level (default 2).
        ccsd_threshold: CCSD convergence threshold.
        ccsd_maxiter: Maximum CCSD iterations.
        scf_type: SCF type (``"restricted"`` or ``"unrestricted"``).
        extra_cc_options: Additional CC block options merged into the input.
        extra_scf_options: Additional SCF block options merged into the input.

    """

    atoms: list[str] = field(default_factory=list)
    basis: str = "cc-pvdz"
    charge: int = 0
    multiplicity: int = 1
    units: str = "angstrom"
    nactive_oa: int = 0
    nactive_ob: int = 0
    nactive_va: int = 0
    nactive_vb: int = 0
    ducc_level: int = 2
    ccsd_threshold: float = 1e-6
    ccsd_maxiter: int = 100
    cd_diagtol: float = 1e-5
    scf_type: str = "restricted"
    noscf: bool = False
    input_prefix: str = "ducc_input"
    extra_cc_options: dict = field(default_factory=dict)
    extra_scf_options: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Convert to ExaChem JSON input format."""
        input_dict = {
            "geometry": {
                "coordinates": list(self.atoms),
                "units": self.units,
            },
            "basis": {
                "basisset": self.basis,
            },
            "common": {
                "maxiter": 100,
            },
            "SCF": {
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "conve": 1e-9,
                "convd": 1e-8,
                "scf_type": self.scf_type,
                **(({"noscf": True}) if self.noscf else {}),
                **self.extra_scf_options,
            },
            "CD": {
                "diagtol": self.cd_diagtol,
            },
            "CC": {
                "threshold": self.ccsd_threshold,
                "ccsd_maxiter": self.ccsd_maxiter,
                "nactive_oa": self.nactive_oa,
                "nactive_ob": self.nactive_ob,
                "nactive_va": self.nactive_va,
                "nactive_vb": self.nactive_vb,
                "ducc_lvl": self.ducc_level,
                "writet": False,
                **self.extra_cc_options,
            },
            "TASK": {
                "scf": not self.noscf,
                "ducc": [True, "default"],
            },
        }
        return input_dict


@dataclass
class CcsdInputConfig:
    """Configuration for an ExaChem CCSD calculation that writes T amplitudes.

    Runs ExaChem's ``ccsd`` task (Coupled Cluster Singles and Doubles) and, when
    ``write_amplitudes`` is set, enables ExaChem's ``CC.PRINT.tamplitudes`` option
    so the converged T1/T2 amplitudes are written to text files
    (``<prefix>.print_t1amp.txt`` and ``<prefix>.print_t2amp.txt``).  No
    perturbative-triples ``(T)`` step is performed, so the T amplitudes ExaChem
    writes are exactly the converged CCSD amplitudes.

    Attributes:
        atoms: List of atom lines, e.g. ``["H 0.0 0.0 0.0", "O 0.0 0.0 1.0"]``.
        basis: Gaussian basis set name, e.g. ``"cc-pvdz"``.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        units: Coordinate units (``"angstrom"`` or ``"bohr"``).
        ccsd_threshold: CCSD convergence threshold.
        ccsd_maxiter: Maximum CCSD iterations.
        cd_diagtol: Cholesky decomposition diagonal tolerance.
        freeze_core: Number of frozen core orbitals (0 = none).
        freeze_virtual: Number of frozen virtual orbitals (0 = none).
        scf_type: SCF type (``"restricted"`` or ``"unrestricted"``).
        noscf: Whether ExaChem should skip its internal SCF and restart.
        write_amplitudes: Enable ``CC.PRINT.tamplitudes`` to write T1/T2 to text.
        amplitude_threshold: Only amplitudes with absolute value strictly greater
            than this are written (``0.0`` writes every nonzero amplitude).
        input_prefix: Base name for the ExaChem input file and restart directory.
        extra_cc_options: Additional CC block options merged into the input.
        extra_scf_options: Additional SCF block options merged into the input.

    """

    atoms: list[str] = field(default_factory=list)
    basis: str = "cc-pvdz"
    charge: int = 0
    multiplicity: int = 1
    units: str = "angstrom"
    ccsd_threshold: float = 1e-6
    ccsd_maxiter: int = 100
    cd_diagtol: float = 1e-5
    freeze_core: int = 0
    freeze_virtual: int = 0
    scf_type: str = "restricted"
    noscf: bool = False
    write_amplitudes: bool = True
    amplitude_threshold: float = 0.0
    input_prefix: str = "ccsd_input"
    extra_cc_options: dict = field(default_factory=dict)
    extra_scf_options: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Convert to ExaChem JSON input format."""
        cc_block: dict = {
            "threshold": self.ccsd_threshold,
            "ccsd_maxiter": self.ccsd_maxiter,
            "writet": False,
        }
        if self.write_amplitudes:
            cc_block["PRINT"] = {"tamplitudes": [True, self.amplitude_threshold]}
        if self.freeze_core or self.freeze_virtual:
            cc_block["freeze"] = {
                "core": self.freeze_core,
                "virtual": self.freeze_virtual,
            }
        cc_block.update(self.extra_cc_options)

        return {
            "geometry": {
                "coordinates": list(self.atoms),
                "units": self.units,
            },
            "basis": {
                "basisset": self.basis,
            },
            "common": {
                "maxiter": 100,
            },
            "SCF": {
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "conve": 1e-9,
                "convd": 1e-8,
                "scf_type": self.scf_type,
                **(({"noscf": True}) if self.noscf else {}),
                **self.extra_scf_options,
            },
            "CD": {
                "diagtol": self.cd_diagtol,
            },
            "CC": cc_block,
            "TASK": {
                "scf": not self.noscf,
                "ccsd": True,
            },
        }


@dataclass
class ExachemResult:
    """Result from an ExaChem DUCC run.

    Attributes:
        input_json: Path to the input JSON file.
        work_dir: Working directory where ExaChem ran.
        fcidump_path: Path to the FCIDUMP output file (if found).
        ducc_results_path: Path to the DUCC results text file (if found).
        ducc_json_path: Path to the DUCC JSON output file (if found).
        stdout: Captured standard output.
        stderr: Captured standard error.
        returncode: Process exit code.

    """

    input_json: Path
    work_dir: Path
    fcidump_path: Path | None
    ducc_results_path: Path | None
    ducc_json_path: Path | None
    stdout: str
    stderr: str
    returncode: int


def run_exachem(
    config: DuccInputConfig | CcsdInputConfig,
    *,
    nprocs: int = 1,
    work_dir: Path | None = None,
    exachem_binary: Path | None = None,
    mpi_extra_args: list[str] | None = None,
    timeout: int | None = None,
    scf_files_prefix: Path | None = None,
    libint_data_path: Path | None = None,
) -> ExachemResult:
    """Run an ExaChem DUCC or CCSD calculation.

    Args:
        config: ExaChem input configuration (:class:`DuccInputConfig` or :class:`CcsdInputConfig`).
        nprocs: Number of MPI processes.
        work_dir: Working directory. If None, creates a temporary directory.
        exachem_binary: Path to ExaChem binary. If None, auto-detects.
        mpi_extra_args: Extra arguments for the MPI launcher (e.g. ``["--bind-to", "core"]``).
        timeout: Timeout in seconds for the subprocess.
        libint_data_path: If given, set ``LIBINT_DATA_PATH`` for the ExaChem
            subprocess so Libint2 reads the basis from
            ``<libint_data_path>/basis/<name>.g94`` (see
            :func:`~qdk_chemistry.plugins.exachem.scf_export.write_qdk_basis_g94`).

    Returns:
        ExachemResult with paths to outputs and captured output.

    Raises:
        ExachemNotFoundError: If ExaChem or MPI launcher cannot be found.
        ExachemRunError: If ExaChem exits with non-zero status.

    """
    if exachem_binary is None:
        exachem_binary = find_exachem_binary()

    cleanup_work_dir = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="exachem_ducc_"))
        cleanup_work_dir = True

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # If noscf mode, set up SCF restart files in the expected directory structure
    if config.noscf and scf_files_prefix is not None:
        _setup_noscf_directory(work_dir, config, scf_files_prefix)

    # Write input JSON
    input_path = work_dir / f"{config.input_prefix}.json"
    input_dict = config.to_json()
    input_path.write_text(json.dumps(input_dict, indent=2))

    # Build command
    launcher = find_mpi_launcher()
    cmd = [*launcher, "-np", str(nprocs)]
    if mpi_extra_args:
        cmd.extend(mpi_extra_args)
    cmd.extend([str(exachem_binary), str(input_path)])

    logger.info("Running ExaChem DUCC: %s", " ".join(cmd))

    run_env = None
    if libint_data_path is not None:
        run_env = {**os.environ, "LIBINT_DATA_PATH": str(libint_data_path)}
        logger.info("Using LIBINT_DATA_PATH=%s for ExaChem basis", libint_data_path)

    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=run_env,
    )

    # Locate output files
    fcidump_path = _find_fcidump(work_dir)
    ducc_results_path = _find_file(work_dir, "*ducc.results.txt")
    ducc_json_path = _find_file(work_dir, "*.ducc.json")

    exachem_result = ExachemResult(
        input_json=input_path,
        work_dir=work_dir,
        fcidump_path=fcidump_path,
        ducc_results_path=ducc_results_path,
        ducc_json_path=ducc_json_path,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )

    if result.returncode != 0:
        logger.error("ExaChem failed (rc=%d):\nstdout: %s\nstderr: %s", result.returncode, result.stdout, result.stderr)
        raise ExachemRunError(
            f"ExaChem exited with code {result.returncode}. "
            f"Check work_dir={work_dir} for details.\n"
            f"stderr: {result.stderr[:500]}"
        )

    if fcidump_path is None and ducc_results_path is None:
        logger.warning("ExaChem completed but no output files found in %s", work_dir)

    return exachem_result


def _setup_noscf_directory(work_dir: Path, config: DuccInputConfig | CcsdInputConfig, scf_files_prefix: Path) -> None:
    """Set up ExaChem directory structure for noscf restart.

    ExaChem expects SCF restart files at a specific path derived from the
    input file name and basis set::

        <work_dir>/<input_prefix>.<basis>_files/<scf_type>/scf/<input_prefix>.<basis>.<ext>

    This function creates that directory structure and copies the
    pre-computed SCF files into it.
    """
    import shutil

    prefix_name = f"{config.input_prefix}.{config.basis}"
    scf_dir = work_dir / f"{prefix_name}_files" / config.scf_type / "scf"
    scf_dir.mkdir(parents=True, exist_ok=True)

    target_prefix = scf_dir / prefix_name
    source_prefix = scf_files_prefix

    # Copy SCF restart files to the expected location
    for ext in (".alpha.movecs", ".alpha.density", ".beta.movecs", ".beta.density"):
        src = Path(str(source_prefix) + ext)
        if src.exists():
            dst = Path(str(target_prefix) + ext)
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))
                logger.debug("Copied SCF file %s -> %s", src, dst)


def _find_file(work_dir: Path, pattern: str) -> Path | None:
    """Search for a file matching a glob pattern in the work directory."""
    for match in work_dir.rglob(pattern):
        if match.is_file():
            return match
    return None


def _find_fcidump(work_dir: Path) -> Path | None:
    """Search for a FCIDUMP file in the work directory."""
    for pattern in ("*FCIDUMP*", "*fcidump*", "*.fcidump"):
        matches = list(work_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None
