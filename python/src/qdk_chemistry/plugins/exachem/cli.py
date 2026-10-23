"""CLI utilities for locating and running the ExaChem binary.

ExaChem requires MPI (via TAMM/Global Arrays) and runs as an external process.
This module handles binary discovery, input file generation, and subprocess
management.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from qdk_chemistry.utils import Logger

__all__ = [
    "CcsdInputConfig",
    "ExachemNotFoundError",
    "ExachemRunError",
    "find_exachem_binary",
    "find_mpi_launcher",
    "run_exachem",
]


class ExachemNotFoundError(RuntimeError):
    """Raised when the ExaChem binary or an MPI launcher is not on ``PATH``."""


class ExachemRunError(RuntimeError):
    """Raised when an ExaChem run fails, times out, or produces output that cannot be used."""


def find_exachem_binary() -> Path:
    """Locate the ExaChem binary on ``PATH``.

    Returns:
        Path to the ExaChem binary.

    Raises:
        ExachemNotFoundError: If the binary cannot be found.

    """
    Logger.trace_entering()
    which = shutil.which("ExaChem")
    if which:
        return Path(which)

    raise ExachemNotFoundError("ExaChem binary not found on PATH. Install ExaChem and add it to PATH.")


def find_mpi_launcher() -> list[str]:
    """Find an MPI launcher (``mpirun`` or ``srun``).

    Returns:
        Command prefix list, e.g. ``["mpirun"]`` or ``["srun"]``.

    Raises:
        ExachemNotFoundError: If no MPI launcher is available.

    """
    Logger.trace_entering()
    for launcher in ("mpirun", "srun"):
        if shutil.which(launcher):
            return [launcher]
    raise ExachemNotFoundError("No MPI launcher found (tried mpirun, srun). Install an MPI runtime.")


@dataclass
class CcsdInputConfig:
    """Configuration for an ExaChem CCSD calculation on exported SCF orbitals.

    Runs ExaChem's ``ccsd`` task (Coupled Cluster Singles and Doubles) on the SCF
    restart files in the working directory (``noscf``) and, when
    ``write_amplitudes`` is set, enables ExaChem's ``CC.PRINT.tamplitudes`` option
    so the converged T1/T2 amplitudes are written to text files
    (``<prefix>.print_t1amp.txt`` and ``<prefix>.print_t2amp.txt``).  No
    perturbative-triples ``(T)`` step is performed, so the T amplitudes ExaChem
    writes are exactly the converged CCSD amplitudes.
    """

    atoms: list[str]
    """List of atom lines in Bohr, e.g. ``["H 0.0 0.0 0.0", "O 0.0 0.0 1.8"]``."""

    basis: str
    """Gaussian basis set name, e.g. ``"cc-pvdz"``."""

    charge: int = 0
    """Molecular charge."""

    multiplicity: int = 1
    """Spin multiplicity (2S+1)."""

    ccsd_threshold: float = 1e-6
    """CCSD convergence threshold."""

    cd_diagtol: float = 1e-5
    """Cholesky decomposition diagonal tolerance."""

    freeze_core: int = 0
    """Number of frozen core orbitals (0 = none)."""

    freeze_virtual: int = 0
    """Number of frozen virtual orbitals (0 = none)."""

    scf_type: str = "restricted"
    """SCF type (``"restricted"`` or ``"unrestricted"``)."""

    write_amplitudes: bool = True
    """Enable ``CC.PRINT.tamplitudes`` to write T1/T2 to text."""

    input_prefix: str = "ccsd_input"
    """Base name for the ExaChem input file and restart directory."""

    atom_ecp: dict = field(default_factory=dict)
    """Element symbol to ECP name in ExaChem's library, for atoms that use an ECP."""

    def to_json(self) -> dict:
        """Convert to ExaChem JSON input format."""
        Logger.trace_entering()
        cc_block: dict = {
            "threshold": self.ccsd_threshold,
            "ccsd_maxiter": 100,
            "writet": False,
        }
        if self.write_amplitudes:
            # TAMM prints 12 decimals, so smaller amplitudes would only add lines that read as zero.
            cc_block["PRINT"] = {"tamplitudes": [True, 5e-13]}
        if self.freeze_core or self.freeze_virtual:
            cc_block["freeze"] = {
                "core": self.freeze_core,
                "virtual": self.freeze_virtual,
            }

        return {
            "geometry": {
                "coordinates": list(self.atoms),
                "units": "bohr",
            },
            "basis": {
                "basisset": self.basis,
                **({"atom_ecp": dict(self.atom_ecp)} if self.atom_ecp else {}),
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
                "noscf": True,
            },
            "CD": {
                "diagtol": self.cd_diagtol,
            },
            "CC": cc_block,
            "TASK": {
                "scf": False,
                "ccsd": True,
            },
        }


# srun spells binding differently from mpirun, so map the common policies.
_SRUN_BIND_POLICIES = {"core": "cores", "hwthread": "threads", "socket": "sockets", "none": "none"}


def _binding_args(launcher: str, bind_to: str) -> list[str]:
    """Translate a binding policy into launcher-specific arguments.

    Args:
        launcher: The launcher executable name, ``"mpirun"`` or ``"srun"``.
        bind_to: Binding policy such as ``"core"``; empty deferring to the launcher default.

    Returns:
        The arguments to append to the launcher invocation, empty when deferring.

    """
    Logger.trace_entering()
    if not bind_to:
        return []
    if launcher.endswith("srun"):
        policy = _SRUN_BIND_POLICIES.get(bind_to)
        if policy is None:
            Logger.warn(f"No srun equivalent for bind-to policy {bind_to!r}; using the srun default.")
            return []
        return [f"--cpu-bind={policy}"]
    return ["--bind-to", bind_to]


def run_exachem(
    config: CcsdInputConfig,
    *,
    nprocs: int,
    work_dir: Path,
    mpi_bind_to: str,
    timeout_secs: int,
    libint_data_path: Path,
) -> str:
    """Run an ExaChem CCSD calculation.

    Args:
        config: ExaChem input configuration (:class:`CcsdInputConfig`).
        nprocs: Number of MPI processes.
        work_dir: Existing working directory holding the SCF restart files; ExaChem writes its outputs here.
        mpi_bind_to: Binding policy per rank (e.g. ``"core"``); empty defers to the launcher default.
        timeout_secs: Maximum seconds to wait for ExaChem to finish.
        libint_data_path: Directory holding the basis and ECP files for ExaChem, exported as ``LIBINT_DATA_PATH``.

    Returns:
        ExaChem's standard output.

    Raises:
        ExachemNotFoundError: If ExaChem or an MPI launcher is not on ``PATH``.
        ExachemRunError: If ExaChem exits with non-zero status or does not finish within ``timeout_secs`` seconds.

    """
    Logger.trace_entering()
    exachem_binary = find_exachem_binary()
    launcher = find_mpi_launcher()

    input_path = Path(work_dir) / f"{config.input_prefix}.json"
    input_path.write_text(json.dumps(config.to_json(), indent=2))

    # mpirun, mpiexec and srun all take -n; srun has no -np.
    cmd = [*launcher, "-n", str(nprocs)]
    cmd.extend(_binding_args(launcher[0], mpi_bind_to))
    cmd.extend([str(exachem_binary), str(input_path)])

    Logger.info(f"Running ExaChem: {' '.join(cmd)}")

    # A copy of the parent environment: assigning into it never mutates os.environ,
    # so the calling process keeps its own OMP_NUM_THREADS. One OpenMP thread per
    # rank avoids oversubscribing the cores the ranks are bound to.
    run_env = {**os.environ, "OMP_NUM_THREADS": "1", "LIBINT_DATA_PATH": str(libint_data_path)}
    Logger.info(f"Using LIBINT_DATA_PATH={libint_data_path} for ExaChem basis and ECP files")

    process = subprocess.Popen(
        cmd, cwd=str(work_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=run_env
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_secs)
    except BaseException as error:
        # SIGTERM lets the launcher stop its ranks; subprocess.run's SIGKILL would orphan them.
        process.terminate()
        process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise ExachemRunError(f"ExaChem did not finish within {timeout_secs} s.") from None
        raise

    if process.returncode != 0:
        Logger.error(f"ExaChem failed (rc={process.returncode}):\nstdout: {stdout}\nstderr: {stderr}")
        output_tail = "\n".join(f"{stdout}\n{stderr}".strip().splitlines()[-5:])
        raise ExachemRunError(f"ExaChem exited with code {process.returncode}. Output ends with:\n{output_tail}")
    return stdout
