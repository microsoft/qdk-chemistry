# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Parsers for ExaChem CCSD output formats.

ExaChem writes its converged coupled-cluster results in two places:

1. **stdout** -- the CCSD/CCSD(T) energy summary, parsed by
   :func:`parse_ccsdt_energy` into :class:`CcsdtEnergies`.
2. **T-amplitude text files** -- written when ``CC.PRINT.tamplitudes`` is
   enabled, parsed by :func:`parse_ccsd_amplitudes_restricted` and
   :func:`parse_ccsd_amplitudes_unrestricted`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class CcsdtEnergies:
    """Energies parsed from an ExaChem CCSD(T) run.

    All energies are absolute totals (including nuclear repulsion) in Hartree,
    except the ``*_correction`` and ``*_correlation`` fields. The perturbative
    triples result is ``ccsd_pt_total`` (the canonical "CCSD(T)" energy); the
    ``ccsd_bracket_t_*`` fields hold the related "CCSD[T]" bracket variant.

    Attributes:
        ccsd_correlation: CCSD correlation energy.
        ccsd_total: CCSD total energy.
        ccsd_bracket_t_correction: [T] correction energy (bracket variant).
        ccsd_bracket_t_total: CCSD[T] total energy.
        ccsd_pt_correction: (T) perturbative correction energy.
        ccsd_pt_total: CCSD(T) total energy (the canonical result).

    """

    ccsd_correlation: float | None = None
    ccsd_total: float | None = None
    ccsd_bracket_t_correction: float | None = None
    ccsd_bracket_t_total: float | None = None
    ccsd_pt_correction: float | None = None
    ccsd_pt_total: float | None = None


def parse_ccsdt_energy(stdout: str) -> CcsdtEnergies:
    """Parse CCSD and CCSD(T) energies from ExaChem stdout.

    ExaChem prints the CCSD and CCSD(T)/CCSD[T] energies as labeled lines (at
    15-digit precision). This extracts them into a :class:`CcsdtEnergies`.

    Args:
        stdout: Captured ExaChem stdout text.

    Returns:
        Parsed :class:`CcsdtEnergies`. Fields that could not be found are ``None``.

    """
    num = r"(-?\d+\.\d+)"
    patterns = {
        "ccsd_correlation": rf"\bCCSD correlation energy / hartree\s*=\s*{num}",
        "ccsd_total": rf"\bCCSD total energy / hartree\s*=\s*{num}",
        "ccsd_bracket_t_correction": rf"\bCCSD\[T\] correction energy / hartree\s*=\s*{num}",
        "ccsd_bracket_t_total": rf"\bCCSD\[T\] total energy / hartree\s*=\s*{num}",
        "ccsd_pt_correction": rf"\bCCSD\(T\) correction energy / hartree\s*=\s*{num}",
        "ccsd_pt_total": rf"\bCCSD\(T\) total energy / hartree\s*=\s*{num}",
    }
    energies = CcsdtEnergies()
    for field_name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            setattr(energies, field_name, float(match.group(1)))
    return energies


def _read_amplitude_file(path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    """Read a TAMM ``print_max_above_threshold`` text file into a dense array.

    Each non-empty line is ``idx0 idx1 ... value``; entries not listed are exact
    zeros (TAMM only writes amplitudes whose magnitude exceeds the print
    threshold).  The returned array has the given ``shape``.
    """
    arr = np.zeros(shape, dtype=float)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != len(shape) + 1:
                continue
            idx = tuple(int(p) for p in parts[:-1])
            arr[idx] = float(parts[-1])
    return arr


def parse_ccsd_amplitudes_restricted(
    t1_path: str | Path, t2_path: str | Path, nocc: int, nvir: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse restricted (RHF) CCSD T amplitudes from ExaChem text files.

    ExaChem stores T1 as ``(vir, occ)`` and T2 as ``(vir, vir, occ, occ)``.
    This returns them in the qdk-chemistry/PySCF layout: ``t1[i, a]`` with shape
    ``(nocc, nvir)`` and ``t2[i, j, a, b]`` with shape
    ``(nocc, nocc, nvir, nvir)`` (the closed-shell ``abab`` amplitude).

    Args:
        t1_path: Path to ``<prefix>.print_t1amp.txt``.
        t2_path: Path to ``<prefix>.print_t2amp.txt``.
        nocc: Number of correlated occupied (spatial) orbitals.
        nvir: Number of correlated virtual (spatial) orbitals.

    Returns:
        ``(t1, t2)`` dense arrays in qdk-chemistry layout.

    """
    t1_vo = _read_amplitude_file(t1_path, (nvir, nocc))
    t2_vvoo = _read_amplitude_file(t2_path, (nvir, nvir, nocc, nocc))
    t1 = np.ascontiguousarray(t1_vo.transpose(1, 0))
    t2 = np.ascontiguousarray(t2_vvoo.transpose(2, 3, 0, 1))
    return t1, t2


def parse_ccsd_amplitudes_unrestricted(
    t1_path: str | Path,
    t2_path: str | Path,
    noa: int,
    nob: int,
    nva: int,
    nvb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse unrestricted (UHF) CCSD T amplitudes from ExaChem MSO text files.

    ExaChem writes a single molecular-spin-orbital (MSO) tensor whose occupied
    block is ordered ``[alpha, beta]`` and whose virtual block is ordered
    ``[alpha, beta]``.  This splits the MSO tensor into the spin blocks and
    returns them in the qdk-chemistry/PySCF layout (occupied indices first):
    ``t1_aa`` ``(noa, nva)``, ``t1_bb`` ``(nob, nvb)``, ``t2_aaaa``
    ``(noa, noa, nva, nva)``, ``t2_abab`` ``(noa, nob, nva, nvb)``, ``t2_bbbb``
    ``(nob, nob, nvb, nvb)``.

    Args:
        t1_path: Path to ``<prefix>.print_t1amp.txt``.
        t2_path: Path to ``<prefix>.print_t2amp.txt``.
        noa: Number of correlated occupied alpha orbitals.
        nob: Number of correlated occupied beta orbitals.
        nva: Number of correlated virtual alpha orbitals.
        nvb: Number of correlated virtual beta orbitals.

    Returns:
        ``(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb)`` dense arrays.

    """
    no, nv = noa + nob, nva + nvb
    t1 = _read_amplitude_file(t1_path, (nv, no))  # (vir_mso, occ_mso)
    t2 = _read_amplitude_file(t2_path, (nv, nv, no, no))  # (v1, v2, o1, o2)
    va, vb = slice(0, nva), slice(nva, nv)
    oa, ob = slice(0, noa), slice(noa, no)
    t1_aa = np.ascontiguousarray(t1[va, oa].transpose(1, 0))
    t1_bb = np.ascontiguousarray(t1[vb, ob].transpose(1, 0))
    t2_aaaa = np.ascontiguousarray(t2[va, va, oa, oa].transpose(2, 3, 0, 1))
    t2_abab = np.ascontiguousarray(t2[va, vb, oa, ob].transpose(2, 3, 0, 1))
    t2_bbbb = np.ascontiguousarray(t2[vb, vb, ob, ob].transpose(2, 3, 0, 1))
    return t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb
