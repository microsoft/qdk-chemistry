"""Parsers for ExaChem CCSD output formats.

ExaChem writes its converged coupled-cluster results in two places:

1. **stdout** -- the SCF and CCSD energies, parsed by
   :func:`parse_ccsd_energies` into :class:`CcsdEnergies`.
2. **T-amplitude text files** -- written when ``CC.PRINT.tamplitudes`` is
   enabled, parsed by :func:`parse_ccsd_amplitudes_restricted` and
   :func:`parse_ccsd_amplitudes_unrestricted`.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.utils import Logger

__all__ = [
    "CcsdEnergies",
    "parse_ccsd_amplitudes_restricted",
    "parse_ccsd_amplitudes_unrestricted",
    "parse_ccsd_energies",
]

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class CcsdEnergies:
    """Energies parsed from an ExaChem CCSD run, as totals (including nuclear repulsion) in Hartree."""

    scf_total: float | None = None
    """SCF total energy of the reference that ExaChem correlated."""

    ccsd_total: float | None = None
    """CCSD total energy."""


def parse_ccsd_energies(stdout: str) -> CcsdEnergies:
    """Parse the SCF and CCSD energies from ExaChem stdout.

    ExaChem prints the SCF and CCSD energies as labeled lines (at 15-digit
    precision). This extracts them into a :class:`CcsdEnergies`.

    Args:
        stdout: Captured ExaChem stdout text.

    Returns:
        Parsed :class:`CcsdEnergies`. Fields that could not be found are ``None``.

    """
    Logger.trace_entering()
    num = r"(-?\d+\.\d+)"
    patterns = {
        "scf_total": rf"\bTotal SCF energy\s*=\s*{num}",
        "ccsd_total": rf"\bCCSD total energy / hartree\s*=\s*{num}",
    }
    energies = CcsdEnergies()
    for field_name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            setattr(energies, field_name, float(match.group(1)))
    return energies


def _amplitude_entries(path: str | Path, rank: int) -> Iterator[tuple[tuple[int, ...], float]]:
    """Yield the ``(indices, value)`` entries of a TAMM ``print_max_above_threshold`` text file.

    Each non-empty line is ``idx0 idx1 ... value``; entries not listed are exact
    zeros (TAMM only writes amplitudes whose magnitude exceeds the print
    threshold).
    """
    Logger.trace_entering()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == rank + 1:
                yield tuple(map(int, parts[:-1])), float(parts[-1])


def _read_amplitude_file(path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    """Read a TAMM ``print_max_above_threshold`` text file into a dense array of the given ``shape``."""
    Logger.trace_entering()
    arr = np.zeros(shape, dtype=float)
    for idx, value in _amplitude_entries(path, len(shape)):
        arr[idx] = value
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
    Logger.trace_entering()
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
    Logger.trace_entering()
    t1_aa, t1_bb = np.zeros((noa, nva)), np.zeros((nob, nvb))
    for (a, i), value in _amplitude_entries(t1_path, 2):
        if a < nva and i < noa:
            t1_aa[i, a] = value
        elif a >= nva and i >= noa:
            t1_bb[i - noa, a - nva] = value

    t2_aaaa = np.zeros((noa, noa, nva, nva))
    t2_abab = np.zeros((noa, nob, nva, nvb))
    t2_bbbb = np.zeros((nob, nob, nvb, nvb))
    # Filling the blocks directly avoids a dense MSO tensor; the abba, baab and baba blocks repeat abab up to sign.
    for (a, b, i, j), value in _amplitude_entries(t2_path, 4):
        if a < nva and b < nva and i < noa and j < noa:
            t2_aaaa[i, j, a, b] = value
        elif a < nva <= b and i < noa <= j:
            t2_abab[i, j - noa, a, b - nva] = value
        elif a >= nva and b >= nva and i >= noa and j >= noa:
            t2_bbbb[i - noa, j - noa, a - nva, b - nva] = value
    return t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb
