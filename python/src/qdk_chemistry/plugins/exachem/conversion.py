# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Parsers for ExaChem DUCC output formats.

Supports two output formats:

1. **DUCC results text** — ExaChem's native DUCC output with labeled integral
   blocks (IJ, IA, AB, IJKL, IJKA, IJAB, AIJB, IABC, ABCD).
2. **FCIDUMP** — standard integral exchange format.

Both are parsed into :class:`FcidumpData` which can be converted to a
:class:`~qdk_chemistry.data.Hamiltonian` via :func:`fcidump_to_hamiltonian`.

References:
    - Knowles & Handy, Comp. Phys. Comm. 54, 75 (1989).
    - N.P. Bauman et al., J. Chem. Phys. 151, 014107 (2019).

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FcidumpData:
    """Parsed FCIDUMP integral data.

    Attributes:
        norb: Number of spatial orbitals.
        nelec: Number of electrons.
        ms2: Spin projection (2*S).
        isym: Spatial symmetry of the state.
        orbsym: Orbital symmetry labels.
        uhf: Whether this is a UHF FCIDUMP (separate alpha/beta blocks).
        one_body: One-electron integrals ``h[p,q]``, shape ``(norb, norb)``.
        two_body: Two-electron integrals ``V[p,q,r,s]`` in chemist notation ``(pq|rs)``, shape ``(norb, norb, norb, norb)``.
        nuclear_repulsion: Nuclear repulsion energy (or frozen-core energy shift).
        one_body_beta: Beta one-electron integrals for UHF (None for RHF).
        two_body_bbbb: Beta-beta two-electron integrals for UHF (None for RHF).
        two_body_aabb: Alpha-beta two-electron integrals for UHF (None for RHF).

    """

    norb: int
    nelec: int
    ms2: int = 0
    isym: int = 1
    orbsym: list[int] = field(default_factory=list)
    uhf: bool = False
    one_body: np.ndarray = field(default_factory=lambda: np.array([]))
    two_body: np.ndarray = field(default_factory=lambda: np.array([]))
    nuclear_repulsion: float = 0.0
    one_body_beta: np.ndarray | None = None
    two_body_bbbb: np.ndarray | None = None
    two_body_aabb: np.ndarray | None = None


def parse_ducc_results(results_path: str | Path, json_path: str | Path) -> FcidumpData:
    """Parse ExaChem DUCC output into :class:`FcidumpData`.

    ExaChem's DUCC driver writes downfolded integrals in a block-labeled text
    format with blocks IJ, IA, AB (1e) and IJKL, IJKA, IJAB, AIJB, IABC,
    ABCD (2e). Indices are 1-based **spin-orbital** indices within the active
    space, with occupied spin orbitals first (alpha then beta) and virtual
    spin orbitals second.

    Args:
        results_path: Path to the ``*.ducc.results.txt`` file.
        json_path: Path to the ``*.ducc.json`` file (for metadata).

    Returns:
        FcidumpData with the downfolded active-space Hamiltonian in the
        spin-orbital basis.

    """
    results_path = Path(results_path)
    json_path = Path(json_path)

    with open(json_path) as f:
        meta = json.load(f)

    cc_input = meta["input"]["CC"]
    noa = cc_input["nactive_oa"]
    nob = cc_input["nactive_ob"]
    nva = cc_input["nactive_va"]
    nvb = cc_input["nactive_vb"]
    nocc = noa + nob  # total active occupied spin orbitals
    nvir = nva + nvb  # total active virtual spin orbitals
    norb = nocc + nvir  # total active spin orbitals
    nelec_alpha = meta["molecule"]["nelectrons_alpha"]
    nelec_beta = meta["molecule"]["nelectrons_beta"]
    # Active electrons = active occupied orbitals
    nelec = nocc
    nuc_rep = meta["output"]["SCF"]["nucl_rep_energy"]

    text = results_path.read_text()

    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))

    blocks = _parse_ducc_blocks(text)

    # 1e blocks: indices are 1-based within their subspace (occ or vir spin orbitals)
    # ExaChem prints only alpha-alpha elements. For restricted, beta = alpha.
    # IJ: occupied-occupied, map i -> i-1 (alpha), copy to i-1+noa (beta)
    for i, j, val in blocks.get("IJ", []):
        ia, ja = i - 1, j - 1
        h1[ia, ja] = val
        h1[ja, ia] = val
        # Beta-beta copy
        h1[noa + ia, noa + ja] = val
        h1[noa + ja, noa + ia] = val

    # IA: occupied-virtual, map i -> i-1, a -> nocc + a - 1
    for i, a, val in blocks.get("IA", []):
        ia, aa = i - 1, nocc + a - 1
        h1[ia, aa] = val
        h1[aa, ia] = val
        # Beta-beta copy
        h1[noa + ia, nva + aa] = val
        h1[nva + aa, noa + ia] = val

    # AB: virtual-virtual, map a -> nocc + a - 1
    for a, b, val in blocks.get("AB", []):
        aa, ba = nocc + a - 1, nocc + b - 1
        h1[aa, ba] = val
        h1[ba, aa] = val
        # Beta-beta copy
        h1[nva + aa, nva + ba] = val
        h1[nva + ba, nva + aa] = val

    # 2e blocks: ExaChem prints only specific alpha-beta patterns.
    # The printed integrals are in physicist notation <pq||rs> (antisymmetrized).
    # We store them in h2[p,q,r,s] and must restore full antisymmetry:
    #   h2[p,q,r,s] = -h2[q,p,r,s] = -h2[p,q,s,r] = h2[q,p,s,r]
    #   h2[p,q,r,s] = h2[r,s,p,q]
    _fill_2e(h2, blocks.get("IJKL", []), (0, 0, 0, 0), nocc)
    _fill_2e(h2, blocks.get("IJKA", []), (0, 0, 0, 1), nocc)
    _fill_2e(h2, blocks.get("IJAB", []), (0, 0, 1, 1), nocc)
    _fill_2e(h2, blocks.get("AIJB", []), (1, 0, 0, 1), nocc)
    _fill_2e(h2, blocks.get("IABC", []), (0, 1, 1, 1), nocc)
    _fill_2e(h2, blocks.get("ABCD", []), (1, 1, 1, 1), nocc)

    # Restore spin-flip symmetry for restricted (closed-shell) systems.
    # ExaChem prints only a subset of spin-orbital integrals (specific
    # alpha/beta patterns). For restricted, the spin-flipped integral
    # is identical: h2[flip(p), flip(q), flip(r), flip(s)] = h2[p,q,r,s].
    _restore_spin_flip_symmetry(h2, noa, nva, nocc)

    return FcidumpData(
        norb=norb,
        nelec=nelec,
        ms2=0,
        one_body=h1,
        two_body=h2,
        nuclear_repulsion=nuc_rep,
    )


def _parse_ducc_blocks(text: str) -> dict[str, list[tuple]]:
    """Parse block-labeled DUCC results into a dict of integral lists."""
    blocks: dict[str, list[tuple]] = {}
    current_block = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Begin ") and line.endswith(" Block"):
            current_block = line[6:-6].strip()
            blocks[current_block] = []
        elif line.startswith("End ") and line.endswith(" Block"):
            current_block = None
        elif current_block is not None and line:
            parts = line.split()
            if len(parts) >= 3:
                indices = tuple(int(x) for x in parts[:-1])
                val = float(parts[-1])
                blocks[current_block].append((*indices, val))

    return blocks


def _fill_2e(
    h2: np.ndarray,
    entries: list[tuple],
    subspace: tuple[int, int, int, int],
    noa: int,
) -> None:
    """Fill 2e integral array from a DUCC block with antisymmetry restoration.

    ExaChem prints only unique spin-orbital elements (specific alpha-beta
    patterns). This function stores the printed value and restores the full
    antisymmetric permutation symmetry of the 2e integrals:

    - ``h2[p,q,r,s] = -h2[q,p,r,s]`` (antisymmetry in bra)
    - ``h2[p,q,r,s] = -h2[p,q,s,r]`` (antisymmetry in ket)
    - ``h2[p,q,r,s] =  h2[r,s,p,q]`` (exchange symmetry)

    Args:
        h2: Target array, shape ``(norb, norb, norb, norb)``.
        entries: List of ``(i, j, k, l, val)`` tuples (1-based within subspace).
        subspace: Tuple of 4 ints, each 0 (occupied) or 1 (virtual).
        noa: Number of active occupied spin orbitals (offset for virtual indices).

    """
    offsets = tuple(noa * s for s in subspace)

    for *indices, val in entries:
        if len(indices) != 4:
            continue
        p = indices[0] - 1 + offsets[0]
        q = indices[1] - 1 + offsets[1]
        r = indices[2] - 1 + offsets[2]
        s = indices[3] - 1 + offsets[3]

        # Store with full antisymmetric permutation symmetry
        h2[p, q, r, s] = val
        h2[q, p, r, s] = -val
        h2[p, q, s, r] = -val
        h2[q, p, s, r] = val
        h2[r, s, p, q] = val
        h2[s, r, p, q] = -val
        h2[r, s, q, p] = -val
        h2[s, r, q, p] = val


def _restore_spin_flip_symmetry(h2: np.ndarray, noa: int, nva: int, nocc: int) -> None:
    """Restore spin-flip symmetry for restricted closed-shell 2e integrals.

    ExaChem prints only unique spin-orbital integrals (specific alpha/beta
    patterns) for restricted systems. This function fills the spin-flipped
    counterparts: ``h2[flip(p), flip(q), flip(r), flip(s)] = h2[p, q, r, s]``.

    The spin-flip mapping swaps alpha ↔ beta within each subspace:

    - Occupied: ``α_i ↔ β_i`` (index ``i ↔ noa + i``)
    - Virtual: ``α_j ↔ β_j`` (index ``nocc + j ↔ nocc + nva + j``)

    Args:
        h2: Spin-orbital 2e integral array, shape ``(norb, norb, norb, norb)``.
        noa: Number of alpha occupied spin orbitals.
        nva: Number of alpha virtual spin orbitals.
        nocc: Total occupied spin orbitals (``noa + nob``).

    """
    norb = h2.shape[0]

    def flip(idx: int) -> int:
        if idx < noa:
            return idx + noa
        if idx < nocc:
            return idx - noa
        if idx < nocc + nva:
            return idx + nva
        return idx - nva

    h2_orig = h2.copy()
    for p in range(norb):
        fp = flip(p)
        for q in range(norb):
            fq = flip(q)
            for r in range(norb):
                fr = flip(r)
                for s in range(norb):
                    fs = flip(s)
                    val = h2_orig[p, q, r, s]
                    if abs(val) > 1e-15:
                        h2[fp, fq, fr, fs] = val


def parse_fcidump(filepath: str | Path) -> FcidumpData:
    """Parse a FCIDUMP file into structured integral data.

    Supports both RHF (single integral block) and UHF (separated alpha/beta
    blocks) formats. For RHF, exploits 8-fold permutation symmetry when
    restoring the full integral arrays.

    Args:
        filepath: Path to the FCIDUMP file.

    Returns:
        FcidumpData with parsed integrals and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header cannot be parsed.

    """
    filepath = Path(filepath)
    text = filepath.read_text()

    # Parse header
    header_match = re.search(r"&FCI(.*?)&END", text, re.DOTALL | re.IGNORECASE)
    if not header_match:
        raise ValueError(f"Could not find &FCI...&END header in {filepath}")

    header = header_match.group(1)
    norb = _parse_header_int(header, "NORB")
    nelec = _parse_header_int(header, "NELEC")
    ms2 = _parse_header_int(header, "MS2", default=0)
    isym = _parse_header_int(header, "ISYM", default=1)
    uhf = _parse_header_int(header, "IUHF", default=0) == 1

    orbsym_match = re.search(r"ORBSYM\s*=\s*([\d,\s]+)", header, re.IGNORECASE)
    orbsym = []
    if orbsym_match:
        orbsym_str = orbsym_match.group(1).strip().rstrip(",")
        if orbsym_str:
            orbsym = [int(x) for x in re.split(r"[,\s]+", orbsym_str) if x.strip()]

    # Parse integral lines (everything after &END)
    body = text[header_match.end() :].strip()
    lines = body.splitlines()

    if uhf:
        return _parse_uhf_integrals(lines, norb, nelec, ms2, isym, orbsym)
    return _parse_rhf_integrals(lines, norb, nelec, ms2, isym, orbsym)


def _parse_header_int(header: str, key: str, *, default: int | None = None) -> int:
    """Extract an integer value from the FCIDUMP header."""
    match = re.search(rf"{key}\s*=\s*(-?\d+)", header, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Required header field {key} not found in FCIDUMP")


def _parse_rhf_integrals(
    lines: list[str], norb: int, nelec: int, ms2: int, isym: int, orbsym: list[int]
) -> FcidumpData:
    """Parse RHF FCIDUMP integrals with 8-fold symmetry restoration."""
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    e_nuc = 0.0

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        val = float(parts[0])
        i, j, k, l = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1

        if i == j == k == l == -1:
            # Nuclear repulsion / frozen core energy
            e_nuc = val
        elif k == l == -1:
            # One-electron integral h[i,j]
            h1[i, j] = val
            h1[j, i] = val  # Hermitian symmetry
        else:
            # Two-electron integral (ij|kl) in chemist notation
            # Restore 8-fold symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
            #                        = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
            h2[i, j, k, l] = val
            h2[j, i, k, l] = val
            h2[i, j, l, k] = val
            h2[j, i, l, k] = val
            h2[k, l, i, j] = val
            h2[l, k, i, j] = val
            h2[k, l, j, i] = val
            h2[l, k, j, i] = val

    return FcidumpData(
        norb=norb,
        nelec=nelec,
        ms2=ms2,
        isym=isym,
        orbsym=orbsym,
        uhf=False,
        one_body=h1,
        two_body=h2,
        nuclear_repulsion=e_nuc,
    )


def _parse_uhf_integrals(
    lines: list[str], norb: int, nelec: int, ms2: int, isym: int, orbsym: list[int]
) -> FcidumpData:
    """Parse UHF FCIDUMP integrals (alpha-alpha, beta-beta, alpha-beta blocks)."""
    h2_aaaa = np.zeros((norb, norb, norb, norb))
    h2_bbbb = np.zeros((norb, norb, norb, norb))
    h2_aabb = np.zeros((norb, norb, norb, norb))
    h1_aa = np.zeros((norb, norb))
    h1_bb = np.zeros((norb, norb))
    e_nuc = 0.0

    # UHF FCIDUMP has 6 sections separated by zero-index sentinel lines:
    # 1) alpha-alpha 2e integrals
    # 2) sentinel (0.0  0 0 0 0)
    # 3) beta-beta 2e integrals
    # 4) sentinel
    # 5) alpha-beta 2e integrals
    # 6) sentinel
    # 7) alpha 1e integrals
    # 8) sentinel
    # 9) beta 1e integrals
    # 10) nuclear repulsion (0 0 0 0)

    section = 0  # 0=aaaa-2e, 1=bbbb-2e, 2=aabb-2e, 3=aa-1e, 4=bb-1e
    targets_2e = [h2_aaaa, h2_bbbb, h2_aabb]
    targets_1e = [h1_aa, h1_bb]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        val = float(parts[0])
        i, j, k, l = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1

        if i == j == k == l == -1:
            if section < 3:
                # Sentinel between 2e blocks
                section += 1 if section < 2 else 1
                # After 3rd sentinel, switch to 1e
                if section == 3:
                    continue
            elif section == 3:
                section = 4
            else:
                e_nuc = val
            continue

        if k == l == -1:
            # One-electron integral
            idx_1e = section - 3 if section >= 3 else 0
            if 0 <= idx_1e < len(targets_1e):
                targets_1e[idx_1e][i, j] = val
                targets_1e[idx_1e][j, i] = val
        elif section < 3:
            # Two-electron integral
            t = targets_2e[section]
            t[i, j, k, l] = val
            t[j, i, k, l] = val
            t[i, j, l, k] = val
            t[j, i, l, k] = val
            t[k, l, i, j] = val
            t[l, k, i, j] = val
            t[k, l, j, i] = val
            t[l, k, j, i] = val

    return FcidumpData(
        norb=norb,
        nelec=nelec,
        ms2=ms2,
        isym=isym,
        orbsym=orbsym,
        uhf=True,
        one_body=h1_aa,
        two_body=h2_aaaa,
        nuclear_repulsion=e_nuc,
        one_body_beta=h1_bb,
        two_body_bbbb=h2_bbbb,
        two_body_aabb=h2_aabb,
    )


def spinorb_to_spatial(fcidump: FcidumpData) -> FcidumpData:
    """Convert spin-orbital integrals to spatial-orbital integrals.

    For closed-shell restricted systems, the spin-orbital Hamiltonian has
    a block structure where alpha-alpha = beta-beta (1e) and 2e integrals
    decompose into spatial Coulomb and exchange integrals. This function
    extracts the spatial-orbital integrals from the spin-orbital ones.

    The spin-orbital ordering is assumed to be interleaved:
    ``[α₁, β₁, α₂, β₂, ...]`` (alpha/beta alternating for each spatial orbital).

    The returned 2e integrals are in **chemist notation** ``(pq|rs)``
    (Coulomb integrals), not physicist notation ``<pq||rs>``.

    Args:
        fcidump: Spin-orbital FcidumpData (e.g. from :func:`parse_ducc_results`).

    Returns:
        New FcidumpData in the spatial-orbital basis.

    """
    norb_spin = fcidump.norb
    if norb_spin % 2 != 0:
        raise ValueError(f"Expected even number of spin-orbitals, got {norb_spin}")
    norb = norb_spin // 2
    nelec = fcidump.nelec // 2  # electrons per spin channel

    h1_spin = fcidump.one_body
    h2_spin = fcidump.two_body

    # Spin-orbital ordering from DUCC parser:
    # [α_occ₁, ..., α_occN, β_occ₁, ..., β_occN, α_vir₁, ..., α_virM, β_vir₁, ..., β_virM]
    # where N = noa = nob, M = nva = nvb (closed-shell restricted)
    #
    # Spatial orbital i maps to:
    #   alpha spin-orbital: i (if occupied), or nocc + i - nocc_spatial (if virtual)
    #   beta spin-orbital:  noa + i (if occupied), or nocc + nva + i - nocc_spatial (if virtual)
    #
    # For nocc_spin = noa + nob = 2*noa, nvir_spin = nva + nvb = 2*nva:
    # spatial occ orbital i (0..noa-1): alpha = i, beta = noa + i
    # spatial vir orbital j (0..nva-1): alpha = nocc + j, beta = nocc + nva + j

    noa = norb_spin // 4  # assuming noa = nob, nva = nvb and norb_spin = 2*(noa+nva)
    # Actually we need the actual split. nelec gives nocc_spin.
    nocc = fcidump.nelec  # total occupied spin-orbitals
    noa_val = nocc // 2
    nvir = norb_spin - nocc
    nva_val = nvir // 2

    def alpha_idx(spatial_i):
        """Map spatial orbital index to alpha spin-orbital index."""
        if spatial_i < noa_val:
            return spatial_i  # occupied alpha
        return nocc + (spatial_i - noa_val)  # virtual alpha

    def beta_idx(spatial_i):
        """Map spatial orbital index to beta spin-orbital index."""
        if spatial_i < noa_val:
            return noa_val + spatial_i  # occupied beta
        return nocc + nva_val + (spatial_i - noa_val)  # virtual beta

    # 1e: spatial h[p,q] = spin h[alpha_p, alpha_q]
    h1 = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            h1[p, q] = h1_spin[alpha_idx(p), alpha_idx(q)]

    # 2e: convert physicist → chemist via cross-spin extraction.
    # Relation: (pq|rs)_chemist = <pr|qs>_physicist.
    # For cross-spin, the exchange term vanishes:
    #   <p_α r_β | q_α s_β> = <p_α r_β || q_α s_β> = (pq|rs)_chemist
    h2 = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    h2[p, q, r, s] = h2_spin[alpha_idx(p), beta_idx(r), alpha_idx(q), beta_idx(s)]

    return FcidumpData(
        norb=norb,
        nelec=nelec * 2,  # total electrons
        ms2=0,
        one_body=h1,
        two_body=h2,
        nuclear_repulsion=fcidump.nuclear_repulsion,
    )


def fcidump_to_hamiltonian(
    fcidump: FcidumpData,
    atoms: list[str],
    basis: str,
    units: str = "angstrom",
    from_spinorb: bool = True,
    core_energy_override: float | None = None,
):
    """Convert FCIDUMP data to a qdk-chemistry Hamiltonian.

    Constructs a :class:`~qdk_chemistry.data.Hamiltonian` from FCIDUMP
    integrals. If the input is in the spin-orbital basis (as produced by
    :func:`parse_ducc_results`), it is first converted to spatial orbitals
    via :func:`spinorb_to_spatial`.

    The ``atoms``, ``basis``, and ``units`` are needed to build the
    :class:`~qdk_chemistry.data.BasisSet` and
    :class:`~qdk_chemistry.data.Orbitals` that the container requires. The
    MO coefficients are set to identity (the DUCC integrals are already in
    the active-space MO basis).

    Args:
        fcidump: Parsed FCIDUMP data from :func:`parse_fcidump` or :func:`parse_ducc_results`.
        atoms: Atom coordinate lines, e.g. ``["H 0.0 0.0 0.0", "O 0.0 0.0 1.0"]``.
        basis: Gaussian basis set name, e.g. ``"cc-pvdz"``.
        units: Coordinate units, ``"angstrom"`` or ``"bohr"`` (default: ``"angstrom"``).
        from_spinorb: If True, convert spin-orbital integrals to spatial before constructing the Hamiltonian (default: True).

    Returns:
        A :class:`~qdk_chemistry.data.Hamiltonian` instance in the spatial-orbital basis.

    """
    from qdk_chemistry.data import (
        BasisSet,
        CanonicalFourCenterHamiltonianContainer,
        Element,
        Hamiltonian,
        Orbitals,
        Structure,
    )

    # Parse atom lines into coordinates and elements
    coords = []
    elements = []
    for line in atoms:
        parts = line.split()
        elements.append(getattr(Element, parts[0]))
        coords.append([float(x) for x in parts[1:4]])

    coords_np = np.array(coords)
    if units.lower() == "angstrom":
        coords_np = coords_np / 0.529177249  # Convert to Bohr

    structure = Structure(coords_np, elements)
    basis_set = BasisSet.from_basis_name(basis, structure)

    # Convert spin-orbital → spatial if needed
    if from_spinorb:
        fcidump = spinorb_to_spatial(fcidump)

    norb = fcidump.norb
    mo_coeff = np.eye(norb)
    orbitals = Orbitals(mo_coeff, None, None, basis_set)
    zero_fock = np.zeros((norb, norb))

    core_energy = core_energy_override if core_energy_override is not None else fcidump.nuclear_repulsion

    if fcidump.uhf:
        container = CanonicalFourCenterHamiltonianContainer(
            fcidump.one_body,
            fcidump.one_body_beta,
            fcidump.two_body.ravel(),
            fcidump.two_body_aabb.ravel(),
            fcidump.two_body_bbbb.ravel(),
            orbitals,
            core_energy,
            zero_fock,
            zero_fock,
        )
    else:
        container = CanonicalFourCenterHamiltonianContainer(
            fcidump.one_body,
            fcidump.two_body.ravel(),
            orbitals,
            core_energy,
            zero_fock,
        )

    return Hamiltonian(container)
