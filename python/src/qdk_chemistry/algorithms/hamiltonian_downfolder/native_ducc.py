# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Native DUCC Hamiltonian downfolding solver (no ExaChem binary required).

Implements the Double Unitary Coupled Cluster (DUCC) downfolding method using
PySCF for CCSD amplitudes and NumPy for tensor contractions. Produces a
Hermitian effective Hamiltonian for the active space that incorporates
dynamical correlation from external orbitals.

The BCH expansion formulas are auto-generated from ExaChem's ``ducc-t_ccsd.hpp``
to ensure exact correspondence. All internal contractions use physicist
antisymmetrized spin-orbital notation, matching ExaChem's convention. The output
is converted to chemist spatial-orbital notation via the existing
:func:`~qdk_chemistry.plugins.exachem.conversion.spinorb_to_spatial` and
:func:`~qdk_chemistry.plugins.exachem.conversion.fcidump_to_hamiltonian`.

References:
    - N.P. Bauman et al., J. Chem. Phys. 151, 014107 (2019)
    - K. Kowalski, J. Chem. Phys. 148, 094104 (2018)
"""

from __future__ import annotations

import logging

import numpy as np
from pyscf import cc as pyscf_cc

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hamiltonian_downfolder import bch_contractions
from qdk_chemistry.plugins.exachem.conversion import (
    FcidumpData,
    fcidump_to_hamiltonian,
    spinorb_to_spatial,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spin-orbital conversion helpers
# ---------------------------------------------------------------------------


def spatial_to_spinorb_h1(h1_spatial):
    """Expand spatial 1e integrals to spin-orbital basis.

    Args:
        h1_spatial: (nmo, nmo) spatial-orbital 1e integrals.

    Returns:
        (2*nmo, 2*nmo) spin-orbital 1e integrals.
        h1_so[2p+s1, 2q+s2] = h1[p,q] * delta(s1,s2)
    """
    nmo = h1_spatial.shape[0]
    nso = 2 * nmo
    h1_so = np.zeros((nso, nso))
    for s in range(2):
        h1_so[s::2, s::2] = h1_spatial
    return h1_so


def spatial_to_spinorb_eri(eri_spatial):
    """Expand spatial chemist ERI to spin-orbital physicist antisymmetrized form.

    Args:
        eri_spatial: (nmo, nmo, nmo, nmo) spatial ERI in chemist notation (pq|rs).

    Returns:
        (2*nmo, 2*nmo, 2*nmo, 2*nmo) physicist antisymmetrized <pq||rs>.
    """
    nmo = eri_spatial.shape[0]
    nso = 2 * nmo

    # Build spin-orbital Coulomb integral (chemist notation)
    # eri_so[2p+s1, 2q+s2, 2r+s3, 2s+s4] = eri[p,q,r,s] * delta(s1,s2) * delta(s3,s4)
    eri_so = np.zeros((nso, nso, nso, nso))
    for s1 in range(2):
        for s2 in range(2):
            eri_so[s1::2, s1::2, s2::2, s2::2] = eri_spatial

    # Convert chemist (pq|rs) to physicist antisymmetrized <pq||rs>
    # <pq||rs> = <pq|rs> - <pq|sr>
    # <pq|rs> = (pr|qs) in chemist notation
    # So: v2[p,q,r,s] = eri_so[p,r,q,s] - eri_so[p,s,q,r]
    # Use explicit einsum for clarity and correctness:
    v2 = np.einsum("prqs->pqrs", eri_so) - np.einsum("psqr->pqrs", eri_so)
    return v2


def spatial_to_spinorb_t1(t1_spatial):
    """Expand spatial T1 amplitudes to spin-orbital form.

    PySCF convention: t1[i,a] where i=occ, a=vir (spatial).
    ExaChem convention: t1[a,i] where a=vir_so, i=occ_so (spin-orbital).

    Note: ExaChem stores t1 as t1(virt, occ) = t1[e, m].

    Args:
        t1_spatial: (nocc, nvir) spatial T1 amplitudes.

    Returns:
        (nso, nso) spin-orbital T1 in full-array form t1_so[p,q].
        Only vir->occ elements are nonzero: t1_so[2a+s, 2i+s] = t1[i,a].
    """
    nocc, nvir = t1_spatial.shape
    nmo = nocc + nvir
    nso = 2 * nmo
    t1_so = np.zeros((nso, nso))
    for s in range(2):
        for i in range(nocc):
            for a in range(nvir):
                # ExaChem: t1(virt_idx, occ_idx)
                t1_so[2 * (nocc + a) + s, 2 * i + s] = t1_spatial[i, a]
    return t1_so


def spatial_to_spinorb_t2(t2_spatial):
    """Expand spatial T2 amplitudes to spin-orbital form.

    PySCF convention: t2[i,j,a,b] where i,j=occ, a,b=vir (spatial).
    ExaChem convention: t2[a,b,i,j] where a,b=vir_so, i,j=occ_so.

    Args:
        t2_spatial: (nocc, nocc, nvir, nvir) spatial T2 amplitudes.

    Returns:
        (nso, nso, nso, nso) spin-orbital T2 in full-array form.
        Only (vir,vir,occ,occ) elements are nonzero.
        Fully antisymmetric: t2[e,f,m,n] = -t2[f,e,m,n] = -t2[e,f,n,m].
    """
    nocc, _, nvir, _ = t2_spatial.shape
    nmo = nocc + nvir
    nso = 2 * nmo
    t2_so = np.zeros((nso, nso, nso, nso))

    # Fill raw (un-antisymmetrized) Coulomb-like values:
    # t2_raw[2a+s1, 2b+s2, 2i+s1, 2j+s2] = t2_spatial[i,j,a,b]
    for s1 in range(2):
        for s2 in range(2):
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            p = 2 * (nocc + a) + s1
                            q = 2 * (nocc + b) + s2
                            r = 2 * i + s1
                            s = 2 * j + s2
                            t2_so[p, q, r, s] = t2_spatial[i, j, a, b]

    # Antisymmetrize in first index pair (e,f):
    #   t2[e,f,m,n] -> t2[e,f,m,n] - t2[f,e,m,n]
    # For same-spin: t_raw[a,b,i,j] - t_raw[b,a,i,j] = t[i,j,a,b] - t[i,j,b,a]
    # For cross-spin: t_raw[α_a,β_b,α_i,β_j] - 0 = t[i,j,a,b]
    #   and: 0 - t_raw[α_a,β_b,α_i,β_j] fills the partner [β_b,α_a,α_i,β_j]
    t2_so = t2_so - t2_so.transpose(1, 0, 2, 3)

    return t2_so


def reorder_to_exachem_layout(arr, nocc_alpha, nvir_alpha, nmo):
    """Reorder spin-orbital indices from interleaved to ExaChem block layout.

    Input layout (interleaved):  [α₀,β₀, α₁,β₁, ...]
    Output layout (ExaChem):     [occ_α_ext|occ_α_int|occ_β_ext|occ_β_int|vir_α_int|vir_α_ext|vir_β_int|vir_β_ext]

    For the initial expansion (before active-space partitioning), we just need
    the block layout: [occ_α | occ_β | vir_α | vir_β]

    Args:
        arr: array with interleaved spin-orbital indices.
        nocc_alpha: number of alpha occupied orbitals.
        nvir_alpha: number of alpha virtual orbitals.
        nmo: total spatial orbitals.

    Returns:
        Reordered array and the permutation used.
    """
    nso = 2 * nmo
    # Interleaved: even indices = alpha, odd = beta
    # Target: [occ_α (0..nocc_α-1), occ_β, vir_α, vir_β]
    perm = np.zeros(nso, dtype=int)
    idx = 0
    # occ alpha
    for i in range(nocc_alpha):
        perm[idx] = 2 * i  # alpha
        idx += 1
    # occ beta
    for i in range(nocc_alpha):
        perm[idx] = 2 * i + 1  # beta
        idx += 1
    # vir alpha
    for a in range(nvir_alpha):
        perm[idx] = 2 * (nocc_alpha + a)
        idx += 1
    # vir beta
    for a in range(nvir_alpha):
        perm[idx] = 2 * (nocc_alpha + a) + 1
        idx += 1
    return perm


def apply_permutation(arr, perm):
    """Apply a permutation to all axes of an array."""
    ndim = arr.ndim
    result = arr
    for axis in range(ndim):
        result = np.take(result, perm, axis=axis)
    return result


def build_fock_matrix(h1_so, v2_so, nocc_so):
    """Build the Fock matrix from 1e integrals and 2e integrals.

    f[p,q] = h1[p,q] + sum_i <pi||qi> = h1[p,q] + sum_i v2[p,i,q,i]

    Args:
        h1_so: spin-orbital 1e integrals (nso, nso).
        v2_so: spin-orbital antisymmetrized 2e integrals (nso, nso, nso, nso).
        nocc_so: number of occupied spin-orbitals.

    Returns:
        Fock matrix (nso, nso).
    """
    f1 = h1_so.copy()
    occ = slice(0, nocc_so)
    f1 += np.einsum("piqi->pq", v2_so[:, :nocc_so, :, :nocc_so])
    return f1


# ---------------------------------------------------------------------------
# Factory and Algorithm
# ---------------------------------------------------------------------------


class NativeDuccFactory(AlgorithmFactory):
    """Factory for native DUCC solver."""

    def algorithm_type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def default_algorithm_name(self) -> str:
        """Return ``"native_ducc"``."""
        return "native_ducc"


class NativeDuccSolver(Algorithm):
    """Native DUCC Hamiltonian downfolding (no ExaChem binary).

    Uses PySCF for CCSD amplitudes and NumPy einsum contractions for the
    BCH expansion. The formulas are auto-generated from ExaChem's source
    to ensure exact correspondence.

    Settings:
        nactive_oa (int): Active occupied alpha orbitals (default: 0).
        nactive_ob (int): Active occupied beta orbitals (default: 0).
        nactive_va (int): Active virtual alpha orbitals (default: 0).
        nactive_vb (int): Active virtual beta orbitals (default: 0).
        ducc_level (int): BCH truncation level 0/1/2 (default: 2).
    """

    def __init__(self):
        super().__init__()
        s = self._settings
        s._set_default("nactive_oa", "int", 0)
        s._set_default("nactive_ob", "int", 0)
        s._set_default("nactive_va", "int", 0)
        s._set_default("nactive_vb", "int", 0)
        s._set_default("ducc_level", "int", 2)

    def type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def name(self) -> str:
        """Return ``"native_ducc"``."""
        return "native_ducc"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["native_ducc"]

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        """Run native DUCC and return downfolded Hamiltonian.

        Args:
            hamiltonian: qdk_chemistry Hamiltonian (chemist spatial notation).
            n_alpha: number of alpha electrons.
            n_beta: number of beta electrons.

        Returns:
            Downfolded Hamiltonian.
        """
        s = self._settings
        noa = s.get("nactive_oa")
        nob = s.get("nactive_ob")
        nva = s.get("nactive_va")
        nvb = s.get("nactive_vb")
        ducc_level = s.get("ducc_level")

        if noa != nob:
            raise ValueError("Native DUCC only supports closed-shell (nactive_oa == nactive_ob).")
        if nva != nvb:
            raise ValueError("Native DUCC only supports nactive_va == nactive_vb.")
        if n_alpha != n_beta:
            raise ValueError("Native DUCC only supports closed-shell (n_alpha == n_beta).")

        # ── Step 1: Extract integrals (chemist spatial) ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc = n_alpha  # spatial occupied
        nvir = nmo - nocc

        h1_a, _ = hamiltonian.get_one_body_integrals()
        eri_flat, _, _ = hamiltonian.get_two_body_integrals()
        eri = np.array(eri_flat).reshape(nmo, nmo, nmo, nmo)
        core_energy_input = hamiltonian.get_core_energy()

        logger.info("nmo=%d, nocc=%d, nvir=%d, active=(%d,%d,%d,%d), level=%d",
                     nmo, nocc, nvir, noa, nob, nva, nvb, ducc_level)

        # ── Step 2: Get CCSD amplitudes from PySCF ──
        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf

        alpha_occ = np.zeros(nmo)
        alpha_occ[:nocc] = 1.0
        pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, alpha_occ)
        mycc = pyscf_cc.CCSD(pyscf_scf)
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError("PySCF CCSD did not converge.")
        t1_spatial, t2_spatial = mycc.t1, mycc.t2
        logger.info("CCSD converged: E_corr = %.10f", mycc.e_corr)

        # ── Step 3: Convert to spin-orbital physicist notation ──
        h1_so_interleaved = spatial_to_spinorb_h1(np.array(h1_a))
        v2_so_interleaved = spatial_to_spinorb_eri(eri)
        t1_so_interleaved = spatial_to_spinorb_t1(t1_spatial)
        t2_so_interleaved = spatial_to_spinorb_t2(t2_spatial)

        # Reorder from interleaved [α₀β₀α₁β₁...] to block [occ_α|occ_β|vir_α|vir_β]
        perm = reorder_to_exachem_layout(h1_so_interleaved, nocc, nvir, nmo)
        h1_so = apply_permutation(h1_so_interleaved, perm)
        v2_so = apply_permutation(v2_so_interleaved, perm)
        t1_so = apply_permutation(t1_so_interleaved, perm)
        t2_so = apply_permutation(t2_so_interleaved, perm)

        nso = 2 * nmo
        nocc_so = 2 * nocc  # total occupied spin-orbitals

        # ── Step 4: Build Fock matrix ──
        f1 = build_fock_matrix(h1_so, v2_so, nocc_so)

        # ── Step 5: Define active/external index partitions ──
        # ExaChem layout: [occ_α_ext|occ_α_int|occ_β_ext|occ_β_int|vir_α_int|vir_α_ext|vir_β_int|vir_β_ext]
        # In our block layout: [occ_α(0..nocc-1)|occ_β(nocc..2nocc-1)|vir_α(2nocc..2nocc+nvir-1)|vir_β(...)]
        # occ_α_int = last noa of occ_α = indices [nocc-noa, ..., nocc-1]
        # occ_β_int = last nob of occ_β = indices [2*nocc-nob, ..., 2*nocc-1]
        # vir_α_int = first nva of vir_α = indices [2*nocc, ..., 2*nocc+nva-1]
        # vir_β_int = first nvb of vir_β = indices [2*nocc+nvir, ..., 2*nocc+nvir+nvb-1]
        occ_alpha_int = list(range(nocc - noa, nocc))
        occ_beta_int = list(range(2 * nocc - nob, 2 * nocc))
        vir_alpha_int = list(range(2 * nocc, 2 * nocc + nva))
        vir_beta_int = list(range(2 * nocc + nvir, 2 * nocc + nvir + nvb))

        OI = np.array(occ_alpha_int + occ_beta_int, dtype=int)
        VI = np.array(vir_alpha_int + vir_beta_int, dtype=int)
        O = np.arange(nocc_so, dtype=int)
        V = np.arange(nocc_so, nso, dtype=int)

        idx = {"O": O, "V": V, "OI": OI, "VI": VI}

        # ── Step 6: Zero T_int ──
        for ii in OI:
            for aa in VI:
                t1_so[aa, ii] = 0.0
        for ii in OI:
            for jj in OI:
                for aa in VI:
                    for bb in VI:
                        t2_so[aa, bb, ii, jj] = 0.0

        # ── Step 7: Compute SCF energy and initialize total_shift ──
        # E_SCF = V_nuc + sum_i h1[i,i] + 0.5 * sum_{ij} v2[i,j,i,j]
        rep_energy = pyscf_scf.energy_nuc()
        e_electronic = 0.0
        for i in range(nocc_so):
            e_electronic += h1_so[i, i]
            for j in range(nocc_so):
                e_electronic += 0.5 * v2_so[i, j, i, j]
        full_scf_energy = rep_energy + e_electronic

        # total_shift = E_SCF - V_nuc (electronic SCF energy)
        total_shift = e_electronic

        logger.info("Full SCF energy (spin-orb): %.10f", full_scf_energy)
        logger.info("V_nuc: %.10f", rep_energy)

        # ── Step 8: Initialize active-space tensors ──
        n_oi = len(OI)
        n_vi = len(VI)
        ft = {
            "ij": np.zeros((n_oi, n_oi)),
            "ia": np.zeros((n_oi, n_vi)),
            "ab": np.zeros((n_vi, n_vi)),
        }
        vt = {
            "ijkl": np.zeros((n_oi, n_oi, n_oi, n_oi)),
            "ijka": np.zeros((n_oi, n_oi, n_oi, n_vi)),
            "aijb": np.zeros((n_vi, n_oi, n_oi, n_vi)),
            "ijab": np.zeros((n_oi, n_oi, n_vi, n_vi)),
            "iabc": np.zeros((n_oi, n_vi, n_vi, n_vi)),
            "abcd": np.zeros((n_vi, n_vi, n_vi, n_vi)),
        }

        # ── Level 0: Bare Hamiltonian projection ──
        logger.info("Computing H_0 (bare Hamiltonian)...")
        ft["ij"] = f1[np.ix_(OI, OI)].copy()
        ft["ia"] = f1[np.ix_(OI, VI)].copy()
        ft["ab"] = f1[np.ix_(VI, VI)].copy()
        vt["ijkl"] = v2_so[np.ix_(OI, OI, OI, OI)].copy()
        vt["ijka"] = v2_so[np.ix_(OI, OI, OI, VI)].copy()
        vt["ijab"] = v2_so[np.ix_(OI, OI, VI, VI)].copy()
        vt["iabc"] = v2_so[np.ix_(OI, VI, VI, VI)].copy()
        vt["abcd"] = v2_so[np.ix_(VI, VI, VI, VI)].copy()
        # vtaijb = -v2iajb (sign flip, matching ExaChem H_0)
        vt["aijb"] = -v2_so[np.ix_(OI, VI, OI, VI)].transpose(1, 0, 2, 3)
        # More precisely: vtaijb(a,i,j,b) = -v2(i,a,j,b)
        vt["aijb"] = -v2_so[np.ix_(OI, VI, OI, VI)]  # v2[i,a,j,b] with OI,VI,OI,VI
        # Need: vt[a,i,j,b] = -v2[i,a,j,b], so transpose (1,0,2,3) on [i,a,j,b]
        v2_iajb = v2_so[np.ix_(OI, VI, OI, VI)]  # shape (n_oi, n_vi, n_oi, n_vi)
        vt["aijb"] = -v2_iajb.transpose(1, 0, 2, 3)  # -> (n_vi, n_oi, n_oi, n_vi)

        # ── Level 1: [F,T] + [V,T] + [[F,T],T] ──
        if ducc_level >= 1:
            logger.info("Computing F_1 (single commutator of F)...")
            scalar_f1 = bch_contractions.f_1(ft, vt, idx, f1, t1_so, t2_so)

            # Scalar: Tr(f1 * t1) + Tr(t1 * f1)
            adj = np.einsum("hp,ph->", f1[:nocc_so, nocc_so:], t1_so[nocc_so:, :nocc_so])
            adj += np.einsum("ph,hp->", t1_so[nocc_so:, :nocc_so], f1[:nocc_so, nocc_so:])
            total_shift += adj

            logger.info("Computing V_1 (single commutator of V)...")
            scalar_v1 = bch_contractions.v_1(ft, vt, idx, v2_so, t1_so, t2_so)

            # Scalar: (1/4) * v2ijab * t2 + (1/4) * t2 * v2ijab
            adj = 0.25 * np.einsum("ijab,abij->", v2_so[:nocc_so, :nocc_so, nocc_so:, nocc_so:],
                                    t2_so[nocc_so:, nocc_so:, :nocc_so, :nocc_so])
            adj += 0.25 * np.einsum("abij,ijab->", t2_so[nocc_so:, nocc_so:, :nocc_so, :nocc_so],
                                     v2_so[:nocc_so, :nocc_so, nocc_so:, nocc_so:])
            total_shift += adj

            logger.info("Computing F_2 (double commutator of F)...")
            scalar_f2 = bch_contractions.f_2(ft, vt, idx, f1, t1_so, t2_so)
            total_shift += scalar_f2

        # ── Level 2: [[V,T],T] + [[[F,T],T],T] ──
        if ducc_level >= 2:
            logger.info("Computing V_2 (double commutator of V)...")
            scalar_v2 = bch_contractions.v_2(ft, vt, idx, v2_so, t1_so, t2_so)
            total_shift += scalar_v2

            logger.info("Computing F_3 (triple commutator of F)...")
            scalar_f3 = bch_contractions.f_3(ft, vt, idx, f1, t1_so, t2_so)
            total_shift += scalar_f3

        # ── Step 9: Fock → 1e transform ──
        # Subtract mean-field of active occupied from Fock to get bare 1e
        # (matching ExaChem ducc-t_ccsd.cpp lines 340-357)
        # The ft tensors are in Fock-operator form. We need to convert to bare 1e
        # by subtracting the mean-field contributions of active occupied orbitals.
        logger.info("Fock -> 1e transform...")
        delta_oi = np.eye(n_oi)

        if n_vi > 0:
            # ftab(a,b) += delta(i,j) * vtaijb(a,i,j,b)
            ft["ab"] += np.einsum("ij,aijb->ab", delta_oi, vt["aijb"])

            # ftia(h3,p1) += -0.5 * delta(h1,h2) * vtijka(h1,h3,h2,p1)
            #              +  0.5 * delta(h1,h2) * vtijka(h3,h1,h2,p1)
            for h3 in range(n_oi):
                for p1 in range(n_vi):
                    correction = 0.0
                    for h12 in range(n_oi):
                        correction += -0.5 * vt["ijka"][h12, h3, h12, p1]
                        correction += 0.5 * vt["ijka"][h3, h12, h12, p1]
                    ft["ia"][h3, p1] += correction

        # ftij corrections (always applied, even if no virtual active orbitals)
        # ftij(h3,h4) += -0.25*delta(h1,h2)*vtijkl(h3,h1,h4,h2)
        #              +  0.25*delta(h1,h2)*vtijkl(h3,h1,h2,h4)
        #              +  0.25*delta(h1,h2)*vtijkl(h1,h3,h4,h2)
        #              + -0.25*delta(h1,h2)*vtijkl(h1,h3,h2,h4)
        for h3 in range(n_oi):
            for h4 in range(n_oi):
                correction = 0.0
                for h12 in range(n_oi):
                    correction += -0.25 * vt["ijkl"][h3, h12, h4, h12]
                    correction += 0.25 * vt["ijkl"][h3, h12, h12, h4]
                    correction += 0.25 * vt["ijkl"][h12, h3, h4, h12]
                    correction += -0.25 * vt["ijkl"][h12, h3, h12, h4]
                ft["ij"][h3, h4] += correction

        # ── Compute active-space SCF energy and adjust total_shift ──
        # Match ExaChem driver (ducc-t_ccsd.cpp lines 361-373):
        # adj_scalar = Tr(delta_oi * ftij)
        #            + 0.25 * delta(h1,h2) * delta(h3,h4) * vtijkl(h3,h1,h4,h2)
        #            - 0.25 * delta(h1,h2) * delta(h3,h4) * vtijkl(h3,h1,h2,h4)
        # This computes the active-space SCF energy (after Fock→1e transform).
        active_scf = np.trace(ft["ij"])
        for i in range(n_oi):
            for j in range(n_oi):
                active_scf += 0.25 * vt["ijkl"][j, i, j, i]
                active_scf -= 0.25 * vt["ijkl"][j, i, i, j]
        total_shift -= active_scf
        core_energy = total_shift + rep_energy

        logger.info("Active-space SCF energy: %.10f", active_scf + rep_energy)
        logger.info("Total energy shift: %.10f", total_shift)
        logger.info("Core energy: %.10f", core_energy)

        # ── Step 10: Pack into FcidumpData and convert ──
        nocc_active = noa + nob
        nvir_active = nva + nvb
        norb_active = nocc_active + nvir_active

        h1_active = np.zeros((norb_active, norb_active))
        h2_active = np.zeros((norb_active, norb_active, norb_active, norb_active))

        # Pack 1e blocks
        h1_active[:nocc_active, :nocc_active] = ft["ij"]
        if n_vi > 0:
            h1_active[:nocc_active, nocc_active:] = ft["ia"]
            h1_active[nocc_active:, :nocc_active] = ft["ia"].T
            h1_active[nocc_active:, nocc_active:] = ft["ab"]

        # Pack 2e blocks (all 9 blocks into full array)
        OI_l = list(range(nocc_active))
        VI_l = list(range(nocc_active, norb_active))
        OI_a = np.array(OI_l)
        VI_a = np.array(VI_l)

        h2_active[np.ix_(OI_a, OI_a, OI_a, OI_a)] = vt["ijkl"]
        if n_vi > 0:
            h2_active[np.ix_(OI_a, OI_a, OI_a, VI_a)] = vt["ijka"]
            h2_active[np.ix_(VI_a, OI_a, OI_a, VI_a)] = vt["aijb"]
            h2_active[np.ix_(OI_a, OI_a, VI_a, VI_a)] = vt["ijab"]
            h2_active[np.ix_(OI_a, VI_a, VI_a, VI_a)] = vt["iabc"]
            h2_active[np.ix_(VI_a, VI_a, VI_a, VI_a)] = vt["abcd"]

        # Enforce full antisymmetry: the 9 blocks only populate their natural
        # index patterns. spinorb_to_spatial needs ALL antisymmetric partners.
        # Propagate each nonzero element to all 8 partner positions.
        n = norb_active
        h2_full = np.zeros_like(h2_active)
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        val = h2_active[p, q, r, s]
                        if abs(val) > 1e-15:
                            h2_full[p, q, r, s] = val
                            h2_full[q, p, r, s] = -val
                            h2_full[p, q, s, r] = -val
                            h2_full[q, p, s, r] = val
                            h2_full[r, s, p, q] = val
                            h2_full[s, r, p, q] = -val
                            h2_full[r, s, q, p] = -val
                            h2_full[s, r, q, p] = val
        h2_active = h2_full

        fcidump = FcidumpData(
            norb=norb_active,
            nelec=nocc_active,
            ms2=0,
            one_body=h1_active,
            two_body=h2_active,
            nuclear_repulsion=rep_energy,
        )

        # Get atom info from hamiltonian for fcidump_to_hamiltonian
        # We need atoms/basis/units but those aren't stored in the Hamiltonian.
        # Use spinorb_to_spatial + fcidump_to_hamiltonian with core_energy_override.
        # Actually, we already have the spin-orbital integrals packed.
        # Use the same conversion path as ExaChem plugin.
        spatial_fcidump = spinorb_to_spatial(fcidump)

        # Build output Hamiltonian directly from spatial integrals
        from qdk_chemistry.data import (
            CanonicalFourCenterHamiltonianContainer,
            Hamiltonian,
            ModelOrbitals,
        )

        n_spatial_active = norb_active // 2
        active_orbitals = ModelOrbitals(n_spatial_active, True)  # restricted

        container = CanonicalFourCenterHamiltonianContainer(
            spatial_fcidump.one_body,
            spatial_fcidump.two_body.ravel(),
            active_orbitals,
            core_energy,
            np.zeros((n_spatial_active, n_spatial_active)),
        )

        return Hamiltonian(container)
