# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Wicked-based DUCC Hamiltonian downfolding.

Uses the `wicked` library for symbolic BCH expansion and Wick contraction
to downfold a full-space Hamiltonian into an active-space effective
Hamiltonian. This is an alternative to :class:`NativeDuccSolver` that uses
wicked's symbolic algebra instead of auto-generated tensor contractions.

The BCH expansion follows the DUCC paper (Bauman et al., JCP 151, 014107):

- Level 0: bare Hamiltonian restricted to active space
- Level 1: H + [H_N, σ_ext] + ½[[F, σ_ext], σ_ext]
- Level 2: H + [H_N, σ_ext] + ½[[H_N, σ_ext], σ_ext] + ⅙[[[F, σ_ext], σ_ext], σ_ext]

where σ_ext = T_ext - T_ext† with all-active T amplitudes zeroed.
"""

from __future__ import annotations

import logging

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm

logger = logging.getLogger(__name__)

# Lazy import — wicked is optional
_wicked = None


def _require_wicked():
    global _wicked
    if _wicked is None:
        try:
            import wicked

            _wicked = wicked
        except ImportError:
            raise ImportError(
                "wicked is required for WickedDuccSolver. "
                "Install from https://github.com/fevangelista/wicked"
            )
    return _wicked


class WickedDuccSolver(Algorithm):
    """DUCC Hamiltonian downfolder using wicked symbolic BCH.

    Takes a full-space qdk-chemistry Hamiltonian (spatial, chemist notation),
    runs PySCF CCSD for T amplitudes, performs the BCH expansion via wicked,
    restricts to the active space, and returns the downfolded Hamiltonian.

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc",
                         nactive_oa=2, nactive_va=3, ducc_level=2)
        downfolded = solver.run(hamiltonian, n_alpha, n_beta)
    """

    def __init__(self) -> None:
        super().__init__()
        s = self.settings()
        s._set_default("nactive_oa", "int", 0, "Number of active occupied alpha orbitals")
        s._set_default("nactive_ob", "int", 0, "Number of active occupied beta orbitals")
        s._set_default("nactive_va", "int", 0, "Number of active virtual alpha orbitals")
        s._set_default("nactive_vb", "int", 0, "Number of active virtual beta orbitals")
        s._set_default("ducc_level", "int", 2, "BCH truncation level (0, 1, or 2)")

    @staticmethod
    def type_name() -> str:
        return "hamiltonian_downfolder"

    @staticmethod
    def name() -> str:
        return "wicked_ducc"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        """Run DUCC downfolding.

        Args:
            hamiltonian: Full-space qdk-chemistry Hamiltonian (spatial, chemist).
            n_alpha: Number of alpha electrons.
            n_beta: Number of beta electrons.

        Returns:
            Downfolded active-space Hamiltonian (spatial, chemist).
        """
        w = _require_wicked()
        s = self.settings()
        nactive_oa = s["nactive_oa"]
        nactive_ob = s["nactive_ob"]
        nactive_va = s["nactive_va"]
        nactive_vb = s["nactive_vb"]
        ducc_level = s["ducc_level"]

        if nactive_oa != nactive_ob:
            raise ValueError("WickedDuccSolver requires nactive_oa == nactive_ob (closed-shell)")
        if nactive_va != nactive_vb:
            raise ValueError("WickedDuccSolver requires nactive_va == nactive_vb (closed-shell)")
        if n_alpha != n_beta:
            raise ValueError("WickedDuccSolver requires n_alpha == n_beta (closed-shell)")

        # ── 1. Extract integrals from Hamiltonian ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc = n_alpha  # spatial occupied count

        h1_list, _ = hamiltonian.get_one_body_integrals()
        h1_spatial = np.array(h1_list).reshape(nmo, nmo)

        eri_list, _, _ = hamiltonian.get_two_body_integrals()
        eri_spatial = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)

        core_energy_input = hamiltonian.get_core_energy()

        logger.info(
            "WickedDuccSolver: nmo=%d, nocc=%d, active=(%d,%d), level=%d",
            nmo, nocc, nactive_oa, nactive_va, ducc_level,
        )

        # ── 2. Convert spatial → spin-orbital (interleaved α,β) ──
        nso = 2 * nmo
        nocc_so = 2 * nocc
        nvir_so = nso - nocc_so

        # 1-body: h1_so[2p+s, 2q+s] = h1[p,q]
        h1_so = np.zeros((nso, nso))
        for spin in range(2):
            h1_so[spin::2, spin::2] = h1_spatial

        # 2-body: chemist → spin-orbital chemist → physicist antisymmetrized
        eri_so = np.zeros((nso,) * 4)
        for s1 in range(2):
            for s2 in range(2):
                eri_so[s1::2, s1::2, s2::2, s2::2] = eri_spatial
        h2_so = np.einsum("prqs->pqrs", eri_so) - np.einsum("psqr->pqrs", eri_so)

        # ── 3. Normal-order w.r.t. HF ──
        occ_idx = list(range(nocc_so))
        E0_hf = core_energy_input
        for m in occ_idx:
            E0_hf += h1_so[m, m]
        E0_hf += 0.5 * sum(h2_so[m, n, m, n] for m in occ_idx for n in occ_idx)

        f_no = h1_so.copy()
        for m in occ_idx:
            f_no += h2_so[:, m, :, m]
        v_no = h2_so

        # ── 4. CCSD amplitudes via PySCF ──
        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf

        alpha_occ = np.zeros(nmo)
        alpha_occ[:nocc] = 1.0
        pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, alpha_occ)

        from pyscf import cc

        mycc = cc.CCSD(pyscf_scf).run()
        logger.info("CCSD energy: %.10f", mycc.e_tot)

        # Convert spatial T1/T2 → spin-orbital (interleaved)
        t1_spatial = mycc.t1  # [nocc, nvir]
        t2_spatial = mycc.t2  # [nocc, nocc, nvir, nvir]
        nvir = nmo - nocc

        t1_so = np.zeros((nocc_so, nvir_so))
        for spin in range(2):
            t1_so[spin::2, spin::2] = t1_spatial

        t2_so = np.zeros((nocc_so, nocc_so, nvir_so, nvir_so))
        for s1 in range(2):
            for s2 in range(2):
                t2_so[s1::2, s2::2, s1::2, s2::2] = t2_spatial
        # Antisymmetrize T2: t2[i,j,a,b] = t2[i,j,a,b] - t2[i,j,b,a]
        # (spatial T2 from RHF CCSD is NOT antisymmetrized)
        t2_so = t2_so - t2_so.transpose(0, 1, 3, 2)

        # ── 5. Active space definition (interleaved) ──
        nocc_spatial = nocc
        ncore_spatial = nocc_spatial - nactive_oa
        active_so = []
        for i in range(ncore_spatial, nocc_spatial):
            active_so.extend([2 * i, 2 * i + 1])
        for i in range(nactive_va):
            active_so.extend([nocc_so + 2 * i, nocc_so + 2 * i + 1])
        active_so = sorted(active_so)
        nact = len(active_so)
        nocc_act = 2 * nactive_oa

        # Zero all-active T amplitudes → σ_ext
        t1_ext = t1_so.copy()
        t2_ext = t2_so.copy()
        aoc = [g for g in active_so if g < nocc_so]
        avl = [g - nocc_so for g in active_so if g >= nocc_so]
        for i in aoc:
            for a in avl:
                t1_ext[i, a] = 0.0
        for i in aoc:
            for j in aoc:
                for a in avl:
                    for b in avl:
                        t2_ext[i, j, a, b] = 0.0

        # ── 6. Wicked BCH expansion ──
        chi_1, chi_2, C = self._wicked_bch(
            w, ducc_level, f_no, v_no, t1_ext, t2_ext, E0_hf, nocc_so, nso, active_so
        )

        # ── 7. Convert chi (interleaved spin-orbital) → spatial chemist ──
        from qdk_chemistry.plugins.exachem.conversion import FcidumpData, spinorb_to_spatial
        from qdk_chemistry.data import (
            CanonicalFourCenterHamiltonianContainer,
            Hamiltonian,
            ModelOrbitals,
        )

        # Pack chi into FcidumpData (interleaved spin-orbital physicist)
        fcidump = FcidumpData(
            norb=nact,
            nelec=nocc_act,
            ms2=0,
            one_body=chi_1,
            two_body=chi_2,
            nuclear_repulsion=C,
        )

        # spinorb_to_spatial expects blocked ordering [α_occ, β_occ, α_vir, β_vir].
        # Our chi is interleaved [α₁, β₁, α₂, β₂, ...]. Reorder.
        noa = nocc_act // 2
        nva = (nact - nocc_act) // 2
        il_to_blocked = np.empty(nact, dtype=int)
        for i in range(noa):
            il_to_blocked[2 * i] = i
            il_to_blocked[2 * i + 1] = noa + i
        for i in range(nva):
            il_to_blocked[nocc_act + 2 * i] = nocc_act + i
            il_to_blocked[nocc_act + 2 * i + 1] = nocc_act + nva + i

        chi1_blocked = np.zeros_like(chi_1)
        chi2_blocked = np.zeros_like(chi_2)
        for p in range(nact):
            for q in range(nact):
                chi1_blocked[il_to_blocked[p], il_to_blocked[q]] = chi_1[p, q]
                for r in range(nact):
                    for s in range(nact):
                        if abs(chi_2[p, q, r, s]) > 1e-20:
                            chi2_blocked[
                                il_to_blocked[p], il_to_blocked[q],
                                il_to_blocked[r], il_to_blocked[s],
                            ] = chi_2[p, q, r, s]

        fcidump_blocked = FcidumpData(
            norb=nact,
            nelec=nocc_act,
            ms2=0,
            one_body=chi1_blocked,
            two_body=chi2_blocked,
            nuclear_repulsion=C,
        )
        spatial = spinorb_to_spatial(fcidump_blocked)

        n_spatial_active = nact // 2
        active_orbitals = ModelOrbitals(n_spatial_active)

        container = CanonicalFourCenterHamiltonianContainer(
            spatial.one_body,
            spatial.two_body.ravel(),
            active_orbitals,
            C,
            np.zeros((n_spatial_active, n_spatial_active)),
        )
        return Hamiltonian(container)

    @staticmethod
    def _wicked_bch(w, bch_order, f_no, v_no, t1_ov, t2_oovv, E0_hf, nocc, nso, active_so):
        """Run wicked BCH and return active-space chi tensors + scalar C.

        Args:
            w: wicked module.
            bch_order: 0, 1, or 2.
            f_no: Normal-ordered Fock matrix [nso, nso].
            v_no: Antisymmetrized 2e integrals [nso, nso, nso, nso].
            t1_ov: T1 amplitudes [nocc, nvir] (σ_ext, with active zeroed).
            t2_oovv: T2 amplitudes [nocc, nocc, nvir, nvir] (σ_ext).
            E0_hf: HF energy scalar.
            nocc: Number of occupied spin-orbitals.
            nso: Total spin-orbitals.
            active_so: List of active spin-orbital indices (interleaved).

        Returns:
            (chi_1, chi_2, C): Active-space chi tensors and scalar.
        """
        nvir = nso - nocc

        w.reset_space()
        w.add_space("o", "fermion", "occupied", list("ijklmnop")[:nocc])
        w.add_space("v", "fermion", "unoccupied", list("abcdefgh")[:nvir])

        E0op = w.op("E0", [""])
        F = w.utils.gen_op("f", 1, "ov", "ov")
        V = w.utils.gen_op("v", 2, "ov", "ov")
        H_N = F + V

        T = w.op("t", ["v+ o", "v+ v+ o o"])
        sigma = w.op("t", ["v+ o", "v+ v+ o o"])
        sigma.add2(T.adjoint(), w.rational(-1))

        # BCH expansion — paper's Eq.(33) and extensions.
        # Equivalently: w.bch_series(E0op + H_N, sigma, 1) gives E0 + H_N + [H_N, σ]
        # (since [E0, σ] = 0), then add the double/triple commutator corrections.
        if bch_order == 0:
            Hbar = E0op + H_N
        elif bch_order == 1:
            comm1 = w.commutator(H_N, sigma)
            comm2_F = w.commutator(F, sigma, sigma)
            Hbar = E0op + H_N + comm1
            Hbar.add2(comm2_F, w.rational(1, 2))
        elif bch_order == 2:
            comm1 = w.commutator(H_N, sigma)
            comm2_HN = w.commutator(H_N, sigma, sigma)
            comm3_F = w.commutator(F, sigma, sigma, sigma)
            Hbar = E0op + H_N + comm1
            Hbar.add2(comm2_HN, w.rational(1, 2))
            Hbar.add2(comm3_F, w.rational(1, 6))
        else:
            raise ValueError(f"Unsupported BCH order {bch_order} (must be 0, 1, or 2)")

        # Wick contraction → many-body equations → einsum evaluation
        expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
        mbeq = expr.to_manybody_equation("r")

        o_sl = slice(0, nocc)
        v_sl = slice(nocc, nso)
        f_dict = {
            "oo": f_no[o_sl, o_sl], "ov": f_no[o_sl, v_sl],
            "vo": f_no[v_sl, o_sl], "vv": f_no[v_sl, v_sl],
        }
        v_dict = {}
        for k1, s1 in [("o", o_sl), ("v", v_sl)]:
            for k2, s2 in [("o", o_sl), ("v", v_sl)]:
                for k3, s3 in [("o", o_sl), ("v", v_sl)]:
                    for k4, s4 in [("o", o_sl), ("v", v_sl)]:
                        v_dict[k1 + k2 + k3 + k4] = v_no[s1, s2, s3, s4]

        t_dict = {
            "ov": t1_ov, "vo": t1_ov.T,
            "oovv": t2_oovv, "vvoo": t2_oovv.transpose(2, 3, 0, 1),
        }

        class _ScalarDict:
            def __init__(self, val):
                self.val = val
            def __getitem__(self, key):
                return self.val

        slices_map = {"o": o_sl, "v": v_sl}
        r1b, r2b, rs = {}, {}, 0.0

        for key, eqs in mbeq.items():
            if not eqs:
                continue
            lower, upper = key.split("|")
            ndim = len(lower) + len(upper)
            fc = eqs[0].compile("einsum")
            rv = fc.split("+=")[0].strip()
            ic = rv[1:]
            shape = [nocc if c == "o" else nvir for c in ic]

            lines = ["def _eval(E0, f, v, t, nocc, nvir):"]
            if ndim == 0:
                lines.append(f"    {rv} = 0.0")
            else:
                lines.append(
                    f"    {rv} = np.zeros(({','.join(str(s) for s in shape)}))"
                )
            for eq in eqs:
                lines.append(f"    {eq.compile('einsum')}")
            lines.append(f"    return {rv}")

            ns = {}
            exec("\n".join(lines), {"np": np}, ns)  # noqa: S102
            result = ns["_eval"](_ScalarDict(E0_hf), f_dict, v_dict, t_dict, nocc, nvir)

            if ndim == 0:
                rs = result
            elif ndim == 2:
                r1b[key] = (rv, ic, np.array(result))
            elif ndim == 4:
                r2b[key] = (rv, ic, np.array(result))

        # Assemble full-space fbar, vbar
        fbar = np.zeros((nso, nso))
        for k, (rv, ic, blk) in r1b.items():
            fbar[tuple(slices_map[c] for c in ic)] += blk

        vr = np.zeros((nso,) * 4)
        for k, (rv, ic, blk) in r2b.items():
            vr[tuple(slices_map[c] for c in ic)] += blk
        vbar = (
            vr
            - vr.transpose(1, 0, 2, 3)
            - vr.transpose(0, 1, 3, 2)
            + vr.transpose(1, 0, 3, 2)
        )

        # Restrict to active space → gamma → chi
        nact = len(active_so)
        gamma_1 = fbar[np.ix_(active_so, active_so)]
        gamma_2 = vbar[np.ix_(active_so, active_so, active_so, active_so)]
        aol = [i for i, g in enumerate(active_so) if g < nocc]

        chi_1 = gamma_1 - np.einsum(
            "pmqm->pq", gamma_2[:, aol, :, :][:, :, :, aol]
        )
        chi_2 = gamma_2.copy()

        # Physical-vacuum scalar
        C = rs
        for m in aol:
            C -= chi_1[m, m]
            for n in aol:
                C -= 0.5 * chi_2[m, n, m, n]

        return chi_1, chi_2, C
