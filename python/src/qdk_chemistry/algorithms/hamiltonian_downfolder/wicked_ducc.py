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
            import wickd as wicked

            _wicked = wicked
        except ImportError:
            raise ImportError(
                "wickd is required for WickedDuccSolver. Install from https://github.com/fevangelista/wicked"
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

        # ── Steps 1-6: spin-orbital active-space χ tensors (open- and closed-shell) ──
        chi_1, chi_2, C, meta = self._build_active_chi(
            w, hamiltonian, n_alpha, n_beta, nactive_oa, nactive_ob, nactive_va, nactive_vb, ducc_level
        )

        # ── Step 7: pack χ into a chemist-notation active-space Hamiltonian ──
        return self._assemble_hamiltonian(chi_1, chi_2, C, meta, n_alpha, n_beta, nactive_oa, nactive_ob, nactive_va, nactive_vb)

    def _assemble_hamiltonian(self, chi_1, chi_2, C, meta, n_alpha, n_beta, noa, nob, nva, nvb):
        """Step 7: pack spin-orbital χ into a chemist-notation active-space Hamiltonian.

        Slices the spin-orbital χ tensors into spin-blocked **chemist** integrals
        and stores them in a :class:`CanonicalFourCenterHamiltonianContainer`.
        No spatial collapse — the α≠β structure is preserved for open-shell.

        The spin-orbital chemist integral is ``g[P,Q,R,S] = ½·χ₂[P,R,Q,S]`` (the
        tensor validated against pyscf's spin-orbital FCI).  Spin blocks map to
        spin-blocked chemist integrals as:

        * same-spin  ``(pq|rs)_αα = g[a,a,a,a]`` (χ₂ only fixes the antisymmetric
          part; the CI solver re-antisymmetrizes internally),
        * opposite-spin ``(pq|rs)_αβ = 2·g[a,a,b,b] = χ₂[a,b,a,b]`` (full Coulomb).

        Restricted is the special case where the three 2e blocks coincide, so a
        single chemist ``V`` (the αβ Coulomb slice) suffices.
        """
        from qdk_chemistry.data import (
            CanonicalFourCenterHamiltonianContainer,
            Hamiltonian,
            ModelOrbitals,
        )
        from qdk_chemistry.data.symmetry import SymmetryProduct, axes

        a_local = meta["a_local"]
        b_local = meta["b_local"]
        na = len(a_local)
        nb = len(b_local)
        if na != nb:
            raise ValueError(
                f"DUCC active space must have equal α/β active orbital counts, got {na} α and {nb} β. "
                f"Set nactive_oa+nactive_va == nactive_ob+nactive_vb."
            )

        # Spin-orbital chemist integrals (PQ|RS) = ½ <PR||QS>-derived.
        g = 0.5 * chi_2.transpose(0, 2, 1, 3)

        restricted = n_alpha == n_beta and noa == nob and nva == nvb
        if restricted:
            h1 = np.ascontiguousarray(chi_1[np.ix_(a_local, a_local)])
            # Single spatial chemist V = full αβ Coulomb slice.
            v = np.ascontiguousarray(2.0 * g[np.ix_(a_local, a_local, b_local, b_local)])
            orbitals = ModelOrbitals(na)
            container = CanonicalFourCenterHamiltonianContainer(
                h1, v.ravel(), orbitals, C, np.zeros((na, na))
            )
        else:
            h1_a = np.ascontiguousarray(chi_1[np.ix_(a_local, a_local)])
            h1_b = np.ascontiguousarray(chi_1[np.ix_(b_local, b_local)])
            v_aaaa = np.ascontiguousarray(g[np.ix_(a_local, a_local, a_local, a_local)])
            v_bbbb = np.ascontiguousarray(g[np.ix_(b_local, b_local, b_local, b_local)])
            v_aabb = np.ascontiguousarray(2.0 * g[np.ix_(a_local, a_local, b_local, b_local)])
            orbitals = ModelOrbitals(na, SymmetryProduct([axes.spin(1, False)]))
            container = CanonicalFourCenterHamiltonianContainer(
                h1_a,
                h1_b,
                v_aaaa.ravel(),
                v_aabb.ravel(),
                v_bbbb.ravel(),
                orbitals,
                C,
                np.zeros((na, na)),
                np.zeros((nb, nb)),
            )
        return Hamiltonian(container)

    def _build_active_chi(self, w, hamiltonian, n_alpha, n_beta, noa, nob, nva, nvb, ducc_level):
        """Steps 1-6 (spin-orbital, open- and closed-shell): active-space χ tensors.

        Builds spin-orbital integrals from the qdk Hamiltonian's spin-blocked
        MO integrals (``get_one_body_integrals`` → ``(h1_α, h1_β)``,
        ``get_two_body_integrals`` → ``(V_αααα, V_ααββ, V_ββββ)`` chemist), gets
        CCSD amplitudes via the shared qdk CC plugin, forms σ_ext, and runs the
        wicked BCH.  All integrals come from qdk-data — no pyscf integral reads.

        Returns:
            ``(chi_1, chi_2, C, meta)`` where *meta* carries active-space
            metadata and the spin-orbital inputs for debugging.
        """
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_common import build_ccsd_amplitudes

        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc_a, nocc_b = n_alpha, n_beta
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b

        # ── qdk spin-blocked MO integrals (2e in chemist (pq|rs)) ──
        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h1_a = np.array(h1_alpha).reshape(nmo, nmo)
        h1_b = np.array(h1_beta).reshape(nmo, nmo)
        v_aaaa, v_aabb, v_bbbb = hamiltonian.get_two_body_integrals()
        eri_aa = np.array(v_aaaa).reshape(nmo, nmo, nmo, nmo)
        eri_bb = np.array(v_bbbb).reshape(nmo, nmo, nmo, nmo)
        eri_ab = np.array(v_aabb).reshape(nmo, nmo, nmo, nmo)
        core = hamiltonian.get_core_energy()

        # ── Spin-orbital layout: occupied-first [α-occ, β-occ, α-vir, β-vir] ──
        occ_so = [(p, 0) for p in range(nocc_a)] + [(p, 1) for p in range(nocc_b)]
        vir_so = [(p, 0) for p in range(nocc_a, nmo)] + [(p, 1) for p in range(nocc_b, nmo)]
        so_list = occ_so + vir_so
        nso = 2 * nmo
        nocc_so = nocc_a + nocc_b
        spin = np.array([s for _, s in so_list])
        spat = np.array([p for p, _ in so_list])
        a_idx = np.where(spin == 0)[0]
        b_idx = np.where(spin == 1)[0]
        sa, sb = spat[a_idx], spat[b_idx]

        # ── Spin-orbital 1e/2e integrals (from spin-blocked qdk integrals) ──
        h1_so = np.zeros((nso, nso))
        h1_so[np.ix_(a_idx, a_idx)] = h1_a[np.ix_(sa, sa)]
        h1_so[np.ix_(b_idx, b_idx)] = h1_b[np.ix_(sb, sb)]

        # Chemist (PQ|RS): nonzero only for σ_P=σ_Q and σ_R=σ_S.
        eri_so = np.zeros((nso, nso, nso, nso))
        eri_so[np.ix_(a_idx, a_idx, a_idx, a_idx)] = eri_aa[np.ix_(sa, sa, sa, sa)]
        eri_so[np.ix_(b_idx, b_idx, b_idx, b_idx)] = eri_bb[np.ix_(sb, sb, sb, sb)]
        eri_so[np.ix_(a_idx, a_idx, b_idx, b_idx)] = eri_ab[np.ix_(sa, sa, sb, sb)]
        eri_so[np.ix_(b_idx, b_idx, a_idx, a_idx)] = eri_ab[np.ix_(sa, sa, sb, sb)].transpose(2, 3, 0, 1)
        v_phys = eri_so.transpose(0, 2, 1, 3)  # <PQ|RS> = (PR|QS)
        v_no = v_phys - v_phys.transpose(0, 1, 3, 2)  # <PQ||RS>

        occ_idx = list(range(nocc_so))
        E0_hf = core + sum(h1_so[m, m] for m in occ_idx) + 0.5 * sum(
            v_no[m, n, m, n] for m in occ_idx for n in occ_idx
        )
        f_no = h1_so.copy()
        for m in occ_idx:
            f_no += v_no[:, m, :, m]

        # ── CCSD amplitudes (shared qdk CC plugin, spin-blocked) → spin-orbital ──
        t_dict = build_ccsd_amplitudes(hamiltonian, nmo, nocc_a, nocc_b)
        t1_so, t2_so = self._spinblocked_amplitudes_to_so(t_dict, occ_so, vir_so, nocc_a, nocc_b)

        # ── Active space (per-spin, index-based) in the spin-blocked SO layout ──
        # α-occ: top noa of [0, nocc_a);         β-occ: top nob of the β-occ block.
        # α-vir: bottom nva of the α-vir block;  β-vir: bottom nvb of the β-vir block.
        active_a_occ = list(range(nocc_a - noa, nocc_a))
        active_b_occ = list(range(nocc_so - nob, nocc_so))
        active_a_vir = list(range(nocc_so, nocc_so + nva))
        active_b_vir = list(range(nocc_so + nvir_a, nocc_so + nvir_a + nvb))
        active_so = sorted(active_a_occ + active_b_occ + active_a_vir + active_b_vir)

        # ── σ_ext: zero all-active amplitudes ──
        a_occ = [g for g in active_so if g < nocc_so]
        a_vir = [g - nocc_so for g in active_so if g >= nocc_so]
        t1_ext = t1_so.copy()
        t2_ext = t2_so.copy()
        for i in a_occ:
            for a in a_vir:
                t1_ext[i, a] = 0.0
        for i in a_occ:
            for j in a_occ:
                for a in a_vir:
                    for b in a_vir:
                        t2_ext[i, j, a, b] = 0.0

        # ── Wicked BCH ──
        chi_1, chi_2, C = self._wicked_bch(
            w, ducc_level, f_no, v_no, t1_ext, t2_ext, E0_hf, nocc_so, nso, active_so
        )

        # Partition active spin-orbitals by spin (local active-space indices).
        a_local = [k for k, g in enumerate(active_so) if spin[g] == 0]
        b_local = [k for k, g in enumerate(active_so) if spin[g] == 1]
        # Active-occupied (electrons) per spin: active SO below the Fermi level.
        n_act_alpha_elec = sum(1 for k in a_local if active_so[k] < nocc_so)
        n_act_beta_elec = sum(1 for k in b_local if active_so[k] < nocc_so)

        meta = {
            "nact": len(active_so),
            "nelec_active": len(a_occ),
            "active_so": active_so,
            "nocc_so": nocc_so,
            "nso": nso,
            "E0_hf": E0_hf,
            "a_local": a_local,
            "b_local": b_local,
            "n_act_alpha_elec": n_act_alpha_elec,
            "n_act_beta_elec": n_act_beta_elec,
        }
        return chi_1, chi_2, C, meta

    @staticmethod
    def _spinblocked_amplitudes_to_so(t_dict, occ_so, vir_so, nocc_a, nocc_b):
        """Map spin-blocked spatial CCSD amplitudes to the occupied-first SO layout.

        Args:
            t_dict: Spin-blocked amplitudes ``{"ov", "OV", "oovv", "OOVV", "oOvV"}``
                (same-spin T2 already antisymmetrized).
            occ_so, vir_so: SO layout as ``(spatial, spin)`` lists.
            nocc_a, nocc_b: α/β occupied counts (for the virtual local offset).

        Returns:
            ``(t1_so, t2_so)`` full spin-orbital amplitudes (t2 antisymmetric).
        """
        t1_aa, t1_bb = t_dict["ov"], t_dict["OV"]
        t2_aa, t2_bb, t2_ab = t_dict["oovv"], t_dict["OOVV"], t_dict["oOvV"]
        nocc_so, nvir_so = len(occ_so), len(vir_so)

        occ_sg = np.array([s for _, s in occ_so])
        occ_sp = np.array([p for p, _ in occ_so])
        vir_sg = np.array([s for _, s in vir_so])
        vir_lc = np.array([(p - nocc_a) if s == 0 else (p - nocc_b) for p, s in vir_so])

        oa, ob = np.where(occ_sg == 0)[0], np.where(occ_sg == 1)[0]
        va, vb = np.where(vir_sg == 0)[0], np.where(vir_sg == 1)[0]

        t1_so = np.zeros((nocc_so, nvir_so))
        t1_so[np.ix_(oa, va)] = t1_aa[np.ix_(occ_sp[oa], vir_lc[va])]
        t1_so[np.ix_(ob, vb)] = t1_bb[np.ix_(occ_sp[ob], vir_lc[vb])]

        t2_so = np.zeros((nocc_so, nocc_so, nvir_so, nvir_so))
        t2_so[np.ix_(oa, oa, va, va)] = t2_aa[np.ix_(occ_sp[oa], occ_sp[oa], vir_lc[va], vir_lc[va])]
        t2_so[np.ix_(ob, ob, vb, vb)] = t2_bb[np.ix_(occ_sp[ob], occ_sp[ob], vir_lc[vb], vir_lc[vb])]
        ab = t2_ab[np.ix_(occ_sp[oa], occ_sp[ob], vir_lc[va], vir_lc[vb])]  # [i_α, J_β, a_α, B_β]
        t2_so[np.ix_(oa, ob, va, vb)] = ab
        t2_so[np.ix_(ob, oa, vb, va)] = ab.transpose(1, 0, 3, 2)
        t2_so[np.ix_(oa, ob, vb, va)] = -ab.transpose(0, 1, 3, 2)
        t2_so[np.ix_(ob, oa, va, vb)] = -ab.transpose(1, 0, 2, 3)
        return t1_so, t2_so

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

        # BCH expansion of H̄ = e^{-σ} H e^{σ} truncated at each level.
        # [Bauman et al., JCP 151, 014107, Eqs. (20)-(24)]
        #
        # Level 0: H̄ = E_0 + H_N  (bare, no dressing)
        # Level 1: H̄ = E_0 + H_N + [H_N, σ] + ½[[F, σ], σ]
        #          (single commutator of full H_N, double commutator of F only)
        # Level 2: H̄ = E_0 + H_N + [H_N, σ] + ½[[H_N, σ], σ] + ⅙[[[F, σ], σ], σ]
        #          (double commutator of full H_N, triple commutator of F only)
        #
        # The asymmetry (F vs H_N) comes from the DUCC truncation: higher-body
        # commutators of V produce 3-body+ operators that are discarded, while
        # F commutators stay at most 2-body at one order higher.
        #
        # Note: wicked's w.bch_series(E0 + H_N, σ, n) computes the standard BCH
        # e^{-σ} H e^{σ} = H + [H,σ] + ½[[H,σ],σ] + ... truncated at order n,
        # treating F and V symmetrically. The DUCC levels differ from this because
        # they use F at one order higher than V (e.g., level 1 has [[F,σ],σ] but
        # only [H_N,σ]). We therefore build the commutators explicitly rather than
        # calling bch_series.
        if bch_order == 0:
            Hbar = E0op + H_N
        elif bch_order == 1:
            comm1 = w.commutator(H_N, sigma)  # [H_N, σ]
            comm2_F = w.commutator(F, sigma, sigma)  # [[F, σ], σ]
            Hbar = E0op + H_N + comm1
            Hbar.add2(comm2_F, w.rational(1, 2))  # + ½[[F, σ], σ]
        elif bch_order == 2:
            comm1 = w.commutator(H_N, sigma)  # [H_N, σ]
            comm2_HN = w.commutator(H_N, sigma, sigma)  # [[H_N, σ], σ]
            comm3_F = w.commutator(F, sigma, sigma, sigma)  # [[[F, σ], σ], σ]
            Hbar = E0op + H_N + comm1
            Hbar.add2(comm2_HN, w.rational(1, 2))  # + ½[[H_N, σ], σ]
            Hbar.add2(comm3_F, w.rational(1, 6))  # + ⅙[[[F, σ], σ], σ]
        else:
            raise ValueError(f"Unsupported BCH order {bch_order} (must be 0, 1, or 2)")

        # Apply Wick's theorem to fully contract H̄ into normal-ordered
        # 0-, 1-, and 2-body components (discarding 3-body and higher).
        # The result is a set of many-body equations: einsum expressions
        # that evaluate each block (oo, ov, vo, vv, oovv, ...) of the
        # dressed operators f̄ and v̄.  [Kutzelnigg & Mukherjee, JCP 107, 432]
        expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
        mbeq = expr.to_manybody_equation("r")

        o_sl = slice(0, nocc)
        v_sl = slice(nocc, nso)
        f_dict = {
            "oo": f_no[o_sl, o_sl],
            "ov": f_no[o_sl, v_sl],
            "vo": f_no[v_sl, o_sl],
            "vv": f_no[v_sl, v_sl],
        }
        v_dict = {}
        for k1, s1 in [("o", o_sl), ("v", v_sl)]:
            for k2, s2 in [("o", o_sl), ("v", v_sl)]:
                for k3, s3 in [("o", o_sl), ("v", v_sl)]:
                    for k4, s4 in [("o", o_sl), ("v", v_sl)]:
                        v_dict[k1 + k2 + k3 + k4] = v_no[s1, s2, s3, s4]

        t_dict = {
            "ov": t1_ov,
            "vo": t1_ov.T,
            "oovv": t2_oovv,
            "vvoo": t2_oovv.transpose(2, 3, 0, 1),
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
                lines.append(f"    {rv} = np.zeros(({','.join(str(s) for s in shape)}))")
            for eq in eqs:
                lines.append(f"    {eq.compile('einsum')}")
            lines.append(f"    return {rv}")

            ns = {}
            exec("\n".join(lines), {"np": np}, ns)
            result = ns["_eval"](_ScalarDict(E0_hf), f_dict, v_dict, t_dict, nocc, nvir)

            if ndim == 0:
                rs = result
            elif ndim == 2:
                r1b[key] = (rv, ic, np.array(result))
            elif ndim == 4:
                r2b[key] = (rv, ic, np.array(result))

        # Assemble full-space dressed operators f̄_{pq} and v̄_{pqrs}.
        # The 1-body blocks (oo, ov, vo, vv) form f̄.
        fbar = np.zeros((nso, nso))
        for k, (rv, ic, blk) in r1b.items():
            fbar[tuple(slices_map[c] for c in ic)] += blk

        # The 2-body blocks are NOT yet antisymmetrized; wicked produces
        # one representative ordering per block. Antisymmetrize:
        # v̄_{pqrs} = r_{pqrs} - r_{qprs} - r_{pqsr} + r_{qpsr}
        vr = np.zeros((nso,) * 4)
        for k, (rv, ic, blk) in r2b.items():
            vr[tuple(slices_map[c] for c in ic)] += blk
        vbar = vr - vr.transpose(1, 0, 2, 3) - vr.transpose(0, 1, 3, 2) + vr.transpose(1, 0, 3, 2)

        # Restrict to active space → γ (Fermi-vacuum normal-ordered)
        # γ₁[PQ] = f̄[P,Q]  restricted to active indices
        # γ₂[PQRS] = v̄[P,Q,R,S]  restricted to active indices
        nact = len(active_so)
        gamma_1 = fbar[np.ix_(active_so, active_so)]
        gamma_2 = vbar[np.ix_(active_so, active_so, active_so, active_so)]
        aol = [i for i, g in enumerate(active_so) if g < nocc]

        # Convert γ → χ (physical-vacuum normal-ordered).
        # χ₁[PQ] = γ₁[PQ] - Σ_M γ₂[PM,QM]   (subtract active-occupied contraction)
        # χ₂[PQRS] = γ₂[PQRS]                (unchanged)
        # This re-normal-orders from the Fermi vacuum |Φ⟩ to the physical
        # vacuum |0⟩, absorbing the active-occupied contractions into the
        # 1-body term.  [Bauman et al., JCP 151, 014107, Eq. (30)-(31)]
        chi_1 = gamma_1 - np.einsum("pmqm->pq", gamma_2[:, aol, :, :][:, :, :, aol])
        chi_2 = gamma_2.copy()

        # Physical-vacuum scalar (core energy of the downfolded Hamiltonian).
        # C = E₀(scalar from BCH) - Σ_M χ₁[MM] - ½ Σ_{MN} χ₂[MN,MN]
        # This absorbs the active-occupied orbital energies into C so that
        # E_total = E_CI(active) + C.  [Bauman et al., Eq. (32)]
        C = rs
        for m in aol:
            C -= chi_1[m, m]
            for n in aol:
                C -= 0.5 * chi_2[m, n, m, n]

        return chi_1, chi_2, C
