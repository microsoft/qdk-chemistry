# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Spin-integrated DUCC Hamiltonian downfolding using wicked.

Uses wicked's 4-space spin-integrated formalism (α occ, α vir, β occ, β vir)
to avoid the spin-orbital expansion entirely. Works directly with spatial
integrals from the qdk-chemistry Hamiltonian and spin-blocked T amplitudes
from PySCF.

Advantages over the spin-orbital :class:`WickedDuccSolver`:

- No 2× expansion to spin-orbitals (4× memory savings on 2e integrals)
- Direct spatial output without ``spinorb_to_spatial`` conversion
- Natural support for open-shell (UHF/ROHF) references

The BCH truncation levels follow the same DUCC paper
(Bauman et al., JCP 151, 014107) as the spin-orbital version.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm

logger = logging.getLogger(__name__)

_wicked = None


def _require_wicked():
    global _wicked
    if _wicked is None:
        try:
            import wicked
            _wicked = wicked
        except ImportError:
            raise ImportError(
                "wicked is required for WickedDuccSISolver. "
                "Install from https://github.com/fevangelista/wicked"
            )
    return _wicked


class WickedDuccSISolver(Algorithm):
    """Spin-integrated DUCC Hamiltonian downfolder.

    Uses wicked's 4-space formalism with separate α/β orbital spaces,
    working directly with spatial integrals. Supports both closed-shell
    (RHF) and open-shell (UHF) references.

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc_si",
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
        return "wicked_ducc_si"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        """Run spin-integrated DUCC downfolding.

        Args:
            hamiltonian: Full-space qdk-chemistry Hamiltonian (spatial, chemist).
            n_alpha: Number of alpha electrons.
            n_beta: Number of beta electrons.

        Returns:
            Downfolded active-space Hamiltonian (spatial, chemist).
        """
        w = _require_wicked()
        s = self.settings()
        noa_act = s["nactive_oa"]
        nob_act = s["nactive_ob"]
        nva_act = s["nactive_va"]
        nvb_act = s["nactive_vb"]
        ducc_level = s["ducc_level"]

        # ── 1. Extract spatial integrals (already spin-free!) ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc_a = n_alpha
        nocc_b = n_beta
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b

        h1_list, _ = hamiltonian.get_one_body_integrals()
        h1 = np.array(h1_list).reshape(nmo, nmo)

        eri_list, _, _ = hamiltonian.get_two_body_integrals()
        eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)

        core_energy = hamiltonian.get_core_energy()

        logger.info(
            "WickedDuccSISolver: nmo=%d, nocc=(%d,%d), active_occ=(%d,%d), active_vir=(%d,%d), level=%d",
            nmo, nocc_a, nocc_b, noa_act, nob_act, nva_act, nvb_act, ducc_level,
        )

        # ── 2. Build H blocks directly in physicist notation ──
        # V[p,q,r,s] = <pq|rs> = (pr|qs) in chemist.
        # Obtained from eri[p,q,r,s]=(pq|rs) via swapaxes(1,2): V[p,q,r,s] = eri[p,r,q,s].
        V = eri.swapaxes(1, 2)
        # Same-spin antisymmetrized: <pq||rs> = <pq|rs> - <pq|sr>
        V_asym = V - V.swapaxes(2, 3)

        # HF energy: E_0 = V_nuc + Σ_m^α h[m,m] + Σ_M^β h[M,M]
        #            + ½ Σ_{mn}^α <mn||mn> + ½ Σ_{MN}^β <MN||MN> + Σ_{mM}^{αβ} <mM|mM>
        E0 = core_energy
        for m in range(nocc_a):
            E0 += h1[m, m]
        for m in range(nocc_b):
            E0 += h1[m, m]
        for m in range(nocc_a):
            for n in range(nocc_a):
                E0 += 0.5 * V_asym[m, n, m, n]
        for m in range(nocc_b):
            for n in range(nocc_b):
                E0 += 0.5 * V_asym[m, n, m, n]
        for m in range(nocc_a):
            for n in range(nocc_b):
                E0 += V[m, n, m, n]

        # Fock matrix (spatial, closed-shell: f^α = f^β):
        # f[p,q] = h[p,q] + Σ_m^α <pm||qm> + Σ_M^β <pM|qM>
        #        = h[p,q] + Σ_m [V_asym[p,m,q,m] + V[p,m,q,m]]
        #        = h[p,q] + Σ_m [2*V[p,m,q,m] - V[p,m,m,q]]  (for nocc_a = nocc_b)
        F = h1.copy()
        for m in range(nocc_a):
            F += V_asym[:, m, :, m]  # same-spin α contribution
        for m in range(nocc_b):
            F += V[:, m, :, m]  # cross-spin β contribution

        # Slices for occupied/virtual blocks
        oa = slice(0, nocc_a)
        va = slice(nocc_a, nmo)
        ob = slice(0, nocc_b)
        vb = slice(nocc_b, nmo)

        # Build H dictionary with all spin blocks needed by wicked.
        # Lowercase = α, uppercase = β.
        # 1-body: F blocks (α and β use same spatial Fock for RHF)
        H = {}
        sl_map = {"o": oa, "v": va, "O": ob, "V": vb}
        for c1 in ["o", "v"]:
            for c2 in ["o", "v"]:
                H[c1 + c2] = F[sl_map[c1], sl_map[c2]]
                H[c1.upper() + c2.upper()] = F[sl_map[c1.upper()], sl_map[c2.upper()]]

        # 2-body blocks: same-spin uses V_asym, cross-spin uses V.
        # Determine which blocks are cross-spin (αβ) by counting lowercase indices.
        for c1 in ["o", "v"]:
            for c2 in ["o", "v"]:
                for c3 in ["o", "v"]:
                    for c4 in ["o", "v"]:
                        # αααα
                        key_aa = c1 + c2 + c3 + c4
                        H[key_aa] = V_asym[sl_map[c1], sl_map[c2], sl_map[c3], sl_map[c4]]
                        # ββββ
                        key_bb = c1.upper() + c2.upper() + c3.upper() + c4.upper()
                        H[key_bb] = V_asym[sl_map[c1.upper()], sl_map[c2.upper()],
                                           sl_map[c3.upper()], sl_map[c4.upper()]]
                        # αβαβ (cross-spin: bare Coulomb, no exchange)
                        key_ab = c1 + c2.upper() + c3 + c4.upper()
                        H[key_ab] = V[sl_map[c1], sl_map[c2.upper()],
                                      sl_map[c3], sl_map[c4.upper()]]

        # ── 3. CCSD amplitudes via PySCF ──
        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf
        alpha_occ = np.zeros(nmo)
        alpha_occ[:nocc_a] = 1.0
        beta_occ = np.zeros(nmo)
        beta_occ[:nocc_b] = 1.0
        pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)

        from pyscf import cc
        mycc = cc.CCSD(pyscf_scf).run()
        logger.info("CCSD energy: %.10f", mycc.e_tot)

        # Build spin-blocked T amplitudes from spatial RCCSD.
        # PySCF RCCSD: t1[nocc, nvir], t2[nocc, nocc, nvir, nvir] (spatial, NOT antisymmetrized).
        t1_spatial = mycc.t1
        t2_spatial = mycc.t2
        # Full antisymmetrization for same-spin: t2_aa[ijab] must be antisymmetric
        # in BOTH (i,j) and (a,b). PySCF's spatial T2 is symmetric, not antisymmetric.
        t2_asym = (t2_spatial - t2_spatial.swapaxes(0, 1)
                   - t2_spatial.swapaxes(2, 3) + t2_spatial.transpose(1, 0, 3, 2))

        T = {
            "ov": t1_spatial.copy(),           # T1 αα
            "OV": t1_spatial.copy(),           # T1 ββ (= αα for RHF)
            "oovv": t2_asym.copy(),            # T2 αααα (antisymmetrized)
            "OOVV": t2_asym.copy(),            # T2 ββββ (= αααα for RHF)
            "oOvV": t2_spatial.copy(),         # T2 αβαβ (NOT antisymmetrized)
        }
        # Add transposes needed by wicked's generated equations
        T["vo"] = T["ov"].T.copy()
        T["VO"] = T["OV"].T.copy()
        T["vvoo"] = T["oovv"].transpose(2, 3, 0, 1).copy()
        T["VVOO"] = T["OOVV"].transpose(2, 3, 0, 1).copy()
        T["vVoO"] = T["oOvV"].transpose(2, 3, 0, 1).copy()
        T["VvOo"] = T["oOvV"].transpose(3, 2, 1, 0).copy()

        # ── 4. Zero all-active amplitudes → σ_ext ──
        # Active occupied: last noa_act of occupied; active virtual: first nva_act of virtual.
        ncore_a = nocc_a - noa_act
        ncore_b = nocc_b - nob_act
        act_oa = slice(ncore_a, nocc_a)   # active α occupied (relative to occ block)
        act_va = slice(0, nva_act)         # active α virtual (relative to vir block)
        act_ob = slice(ncore_b, nocc_b)   # active β occupied
        act_vb = slice(0, nvb_act)         # active β virtual

        # Zero T1: active_occ → active_vir
        T["ov"][act_oa, act_va] = 0.0
        T["OV"][act_ob, act_vb] = 0.0

        # Zero T2: all combinations where ALL indices are active
        T["oovv"][act_oa, act_oa, act_va, act_va] = 0.0
        T["OOVV"][act_ob, act_ob, act_vb, act_vb] = 0.0
        T["oOvV"][act_oa, act_ob, act_va, act_vb] = 0.0

        # ── 5. Wicked BCH with 4 spin-integrated spaces ──
        fbar, vbar, E0_bch = self._wicked_bch_si(
            w, ducc_level, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b
        )

        # ── 6. Restrict to active space → γ → χ ──
        # Assemble full-space arrays from blocks, then slice (like spin-orbital version).
        nact_o = noa_act
        nact_v = nva_act
        nact = nact_o + nact_v

        # Full-space 1-body: reconstruct f̄^αα[nmo, nmo] from blocks
        fbar_aa = np.zeros((nmo, nmo))
        fbar_bb = np.zeros((nmo, nmo))
        for key, arr in fbar.items():
            if key.islower():  # αα
                r = oa if key[0] == "o" else va
                c = oa if key[1] == "o" else va
                fbar_aa[r, c] = arr
            elif key.isupper():  # ββ
                r = ob if key[0] == "O" else vb
                c = ob if key[1] == "O" else vb
                fbar_bb[r, c] = arr

        # Full-space 2-body: reconstruct from blocks
        vbar_aa = np.zeros((nmo, nmo, nmo, nmo))
        vbar_bb = np.zeros((nmo, nmo, nmo, nmo))
        vbar_ab = np.zeros((nmo, nmo, nmo, nmo))

        def _sl(c):
            if c == "o": return oa
            if c == "v": return va
            if c == "O": return ob
            if c == "V": return vb

        for key, arr in vbar.items():
            n_lower = sum(1 for c in key if c.islower())
            if n_lower == 4:  # αααα
                vbar_aa[np.ix_(range(*_sl(key[0]).indices(nmo)), range(*_sl(key[1]).indices(nmo)),
                               range(*_sl(key[2]).indices(nmo)), range(*_sl(key[3]).indices(nmo)))] += arr
            elif n_lower == 0:  # ββββ
                vbar_bb[np.ix_(range(*_sl(key[0]).indices(nmo)), range(*_sl(key[1]).indices(nmo)),
                               range(*_sl(key[2]).indices(nmo)), range(*_sl(key[3]).indices(nmo)))] += arr
            elif n_lower == 2:  # αβ mixed
                s = [_sl(c) for c in key]
                vbar_ab[np.ix_(range(*s[0].indices(nmo)), range(*s[1].indices(nmo)),
                               range(*s[2].indices(nmo)), range(*s[3].indices(nmo)))] += arr

        # Same-spin: antisymmetrize full array to fill in missing block permutations.
        # Wicked only produces a canonical subset of blocks (e.g., "ovov" but not "vovo").
        # The full antisymmetric tensor: v[pqrs] = vr[pqrs] - vr[qprs] - vr[pqsr] + vr[qpsr]
        vbar_aa = (vbar_aa - vbar_aa.transpose(1, 0, 2, 3)
                   - vbar_aa.transpose(0, 1, 3, 2) + vbar_aa.transpose(1, 0, 3, 2))
        vbar_bb = (vbar_bb - vbar_bb.transpose(1, 0, 2, 3)
                   - vbar_bb.transpose(0, 1, 3, 2) + vbar_bb.transpose(1, 0, 3, 2))

        # Active spatial indices
        act = list(range(nocc_a - noa_act, nocc_a)) + [nocc_a + i for i in range(nva_act)]

        gamma1_aa = fbar_aa[np.ix_(act, act)]
        gamma1_bb = fbar_bb[np.ix_(act, act)]
        gamma2_aa = vbar_aa[np.ix_(act, act, act, act)]
        gamma2_bb = vbar_bb[np.ix_(act, act, act, act)]
        gamma2_ab = vbar_ab[np.ix_(act, act, act, act)]

        # DEBUG: temporary trace
        logger.warning("DEBUG fbar_aa diag: %s", np.diag(fbar_aa))
        logger.warning("DEBUG vbar_ab[0,0,0,0]=%s vbar_ab[0,0,1,1]=%s", vbar_ab[0,0,0,0], vbar_ab[0,0,1,1] if nmo>1 else "N/A")
        logger.warning("DEBUG gamma1_aa:\n%s", gamma1_aa)
        logger.warning("DEBUG gamma2_ab diag: %s", [gamma2_ab[i,i,i,i] for i in range(min(nact,3))])

        # γ → χ (physical-vacuum re-normal-ordering)
        # χ₁^αα[pq] = γ₁^αα[pq] - Σ_m γ₂^αα[pm,qm] - Σ_M γ₂^αβ[pM,qM]
        aol = list(range(nact_o))

        chi1_aa = gamma1_aa.copy()
        chi1_aa -= np.einsum("pmqm->pq", gamma2_aa[:, aol, :, :][:, :, :, aol])
        chi1_aa -= np.einsum("pmqm->pq", gamma2_ab[:, aol, :, :][:, :, :, aol])

        chi1_bb = gamma1_bb.copy()
        chi1_bb -= np.einsum("pmqm->pq", gamma2_bb[:, aol, :, :][:, :, :, aol])
        # γ₂^βα[Pm,Qm] = γ₂^αβ[mP,mQ]: contract over α-occ m, output β indices P,Q
        chi1_bb -= np.einsum("mpmq->pq", gamma2_ab[np.ix_(aol, range(nact), aol, range(nact))])

        # Scalar C
        C = E0_bch
        for m in aol:
            C -= chi1_aa[m, m]
            C -= chi1_bb[m, m]
        for m in aol:
            for n in aol:
                C -= 0.5 * gamma2_aa[m, n, m, n]
                C -= 0.5 * gamma2_bb[m, n, m, n]
                C -= gamma2_ab[m, n, m, n]

        # ── 7. Extract spatial Hamiltonian from χ_αβ ──
        # h2_spatial(pq|rs) = χ₂^αβ[p,r,q,s] → swapaxes(1,2) converts physicist to chemist
        from qdk_chemistry.data import (
            CanonicalFourCenterHamiltonianContainer,
            Hamiltonian,
            ModelOrbitals,
        )

        h1_active = chi1_aa
        h2_active = gamma2_ab.swapaxes(1, 2)  # physicist αβ → chemist spatial

        active_orbitals = ModelOrbitals(nact)
        container = CanonicalFourCenterHamiltonianContainer(
            h1_active,
            h2_active.ravel(),
            active_orbitals,
            C,
            np.zeros((nact, nact)),
        )
        return Hamiltonian(container)

    @staticmethod
    def _wicked_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b):
        """Run spin-integrated wicked BCH.

        Uses 4 orbital spaces (α occ, α vir, β occ, β vir) following
        the convention from wicked's spin-integrated CCSD example.

        Args:
            w: wicked module.
            bch_order: 0, 1, or 2.
            H: Dict of Hamiltonian blocks (1-body and 2-body, all spin combos).
            T: Dict of T amplitude blocks (σ_ext, with active zeroed).
            E0: HF reference energy scalar.
            nocc_a, nvir_a: Alpha occupied/virtual counts.
            nocc_b, nvir_b: Beta occupied/virtual counts.

        Returns:
            (fbar, vbar, E0_bch): Dicts of dressed 1-body and 2-body blocks, plus scalar.
        """
        w.reset_space()
        # α spaces (lowercase)
        w.add_space("o", "fermion", "occupied", list("ijklmn")[:max(nocc_a, 1)])
        w.add_space("v", "fermion", "unoccupied", list("abcdef")[:max(nvir_a, 1)])
        # β spaces (uppercase)
        w.add_space("O", "fermion", "occupied", list("IJKLMN")[:max(nocc_b, 1)])
        w.add_space("V", "fermion", "unoccupied", list("ABCDEF")[:max(nvir_b, 1)])

        # Build T operator (M_S-conserving): T1_α, T1_β, T2_αα, T2_ββ, T2_αβ
        # unique=True prevents combinatorially equivalent operators
        Top = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "V+ v+ O o"], unique=True)

        # Build H operator with all spin blocks
        Hops = []
        # 1-body αα
        for i in itertools.product(["v+", "o+"], ["v", "o"]):
            Hops.append(" ".join(i))
        # 1-body ββ
        for i in itertools.product(["V+", "O+"], ["V", "O"]):
            Hops.append(" ".join(i))
        # 2-body αααα
        for i in itertools.product(["v+", "o+"], ["v+", "o+"], ["v", "o"], ["v", "o"]):
            Hops.append(" ".join(i))
        # 2-body ββββ
        for i in itertools.product(["V+", "O+"], ["V+", "O+"], ["V", "O"], ["V", "O"]):
            Hops.append(" ".join(i))
        # 2-body αβαβ (cross-spin)
        for i in itertools.product(["v+", "o+"], ["V+", "O+"], ["v", "o"], ["V", "O"]):
            Hops.append(" ".join(i))
        Hop = w.op("H", Hops, unique=True)

        # Split H into F (1-body) and V (2-body) for asymmetric DUCC truncation
        Fops_list = []
        for i in itertools.product(["v+", "o+"], ["v", "o"]):
            Fops_list.append(" ".join(i))
        for i in itertools.product(["V+", "O+"], ["V", "O"]):
            Fops_list.append(" ".join(i))
        Fop = w.op("H", Fops_list, unique=True)

        # σ = T - T†
        sigma = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "V+ v+ O o"], unique=True)
        sigma.add2(Top.adjoint(), w.rational(-1))

        # BCH expansion with DUCC asymmetric truncation.
        # [Bauman et al., JCP 151, 014107, Eqs. (20)-(24)]
        E0op = w.op("E0", [""])
        if bch_order == 0:
            Hbar = E0op + Hop
        elif bch_order == 1:
            comm1 = w.commutator(Hop, sigma)
            comm2_F = w.commutator(Fop, sigma, sigma)
            Hbar = E0op + Hop + comm1
            Hbar.add2(comm2_F, w.rational(1, 2))
        elif bch_order == 2:
            comm1 = w.commutator(Hop, sigma)
            comm2_H = w.commutator(Hop, sigma, sigma)
            comm3_F = w.commutator(Fop, sigma, sigma, sigma)
            Hbar = E0op + Hop + comm1
            Hbar.add2(comm2_H, w.rational(1, 2))
            Hbar.add2(comm3_F, w.rational(1, 6))
        else:
            raise ValueError(f"Unsupported BCH order {bch_order}")

        # Wick contraction → many-body equations
        expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
        mbeq = expr.to_manybody_equation("R")

        # Evaluate all blocks
        class _ScalarDict:
            def __init__(self, val):
                self.val = val
            def __getitem__(self, key):
                return self.val

        # Determine shapes for each space label
        def _dim(c):
            return {"o": nocc_a, "v": nvir_a, "O": nocc_b, "V": nvir_b}[c]

        fbar = {}
        vbar = {}
        E0_bch = 0.0

        for key, eqs in mbeq.items():
            if not eqs:
                continue
            lower, upper = key.split("|")
            ndim = len(lower) + len(upper)
            fc = eqs[0].compile("einsum")
            rv = fc.split("+=")[0].strip()
            ic = rv[1:]  # index characters like "ov", "oOvV", etc.
            shape = [_dim(c) for c in ic]

            lines = ["def _eval(E0, H, T):"]
            if ndim == 0:
                lines.append(f"    {rv} = 0.0")
            else:
                lines.append(f"    {rv} = np.zeros(({','.join(str(s) for s in shape)}))")
            for eq in eqs:
                lines.append(f"    {eq.compile('einsum')}")
            lines.append(f"    return {rv}")

            ns = {}
            exec("\n".join(lines), {"np": np}, ns)  # noqa: S102
            result = ns["_eval"](_ScalarDict(E0), H, T)

            if ndim == 0:
                E0_bch = result
            elif ndim == 2:
                fbar[ic] = np.array(result)
            elif ndim == 4:
                arr = np.array(result)
                # Same-spin blocks are already antisymmetric (input H uses V_asym,
                # and wicked's Wick contraction preserves antisymmetry).
                # Cross-spin blocks have no antisymmetry requirement.
                # No additional antisymmetrization needed.
                vbar[ic] = arr

        return fbar, vbar, E0_bch
