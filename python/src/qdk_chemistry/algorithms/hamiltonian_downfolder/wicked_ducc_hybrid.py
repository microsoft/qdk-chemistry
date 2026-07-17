# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Hybrid 4-space DUCC: gen_op H (uniform) + 4-space T (no all-active).

Defines 8 orbital spaces (c/o/v/e × α/β) but uses wicked's ``gen_op`` to
define H as a uniform operator over the union of spaces within each spin.
This drastically reduces the number of H operator components compared to
explicit enumeration, while T retains the 4-space decomposition to
structurally exclude all-active amplitudes from σ_ext.

Scalability compared to the original 4-space implementation:

- H components: ~332 (gen_op) vs ~800 (manual enumeration)
- T components: 37 (same — irreducible for 8 disjoint spaces)
- BCH(2): tractable (~minutes) vs killed (>hours)

The key insight: H is uniform across all orbital sub-spaces (same Fock
matrix, same integrals), so splitting it into sub-blocks at the symbolic
level adds no information — only combinatorial overhead. ``gen_op``
exploits the antisymmetry of the 2-body operator to produce the minimal
canonical component set.
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
            import wickd as wicked

            _wicked = wicked
        except ImportError:
            raise ImportError("wickd is required for WickedDuccHybridSolver. Install with: pip install wickd")
    return _wicked


class WickedDuccHybridSolver(Algorithm):
    """Hybrid 4-space DUCC: gen_op H + structural T exclusion.

    Combines gen_op's compact H definition with the 4-space T decomposition
    that structurally excludes all-active amplitudes. Tractable for BCH(2).

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc_hybrid",
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
        return "wicked_ducc_hybrid"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        w = _require_wicked()
        s = self.settings()
        noa_act = s["nactive_oa"]
        nob_act = s["nactive_ob"]
        nva_act = s["nactive_va"]
        nvb_act = s["nactive_vb"]
        ducc_level = s["ducc_level"]

        # ── 1. Extract spatial integrals ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc_a, nocc_b = n_alpha, n_beta
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b
        ncore_a = nocc_a - noa_act
        ncore_b = nocc_b - nob_act

        h1_list, _ = hamiltonian.get_one_body_integrals()
        h1 = np.array(h1_list).reshape(nmo, nmo)
        eri_list, _, _ = hamiltonian.get_two_body_integrals()
        eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)
        core_energy = hamiltonian.get_core_energy()

        # ── 2. Normal-ordered integrals ──
        V = eri.swapaxes(1, 2)
        V_asym = V - V.swapaxes(2, 3)

        F = h1.copy()
        for m in range(nocc_a):
            F += V_asym[:, m, :, m]
        for m in range(nocc_b):
            F += V[:, m, :, m]

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

        # Sub-space slices
        sl = {
            "c": slice(0, ncore_a),
            "o": slice(ncore_a, nocc_a),
            "v": slice(nocc_a, nocc_a + nva_act),
            "e": slice(nocc_a + nva_act, nmo),
            "C": slice(0, ncore_b),
            "O": slice(ncore_b, nocc_b),
            "V": slice(nocc_b, nocc_b + nvb_act),
            "E": slice(nocc_b + nvb_act, nmo),
        }
        dim = {k: v.stop - v.start for k, v in sl.items()}

        # ── 3. H dictionary ──
        # gen_op produces keys like "cove", "ccvv" etc. — each key maps to a
        # sub-block of the full Fock/V arrays via the space slices.
        H = {}
        spaces_a = ["c", "o", "v", "e"]
        spaces_b = ["C", "O", "V", "E"]
        # 1-body
        for c1 in spaces_a:
            for c2 in spaces_a:
                H[c1 + c2] = F[sl[c1], sl[c2]]
        for c1 in spaces_b:
            for c2 in spaces_b:
                H[c1 + c2] = F[sl[c1], sl[c2]]
        # 2-body αα
        for c1, c2, c3, c4 in itertools.product(spaces_a, repeat=4):
            H[c1 + c2 + c3 + c4] = V_asym[sl[c1], sl[c2], sl[c3], sl[c4]]
        # 2-body ββ
        for c1, c2, c3, c4 in itertools.product(spaces_b, repeat=4):
            H[c1 + c2 + c3 + c4] = V_asym[sl[c1], sl[c2], sl[c3], sl[c4]]
        # 2-body αβ
        for c1, c3 in itertools.product(spaces_a, repeat=2):
            for c2, c4 in itertools.product(spaces_b, repeat=2):
                H[c1 + c2 + c3 + c4] = V[sl[c1], sl[c2], sl[c3], sl[c4]]

        # ── 4. CCSD amplitudes ──
        from pyscf import cc

        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf

        alpha_occ = np.zeros(nmo)
        alpha_occ[:nocc_a] = 1.0
        beta_occ = np.zeros(nmo)
        beta_occ[:nocc_b] = 1.0
        mycc = cc.CCSD(hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)).run()
        logger.info("CCSD energy: %.10f", mycc.e_tot)

        t1 = mycc.t1
        t2 = mycc.t2
        t2_asym = t2 - t2.swapaxes(2, 3)

        # ── 5. T dictionary (4-space, no all-active) ──
        T = self._build_T_dict(t1, t2, t2_asym, sl, nocc_a, nocc_b)

        # ── 6. Wicked BCH (hybrid: gen_op H + 4-space T) ──
        fbar, vbar, E0_bch = self._wicked_bch_hybrid(w, ducc_level, H, T, E0, dim)

        # ── 7. Extract active-space Hamiltonian ──
        nact = noa_act + nva_act
        return self._extract_active(fbar, vbar, E0_bch, nact, noa_act)

    @staticmethod
    def _build_T_dict(t1, t2, t2_asym, sl, nocc_a, nocc_b):
        """Build T amplitude dict. De-excitation direct, excitation via transpose."""
        occ_a, vir_a = ["c", "o"], ["v", "e"]
        occ_b, vir_b = ["C", "O"], ["V", "E"]

        def _aa(o, v):
            return all(c in ("o", "O") for c in o) and all(c in ("v", "V") for c in v)

        T = {}

        # T1
        for v_sp in vir_a:
            for o_sp in occ_a:
                if _aa([o_sp], [v_sp]):
                    continue
                o_sl = sl[o_sp]
                v_sl = slice(sl[v_sp].start - nocc_a, sl[v_sp].stop - nocc_a)
                T[o_sp + v_sp] = t1[o_sl, v_sl].copy()
                T[v_sp + o_sp] = t1[o_sl, v_sl].T.copy()
        for v_sp in vir_b:
            for o_sp in occ_b:
                if _aa([o_sp], [v_sp]):
                    continue
                o_sl = sl[o_sp]
                v_sl = slice(sl[v_sp].start - nocc_b, sl[v_sp].stop - nocc_b)
                T[o_sp + v_sp] = t1[o_sl, v_sl].copy()
                T[v_sp + o_sp] = t1[o_sl, v_sl].T.copy()

        # T2 same-spin
        for o1 in occ_a:
            for o2 in occ_a:
                for v1 in vir_a:
                    for v2 in vir_a:
                        if _aa([o1, o2], [v1, v2]):
                            continue
                        o1_sl, o2_sl = sl[o1], sl[o2]
                        v1_sl = slice(sl[v1].start - nocc_a, sl[v1].stop - nocc_a)
                        v2_sl = slice(sl[v2].start - nocc_a, sl[v2].stop - nocc_a)
                        dk = o1 + o2 + v1 + v2
                        T[dk] = t2_asym[o1_sl, o2_sl, v1_sl, v2_sl].copy()
                        T[dk.upper()] = T[dk].copy()
                        ek = dk[::-1]
                        T[ek] = T[dk].transpose(3, 2, 1, 0).copy()
                        T[ek.upper()] = T[ek].copy()

        # T2 cross-spin
        for oa in occ_a:
            for ob in occ_b:
                for va in vir_a:
                    for vb in vir_b:
                        if _aa([oa, ob], [va, vb]):
                            continue
                        oa_sl, ob_sl = sl[oa], sl[ob]
                        va_sl = slice(sl[va].start - nocc_a, sl[va].stop - nocc_a)
                        vb_sl = slice(sl[vb].start - nocc_b, sl[vb].stop - nocc_b)
                        dk = oa + ob + va + vb
                        T[dk] = t2[oa_sl, ob_sl, va_sl, vb_sl].copy()
                        T[dk[::-1]] = T[dk].transpose(3, 2, 1, 0).copy()
                        T[va + vb + oa + ob] = T[dk].transpose(2, 3, 0, 1).copy()

        return T

    @staticmethod
    def _wicked_bch_hybrid(w, bch_order, H, T, E0, dim):
        """Hybrid BCH: gen_op H (compact) + 4-space T (no all-active)."""
        w.reset_space()

        spaces_a = ["c", "o", "v", "e"]
        spaces_b = ["C", "O", "V", "E"]
        _idx = {
            "c": ["i"],
            "o": list("jk")[: max(dim["o"], 1)],
            "v": list("ab")[: max(dim["v"], 1)],
            "e": list("bcde")[: max(dim["e"], 1)],
            "C": ["I"],
            "O": list("JK")[: max(dim["O"], 1)],
            "V": list("AB")[: max(dim["V"], 1)],
            "E": list("BCDE")[: max(dim["E"], 1)],
        }
        for sp in spaces_a + spaces_b:
            occ = "occupied" if sp.lower() in ("c", "o") else "unoccupied"
            w.add_space(sp, "fermion", occ, _idx[sp])

        # ── H: gen_op over union of spaces (compact, exploits antisymmetry) ──
        Hop = (
            w.utils.gen_op("H", 1, "cove", "cove")
            + w.utils.gen_op("H", 1, "COVE", "COVE")
            + w.utils.gen_op("H", 2, "cove", "cove")
            + w.utils.gen_op("H", 2, "COVE", "COVE")
        )
        # αβ H: manual enumeration (gen_op mixes spin structure incorrectly)
        h2_ab_ops = [
            f"{c1}+ {c2}+ {c4} {c3}"
            for c1, c3 in itertools.product(spaces_a, repeat=2)
            for c2, c4 in itertools.product(spaces_b, repeat=2)
        ]
        Hop += w.op("H", h2_ab_ops, unique=True)

        Fop = w.utils.gen_op("H", 1, "cove", "cove") + w.utils.gen_op("H", 1, "COVE", "COVE")

        # ── T: 4-space decomposition, exclude all-active ──
        occ_a, vir_a = ["c", "o"], ["v", "e"]
        occ_b, vir_b = ["C", "O"], ["V", "E"]

        def _aa(o, v):
            return all(c in ("o", "O") for c in o) and all(c in ("v", "V") for c in v)

        t1_comps = [f"{v}+ {o}" for v in vir_a for o in occ_a if not _aa([o], [v])]
        t1_comps += [f"{v}+ {o}" for v in vir_b for o in occ_b if not _aa([o], [v])]
        t2_aa = [
            f"{a1}+ {a2}+ {i2} {i1}"
            for i1 in occ_a
            for i2 in occ_a
            for a1 in vir_a
            for a2 in vir_a
            if not _aa([i1, i2], [a1, a2])
        ]
        t2_bb = [
            f"{a1}+ {a2}+ {i2} {i1}"
            for i1 in occ_b
            for i2 in occ_b
            for a1 in vir_b
            for a2 in vir_b
            if not _aa([i1, i2], [a1, a2])
        ]
        t2_ab = [
            f"{ab}+ {aa}+ {ib} {ia}"
            for ia in occ_a
            for ib in occ_b
            for aa in vir_a
            for ab in vir_b
            if not _aa([ia, ib], [aa, ab])
        ]
        all_T = t1_comps + t2_aa + t2_bb + t2_ab
        Top = w.op("T", all_T, unique=True)

        # σ = T - T†
        sigma = w.op("T", all_T, unique=True)
        sigma.add2(Top.adjoint(), w.rational(-1))

        # ── BCH expansion ──
        E0op = w.op("E0", [""])
        if bch_order == 0:
            Hbar = E0op + Hop
        elif bch_order == 1:
            Hbar = E0op + Hop + w.commutator(Hop, sigma)
            Hbar.add2(w.commutator(Fop, sigma, sigma), w.rational(1, 2))
        elif bch_order == 2:
            Hbar = E0op + Hop + w.commutator(Hop, sigma)
            Hbar.add2(w.commutator(Hop, sigma, sigma), w.rational(1, 2))
            Hbar.add2(w.commutator(Fop, sigma, sigma, sigma), w.rational(1, 6))
        else:
            raise ValueError(f"Unsupported BCH order {bch_order}")

        expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
        mbeq = expr.to_manybody_equation("R")

        # ── Evaluate only active output blocks ──
        class _S:
            def __init__(self, v):
                self.val = v

            def __getitem__(self, k):
                return self.val

        active = {"o", "v", "O", "V"}

        fbar, vbar, E0_bch = {}, {}, 0.0
        for key, eqs in mbeq.items():
            if not eqs:
                continue
            ndim = len(key.replace("|", ""))
            fc = eqs[0].compile("einsum")
            rv = fc.split("+=")[0].strip()
            ic = rv[1:]
            if ndim > 0 and not all(c in active for c in ic):
                continue
            shape = [dim[c] for c in ic]
            lines = ["def _e(E0,H,T):"]
            lines.append(f"    {rv}={'0.0' if ndim == 0 else 'np.zeros((' + ','.join(str(s) for s in shape) + '))'}")
            for eq in eqs:
                lines.append(f"    {eq.compile('einsum')}")
            lines.append(f"    return {rv}")
            ns = {}
            exec("\n".join(lines), {"np": np}, ns)
            result = ns["_e"](_S(E0), H, T)
            if ndim == 0:
                E0_bch = result
            elif ndim == 2:
                fbar[ic] = np.array(result)
            elif ndim == 4:
                vbar[ic] = np.array(result)

        return fbar, vbar, E0_bch

    @staticmethod
    def _extract_active(fbar, vbar, E0_bch, nact, noa_act):
        """Extract active-space Hamiltonian from BCH output."""
        from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, ModelOrbitals

        _off = {"o": 0, "O": 0, "v": noa_act, "V": noa_act}

        g1_aa = np.zeros((nact, nact))
        g1_bb = np.zeros((nact, nact))
        for key, arr in fbar.items():
            r0, c0 = _off[key[0]], _off[key[1]]
            if key.islower():
                g1_aa[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = arr
            elif key.isupper():
                g1_bb[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = arr

        g2_aa_raw = np.zeros((nact,) * 4)
        g2_bb_raw = np.zeros((nact,) * 4)
        g2_ab = np.zeros((nact,) * 4)
        for key, arr in vbar.items():
            n_lower = sum(1 for c in key if c.islower())
            o = [_off[c] for c in key]
            s = arr.shape
            target = tuple(slice(o[i], o[i] + s[i]) for i in range(4))
            if n_lower == 4:
                g2_aa_raw[target] += arr
            elif n_lower == 0:
                g2_bb_raw[target] += arr
            elif n_lower == 2:
                g2_ab[target] += arr

        g2_aa = (
            g2_aa_raw
            - g2_aa_raw.transpose(1, 0, 2, 3)
            - g2_aa_raw.transpose(0, 1, 3, 2)
            + g2_aa_raw.transpose(1, 0, 3, 2)
        )
        g2_bb = (
            g2_bb_raw
            - g2_bb_raw.transpose(1, 0, 2, 3)
            - g2_bb_raw.transpose(0, 1, 3, 2)
            + g2_bb_raw.transpose(1, 0, 3, 2)
        )

        aol = list(range(noa_act))
        chi1_aa = (
            g1_aa
            - np.einsum("pmqm->pq", g2_aa[:, aol, :, :][:, :, :, aol])
            - np.einsum("pmqm->pq", g2_ab[:, aol, :, :][:, :, :, aol])
        )
        chi1_bb = (
            g1_bb
            - np.einsum("pmqm->pq", g2_bb[:, aol, :, :][:, :, :, aol])
            - np.einsum("mpmq->pq", g2_ab[np.ix_(aol, range(nact), aol, range(nact))])
        )

        C = E0_bch
        for m in aol:
            C -= chi1_aa[m, m] + chi1_bb[m, m]
            for n in aol:
                C -= 0.5 * g2_aa[m, n, m, n] + 0.5 * g2_bb[m, n, m, n] + g2_ab[m, n, m, n]

        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                chi1_aa, g2_ab.swapaxes(1, 2).ravel(), ModelOrbitals(nact), C, np.zeros((nact, nact))
            )
        )
