# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""4-space spin-integrated DUCC Hamiltonian downfolding using wicked.

Uses wicked's 8-space formalism (c/o/v/e × α/β) to decompose orbitals into
core-occupied, active-occupied, active-virtual, and external-virtual sub-spaces.
This avoids contractions over the full occupied/virtual range, operating instead
on smaller sub-blocks whose dimensions are bounded by the number of core, active,
or external orbitals independently.

Space labels:

- ``c`` / ``C``: core (inactive) occupied (α / β)
- ``o`` / ``O``: active occupied (α / β)
- ``v`` / ``V``: active virtual (α / β)
- ``e`` / ``E``: external (inactive) virtual (α / β)

Advantages over the 2-space :class:`WickedDuccSISolver`:

- Wicked generates equations whose tensor contractions scale with sub-block
  dimensions rather than full occupied/virtual dimensions
- Structural zeroing of all-active T components is enforced at the operator
  level (those components are simply not included in the T definition)
- More efficient for large systems where ncore or next >> nact

The BCH truncation levels follow the same DUCC paper
(Bauman et al., JCP 151, 014107) as the other implementations.
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
            raise ImportError("wickd is required for WickedDucc4SpaceSolver. Install with: pip install wickd")
    return _wicked


class WickedDucc4SpaceSolver(Algorithm):
    """4-space spin-integrated DUCC Hamiltonian downfolder.

    Uses wicked's 8-space formalism (c/o/v/e for each spin) to generate
    BCH equations that contract only within sub-block dimensions. Produces
    the same physics as :class:`WickedDuccSISolver` but with smaller
    intermediate tensors.

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc_4space",
                         nactive_oa=2, nactive_va=3, ducc_level=1)
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
        return "wicked_ducc_4space"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        """Run 4-space spin-integrated DUCC downfolding.

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

        # ── 1. Extract spatial integrals ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc_a, nocc_b = n_alpha, n_beta
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b

        ncore_a = nocc_a - noa_act
        ncore_b = nocc_b - nob_act
        next_a = nvir_a - nva_act
        next_b = nvir_b - nvb_act

        h1_list, _ = hamiltonian.get_one_body_integrals()
        h1 = np.array(h1_list).reshape(nmo, nmo)

        eri_list, _, _ = hamiltonian.get_two_body_integrals()
        eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)

        core_energy = hamiltonian.get_core_energy()

        logger.info(
            "WickedDucc4SpaceSolver: nmo=%d, nocc=(%d,%d), ncore=(%d,%d), "
            "nact_occ=(%d,%d), nact_vir=(%d,%d), next=(%d,%d), level=%d",
            nmo,
            nocc_a,
            nocc_b,
            ncore_a,
            ncore_b,
            noa_act,
            nob_act,
            nva_act,
            nvb_act,
            next_a,
            next_b,
            ducc_level,
        )

        # ── 2. Build physicist-notation integrals and sub-space slices ──
        V = eri.swapaxes(1, 2)
        V_asym = V - V.swapaxes(2, 3)

        # Fock matrix: f[p,q] = h[p,q] + Σ_m <pm||qm>_αα + Σ_M <pM|qM>_αβ
        F = h1.copy()
        for m in range(nocc_a):
            F += V_asym[:, m, :, m]
        for m in range(nocc_b):
            F += V[:, m, :, m]

        # E₀ = V_nuc + Σ h[mm] + ½Σ<mn||mn> + Σ<mM|mM>
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

        # Sub-space slices in the full nmo basis
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
        spaces_a = ["c", "o", "v", "e"]
        spaces_b = ["C", "O", "V", "E"]
        dim = {k: v.stop - v.start for k, v in sl.items()}

        # ── 3. Build H dictionary for all sub-space block combinations ──
        H = {}
        # 1-body: same-spin
        for c1 in spaces_a:
            for c2 in spaces_a:
                H[c1 + c2] = F[sl[c1], sl[c2]]
        for c1 in spaces_b:
            for c2 in spaces_b:
                H[c1 + c2] = F[sl[c1], sl[c2]]
        # 2-body: αα antisymmetrized
        for c1, c2, c3, c4 in itertools.product(spaces_a, repeat=4):
            H[c1 + c2 + c3 + c4] = V_asym[sl[c1], sl[c2], sl[c3], sl[c4]]
        # 2-body: ββ antisymmetrized
        for c1, c2, c3, c4 in itertools.product(spaces_b, repeat=4):
            H[c1 + c2 + c3 + c4] = V_asym[sl[c1], sl[c2], sl[c3], sl[c4]]
        # 2-body: αβ (no antisymmetry — Coulomb only)
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

        # ── 5. Build T dictionary with sub-space blocks ──
        T = self._build_T_dict(t1, t2, t2_asym, sl, nocc_a, nocc_b, spaces_a, spaces_b)

        # ── 6. Wicked BCH ──
        fbar, vbar, E0_bch = self._wicked_bch_4space(w, ducc_level, H, T, E0, dim, spaces_a, spaces_b)

        # ── 7. Extract active-space Hamiltonian ──
        nact = noa_act + nva_act
        return self._extract_active_hamiltonian(fbar, vbar, E0_bch, nact, noa_act, dim, sl, nmo)

    @staticmethod
    def _build_T_dict(t1, t2, t2_asym, sl, nocc_a, nocc_b, spaces_a, spaces_b):
        """Build the T amplitude dictionary with proper index conventions.

        Follows the same pattern as the 2-space implementation: store
        de-excitation entries directly from PySCF (natural index order),
        then derive excitation entries via full index reversal (transpose
        all axes). This avoids manual index reordering that can introduce
        sign errors for antisymmetric same-spin amplitudes.
        """
        occ_a = ["c", "o"]
        vir_a = ["v", "e"]
        occ_b = ["C", "O"]
        vir_b = ["V", "E"]

        def _is_all_active(o_chars, v_chars):
            """True if all occupied indices are active and all virtual are active."""
            return all(c in ("o", "O") for c in o_chars) and all(c in ("v", "V") for c in v_chars)

        T = {}

        # ── T1 blocks (same-spin α and β) ──
        # All-active T1 (o→v, O→V) is excluded from σ_ext — no dict entry needed.
        for v_sp in vir_a:
            for o_sp in occ_a:
                if _is_all_active([o_sp], [v_sp]):
                    continue
                o_sl = sl[o_sp]
                v_sl = slice(sl[v_sp].start - nocc_a, sl[v_sp].stop - nocc_a)
                T[v_sp + o_sp] = t1[o_sl, v_sl].T.copy()
                T[o_sp + v_sp] = t1[o_sl, v_sl].copy()
        for v_sp in vir_b:
            for o_sp in occ_b:
                if _is_all_active([o_sp], [v_sp]):
                    continue
                o_sl = sl[o_sp]
                v_sl = slice(sl[v_sp].start - nocc_b, sl[v_sp].stop - nocc_b)
                T[v_sp + o_sp] = t1[o_sl, v_sl].T.copy()
                T[o_sp + v_sp] = t1[o_sl, v_sl].copy()

        # ── T2 same-spin blocks (αα and ββ) ──
        # All-active T2 (oo→vv, OO→VV) is excluded from σ_ext.
        # Store de-excitation directly from PySCF, derive excitation via
        # full index reversal — matches 2-space pattern, avoids manual
        # index reordering that previously caused a sign bug.
        for o1 in occ_a:
            for o2 in occ_a:
                for v1 in vir_a:
                    for v2 in vir_a:
                        if _is_all_active([o1, o2], [v1, v2]):
                            continue
                        o1_sl, o2_sl = sl[o1], sl[o2]
                        v1_sl = slice(sl[v1].start - nocc_a, sl[v1].stop - nocc_a)
                        v2_sl = slice(sl[v2].start - nocc_a, sl[v2].stop - nocc_a)
                        # De-excitation key (natural PySCF order)
                        deexc_key = o1 + o2 + v1 + v2
                        T[deexc_key] = t2_asym[o1_sl, o2_sl, v1_sl, v2_sl].copy()
                        T[deexc_key.upper()] = T[deexc_key].copy()
                        # Excitation key: derive via full index reversal
                        exc_key = deexc_key[::-1]
                        T[exc_key] = T[deexc_key].transpose(3, 2, 1, 0).copy()
                        T[exc_key.upper()] = T[exc_key].copy()

        # ── T2 cross-spin blocks (αβ) ──
        # All-active αβ T2 (oO→vV) is excluded from σ_ext.
        # Same pattern: store de-excitation directly, derive excitation.
        for oa in occ_a:
            for ob in occ_b:
                for va in vir_a:
                    for vb in vir_b:
                        if _is_all_active([oa, ob], [va, vb]):
                            continue
                        oa_sl, ob_sl = sl[oa], sl[ob]
                        va_sl = slice(sl[va].start - nocc_a, sl[va].stop - nocc_a)
                        vb_sl = slice(sl[vb].start - nocc_b, sl[vb].stop - nocc_b)
                        # De-excitation key (natural PySCF order)
                        deexc_key = oa + ob + va + vb
                        T[deexc_key] = t2[oa_sl, ob_sl, va_sl, vb_sl].copy()
                        # Excitation keys: derive via transpose
                        T[deexc_key[::-1]] = T[deexc_key].transpose(3, 2, 1, 0).copy()
                        T[va + vb + oa + ob] = T[deexc_key].transpose(2, 3, 0, 1).copy()

        return T

    @staticmethod
    def _wicked_bch_4space(w, bch_order, H, T, E0, dim, spaces_a, spaces_b):
        """Run 4-space wicked BCH. Returns (fbar_dict, vbar_dict, E0_scalar)."""
        w.reset_space()

        # Define 8 orbital spaces with appropriate index labels
        _idx_labels = {
            "c": ["i"],
            "o": list("jk")[: max(dim["o"], 1)],
            "v": list("ab")[: max(dim["v"], 1)],
            "e": list("bcde")[: max(dim["e"], 1)],
            "C": ["I"],
            "O": list("JK")[: max(dim["O"], 1)],
            "V": list("AB")[: max(dim["V"], 1)],
            "E": list("BCDE")[: max(dim["E"], 1)],
        }
        for space in spaces_a + spaces_b:
            occ_type = "occupied" if space.lower() in ("c", "o") else "unoccupied"
            w.add_space(space, "fermion", occ_type, _idx_labels[space])

        # T operator: all excitation components EXCEPT all-active (structurally zero)
        occ_a = ["c", "o"]
        vir_a = ["v", "e"]
        occ_b = ["C", "O"]
        vir_b = ["V", "E"]

        def _is_all_active(o_chars, v_chars):
            return all(c in ("o", "O") for c in o_chars) and all(c in ("v", "V") for c in v_chars)

        # T1: exclude all-active (o→v, O→V) — structurally zero in σ_ext
        t1_comps = []
        for v_sp in vir_a:
            for o_sp in occ_a:
                if not _is_all_active([o_sp], [v_sp]):
                    t1_comps.append(f"{v_sp}+ {o_sp}")
        for v_sp in vir_b:
            for o_sp in occ_b:
                if not _is_all_active([o_sp], [v_sp]):
                    t1_comps.append(f"{v_sp}+ {o_sp}")

        # T2: exclude all-active (oo→vv, OO→VV, oO→vV)
        t2_aa = [
            f"{a1}+ {a2}+ {i2} {i1}"
            for i1 in occ_a
            for i2 in occ_a
            for a1 in vir_a
            for a2 in vir_a
            if not _is_all_active([i1, i2], [a1, a2])
        ]
        t2_bb = [
            f"{a1}+ {a2}+ {i2} {i1}"
            for i1 in occ_b
            for i2 in occ_b
            for a1 in vir_b
            for a2 in vir_b
            if not _is_all_active([i1, i2], [a1, a2])
        ]
        t2_ab = [
            f"{ab}+ {aa}+ {ib} {ia}"
            for ia in occ_a
            for ib in occ_b
            for aa in vir_a
            for ab in vir_b
            if not _is_all_active([ia, ib], [aa, ab])
        ]

        all_T = t1_comps + t2_aa + t2_bb + t2_ab
        Top = w.op("T", all_T, unique=True)

        # H operator: all 1-body and 2-body components
        h1_ops = [f"{c1}+ {c2}" for c1 in spaces_a for c2 in spaces_a]
        h1_ops += [f"{c1}+ {c2}" for c1 in spaces_b for c2 in spaces_b]

        h2_aa_ops = [f"{c1}+ {c2}+ {c4} {c3}" for c1, c2, c3, c4 in itertools.product(spaces_a, repeat=4)]
        h2_bb_ops = [f"{c1}+ {c2}+ {c4} {c3}" for c1, c2, c3, c4 in itertools.product(spaces_b, repeat=4)]
        h2_ab_ops = [
            f"{c1}+ {c2}+ {c4} {c3}"
            for c1, c3 in itertools.product(spaces_a, repeat=2)
            for c2, c4 in itertools.product(spaces_b, repeat=2)
        ]

        Hop = w.op("H", h1_ops + h2_aa_ops + h2_bb_ops + h2_ab_ops, unique=True)
        Fop = w.op("H", h1_ops, unique=True)

        # σ = T - T†
        sigma = w.op("T", all_T, unique=True)
        sigma.add2(Top.adjoint(), w.rational(-1))

        # BCH expansion
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

        # Evaluate compiled einsum equations
        class _S:
            def __init__(self, v):
                self.val = v

            def __getitem__(self, k):
                return self.val

        # Only evaluate output blocks needed for the active-space Hamiltonian:
        # scalar (E0), active 1-body (oo, ov, vo, vv and β equivalents),
        # and active 2-body (all combinations of o/v and O/V).
        active_a = {"o", "v"}
        active_b = {"O", "V"}

        def _is_active_output(ic):
            """True if all free indices belong to active sub-spaces."""
            return all(c in active_a or c in active_b for c in ic)

        fbar, vbar, E0_bch = {}, {}, 0.0
        n_skipped = 0
        for key, eqs in mbeq.items():
            if not eqs:
                continue
            ndim = len(key.replace("|", ""))
            fc = eqs[0].compile("einsum")
            rv = fc.split("+=")[0].strip()
            ic = rv[1:]
            # Skip non-active output blocks — they don't contribute to the
            # active-space dressed Hamiltonian.
            if ndim > 0 and not _is_active_output(ic):
                n_skipped += 1
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

        logger.info("Skipped %d non-active output blocks", n_skipped)
        return fbar, vbar, E0_bch

    @staticmethod
    def _extract_active_hamiltonian(fbar, vbar, E0_bch, nact, noa_act, dim, sl, nmo):
        """Extract the active-space Hamiltonian from BCH output blocks.

        Assembles only the active (o+v) sub-blocks from the per-sub-space output,
        antisymmetrizes same-spin, and computes χ₁ and C.
        """
        from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, ModelOrbitals

        # Map sub-space characters to their offset in the nact-sized active array
        # Active order: [active_occ(0..noa_act-1), active_vir(noa_act..nact-1)]
        _off = {"o": 0, "O": 0, "v": noa_act, "V": noa_act}
        active_spaces_a = ["o", "v"]
        active_spaces_b = ["O", "V"]

        # Assemble g1 (1-body)
        g1_aa = np.zeros((nact, nact))
        g1_bb = np.zeros((nact, nact))
        for key, arr in fbar.items():
            if len(key) != 2:
                continue
            c0, c1 = key[0], key[1]
            if c0 in active_spaces_a and c1 in active_spaces_a:
                r0, c0_off = _off[c0], _off[c1]
                g1_aa[r0 : r0 + arr.shape[0], c0_off : c0_off + arr.shape[1]] = arr
            elif c0 in active_spaces_b and c1 in active_spaces_b:
                r0, c0_off = _off[c0], _off[c1]
                g1_bb[r0 : r0 + arr.shape[0], c0_off : c0_off + arr.shape[1]] = arr

        # Assemble g2 (2-body) — only active sub-space blocks contribute
        g2_aa_raw = np.zeros((nact,) * 4)
        g2_bb_raw = np.zeros((nact,) * 4)
        g2_ab = np.zeros((nact,) * 4)
        for key, arr in vbar.items():
            if len(key) != 4:
                continue
            n_lower = sum(1 for c in key if c.islower())
            # Check all indices are in active sub-spaces
            if n_lower == 4:
                if all(c in ("o", "v") for c in key):
                    o = [_off[c] for c in key]
                    s = arr.shape
                    target_sl = tuple(slice(o[i], o[i] + s[i]) for i in range(4))
                    g2_aa_raw[target_sl] += arr
            elif n_lower == 0:
                if all(c in ("O", "V") for c in key):
                    o = [_off[c] for c in key]
                    s = arr.shape
                    target_sl = tuple(slice(o[i], o[i] + s[i]) for i in range(4))
                    g2_bb_raw[target_sl] += arr
            elif n_lower == 2:
                # αβ: check alpha chars in active α, beta chars in active β
                alpha_active = all(c in ("o", "v") for c in key if c.islower())
                beta_active = all(c in ("O", "V") for c in key if c.isupper())
                if alpha_active and beta_active:
                    o = [_off[c] for c in key]
                    s = arr.shape
                    target_sl = tuple(slice(o[i], o[i] + s[i]) for i in range(4))
                    g2_ab[target_sl] += arr

        # Antisymmetrize same-spin
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

        # χ₁ = γ₁ - Σ_m γ₂[pm,qm]
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

        # C = E₀_BCH - Σ_m χ₁[mm] - ½ΣΣ γ₂[mn,mn]
        C = E0_bch
        for m in aol:
            C -= chi1_aa[m, m] + chi1_bb[m, m]
            for n in aol:
                C -= 0.5 * g2_aa[m, n, m, n] + 0.5 * g2_bb[m, n, m, n] + g2_ab[m, n, m, n]

        # Package: χ₁_αα as 1-body, g2_ab (chemist notation) as 2-body
        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                chi1_aa, g2_ab.swapaxes(1, 2).ravel(), ModelOrbitals(nact), C, np.zeros((nact, nact))
            )
        )
