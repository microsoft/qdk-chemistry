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

        # ── 1. Extract spatial integrals from Hamiltonian ──
        # No spin-orbital expansion needed — work directly with spatial (nmo × nmo) arrays.
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc_a, nocc_b = n_alpha, n_beta
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b

        h1_list, _ = hamiltonian.get_one_body_integrals()
        h1 = np.array(h1_list).reshape(nmo, nmo)

        eri_list, _, _ = hamiltonian.get_two_body_integrals()
        eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)

        core_energy = hamiltonian.get_core_energy()

        logger.info(
            "WickedDuccSISolver: nmo=%d, nocc=(%d,%d), active=(%d,%d,%d,%d), level=%d",
            nmo, nocc_a, nocc_b, noa_act, nob_act, nva_act, nvb_act, ducc_level,
        )

        # ── 2. Build Hamiltonian blocks in physicist notation ──
        # V[p,q,r,s] = <pq|rs> = (pr|qs)_chemist  [swapaxes(1,2) of eri]
        V = eri.swapaxes(1, 2)
        # Same-spin antisymmetrized: <pq||rs> = <pq|rs> - <pq|sr>
        V_asym = V - V.swapaxes(2, 3)

        # E₀ = V_nuc + Σ_m h[mm] + Σ_M h[MM] + ½Σ_{mn}<mn||mn> + ½Σ_{MN}<MN||MN> + Σ_{mM}<mM|mM>
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

        # f[p,q] = h[p,q] + Σ_m <pm||qm> + Σ_M <pM|qM>
        F = h1.copy()
        for m in range(nocc_a):
            F += V_asym[:, m, :, m]
        for m in range(nocc_b):
            F += V[:, m, :, m]

        # Build H dictionary (lowercase=α, uppercase=β)
        oa, va = slice(0, nocc_a), slice(nocc_a, nmo)
        ob, vb = slice(0, nocc_b), slice(nocc_b, nmo)
        sl_map = {"o": oa, "v": va, "O": ob, "V": vb}

        H = {}
        for c1 in ["o", "v"]:
            for c2 in ["o", "v"]:
                H[c1 + c2] = F[sl_map[c1], sl_map[c2]]
                H[c1.upper() + c2.upper()] = F[sl_map[c1.upper()], sl_map[c2.upper()]]
        for c1 in ["o", "v"]:
            for c2 in ["o", "v"]:
                for c3 in ["o", "v"]:
                    for c4 in ["o", "v"]:
                        H[c1 + c2 + c3 + c4] = V_asym[sl_map[c1], sl_map[c2], sl_map[c3], sl_map[c4]]
                        H[c1.upper() + c2.upper() + c3.upper() + c4.upper()] = (
                            V_asym[sl_map[c1.upper()], sl_map[c2.upper()],
                                   sl_map[c3.upper()], sl_map[c4.upper()]])
                        H[c1 + c2.upper() + c3 + c4.upper()] = (
                            V[sl_map[c1], sl_map[c2.upper()], sl_map[c3], sl_map[c4.upper()]])

        # ── 3. CCSD amplitudes ──
        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf
        from pyscf import cc

        alpha_occ = np.zeros(nmo); alpha_occ[:nocc_a] = 1.0
        beta_occ = np.zeros(nmo); beta_occ[:nocc_b] = 1.0
        mycc = cc.CCSD(hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)).run()
        logger.info("CCSD energy: %.10f", mycc.e_tot)

        # Same-spin T2: antisymmetrize in (a,b) only. For RHF with t2[ij,ab]=t2[ji,ba],
        # this automatically gives full antisymmetry in both pairs.
        t2_asym = mycc.t2 - mycc.t2.swapaxes(2, 3)

        T = {"ov": mycc.t1.copy(), "OV": mycc.t1.copy(),
             "oovv": t2_asym.copy(), "OOVV": t2_asym.copy(), "oOvV": mycc.t2.copy()}

        # ── 4. Zero all-active T → σ_ext ──
        ncore_a, ncore_b = nocc_a - noa_act, nocc_b - nob_act
        act_oa, act_va = slice(ncore_a, nocc_a), slice(0, nva_act)
        act_ob, act_vb = slice(ncore_b, nocc_b), slice(0, nvb_act)

        T["ov"][act_oa, act_va] = 0.0; T["OV"][act_ob, act_vb] = 0.0
        T["oovv"][act_oa, act_oa, act_va, act_va] = 0.0
        T["OOVV"][act_ob, act_ob, act_vb, act_vb] = 0.0
        T["oOvV"][act_oa, act_ob, act_va, act_vb] = 0.0

        # Transposes AFTER zeroing
        T["vo"] = T["ov"].T.copy(); T["VO"] = T["OV"].T.copy()
        T["vvoo"] = T["oovv"].transpose(2, 3, 0, 1).copy()
        T["VVOO"] = T["OOVV"].transpose(2, 3, 0, 1).copy()
        T["vVoO"] = T["oOvV"].transpose(2, 3, 0, 1).copy()
        T["VvOo"] = T["oOvV"].transpose(3, 2, 1, 0).copy()

        # ── 5. Wicked BCH ──
        fbar, vbar, E0_bch = self._wicked_bch_si(w, ducc_level, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b)

        # ── 6. Assemble full-space, restrict to active, γ→χ ──
        nact = noa_act + nva_act

        fbar_aa = np.zeros((nmo, nmo)); fbar_bb = np.zeros((nmo, nmo))
        for key, arr in fbar.items():
            if key.islower():
                fbar_aa[sl_map[key[0]], sl_map[key[1]]] = arr
            elif key.isupper():
                fbar_bb[sl_map[key[0]], sl_map[key[1]]] = arr

        vbar_aa = np.zeros((nmo,)*4); vbar_bb = np.zeros((nmo,)*4); vbar_ab = np.zeros((nmo,)*4)
        for key, arr in vbar.items():
            n_lower = sum(1 for c in key if c.islower())
            s = [sl_map[c] for c in key]
            idx = np.ix_(range(*s[0].indices(nmo)), range(*s[1].indices(nmo)),
                         range(*s[2].indices(nmo)), range(*s[3].indices(nmo)))
            if n_lower == 4:   vbar_aa[idx] += arr
            elif n_lower == 0: vbar_bb[idx] += arr
            elif n_lower == 2: vbar_ab[idx] += arr

        # Antisymmetrize same-spin (fills missing permutation blocks + corrects factors)
        vbar_aa = vbar_aa - vbar_aa.transpose(1,0,2,3) - vbar_aa.transpose(0,1,3,2) + vbar_aa.transpose(1,0,3,2)
        vbar_bb = vbar_bb - vbar_bb.transpose(1,0,2,3) - vbar_bb.transpose(0,1,3,2) + vbar_bb.transpose(1,0,3,2)

        # Restrict to active
        act = list(range(nocc_a - noa_act, nocc_a)) + [nocc_a + i for i in range(nva_act)]
        g1_aa = fbar_aa[np.ix_(act, act)]; g1_bb = fbar_bb[np.ix_(act, act)]
        g2_aa = vbar_aa[np.ix_(act, act, act, act)]
        g2_bb = vbar_bb[np.ix_(act, act, act, act)]
        g2_ab = vbar_ab[np.ix_(act, act, act, act)]

        # χ₁^αα = γ₁^αα - Σ_m γ₂^αα[pm,qm] - Σ_M γ₂^αβ[pM,qM]
        # χ₁^ββ = γ₁^ββ - Σ_M γ₂^ββ[pM,qM] - Σ_m γ₂^αβ[mp,mq]
        aol = list(range(noa_act))
        chi1_aa = g1_aa - np.einsum("pmqm->pq", g2_aa[:, aol, :, :][:, :, :, aol]) \
                        - np.einsum("pmqm->pq", g2_ab[:, aol, :, :][:, :, :, aol])
        chi1_bb = g1_bb - np.einsum("pmqm->pq", g2_bb[:, aol, :, :][:, :, :, aol]) \
                        - np.einsum("mpmq->pq", g2_ab[np.ix_(aol, range(nact), aol, range(nact))])

        # C = E₀ - Σ_m χ₁^αα[mm] - Σ_M χ₁^ββ[MM] - ½ΣΣ γ₂^αα - ½ΣΣ γ₂^ββ - ΣΣ γ₂^αβ
        C = E0_bch
        for m in aol:
            C -= chi1_aa[m, m] + chi1_bb[m, m]
            for n in aol:
                C -= 0.5 * g2_aa[m, n, m, n] + 0.5 * g2_bb[m, n, m, n] + g2_ab[m, n, m, n]

        # ── 7. Package Hamiltonian ──
        # h1 = χ₁^αα;  h2_chemist = γ₂^αβ.swapaxes(1,2) (physicist→chemist)
        from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, ModelOrbitals

        return Hamiltonian(CanonicalFourCenterHamiltonianContainer(
            chi1_aa, g2_ab.swapaxes(1, 2).ravel(), ModelOrbitals(nact), C, np.zeros((nact, nact))))

    @staticmethod
    def _wicked_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b):
        """Run spin-integrated wicked BCH. Returns (fbar_dict, vbar_dict, E0_scalar)."""
        w.reset_space()
        w.add_space("o", "fermion", "occupied", list("ijklmn")[:max(nocc_a, 1)])
        w.add_space("v", "fermion", "unoccupied", list("abcdef")[:max(nvir_a, 1)])
        w.add_space("O", "fermion", "occupied", list("IJKLMN")[:max(nocc_b, 1)])
        w.add_space("V", "fermion", "unoccupied", list("ABCDEF")[:max(nvir_b, 1)])

        Top = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "V+ v+ O o"], unique=True)
        Hops = []
        for i in itertools.product(["v+", "o+"], ["v", "o"]): Hops.append(" ".join(i))
        for i in itertools.product(["V+", "O+"], ["V", "O"]): Hops.append(" ".join(i))
        for i in itertools.product(["v+", "o+"], ["v+", "o+"], ["v", "o"], ["v", "o"]): Hops.append(" ".join(i))
        for i in itertools.product(["V+", "O+"], ["V+", "O+"], ["V", "O"], ["V", "O"]): Hops.append(" ".join(i))
        for i in itertools.product(["v+", "o+"], ["V+", "O+"], ["v", "o"], ["V", "O"]): Hops.append(" ".join(i))
        Hop = w.op("H", Hops, unique=True)
        Fops = []
        for i in itertools.product(["v+", "o+"], ["v", "o"]): Fops.append(" ".join(i))
        for i in itertools.product(["V+", "O+"], ["V", "O"]): Fops.append(" ".join(i))
        Fop = w.op("H", Fops, unique=True)

        sigma = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "V+ v+ O o"], unique=True)
        sigma.add2(Top.adjoint(), w.rational(-1))

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

        class _S:
            def __init__(self, v): self.val = v
            def __getitem__(self, k): return self.val

        def _dim(c):
            return {"o": nocc_a, "v": nvir_a, "O": nocc_b, "V": nvir_b}[c]

        fbar, vbar, E0_bch = {}, {}, 0.0
        for key, eqs in mbeq.items():
            if not eqs: continue
            ndim = len(key.replace("|", ""))
            fc = eqs[0].compile("einsum")
            rv = fc.split("+=")[0].strip()
            ic = rv[1:]
            shape = [_dim(c) for c in ic]
            lines = ["def _e(E0,H,T):"]
            lines.append(f"    {rv}={'0.0' if ndim==0 else 'np.zeros(('+','.join(str(s) for s in shape)+'))'}")
            for eq in eqs: lines.append(f"    {eq.compile('einsum')}")
            lines.append(f"    return {rv}")
            ns = {}; exec("\n".join(lines), {"np": np}, ns)  # noqa: S102
            result = ns["_e"](_S(E0), H, T)
            if ndim == 0: E0_bch = result
            elif ndim == 2: fbar[ic] = np.array(result)
            elif ndim == 4: vbar[ic] = np.array(result)

        return fbar, vbar, E0_bch
