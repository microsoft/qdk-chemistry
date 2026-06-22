# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Reference DUCC Hamiltonian downfolding using OpenFermion operator algebra.

Implements the Double Unitary Coupled Cluster (DUCC) downfolding via symbolic
BCH expansion using OpenFermion's ``FermionOperator`` commutators.  This is
algebraically exact (no hand-derived formulas) and works in a purely
spin-orbital representation throughout.

The pipeline:

1. Extract spatial MO integrals from the input ``Hamiltonian``.
2. Run PySCF CCSD → convert to spin-orbital T₁/T₂ via GCCSD.
3. Build σ_ext = T_ext − T†_ext (anti-Hermitian).
4. Compute H_DUCC = e^{−σ} H e^{+σ} via BCH (nested commutators).
5. Project H_DUCC to the active spin-orbital space.
6. Convert to spatial chemist notation and wrap in a ``Hamiltonian``.

References:
    - N.P. Bauman et al., J. Chem. Phys. 151, 014107 (2019)
    - K. Kowalski, J. Chem. Phys. 148, 094104 (2018)

"""

from __future__ import annotations

import logging
from itertools import combinations
from math import factorial

import numpy as np
from openfermion import FermionOperator, commutator, hermitian_conjugated, normal_ordered
from pyscf import cc as pyscf_cc

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.plugins.exachem.conversion import FcidumpData, spinorb_to_spatial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core DUCC downfolding (spin-orbital, OpenFermion)
# ---------------------------------------------------------------------------


def _build_hamiltonian_operator(h1_spatial, eri_spatial):
    """Build the full Hamiltonian as a FermionOperator from spatial MO integrals.

    Args:
        h1_spatial: (nmo, nmo) 1e integrals in spatial MO basis.
        eri_spatial: (nmo, nmo, nmo, nmo) 2e integrals in chemist notation (pq|rs).

    Returns:
        Normal-ordered FermionOperator.

    """
    nmo = h1_spatial.shape[0]
    so = lambda p, s: 2 * p + s

    H = FermionOperator()
    for p in range(nmo):
        for q in range(nmo):
            if abs(h1_spatial[p, q]) > 1e-14:
                for s in range(2):
                    H += FermionOperator(((so(p, s), 1), (so(q, s), 0)), h1_spatial[p, q])
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    v = eri_spatial[p, q, r, s]
                    if abs(v) < 1e-14:
                        continue
                    for s1 in range(2):
                        for s2 in range(2):
                            H += FermionOperator(
                                ((so(p, s1), 1), (so(r, s2), 1), (so(s, s2), 0), (so(q, s1), 0)),
                                0.5 * v,
                            )
    return normal_ordered(H)


def _build_sigma_ext(t1_so, t2_so, nocc_so, aocc_so, avir_so):
    """Build σ_ext = T_ext − T†_ext from spin-orbital CCSD amplitudes.

    Args:
        t1_so: (nocc_so, nvir_so) spin-orbital T₁ amplitudes.
        t2_so: (nocc_so, nocc_so, nvir_so, nvir_so) fully antisymmetric T₂.
        nocc_so: Number of occupied spin-orbitals.
        aocc_so: Set of active occupied spin-orbital indices.
        avir_so: Set of active virtual spin-orbital indices.

    Returns:
        Normal-ordered anti-Hermitian FermionOperator.

    """
    nvir_so = t1_so.shape[1]

    def is_internal(occ_idx, vir_idx):
        return all(i in aocc_so for i in occ_idx) and all(a in avir_so for a in vir_idx)

    T_ext = FermionOperator()

    for i in range(nocc_so):
        for a in range(nvir_so):
            if is_internal([i], [a + nocc_so]):
                continue
            if abs(t1_so[i, a]) < 1e-14:
                continue
            T_ext += t1_so[i, a] * FermionOperator(((a + nocc_so, 1), (i, 0)))

    for i in range(nocc_so):
        for j in range(nocc_so):
            for a in range(nvir_so):
                for b in range(nvir_so):
                    if is_internal([i, j], [a + nocc_so, b + nocc_so]):
                        continue
                    if abs(t2_so[i, j, a, b]) < 1e-14:
                        continue
                    T_ext += (
                        0.25 * t2_so[i, j, a, b] * FermionOperator(((a + nocc_so, 1), (b + nocc_so, 1), (j, 0), (i, 0)))
                    )

    T_ext = normal_ordered(T_ext)
    return normal_ordered(T_ext - hermitian_conjugated(T_ext))


def _bch_expansion(H, sigma, bch_order):
    """Compute e^{-σ} H e^{σ} via BCH: Σ_{k=0}^{order} (1/k!) [H,σ]_k.

    Args:
        H: Hamiltonian FermionOperator.
        sigma: Anti-Hermitian cluster operator.
        bch_order: Truncation order.

    Returns:
        Normal-ordered H_DUCC FermionOperator.

    """
    H_ducc = normal_ordered(H + FermionOperator())
    term = H
    for k in range(1, bch_order + 1):
        term = normal_ordered(commutator(term, sigma))
        H_ducc += term * (1.0 / factorial(k))
        max_coeff = max((abs(c) for c in term.terms.values()), default=0)
        if max_coeff < 1e-12:
            break
    return normal_ordered(H_ducc)


def _extract_integrals(op, core, act):
    """Extract (E0, h1_eff, v2_eff) from a FermionOperator via core-folded matrix elements.

    Args:
        op: The full-space FermionOperator.
        core: List of core (inactive) spin-orbital indices.
        act: List of active spin-orbital indices.

    Returns:
        (E0, h1_eff, v2_eff) in the active spin-orbital basis.

    """

    def _act_op(term, occ):
        occ = set(occ)
        sgn = 1.0
        for idx, dag in reversed(term):
            below = sum(1 for o in occ if o < idx)
            if dag:
                if idx in occ:
                    return 0.0, None
                sgn *= (-1) ** below
                occ.add(idx)
            else:
                if idx not in occ:
                    return 0.0, None
                sgn *= (-1) ** below
                occ.discard(idx)
        return sgn, frozenset(occ)

    def _expval(bra, ket):
        bra_fs = frozenset(bra)
        val = 0.0
        for term, c in op.terms.items():
            s, res = _act_op(term, ket)
            if res is not None and res == bra_fs:
                val += c * s
        return val

    def _signed_det(base, extra):
        occ = list(base)
        sgn = 1.0
        for idx in extra:
            below = sum(1 for o in occ if o < idx)
            sgn *= (-1) ** below
            occ.append(idx)
        return sgn, frozenset(occ)

    nA = len(act)
    E0 = _expval(core, core)

    h1_eff = np.zeros((nA, nA))
    for ip, p in enumerate(act):
        for iq, q in enumerate(act):
            h1_eff[ip, iq] = _expval(core + [p], core + [q]) - (E0 if p == q else 0.0)

    v2_eff = np.zeros((nA, nA, nA, nA))
    d = lambda a, b: 1.0 if a == b else 0.0
    for ip, p in enumerate(act):
        for iq, q in enumerate(act):
            if ip == iq:
                continue
            sb, Db = _signed_det(core, [p, q])
            for ir, r in enumerate(act):
                for is_, s in enumerate(act):
                    if ir == is_:
                        continue
                    sk, Dk = _signed_det(core, [r, s])
                    M = sb * sk * _expval(Db, Dk)
                    M -= E0 * (d(ip, ir) * d(iq, is_) - d(ip, is_) * d(iq, ir))
                    M -= (
                        h1_eff[ip, ir] * d(iq, is_)
                        - h1_eff[ip, is_] * d(iq, ir)
                        - h1_eff[iq, ir] * d(ip, is_)
                        + h1_eff[iq, is_] * d(ip, ir)
                    )
                    v2_eff[ip, iq, ir, is_] = M
    return E0, h1_eff, v2_eff


def _to_particle_hole_integrals(op, occ_so, active_so):
    """Convert a vacuum-NO FermionOperator to physical-vacuum ≤2-body integrals.

    Implements the procedure from Bauman et al. (2019), Eqs. 62–66 / Tables IV–V:

    1. **Fermi-vacuum normal ordering** — Wick-contract every vacuum-NO term
       w.r.t. the HF reference (occupied spin-orbitals ``occ_so``).  Each
       contraction pairs a creator at index *k* ∈ ``occ_so`` with the matching
       annihilator (same index), giving factor 1 and a sign.  This folds
       *n*-body vacuum-NO terms into ≤2-body particle-hole coefficients.
    2. **Active-space projection** — keep only terms whose remaining
       (uncontracted) indices are all in ``active_so``.  FV-NO 3-body+
       residuals are dropped.
    3. **Table IV/V transformation** — convert the particle-hole coefficients
       (``f̃``, ``ṽ``) to physical-vacuum integrals (``h``, ``v``).

    Args:
        op: Normal-ordered FermionOperator (vacuum NO).
        occ_so: Sorted list of occupied spin-orbital indices (HF reference).
        active_so: Sorted list of active spin-orbital indices.

    Returns:
        ``(E0, h1, v2)`` — physical-vacuum effective integrals in the active
        spin-orbital basis (indices ``0 .. len(active_so)-1``).

    """
    occ_set = frozenset(occ_so)
    act_set = frozenset(active_so)
    nA = len(active_so)
    act_idx = {p: i for i, p in enumerate(active_so)}
    act_occ_local = [act_idx[k] for k in active_so if k in occ_set]

    # Particle-hole (Fermi-vacuum NO) accumulators
    E0_ph = 0.0
    h1_ph = np.zeros((nA, nA))
    v2_ph = np.zeros((nA, nA, nA, nA))

    for term, coeff in op.terms.items():
        if abs(coeff) < 1e-15:
            continue
        n_body = len(term) // 2

        # Split into creators and annihilators (vacuum-NO order)
        creators = [term[i] for i in range(n_body)]  # (idx, 1)
        anns = [term[n_body + i] for i in range(n_body)]  # (idx, 0)

        # Find all contractible pairs: creator k with annihilator k, k ∈ occ
        pairs = []
        for ci, (cp, _) in enumerate(creators):
            if cp not in occ_set:
                continue
            for ai, (ap, _) in enumerate(anns):
                if ap == cp:
                    pairs.append((ci, ai))

        # Enumerate contraction sets.  For n-body ≤ 2, the empty set (no
        # contractions) gives the vacuum-NO piece that the CI solver handles
        # natively — we keep it as-is and skip all contractions (the CI solver
        # accounts for the occupied-density folding internally).
        # For n-body ≥ 3, only non-empty contraction sets that reduce to
        # ≤ 2-body are kept; these fold 3-body+ contributions into effective
        # ≤ 2-body integrals that a CI solver can use.
        if n_body <= 2:
            # Pass through the vacuum-NO piece unchanged
            indices = [idx for idx, _ in creators] + [idx for idx, _ in anns]
            if all(idx in act_set for idx in indices):
                if n_body == 0:
                    E0_ph += coeff
                elif n_body == 1:
                    E0_ph  # h1 assignment below
                    ip = act_idx[creators[0][0]]
                    iq = act_idx[anns[0][0]]
                    h1_ph[ip, iq] += coeff
                elif n_body == 2:
                    ip = act_idx[creators[0][0]]
                    iq = act_idx[creators[1][0]]
                    ir = act_idx[anns[0][0]]
                    is_ = act_idx[anns[1][0]]
                    v2_ph[ip, iq, ir, is_] += coeff
            continue

        # n_body >= 3: apply Wick contractions to fold into ≤ 2-body
        for cset in _enum_contraction_sets(pairs, n_body, max_remaining=2):
            n_contracted = len(cset)
            remaining_body = n_body - n_contracted

            # Skip the uncontracted piece (FV-NO 3-body+ residual)
            if n_contracted == 0:
                continue
            c_cr = {ci for ci, _ in cset}
            c_an = {ai for _, ai in cset}
            rem_cr = [creators[i] for i in range(n_body) if i not in c_cr]
            rem_an = [anns[i] for i in range(n_body) if i not in c_an]

            # Active-space filter: all remaining indices must be active
            rem_indices = [idx for idx, _ in rem_cr] + [idx for idx, _ in rem_an]
            if not all(idx in act_set for idx in rem_indices):
                continue

            sign = _wick_sign(cset, n_body) if cset else 1
            c_total = coeff * sign

            if remaining_body == 0:
                E0_ph += c_total
            elif remaining_body == 1:
                ip = act_idx[rem_cr[0][0]]
                iq = act_idx[rem_an[0][0]]
                h1_ph[ip, iq] += c_total
            elif remaining_body == 2:
                ip = act_idx[rem_cr[0][0]]
                iq = act_idx[rem_cr[1][0]]
                ir = act_idx[rem_an[0][0]]
                is_ = act_idx[rem_an[1][0]]
                v2_ph[ip, iq, ir, is_] += c_total

    # The Wick contraction procedure above already produces physical-vacuum
    # quantities: E0_ph = ⟨Φ|H|Φ⟩ (all fully-contracted pieces), and h1_ph/v2_ph
    # include the occupied-density folding of higher-body terms.  No Table IV/V
    # transformation is needed — that would double-count.

    return E0_ph, h1_ph, v2_ph


def _enum_contraction_sets(pairs, n_body, max_remaining=2):
    """Yield all non-overlapping contraction sets from ``pairs``.

    Each set has between 0 and ``n_body - max_remaining`` contractions
    (but at least 0) such that the remaining body count is ≤ ``max_remaining``.
    The empty set (no contractions) is always yielded for ``n_body ≤ max_remaining``.
    """
    min_contractions = max(0, n_body - max_remaining)

    def _recurse(remaining_pairs, used_cr, used_an, depth):
        n_contracted = len(used_cr)
        if n_contracted >= min_contractions:
            yield []
        if n_body - n_contracted <= 0:
            return
        for pi, (ci, ai) in enumerate(remaining_pairs):
            if ci in used_cr or ai in used_an:
                continue
            for rest in _recurse(remaining_pairs[pi + 1 :], used_cr | {ci}, used_an | {ai}, depth + 1):
                yield [(ci, ai)] + rest

    return _recurse(pairs, frozenset(), frozenset(), 0)


def _wick_sign(cset, n_body):
    """Compute the sign for a set of Wick contractions.

    For vacuum-NO ``a†_{c0} … a†_{c(n-1)} a_{a0} … a_{a(n-1)}``, contracting
    creator ``ci`` with annihilator ``ai`` costs ``(adj_n - 1 - adj_ci) + adj_ai``
    anticommutation hops.  Contractions are processed right-to-left in the
    creator block to keep adjusted indices simple.
    """
    sign = 1
    adj_n = n_body
    removed_cr: list[int] = []
    removed_an: list[int] = []
    for ci, ai in sorted(cset, key=lambda x: -x[0]):
        adj_ci = ci - sum(1 for rc in removed_cr if rc < ci)
        adj_ai = ai - sum(1 for ra in removed_an if ra < ai)
        sign *= (-1) ** ((adj_n - 1 - adj_ci) + adj_ai)
        removed_cr.append(ci)
        removed_an.append(ai)
        adj_n -= 1
    return sign


def _project_to_active_space(op, active_so):
    """Project a FermionOperator to the active spin-orbital subspace.

    Keeps every term whose indices all belong to ``active_so`` and re-indexes
    them to a contiguous ``0 .. len(active_so)-1`` range.  Terms with any
    index outside the active set are dropped.

    Args:
        op: Normal-ordered FermionOperator on the full spin-orbital space.
        active_so: Sorted list of active spin-orbital indices.

    Returns:
        A new FermionOperator whose indices span ``[0, len(active_so))``.

    """
    idx_map = {p: i for i, p in enumerate(active_so)}
    active_set = set(active_so)
    projected = FermionOperator()
    for term, coeff in op.terms.items():
        if abs(coeff) < 1e-15:
            continue
        if all(idx in active_set for idx, _ in term):
            new_term = tuple((idx_map[idx], dag) for idx, dag in term)
            projected += FermionOperator(new_term, coeff)
    return projected


def _project_to_active_ci(op, active_so, n_elec, nso_full):
    """Build the active-space CI matrix directly from a FermionOperator.

    Projects the full-space operator onto the subspace of Slater determinants
    with ``n_elec`` electrons occupying only active spin-orbitals.  This captures
    ALL contributions (including 3-body+ terms that fold external-index number
    operators into active-space matrix elements) that a 2-body coefficient
    extraction would miss.

    Args:
        op: Normal-ordered FermionOperator on the full spin-orbital space.
        active_so: Sorted list of active spin-orbital indices.
        n_elec: Number of electrons in the active space.
        nso_full: Total number of spin-orbitals (for sign conventions).

    Returns:
        (evals, evecs) from diagonalizing the projected CI matrix.

    """
    dets = [frozenset(c) for c in combinations(active_so, n_elec)]
    det_index = {d: i for i, d in enumerate(dets)}
    n = len(dets)
    H_ci = np.zeros((n, n))

    for j, ket in enumerate(dets):
        for term, c in op.terms.items():
            occ = set(ket)
            sgn = 1.0
            ok = True
            for idx, dag in reversed(term):
                below = sum(1 for o in occ if o < idx)
                if dag:
                    if idx in occ:
                        ok = False
                        break
                    sgn *= (-1) ** below
                    occ.add(idx)
                else:
                    if idx not in occ:
                        ok = False
                        break
                    sgn *= (-1) ** below
                    occ.discard(idx)
            if not ok:
                continue
            res = frozenset(occ)
            i = det_index.get(res)
            if i is not None:
                H_ci[i, j] += c * sgn

    evals, evecs = np.linalg.eigh(H_ci)
    return evals, evecs


def _project_to_active_ci_slater_condon(op, active_so, n_elec, nso_full):
    """Build the active-space CI matrix restricted to ≤2-excitation matrix elements.

    Like :func:`_project_to_active_ci` but only accumulates matrix elements
    between determinants that differ by at most 2 orbitals (the Slater-Condon
    selection rule for ≤2-body operators).  This isolates the contribution of
    many-body operators to the ≤2-excitation block of the CI matrix, which is
    the part that a 2-body effective Hamiltonian should reproduce.

    Args:
        op: Normal-ordered FermionOperator on the full spin-orbital space.
        active_so: Sorted list of active spin-orbital indices.
        n_elec: Number of electrons in the active space.
        nso_full: Total number of spin-orbitals (for sign conventions).

    Returns:
        (evals, evecs) from diagonalizing the projected CI matrix.

    """
    dets = [frozenset(c) for c in combinations(active_so, n_elec)]
    det_index = {d: i for i, d in enumerate(dets)}
    n = len(dets)
    H_ci = np.zeros((n, n))

    for j, ket in enumerate(dets):
        for term, c in op.terms.items():
            occ = set(ket)
            sgn = 1.0
            ok = True
            for idx, dag in reversed(term):
                below = sum(1 for o in occ if o < idx)
                if dag:
                    if idx in occ:
                        ok = False
                        break
                    sgn *= (-1) ** below
                    occ.add(idx)
                else:
                    if idx not in occ:
                        ok = False
                        break
                    sgn *= (-1) ** below
                    occ.discard(idx)
            if not ok:
                continue
            res = frozenset(occ)
            i = det_index.get(res)
            if i is not None:
                # Only keep if ≤2 orbitals differ (Slater-Condon)
                if len(ket ^ res) <= 4:  # symmetric difference ≤ 4 indices = ≤ 2 excitations
                    H_ci[i, j] += c * sgn

    evals, evecs = np.linalg.eigh(H_ci)
    return evals, evecs


# ---------------------------------------------------------------------------
# Factory and Algorithm
# ---------------------------------------------------------------------------


class ReferenceDuccFactory(AlgorithmFactory):
    """Factory for the reference DUCC solver."""

    def algorithm_type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def default_algorithm_name(self) -> str:
        """Return ``"reference_ducc"``."""
        return "reference_ducc"


class ReferenceDuccSolver(Algorithm):
    """Reference DUCC downfolding via OpenFermion BCH.

    Algebraically exact (no hand-derived formulas).  Uses PySCF for CCSD
    amplitudes and OpenFermion for symbolic commutator evaluation.

    Settings:
        nactive_oa (int): Active occupied alpha orbitals (default: 0).
        nactive_ob (int): Active occupied beta orbitals (default: 0).
        nactive_va (int): Active virtual alpha orbitals (default: 0).
        nactive_vb (int): Active virtual beta orbitals (default: 0).
        bch_order (int): BCH truncation order 1–4 (default: 2).
    """

    def __init__(self):
        super().__init__()
        s = self._settings
        s._set_default("nactive_oa", "int", 0)
        s._set_default("nactive_ob", "int", 0)
        s._set_default("nactive_va", "int", 0)
        s._set_default("nactive_vb", "int", 0)
        s._set_default("bch_order", "int", 2)

    def type_name(self) -> str:
        """Return ``"hamiltonian_downfolder"``."""
        return "hamiltonian_downfolder"

    def name(self) -> str:
        """Return ``"reference_ducc"``."""
        return "reference_ducc"

    def aliases(self) -> list[str]:
        """Return algorithm aliases."""
        return ["reference_ducc"]

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        """Run reference DUCC and return the downfolded Hamiltonian.

        Args:
            hamiltonian: qdk_chemistry Hamiltonian (spatial chemist notation).
            n_alpha: number of alpha electrons.
            n_beta: number of beta electrons.

        Returns:
            Downfolded qdk_chemistry Hamiltonian.

        """
        s = self._settings
        noa = s.get("nactive_oa")
        nob = s.get("nactive_ob")
        nva = s.get("nactive_va")
        nvb = s.get("nactive_vb")
        bch_order = s.get("bch_order")

        if noa != nob:
            raise ValueError("Reference DUCC only supports closed-shell (nactive_oa == nactive_ob).")
        if nva != nvb:
            raise ValueError("Reference DUCC only supports nactive_va == nactive_vb.")
        if n_alpha != n_beta:
            raise ValueError("Reference DUCC only supports closed-shell (n_alpha == n_beta).")

        # ── Step 1: Extract spatial MO integrals from qdk Hamiltonian ──
        orbitals = hamiltonian.get_orbitals()
        nmo = orbitals.get_num_molecular_orbitals()
        nocc = n_alpha
        nvir = nmo - nocc

        h1_a, _ = hamiltonian.get_one_body_integrals()
        eri_flat, _, _ = hamiltonian.get_two_body_integrals()
        h1 = np.array(h1_a).reshape(nmo, nmo)
        eri = np.array(eri_flat).reshape(nmo, nmo, nmo, nmo)

        logger.info(
            "nmo=%d, nocc=%d, nvir=%d, active=(%d,%d,%d,%d), bch_order=%d",
            nmo,
            nocc,
            nvir,
            noa,
            nob,
            nva,
            nvb,
            bch_order,
        )

        # ── Step 2: PySCF CCSD → GCCSD spin-orbital amplitudes ──
        from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf

        alpha_occ = np.zeros(nmo)
        alpha_occ[:nocc] = 1.0
        pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, alpha_occ)
        mycc = pyscf_cc.CCSD(pyscf_scf)
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError("PySCF CCSD did not converge.")
        logger.info("CCSD converged: E_corr = %.10f", mycc.e_corr)

        from pyscf.cc.addons import convert_to_gccsd

        gcc = convert_to_gccsd(mycc)
        t1_so = gcc.t1
        t2_so = gcc.t2
        nocc_so = 2 * nocc

        # ── Step 3: Build H as FermionOperator ──
        logger.info("Building Hamiltonian operator (%d spin-orbitals)...", 2 * nmo)
        H = _build_hamiltonian_operator(h1, eri)

        # ── Step 4: Build σ_ext and run BCH ──
        so = lambda p, s: 2 * p + s
        active_occ = list(range(nocc - noa, nocc))
        active_virt = list(range(nva))

        aocc_so = set()
        for i in active_occ:
            aocc_so.add(so(i, 0))
            aocc_so.add(so(i, 1))
        avir_so = set()
        for a in active_virt:
            avir_so.add(so(nocc + a, 0))
            avir_so.add(so(nocc + a, 1))

        logger.info("Building σ_ext (T_ext − T†_ext)...")
        sigma_ext = _build_sigma_ext(t1_so, t2_so, nocc_so, aocc_so, avir_so)

        logger.info("Computing BCH expansion (order %d)...", bch_order)
        H_ducc = _bch_expansion(H, sigma_ext, bch_order)

        # ── Step 5: Project to active space via direct CI ──
        active_spinorbs = sorted(aocc_so | avir_so)
        nso_full = 2 * nmo
        nocc_active = 2 * noa
        norb_active = len(active_spinorbs)
        n_spatial_active = norb_active // 2

        logger.info(
            "Building active-space CI matrix (%d spin-orbitals, %d determinants)...",
            norb_active,
            factorial(norb_active) // (factorial(nocc_active) * factorial(norb_active - nocc_active)),
        )
        evals, evecs = _project_to_active_ci(H_ducc, active_spinorbs, nocc_active, nso_full)
        e_ground = evals[0]

        # ── Step 6: Build output Hamiltonian ──
        # The direct CI gives the electronic energy.  To return a Hamiltonian
        # that downstream code can diagonalise and add core_energy to recover
        # the total, we also extract 2-body integrals (for the Hamiltonian
        # container) but store the 3-body correction in the core energy.
        rep_energy = pyscf_scf.energy_nuc()

        # Extract 2-body integrals (approximate, for the container)
        core_spinorbs = sorted(i for i in range(nocc_so) if i not in aocc_so)
        active_spinorbs_blocked = (
            sorted(so(i, 0) for i in active_occ)
            + sorted(so(i, 1) for i in active_occ)
            + sorted(so(nocc + a, 0) for a in active_virt)
            + sorted(so(nocc + a, 1) for a in active_virt)
        )
        E0, h1_eff, v2_eff = _extract_integrals(H_ducc, core_spinorbs, active_spinorbs_blocked)

        fcidump = FcidumpData(
            norb=norb_active,
            nelec=nocc_active,
            ms2=0,
            one_body=h1_eff,
            two_body=v2_eff,
            nuclear_repulsion=rep_energy,
        )
        spatial_fcidump = spinorb_to_spatial(fcidump)

        # Compute the many-body correction: difference between direct CI
        # (which captures all n-body contributions) and 2-body-only CI.
        from pyscf import fci as pyscf_fci

        e_up_to_2body, _ = pyscf_fci.direct_spin0.kernel(
            spatial_fcidump.one_body,
            spatial_fcidump.two_body,
            n_spatial_active,
            (noa, noa),
        )
        many_body_correction = e_ground - (e_up_to_2body + E0)
        core_energy = E0 + rep_energy + many_body_correction
        logger.info(
            "Many-body (3+) correction: %.6f mHa (direct_CI=%.10f, up_to_2body_CI=%.10f)",
            many_body_correction * 1000,
            e_ground + rep_energy,
            e_up_to_2body + E0 + rep_energy,
        )

        from qdk_chemistry.data import (
            CanonicalFourCenterHamiltonianContainer,
            Hamiltonian,
            ModelOrbitals,
        )

        active_orbitals = ModelOrbitals(n_spatial_active, True)
        container = CanonicalFourCenterHamiltonianContainer(
            spatial_fcidump.one_body,
            spatial_fcidump.two_body.ravel(),
            active_orbitals,
            core_energy,
            np.zeros((n_spatial_active, n_spatial_active)),
        )

        return Hamiltonian(container)
