# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Shared helpers for spin-integrated DUCC implementations.

Contains the wicked-independent building blocks used by both the full-space
:mod:`wicked_ducc_si` and the pre-sliced :mod:`wicked_ducc_si_presliced`
solvers:

- **Integral extraction** — Hamiltonian → physicist-convention H/T dicts
- **Wicked symbolic BCH** — operator algebra, Wick contraction, ``mbeq``
- **Active-space assembly** — γ→χ conversion and Hamiltonian packaging
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

_wicked = None


def require_wicked():
    """Import and cache the wicked module, raising on failure."""
    global _wicked
    if _wicked is None:
        try:
            import wickd as wicked

            _wicked = wicked
        except ImportError:
            raise ImportError("wickd is required for WickedDuccSISolver. Install with: pip install wickd")
    return _wicked


class ScalarProxy:
    """Wraps a scalar so ``E0[""]`` returns the scalar value.

    Wicked's compiled einsum references the reference energy as ``E0[""]``.
    This proxy makes a plain float subscriptable.
    """

    def __init__(self, v):
        self.val = v

    def __getitem__(self, k):
        return self.val


# ── Step 1: Extract integrals from Hamiltonian ────────────────────────────────


def build_integrals(hamiltonian, nocc_a, nocc_b):
    """Extract spatial integrals and build spin-blocked H/F/V dictionaries.

    Converts the qdk-chemistry Hamiltonian (chemist convention) into
    physicist-convention Fock (F) and two-electron (V) arrays, then slices
    them into occ/vir blocks keyed by space strings (``"ov"``, ``"oOvV"``,
    etc.).

    Args:
        hamiltonian: Full-space qdk-chemistry Hamiltonian.
        nocc_a: Number of alpha occupied orbitals.
        nocc_b: Number of beta occupied orbitals.

    Returns:
        ``(H, E0, nmo)`` where *H* is a dict of spin-blocked integral arrays,
        *E0* is the scalar reference energy, and *nmo* is the number of MOs.

    """
    orbitals = hamiltonian.get_orbitals()
    nmo = orbitals.get_num_molecular_orbitals()

    h1_list, _ = hamiltonian.get_one_body_integrals()
    h1 = np.array(h1_list).reshape(nmo, nmo)

    eri_list, _, _ = hamiltonian.get_two_body_integrals()
    eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)

    core_energy = hamiltonian.get_core_energy()

    # Physicist notation: V[p,q,r,s] = <pq|rs> = (pr|qs)_chemist
    V = eri.swapaxes(1, 2)
    V_asym = V - V.swapaxes(2, 3)

    # Reference energy E₀
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

    # Fock matrix: f[p,q] = h[p,q] + Σ_m <pm||qm> + Σ_M <pM|qM>
    F = h1.copy()
    for m in range(nocc_a):
        F += V_asym[:, m, :, m]
    for m in range(nocc_b):
        F += V[:, m, :, m]

    # Spin-blocked dictionaries (lowercase=α, uppercase=β)
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
                    H[c1.upper() + c2.upper() + c3.upper() + c4.upper()] = V_asym[
                        sl_map[c1.upper()], sl_map[c2.upper()], sl_map[c3.upper()], sl_map[c4.upper()]
                    ]
                    H[c1 + c2.upper() + c3 + c4.upper()] = V[
                        sl_map[c1], sl_map[c2.upper()], sl_map[c3], sl_map[c4.upper()]
                    ]

    return H, E0, nmo


# ── Step 2: CCSD amplitudes ──────────────────────────────────────────────────


def build_ccsd_amplitudes(hamiltonian, nmo, nocc_a, nocc_b, wavefunction=None):
    """Run CCSD via PyscfCoupledClusterCalculator and return spin-blocked T dicts.

    Uses the qdk-chemistry CC plugin.  Does NOT re-run SCF — the plugin
    internally calls ``hamiltonian_to_scf`` which feeds our pre-computed
    orbitals directly to PySCF's CCSD solver.

    RHF vs UHF is detected from the container's storage convention:
    for RHF the plugin stores a single spatial T2 for all spin cases
    (``t2_aa == t2_ab``); for UHF they are independent arrays.  Same-spin
    T2 is antisymmetrized only for RHF (UHF already stores them antisymmetric).

    Args:
        hamiltonian: Full-space qdk-chemistry Hamiltonian.
        nmo: Number of molecular orbitals.
        nocc_a: Number of alpha occupied orbitals.
        nocc_b: Number of beta occupied orbitals.
        wavefunction: SCF wavefunction (provides orbital occupations for CCSD).
            If None, a HF-determinant wavefunction is constructed automatically.

    Returns:
        Dict of T-amplitude blocks: ``"ov"``, ``"OV"``, ``"oovv"``,
        ``"OOVV"``, ``"oOvV"`` (excitation components only; transposes
        are added later in :func:`zero_all_active_and_transpose`).

    """
    from qdk_chemistry.data import Ansatz, Configuration, StateVectorContainer, Wavefunction
    from qdk_chemistry.plugins.pyscf.coupled_cluster import PyscfCoupledClusterCalculator

    nvir_a = nmo - nocc_a
    nvir_b = nmo - nocc_b

    # Construct a minimal HF wavefunction if not provided.
    if wavefunction is None:
        orbitals = hamiltonian.get_orbitals()
        # Build HF determinant string: '2'=doubly occ, 'u'=α only, 'd'=β only, '0'=empty
        n_doubly = min(nocc_a, nocc_b)
        n_alpha_only = nocc_a - n_doubly
        n_beta_only = nocc_b - n_doubly
        det_str = "2" * n_doubly + "u" * n_alpha_only + "d" * n_beta_only + "0" * (nmo - max(nocc_a, nocc_b))
        det = Configuration.from_spin_half_string(det_str)
        container = StateVectorContainer(np.array([1.0]), [det], orbitals)
        wavefunction = Wavefunction(container)

    # Run CCSD via the plugin.
    ansatz = Ansatz(hamiltonian, wavefunction)
    cc_calc = PyscfCoupledClusterCalculator()
    cc_calc.settings().set("store_amplitudes", True)
    energy, cc_wfn, _ = cc_calc.run(ansatz)
    logger.info("CCSD energy: %.10f", energy)

    # Extract flattened amplitudes from the container.
    container = cc_wfn.get_container()
    t1_aa_flat, t1_bb_flat = container.get_t1_amplitudes()
    t2_ab_flat, t2_aa_flat, t2_bb_flat = container.get_t2_amplitudes()

    # Reshape to spatial tensor form.
    t1_aa = np.array(t1_aa_flat).reshape(nocc_a, nvir_a)
    t1_bb = np.array(t1_bb_flat).reshape(nocc_b, nvir_b)
    t2_ab = np.array(t2_ab_flat).reshape(nocc_a, nocc_b, nvir_a, nvir_b)
    t2_aa = np.array(t2_aa_flat).reshape(nocc_a, nocc_a, nvir_a, nvir_a)
    t2_bb = np.array(t2_bb_flat).reshape(nocc_b, nocc_b, nvir_b, nvir_b)

    # Detect RHF: the plugin stores a single spatial T2 for all spin blocks,
    # so t2_aa_flat and t2_ab_flat are identical arrays.
    is_rhf = np.array_equal(t2_aa_flat, t2_ab_flat)
    if is_rhf:
        # RHF: same-spin T2 is the raw spatial t2, needs antisymmetrization.
        t2_aa = t2_ab - t2_ab.swapaxes(2, 3)
        t2_bb = t2_aa.copy()
    # else: UHF — same-spin T2 already antisymmetric from PySCF.

    return {
        "ov": t1_aa,
        "OV": t1_bb,
        "oovv": t2_aa,
        "OOVV": t2_bb,
        "oOvV": t2_ab,
    }


# ── Step 3: Zero all-active T → σ_ext ────────────────────────────────────────


def zero_all_active_and_transpose(T, nocc_a, nocc_b, noa_act, nob_act, nva_act, nvb_act):
    """Zero all-active T amplitudes and build de-excitation transposes.

    Modifies *T* in place: sets amplitudes where all indices fall within
    the active window to zero (so σ = T - T† contains only external
    excitations), then adds the transposed blocks needed by wicked.

    Args:
        T: Dict of T-amplitude blocks (modified in place).
        nocc_a, nocc_b: Occupied orbital counts.
        noa_act, nob_act: Active occupied counts.
        nva_act, nvb_act: Active virtual counts.

    Returns:
        ``(act_oa, act_va, act_ob, act_vb)`` — active slices within each space.

    """
    ncore_a, ncore_b = nocc_a - noa_act, nocc_b - nob_act
    act_oa, act_va = slice(ncore_a, nocc_a), slice(0, nva_act)
    act_ob, act_vb = slice(ncore_b, nocc_b), slice(0, nvb_act)

    T["ov"][act_oa, act_va] = 0.0
    T["OV"][act_ob, act_vb] = 0.0
    T["oovv"][act_oa, act_oa, act_va, act_va] = 0.0
    T["OOVV"][act_ob, act_ob, act_vb, act_vb] = 0.0
    T["oOvV"][act_oa, act_ob, act_va, act_vb] = 0.0

    T["vo"] = T["ov"].T.copy()
    T["VO"] = T["OV"].T.copy()
    T["vvoo"] = T["oovv"].transpose(2, 3, 0, 1).copy()
    T["VVOO"] = T["OOVV"].transpose(2, 3, 0, 1).copy()
    T["vVoO"] = T["oOvV"].transpose(2, 3, 0, 1).copy()
    T["VvOo"] = T["oOvV"].transpose(3, 2, 1, 0).copy()

    return act_oa, act_va, act_ob, act_vb


# ── Step 4: Wicked symbolic BCH ──────────────────────────────────────────────


def build_ducc_bch(w, bch_order):
    """Build the DUCC BCH expansion symbolically and return ``mbeq``.

    Sets up four orbital spaces (α occ/vir, β occ/vir) with six indices
    each, constructs H, F, T, σ = T - T† operators, builds the BCH-truncated
    Hbar, applies Wick's theorem, and returns the many-body equations.

    The result depends only on *bch_order* — not on the molecule, orbital
    counts, or active space — so it can be computed once and reused.

    Args:
        w: The wicked module.
        bch_order: BCH truncation level (0, 1, or 2).

    Returns:
        ``(mbeq, osi)`` where *mbeq* is the dict from
        ``Expression.to_manybody_equation("R")`` and *osi* is the
        ``OrbitalSpaceInfo`` needed to interpret index space IDs.

    """
    w.reset_space()
    w.add_space("o", "fermion", "occupied", list("ijklmn"))
    w.add_space("v", "fermion", "unoccupied", list("abcdef"))
    w.add_space("O", "fermion", "occupied", list("IJKLMN"))
    w.add_space("V", "fermion", "unoccupied", list("ABCDEF"))

    Top = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "v+ V+ O o"], unique=True)

    # Full Hamiltonian operator
    Hops = []
    for i in itertools.product(["v+", "o+"], ["v", "o"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V", "O"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["v+", "o+"], ["v+", "o+"], ["v", "o"], ["v", "o"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V+", "O+"], ["V", "O"], ["V", "O"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["v+", "o+"], ["V+", "O+"], ["v", "o"], ["V", "O"]):
        Hops.append(" ".join(i))
    Hop = w.op("H", Hops, unique=True)

    # Fock operator (1-body only, used in higher BCH terms)
    Fops = []
    for i in itertools.product(["v+", "o+"], ["v", "o"]):
        Fops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V", "O"]):
        Fops.append(" ".join(i))
    Fop = w.op("H", Fops, unique=True)

    sigma = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "v+ V+ O o"], unique=True)
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
    elif bch_order == 3:
        Hbar = E0op + Hop + w.commutator(Hop, sigma)
        Hbar.add2(w.commutator(Hop, sigma, sigma), w.rational(1, 2))
        Hbar.add2(w.commutator(Hop, sigma, sigma, sigma), w.rational(1, 6))
        Hbar.add2(w.commutator(Fop, sigma, sigma, sigma, sigma), w.rational(1, 24))
    else:
        raise ValueError(f"Unsupported BCH order {bch_order}")

    expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
    mbeq = expr.to_manybody_equation("R")
    return mbeq, w.osi()


# ── Step 6: Assemble active-space Hamiltonian ────────────────────────────────


def assemble_active_hamiltonian(fbar, vbar, E0_bch, noa_act, nva_act,
                                input_orbitals=None, nocc_a=None):
    """Assemble active-space γ tensors, convert γ→χ, and package Hamiltonian.

    Takes the active-sized 1-body (*fbar*) and 2-body (*vbar*) output from
    the BCH evaluation and builds the final downfolded Hamiltonian.

    Args:
        fbar: Dict of 1-body blocks at active size (e.g. ``"ov"`` → array).
        vbar: Dict of 2-body blocks at active size.
        E0_bch: Scalar reference energy from BCH.
        noa_act: Number of active occupied α orbitals.
        nva_act: Number of active virtual α orbitals.
        input_orbitals: Full-space Orbitals from the input Hamiltonian.
            If provided, the active MO coefficients are extracted and
            attached to the output Hamiltonian.
        nocc_a: Number of occupied α orbitals (needed to locate active
            columns when *input_orbitals* is given).

    Returns:
        A qdk-chemistry Hamiltonian object for the active space.

    """
    nact = noa_act + nva_act

    def _off(c):
        return 0 if c in ("o", "O") else noa_act

    # 1-body
    g1_aa = np.zeros((nact, nact))
    g1_bb = np.zeros((nact, nact))
    for key, sub in fbar.items():
        r0, c0 = _off(key[0]), _off(key[1])
        if key.islower():
            g1_aa[r0 : r0 + sub.shape[0], c0 : c0 + sub.shape[1]] = sub
        elif key.isupper():
            g1_bb[r0 : r0 + sub.shape[0], c0 : c0 + sub.shape[1]] = sub

    # 2-body
    g2_aa_raw = np.zeros((nact,) * 4)
    g2_bb_raw = np.zeros((nact,) * 4)
    g2_ab = np.zeros((nact,) * 4)
    for key, sub in vbar.items():
        n_lower = sum(1 for c in key if c.islower())
        o = [_off(c) for c in key]
        s = sub.shape
        sl = tuple(slice(o[i], o[i] + s[i]) for i in range(4))
        if n_lower == 4:
            g2_aa_raw[sl] += sub
        elif n_lower == 0:
            g2_bb_raw[sl] += sub
        elif n_lower == 2:
            g2_ab[sl] += sub

    # Antisymmetrize same-spin
    g2_aa = (
        g2_aa_raw - g2_aa_raw.transpose(1, 0, 2, 3) - g2_aa_raw.transpose(0, 1, 3, 2) + g2_aa_raw.transpose(1, 0, 3, 2)
    )
    g2_bb = (
        g2_bb_raw - g2_bb_raw.transpose(1, 0, 2, 3) - g2_bb_raw.transpose(0, 1, 3, 2) + g2_bb_raw.transpose(1, 0, 3, 2)
    )

    # γ → χ
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

    # Scalar constant
    C = E0_bch
    for m in aol:
        C -= chi1_aa[m, m] + chi1_bb[m, m]
        for n in aol:
            C -= 0.5 * g2_aa[m, n, m, n] + 0.5 * g2_bb[m, n, m, n] + g2_ab[m, n, m, n]

    # Package
    from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, ModelOrbitals

    # Build active-space orbitals: extract the active MO columns from the
    # full coefficient matrix if real orbitals are available.
    active_orbitals = ModelOrbitals(nact)
    if input_orbitals is not None and nocc_a is not None:
        try:
            from qdk_chemistry.data import Orbitals
            coeffs, _ = input_orbitals.get_coefficients()
            C_full = np.array(coeffs)
            ncore = nocc_a - noa_act
            # Active columns: [ncore..nocc_a) occupied, [nocc_a..nocc_a+nva_act) virtual
            act_cols = list(range(ncore, nocc_a)) + list(range(nocc_a, nocc_a + nva_act))
            C_act = C_full[:, act_cols]
            basis_set = input_orbitals.get_basis_set()
            active_orbitals = Orbitals(C_act, None, None, basis_set)
        except (RuntimeError, AttributeError):
            pass  # Model Hamiltonian without real MO coefficients — fall back

    return Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            chi1_aa, g2_ab.swapaxes(1, 2).ravel(), active_orbitals, C, np.zeros((nact, nact))
        )
    )
