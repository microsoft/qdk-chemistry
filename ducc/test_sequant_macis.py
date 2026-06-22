"""SeQuant->Python active-space downfolding through qdk-chemistry + MACIS.

End-to-end test using qdk-chemistry throughout to minimize conversions:
  1. qdk-chemistry SCF  (create("scf_solver"))
  2. qdk-chemistry full molecular Hamiltonian (create("hamiltonian_constructor"))
  3. Extract spatial-chemist integrals -> spin-orbital physicist (h, v)
  4. SeQuant-derived downfolding einsum -> active gamma -> chi (Table V) + scalar
  5. Convert chi (spin-orbital physicist) -> spatial chemist -> qdk Hamiltonian
  6. MACIS FCI on the chi Hamiltonian (active space)
  7. MACIS CASCI on the ORIGINAL Hamiltonian (qdk active_space_selector)
  8. Compare chi-FCI vs CASCI

The SeQuant-derived einsum (examples/ducc_active_test.cpp):
    gamma_xx += np.einsum('mxma->xa', v_mxmx)   # active Fock (occupied m={o,i})
    gamma_xx += np.einsum('xa->xa', h_xx)
    gamma_xxxx += 1/4 * np.einsum('xabc->xacb', v_xxxx)
    gamma_0  = h^m_m + 1/2 v^{mn}_{mn}           # SCF electronic energy
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "python/src")

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    ModelOrbitals,
    Structure,
)
from qdk_chemistry.data import Element


def spatial_to_spinorb_h1(h1_spatial):
    """Expand spatial 1e integrals to spin-orbital basis.

    h1_so[2p+s1, 2q+s2] = h1[p,q] * delta(s1,s2)
    """
    nmo = h1_spatial.shape[0]
    h1_so = np.zeros((2 * nmo, 2 * nmo))
    for s in range(2):
        h1_so[s::2, s::2] = h1_spatial
    return h1_so


def spatial_to_spinorb_eri(eri_spatial):
    """Expand spatial chemist ERI ``(pq|rs)`` to spin-orbital physicist antisymmetrized ``<pq||rs>``."""
    nmo = eri_spatial.shape[0]
    eri_so = np.zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))
    for s1 in range(2):
        for s2 in range(2):
            eri_so[s1::2, s1::2, s2::2, s2::2] = eri_spatial
    # chemist (pq|rs) -> physicist antisymmetrized <pq||rs> = (pr|qs) - (ps|qr)
    return np.einsum("prqs->pqrs", eri_so) - np.einsum("psqr->pqrs", eri_so)


# ── Active-space configurations (spatial: n_active_occ, n_active_vir) ──
# LiH/STO-3G: 2 occupied + 4 virtual spatial orbitals.
CONFIGS = [
    ("full active (FCI)",       2, 4),
    ("frozen core only",        1, 4),
    ("frozen virtual only",     2, 1),
    ("frozen core + virtual",   1, 1),
    ("frozen core + 2 virtual", 1, 2),
]


def build_qdk_original():
    """qdk-chemistry SCF + full molecular Hamiltonian. Returns integrals."""
    # LiH at 1.6 Angstrom
    coords_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]) / 0.529177249
    structure = Structure(coords_bohr, [Element.Li, Element.H])

    scf_solver = create("scf_solver")
    e_hf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1,
                               basis_or_guess="sto-3g")

    ham_ctor = create("hamiltonian_constructor")
    ham = ham_ctor.run(wfn.get_orbitals())

    nmo = wfn.get_orbitals().get_num_molecular_orbitals()
    h1_a, _ = ham.get_one_body_integrals()                # spatial chemist 1e
    eri_flat, _, _ = ham.get_two_body_integrals()         # spatial chemist 2e
    eri = np.array(eri_flat).reshape(nmo, nmo, nmo, nmo)  # (pq|rs)
    e_nuc = ham.get_core_energy()                         # nuclear repulsion

    return structure, wfn, ham, np.array(h1_a), eri, e_nuc, nmo, e_hf


def partition(nocc_sp, n_act_occ, n_act_vir):
    act_occ_sp = list(range(nocc_sp - n_act_occ, nocc_sp))
    act_vir_sp = list(range(nocc_sp, nocc_sp + n_act_vir))
    inact_occ_sp = list(range(0, nocc_sp - n_act_occ))

    def so(sp):
        return [2 * p + s for p in sp for s in range(2)]

    o_idx = np.array(so(inact_occ_sp), dtype=int)
    i_idx = np.array(so(act_occ_sp), dtype=int)
    a_idx = np.array(so(act_vir_sp), dtype=int)
    m_idx = np.array(sorted(o_idx.tolist() + i_idx.tolist()), dtype=int)
    x_idx = np.array(i_idx.tolist() + a_idx.tolist(), dtype=int)
    return o_idx, i_idx, a_idx, m_idx, x_idx


def sequant_downfold(h_full, v_full, m_idx, x_idx, i_idx):
    """Run SeQuant-derived einsum -> active gamma -> chi (Table V) + scalar."""
    nact = len(x_idx)
    # SeQuant-generated tensors
    h_xx = h_full[np.ix_(x_idx, x_idx)]
    v_mxmx = v_full[np.ix_(m_idx, x_idx, m_idx, x_idx)]
    v_xxxx = v_full[np.ix_(x_idx, x_idx, x_idx, x_idx)]
    v_mmmm = v_full[np.ix_(m_idx, m_idx, m_idx, m_idx)]
    h_mm = h_full[np.ix_(m_idx, m_idx)]

    # gamma_1 (active Fock), gamma_2, gamma_0 (scalar) -- SeQuant einsum
    gamma_xx = np.einsum("mxma->xa", v_mxmx, optimize=True) \
        + np.einsum("xa->xa", h_xx, optimize=True)
    gamma_xxxx = 1 / 4 * np.einsum("xabc->xacb", v_xxxx, optimize=True)
    gamma_0 = np.einsum("mm->", h_mm, optimize=True) \
        + 0.5 * np.einsum("mnmn->", v_mmmm, optimize=True)

    # Reconstruct active v from gamma_2 (1/4 v reordered)
    v_act = -4.0 * gamma_xxxx

    # gamma -> chi (Table V): M over ACTIVE occupied (i)
    x_list = x_idx.tolist()
    act_occ_pos = [x_list.index(int(ii)) for ii in i_idx]
    chi_1 = gamma_xx.copy()
    for M in act_occ_pos:
        chi_1 -= v_act[M, :, M, :]
    chi_2 = v_act.copy()
    return chi_1, chi_2, gamma_0, act_occ_pos


def chi_to_qdk_hamiltonian(chi_1, chi_2, core_energy):
    """Convert chi (interleaved spin-orbital physicist) -> qdk Hamiltonian.

    Active spatial p -> alpha=2p, beta=2p+1.
      h1_spatial[p,q]      = chi_1[2p, 2q]
      (pq|rs)_chemist      = chi_2[2p, 2r+1, 2q, 2s+1]   (cross-spin)
    """
    n_act = chi_1.shape[0]
    n_sp = n_act // 2
    h1_sp = np.zeros((n_sp, n_sp))
    h2_sp = np.zeros((n_sp, n_sp, n_sp, n_sp))
    for p in range(n_sp):
        for q in range(n_sp):
            h1_sp[p, q] = chi_1[2 * p, 2 * q]
            for r in range(n_sp):
                for s in range(n_sp):
                    h2_sp[p, q, r, s] = chi_2[2 * p, 2 * r + 1, 2 * q, 2 * s + 1]

    orbitals = ModelOrbitals(n_sp, True)
    container = CanonicalFourCenterHamiltonianContainer(
        h1_sp, h2_sp.ravel(), orbitals, core_energy, np.zeros((n_sp, n_sp)))
    return Hamiltonian(container)


def run_config(label, n_act_occ, n_act_vir, wfn, h1, eri, e_nuc, nmo, macis):
    nocc_sp = 2  # LiH closed-shell: 2 doubly-occupied spatial

    # Spin-orbital physicist integrals (full)
    h_full = spatial_to_spinorb_h1(h1)
    eri_so = spatial_to_spinorb_eri(eri)
    v_full = np.einsum("prqs->pqrs", eri_so) - np.einsum("psqr->pqrs", eri_so)

    o_idx, i_idx, a_idx, m_idx, x_idx = partition(nocc_sp, n_act_occ, n_act_vir)

    # SeQuant downfolding -> chi + scalar
    chi_1, chi_2, gamma_0, act_occ_pos = sequant_downfold(
        h_full, v_full, m_idx, x_idx, i_idx)

    # Core energy = E_nuc + gamma_0 (SCF electronic) - active reference(chi)
    e_ref = e_nuc + gamma_0
    active_ref = sum(chi_1[p, p] for p in act_occ_pos)
    active_ref += 0.5 * sum(chi_2[p, q, p, q]
                            for p in act_occ_pos for q in act_occ_pos)
    core_energy = e_ref - active_ref

    # Build chi qdk Hamiltonian and run MACIS FCI (active space)
    chi_ham = chi_to_qdk_hamiltonian(chi_1, chi_2, core_energy)
    e_chi, _ = macis.run(chi_ham, n_act_occ, n_act_occ)

    # CASCI reference: qdk active_space_selector + hamiltonian_constructor + MACIS
    # qdk_valence selects frontier orbitals (active occ down from HOMO, active
    # virt up from LUMO) -- matches our partition().
    selector = create("active_space_selector", "qdk_valence")
    selector.settings().set("num_active_electrons", 2 * n_act_occ)
    selector.settings().set("num_active_orbitals", n_act_occ + n_act_vir)
    active_wfn = selector.run(wfn)
    active_ham = create("hamiltonian_constructor").run(active_wfn.get_orbitals())
    e_casci, _ = macis.run(active_ham, n_act_occ, n_act_occ)

    diff = abs(e_chi - e_casci)
    ok = diff < 1e-8
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:26s} CAS({2*n_act_occ}e,{n_act_occ+n_act_vir}o)  "
          f"chi-FCI={e_chi:.8f}  CASCI={e_casci:.8f}  diff={diff:.1e}")
    return ok


def main():
    structure, wfn, ham, h1, eri, e_nuc, nmo, e_hf = build_qdk_original()
    print(f"qdk-chemistry LiH/STO-3G: SCF={e_hf:.8f}, nmo={nmo}, E_nuc={e_nuc:.6f}\n")
    print("SeQuant downfold -> MACIS chi-FCI  vs  qdk-chemistry MACIS CASCI:")

    macis = create("multi_configuration_calculator")

    all_ok = True
    for label, n_occ, n_vir in CONFIGS:
        ok = run_config(label, n_occ, n_vir, wfn, h1, eri, e_nuc, nmo, macis)
        all_ok = all_ok and ok

    print(f"\n  {'ALL PASS' if all_ok else 'SOME FAILED'}: "
          f"SeQuant downfold (MACIS FCI) = qdk-chemistry MACIS CASCI")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
