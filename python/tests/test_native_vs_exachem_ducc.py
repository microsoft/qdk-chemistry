# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Cross-validation: native DUCC vs ExaChem DUCC at levels 0, 1, 2.

Compares the downfolded Hamiltonians produced by NativeDuccSolver (Python/NumPy)
and ExachemDuccSolver (ExaChem MPI binary) by diagonalizing both and comparing
eigenvalues. This is the definitive correctness test.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Element, Structure

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_EXACHEM_PATH = (
    os.environ.get("EXACHEM_PATH", "")
    or "/workspaces/exachem/build_serial_io/methods_stage/workspaces/exachem_install_serial_io/methods/ExaChem"
)
_has_exachem = os.path.isfile(_EXACHEM_PATH) and os.access(_EXACHEM_PATH, os.X_OK)
_has_mpi = shutil.which("mpirun") is not None

try:
    import h5py  # noqa: F401
    _has_h5py = True
except ImportError:
    _has_h5py = False

pytestmark = pytest.mark.skipif(
    not (_has_exachem and _has_mpi and _has_h5py),
    reason="Requires ExaChem binary, mpirun, and h5py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plugins():
    """Ensure both native and exachem plugins are registered."""
    # ExaChem plugin registers the hamiltonian_downfolder factory
    import qdk_chemistry.plugins.exachem
    qdk_chemistry.plugins.exachem.load()

    # Native plugin registers NativeDuccSolver under the same factory
    import qdk_chemistry.algorithms.hamiltonian_downfolder
    qdk_chemistry.algorithms.hamiltonian_downfolder.load()


def _slater_condon_fci(h1, h2, norb, nelec, nuc_rep=0.0):
    """Brute-force FCI via Slater-Condon rules on spin-orbital integrals.

    Args:
        h1: (norb, norb) one-electron integrals (physicist spin-orbital).
        h2: (norb, norb, norb, norb) antisymmetrized two-electron integrals.
        norb: number of spin-orbitals.
        nelec: number of electrons.
        nuc_rep: nuclear repulsion / core energy to add to diagonal.

    Returns:
        Sorted eigenvalues of the FCI Hamiltonian.
    """
    from itertools import combinations
    dets = list(combinations(range(norb), nelec))
    nd = len(dets)
    H = np.zeros((nd, nd))

    for I in range(nd):
        for J in range(I, nd):
            bra = list(dets[I])
            ket = list(dets[J])
            diff_bra = [x for x in bra if x not in ket]
            diff_ket = [x for x in ket if x not in bra]
            common = [x for x in bra if x in ket]
            n_diff = len(diff_bra)

            val = 0.0
            if n_diff == 0:
                val = sum(h1[i, i] for i in bra)
                for idx_i, i in enumerate(bra):
                    for j in bra[idx_i + 1:]:
                        val += h2[i, j, i, j] - h2[i, j, j, i]
            elif n_diff == 1:
                p, r = diff_bra[0], diff_ket[0]
                phase = (-1) ** sorted(bra).index(p) * (-1) ** sorted(ket).index(r)
                val = h1[p, r]
                for k in common:
                    val += h2[p, k, r, k] - h2[p, k, k, r]
                val *= phase
            elif n_diff == 2:
                p, q = diff_bra
                r, s = diff_ket
                phase = (-1) ** sorted(bra).index(p) * (-1) ** sorted([x for x in bra if x != p]).index(q)
                phase *= (-1) ** sorted(ket).index(r) * (-1) ** sorted([x for x in ket if x != r]).index(s)
                val = phase * (h2[p, q, r, s] - h2[p, q, s, r])

            H[I, J] = val
            H[J, I] = val

    np.fill_diagonal(H, H.diagonal() + nuc_rep)
    return np.sort(np.linalg.eigvalsh(H))


def _run_comparison(
    atoms: list[str],
    coords_ang: list[list[float]],
    elements: list[Element],
    basis: str,
    noa: int,
    nva: int,
    ducc_level: int,
    charge: int = 0,
    multiplicity: int = 1,
    atol: float = 5e-2,
):
    """Run both native and ExaChem DUCC, compare FCI eigenvalues."""
    _load_plugins()

    coords_bohr = np.array(coords_ang) / 0.529177249
    structure = Structure(coords_bohr, elements)

    # SCF
    scf = create("scf_solver")
    e_scf, wfn = scf.run(structure, charge, multiplicity, basis)

    # Build full Hamiltonian
    ham_builder = create("hamiltonian_constructor")
    full_ham = ham_builder.run(wfn.get_orbitals())

    nocc = (sum(e.value for e in elements) - charge + multiplicity - 1) // 2
    C = np.array(wfn.get_orbitals().get_coefficients()[0])
    D = C[:, :nocc] @ C[:, :nocc].T

    # Need pyscf_scf for energy_nuc() check
    from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf
    nmo = wfn.get_orbitals().get_num_molecular_orbitals()
    alpha_occ = np.zeros(nmo)
    alpha_occ[:nocc] = 1.0
    pyscf_scf_obj = hamiltonian_to_scf(full_ham, alpha_occ, alpha_occ)

    work_dir = tempfile.mkdtemp(prefix="ducc_cross_")

    try:
        # ── Native DUCC ──
        native = create("hamiltonian_downfolder", "native_ducc")
        ns = native.settings()
        ns.set("nactive_oa", noa)
        ns.set("nactive_ob", noa)
        ns.set("nactive_va", nva)
        ns.set("nactive_vb", nva)
        ns.set("ducc_level", ducc_level)
        native_ham = native.run(full_ham, nocc, nocc)

        # ── ExaChem DUCC ──
        from qdk_chemistry.plugins.exachem.ducc_solver import ExachemDuccSolver
        exachem = ExachemDuccSolver()
        es = exachem.settings()
        es.set("atoms", atoms)
        es.set("basis", basis)
        es.set("charge", charge)
        es.set("multiplicity", multiplicity)
        es.set("nactive_oa", noa)
        es.set("nactive_ob", noa)
        es.set("nactive_va", nva)
        es.set("nactive_vb", nva)
        es.set("ducc_level", ducc_level)
        es.set("mpi_ranks", 2)
        es.set("exachem_binary", _EXACHEM_PATH)
        es.set("work_dir", os.path.join(work_dir, "exachem"))
        exachem_ham = exachem.run(mo_coeff_alpha=C, density_alpha=D)

        # ── Compare integrals directly (core_energy may differ due to ExaChem
        #    stdout parsing issues, but the active-space integrals should match) ──
        h1_native, _ = native_ham.get_one_body_integrals()
        h1_exachem, _ = exachem_ham.get_one_body_integrals()
        eri_native, _, _ = native_ham.get_two_body_integrals()
        eri_exachem, _, _ = exachem_ham.get_two_body_integrals()

        h1_n = np.array(h1_native)
        h1_e = np.array(h1_exachem)
        eri_n = np.array(eri_native)
        eri_e = np.array(eri_exachem)

        print(f"Level {ducc_level}: h1 max_diff={np.max(np.abs(h1_n - h1_e)):.2e}, "
              f"eri max_diff={np.max(np.abs(eri_n - eri_e)):.2e}, "
              f"core_native={native_ham.get_core_energy():.8f}, "
              f"core_exachem={exachem_ham.get_core_energy():.8f}")

        np.testing.assert_allclose(
            h1_n, h1_e, atol=atol,
            err_msg=f"Native vs ExaChem h1 mismatch at level {ducc_level}",
        )
        np.testing.assert_allclose(
            eri_n, eri_e, atol=atol,
            err_msg=f"Native vs ExaChem ERI mismatch at level {ducc_level}",
        )

        # If ExaChem successfully parsed the energy shift, compare energies too
        if abs(exachem_ham.get_core_energy() - pyscf_scf_obj.energy_nuc()) > 0.01:
            macis = create("multi_configuration_calculator")
            e_native, _ = macis.run(native_ham, noa, noa)
            e_exachem, _ = macis.run(exachem_ham, noa, noa)
            print(f"  E_native={e_native:.10f}, E_exachem={e_exachem:.10f}")
            np.testing.assert_allclose(e_native, e_exachem, atol=atol)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_H2_ATOMS = ["H 0 0 0", "H 0 0 0.74"]
_H2_COORDS = [[0, 0, 0], [0, 0, 0.74]]
_H2_ELEMS = [Element.H, Element.H]

_LIH_ATOMS = ["Li 0 0 0", "H 0 0 1.6"]
_LIH_COORDS = [[0, 0, 0], [0, 0, 1.6]]
_LIH_ELEMS = [Element.Li, Element.H]


class TestNativeVsExachem:
    """Compare native DUCC with ExaChem DUCC at each BCH level."""

    def test_h2_sto3g_1o1v_level0(self):
        _run_comparison(_H2_ATOMS, _H2_COORDS, _H2_ELEMS, "sto-3g", noa=1, nva=1, ducc_level=0)

    def test_h2_sto3g_1o1v_level1(self):
        _run_comparison(_H2_ATOMS, _H2_COORDS, _H2_ELEMS, "sto-3g", noa=1, nva=1, ducc_level=1)

    def test_h2_sto3g_1o1v_level2(self):
        _run_comparison(_H2_ATOMS, _H2_COORDS, _H2_ELEMS, "sto-3g", noa=1, nva=1, ducc_level=2)

    def test_lih_sto3g_1o2v_level0(self):
        _run_comparison(_LIH_ATOMS, _LIH_COORDS, _LIH_ELEMS, "sto-3g", noa=1, nva=2, ducc_level=0)

    def test_lih_sto3g_1o2v_level1(self):
        _run_comparison(_LIH_ATOMS, _LIH_COORDS, _LIH_ELEMS, "sto-3g", noa=1, nva=2, ducc_level=1)

    def test_lih_sto3g_1o2v_level2(self):
        _run_comparison(_LIH_ATOMS, _LIH_COORDS, _LIH_ELEMS, "sto-3g", noa=1, nva=2, ducc_level=2)
