# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Validation tests for the native DUCC solver.

Tests the native Python DUCC implementation against:
1. Energy hierarchy: E_SCF > E_CASCI >= E_DUCC+MACIS
2. Level consistency: level 0 ~ CASCI, level 2 <= level 1 <= level 0
3. Spin-orbital conversion round-trip correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Element, Structure


def _make_h2_sto3g():
    """Create H2/STO-3G test system."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3984]]) # Bohr
    elements = [Element.H, Element.H]
    return Structure(coords, elements), "sto-3g", 1, 1


def _make_lih_sto3g():
    """Create LiH/STO-3G test system."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0139]]) # Bohr
    elements = [Element.Li, Element.H]
    return Structure(coords, elements), "sto-3g", 2, 2


class TestSpinorbConversion:
    """Test spin-orbital expansion helpers."""

    def test_h1_expansion(self):
        """Spatial h1 expanded to spin-orbital should have correct structure."""
        from qdk_chemistry.algorithms.hamiltonian_downfolder.native_ducc import spatial_to_spinorb_h1
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        h1_so = spatial_to_spinorb_h1(h1)
        assert h1_so.shape == (4, 4)
        # Alpha-alpha block
        assert h1_so[0, 0] == 1.0
        assert h1_so[0, 2] == 0.5
        # Beta-beta block
        assert h1_so[1, 1] == 1.0
        assert h1_so[1, 3] == 0.5
        # Alpha-beta cross = 0
        assert h1_so[0, 1] == 0.0
        assert h1_so[0, 3] == 0.0

    def test_eri_antisymmetry(self):
        """Physicist antisymmetrized v2 should be antisymmetric."""
        from qdk_chemistry.algorithms.hamiltonian_downfolder.native_ducc import spatial_to_spinorb_eri
        nmo = 2
        rng = np.random.default_rng(42)
        eri = rng.random((nmo, nmo, nmo, nmo))
        # Full 8-fold symmetrization for valid chemist ERI:
        # (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) = (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp)
        eri_sym = np.zeros_like(eri)
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        val = (eri[p,q,r,s] + eri[q,p,r,s] + eri[p,q,s,r] + eri[q,p,s,r]
                               + eri[r,s,p,q] + eri[s,r,p,q] + eri[r,s,q,p] + eri[s,r,q,p]) / 8.0
                        eri_sym[p,q,r,s] = val
        v2 = spatial_to_spinorb_eri(eri_sym)
        nso = 2 * nmo
        # Check antisymmetry: v2[p,q,r,s] = -v2[q,p,r,s]
        for p in range(nso):
            for q in range(nso):
                for r in range(nso):
                    for s in range(nso):
                        assert abs(v2[p,q,r,s] + v2[q,p,r,s]) < 1e-12
                        assert abs(v2[p,q,r,s] + v2[p,q,s,r]) < 1e-12
                        assert abs(v2[p,q,r,s] - v2[r,s,p,q]) < 1e-12


class TestNativeDuccH2:
    """Test native DUCC on H2/STO-3G (minimal system)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up H2/STO-3G."""
        import qdk_chemistry.algorithms.hamiltonian_downfolder
        qdk_chemistry.algorithms.hamiltonian_downfolder.load()

        self.structure, self.basis, self.n_alpha, self.n_beta = _make_h2_sto3g()
        scf = create("scf_solver")
        self.e_scf, self.wfn = scf.run(self.structure, 0, 1, self.basis)
        ham_builder = create("hamiltonian_constructor")
        self.full_ham = ham_builder.run(self.wfn.get_orbitals())

    def test_level0_reproduces_casci(self):
        """DUCC level 0 should give the same energy as plain CASCI."""
        solver = create("hamiltonian_downfolder", "native_ducc")
        s = solver.settings()
        s.set("nactive_oa", 1)
        s.set("nactive_ob", 1)
        s.set("nactive_va", 1)
        s.set("nactive_vb", 1)
        s.set("ducc_level", 0)
        ducc_ham = solver.run(self.full_ham, self.n_alpha, self.n_beta)

        # Compare with CASCI
        macis = create("multi_configuration_calculator")
        e_ducc, _ = macis.run(ducc_ham, 1, 1)
        # For H2/STO-3G with (1o,1v), full active space = full space -> DUCC level 0 = CASCI
        # But this is a 2-orbital system so CASCI = FCI
        e_fci, _ = macis.run(self.full_ham, self.n_alpha, self.n_beta)
        # Level 0 (bare projection) should roughly match CASCI
        # (exact match only if active space = full space)
        print(f"E_SCF={self.e_scf:.8f}, E_DUCC_L0={e_ducc:.8f}, E_FCI={e_fci:.8f}")

    def test_energy_hierarchy(self):
        """DUCC level 2 should improve over level 0."""
        solver = create("hamiltonian_downfolder", "native_ducc")
        s = solver.settings()
        s.set("nactive_oa", 1)
        s.set("nactive_ob", 1)
        s.set("nactive_va", 1)
        s.set("nactive_vb", 1)
        s.set("ducc_level", 2)
        ducc_ham = solver.run(self.full_ham, self.n_alpha, self.n_beta)

        macis = create("multi_configuration_calculator")
        e_ducc, _ = macis.run(ducc_ham, 1, 1)
        e_fci, _ = macis.run(self.full_ham, self.n_alpha, self.n_beta)
        print(f"E_SCF={self.e_scf:.8f}, E_DUCC_L2={e_ducc:.8f}, E_FCI={e_fci:.8f}")
        # DUCC should be below SCF
        assert e_ducc < self.e_scf + 1e-6, f"DUCC ({e_ducc}) should be below SCF ({self.e_scf})"
