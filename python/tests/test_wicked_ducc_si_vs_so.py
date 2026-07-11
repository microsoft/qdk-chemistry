# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests comparing spin-integrated (SI) vs spin-orbital (SO) WickedDucc.

Validates that WickedDuccSISolver produces identical results to WickedDuccSolver
for all systems, active spaces, and BCH levels.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

wicked = pytest.importorskip("wicked")
pytest.importorskip("pyscf")

from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

import qdk_chemistry.algorithms.hamiltonian_downfolder  # noqa: E402

qdk_chemistry.algorithms.hamiltonian_downfolder.load()


def _compare_so_si(xyz, basis, nocc, noa, nva, ducc_level, atol=1e-10):
    """Run SO and SI solvers, assert energies match."""
    structure = Structure.from_xyz(xyz)
    _, wfn = create("scf_solver").run(structure, charge=0, spin_multiplicity=1, basis_or_guess=basis)
    full_ham = create("hamiltonian_constructor").run(wfn.get_orbitals())

    so = create("hamiltonian_downfolder", "wicked_ducc")
    for k in ["nactive_oa", "nactive_ob"]:
        so.settings().set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        so.settings().set(k, nva)
    so.settings().set("ducc_level", ducc_level)
    e_so, _ = create("multi_configuration_calculator", "macis_cas").run(so.run(full_ham, nocc, nocc), noa, noa)

    si = create("hamiltonian_downfolder", "wicked_ducc_si")
    for k in ["nactive_oa", "nactive_ob"]:
        si.settings().set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        si.settings().set(k, nva)
    si.settings().set("ducc_level", ducc_level)
    e_si, _ = create("multi_configuration_calculator", "macis_cas").run(si.run(full_ham, nocc, nocc), noa, noa)

    assert abs(e_so - e_si) < atol, f"SO={e_so:.12f} SI={e_si:.12f} diff={abs(e_so-e_si):.2e}"


class TestWickedDuccSOvsSI:
    """Spin-orbital vs spin-integrated: must give identical results."""

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2_sto3g_cas11(self, ducc_level):
        """H2/STO-3G CAS(1,1) — all orbitals active."""
        _compare_so_si("2\nH2\nH 0 0 0\nH 0 0 0.740848\n", "sto-3g", 1, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas11(self, ducc_level):
        """LiH/STO-3G CAS(1,1) — frozen core + frozen virtuals."""
        _compare_so_si("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas12(self, ducc_level):
        """LiH/STO-3G CAS(1,2) — asymmetric active space."""
        _compare_so_si("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas22(self, ducc_level):
        """LiH/STO-3G CAS(2,2) — all occupied active."""
        _compare_so_si("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 2, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas11(self, ducc_level):
        """H2O/STO-3G CAS(1,1) — large frozen core."""
        _compare_so_si(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g", 5, 1, 1, ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas22(self, ducc_level):
        """H2O/STO-3G CAS(2,2) — medium active space."""
        _compare_so_si(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g", 5, 2, 2, ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h4_sto3g_cas22(self, ducc_level):
        """H4/STO-3G CAS(2,2) — all orbitals active."""
        _compare_so_si("4\nH4\nH 0 0 0\nH 0 0 1.0\nH 0 0 2.0\nH 0 0 3.0\n", "sto-3g", 2, 2, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_n2_sto3g_cas22(self, ducc_level):
        """N2/STO-3G CAS(2,2) — larger molecule, many frozen orbitals."""
        _compare_so_si("2\nN2\nN 0 0 0\nN 0 0 1.098\n", "sto-3g", 7, 2, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_ccpvdz_cas35(self, ducc_level):
        """H2O/cc-pVDZ CAS(3,5) — large active space (3136 determinants)."""
        _compare_so_si(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "cc-pvdz", 5, 3, 5, ducc_level,
        )
