# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests comparing 4-space vs 2-space (SI) WickedDucc.

Validates that WickedDucc4SpaceSolver produces identical results to
WickedDuccSISolver for all systems, active spaces, and BCH levels.
"""

import pytest

# os.environ.setdefault("OMP_NUM_THREADS", "1")

wickd = pytest.importorskip("wickd")
pytest.importorskip("pyscf")

import qdk_chemistry.algorithms.hamiltonian_downfolder  # noqa: E402
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

qdk_chemistry.algorithms.hamiltonian_downfolder.load()


def _compare_si_4space(xyz, basis, nocc, noa, nva, ducc_level, atol=1e-10):
    """Run SI (2-space) and 4-space solvers, assert energies match."""
    structure = Structure.from_xyz(xyz)
    _, wfn = create("scf_solver").run(structure, charge=0, spin_multiplicity=1, basis_or_guess=basis)
    full_ham = create("hamiltonian_constructor").run(wfn.get_orbitals())

    si = create("hamiltonian_downfolder", "wicked_ducc_si")
    for k in ["nactive_oa", "nactive_ob"]:
        si.settings().set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        si.settings().set(k, nva)
    si.settings().set("ducc_level", ducc_level)
    e_si, _ = create("multi_configuration_calculator", "macis_cas").run(si.run(full_ham, nocc, nocc), noa, noa)

    fs = create("hamiltonian_downfolder", "wicked_ducc_4space")
    for k in ["nactive_oa", "nactive_ob"]:
        fs.settings().set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        fs.settings().set(k, nva)
    fs.settings().set("ducc_level", ducc_level)
    e_4s, _ = create("multi_configuration_calculator", "macis_cas").run(fs.run(full_ham, nocc, nocc), noa, noa)

    assert abs(e_si - e_4s) < atol, f"SI={e_si:.12f} 4space={e_4s:.12f} diff={abs(e_si - e_4s):.2e}"


class TestWickedDucc4SpaceVsSI:
    """4-space vs 2-space SI: must give identical results."""

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2_sto3g_cas11(self, ducc_level):
        """H2/STO-3G CAS(1,1) — minimal, all orbitals active."""
        _compare_si_4space("2\nH2\nH 0 0 0\nH 0 0 0.740848\n", "sto-3g", 1, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas11(self, ducc_level):
        """LiH/STO-3G CAS(1,1) — frozen core + frozen virtuals."""
        _compare_si_4space("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas12(self, ducc_level):
        """LiH/STO-3G CAS(1,2) — asymmetric active space."""
        _compare_si_4space("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas22(self, ducc_level):
        """LiH/STO-3G CAS(2,2) — all occupied active."""
        _compare_si_4space("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 2, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas11(self, ducc_level):
        """H2O/STO-3G CAS(1,1) — large frozen core."""
        _compare_si_4space(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g",
            5,
            1,
            1,
            ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas22(self, ducc_level):
        """H2O/STO-3G CAS(2,2) — medium active space, tests cross-space T2 blocks."""
        _compare_si_4space(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g",
            5,
            2,
            2,
            ducc_level,
        )
