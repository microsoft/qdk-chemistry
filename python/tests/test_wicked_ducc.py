# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests for WickedDuccSolver — wicked-based DUCC Hamiltonian downfolding.

Validates WickedDuccSolver against ExaChem's DUCC implementation (noscf mode,
same MOs). Both use the same qdk-chemistry SCF orbitals; differences arise only
from ExaChem's Cholesky decomposition of the 2e integrals.

Requires: wicked, pyscf, h5py, ExaChem binary (EXACHEM_PATH env var), mpirun.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Skip entire module if dependencies missing
wicked = pytest.importorskip("wicked")
pytest.importorskip("pyscf")
pytest.importorskip("h5py")

from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

import qdk_chemistry.algorithms.hamiltonian_downfolder  # noqa: E402

qdk_chemistry.algorithms.hamiltonian_downfolder.load()

_EXACHEM_PATH = os.environ.get("EXACHEM_PATH", os.environ.get("EXACHEM_BIN"))
_requires_exachem = pytest.mark.skipif(
    not _EXACHEM_PATH or not shutil.which("mpirun"),
    reason="Requires EXACHEM_PATH and mpirun",
)


def _run_wicked_ducc(xyz: str, basis: str, nocc: int, noa: int, nva: int, ducc_level: int):
    """Run WickedDuccSolver and return FCI energy."""
    structure = Structure.from_xyz(xyz)
    scf = create("scf_solver")
    _, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess=basis)
    ham_builder = create("hamiltonian_constructor")
    full_ham = ham_builder.run(wfn.get_orbitals())

    solver = create("hamiltonian_downfolder", "wicked_ducc")
    s = solver.settings()
    for k in ["nactive_oa", "nactive_ob"]:
        s.set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        s.set(k, nva)
    s.set("ducc_level", ducc_level)
    ducc_ham = solver.run(full_ham, nocc, nocc)

    macis = create("multi_configuration_calculator", "macis_cas")
    energy, _ = macis.run(ducc_ham, noa, noa)
    return energy, wfn


def _run_exachem_ducc(wfn, atoms: list[str], basis: str, nocc: int, noa: int, nva: int, ducc_level: int):
    """Run ExaChem DUCC (noscf) with same MOs and return FCI energy."""
    from qdk_chemistry.plugins.exachem.ducc_solver import ExachemDuccSolver

    C = np.array(wfn.get_orbitals().get_coefficients()[0])
    D = C[:, :nocc] @ C[:, :nocc].T

    work_dir = tempfile.mkdtemp(prefix="wicked_ducc_test_")
    ec = ExachemDuccSolver()
    es = ec.settings()
    es.set("atoms", atoms)
    es.set("basis", basis)
    for k in ["nactive_oa", "nactive_ob"]:
        es.set(k, noa)
    for k in ["nactive_va", "nactive_vb"]:
        es.set(k, nva)
    es.set("ducc_level", ducc_level)
    es.set("mpi_ranks", 2)
    es.set("exachem_binary", _EXACHEM_PATH)
    es.set("work_dir", work_dir)
    es.set("timeout", 600)

    ec_ham = ec.run(mo_coeff_alpha=C, density_alpha=D)
    macis = create("multi_configuration_calculator", "macis_cas")
    energy, _ = macis.run(ec_ham, noa, noa)

    shutil.rmtree(work_dir, ignore_errors=True)
    return energy


class TestWickedDuccVsExachem:
    """Validate WickedDuccSolver against ExaChem DUCC (noscf, same MOs)."""

    @_requires_exachem
    def test_h2_sto3g_cas11_level1(self):
        """H2/STO-3G CAS(1,1) level 1 — minimal system, all orbitals active."""
        xyz = "2\nH2\nH 0 0 0\nH 0 0 0.740848\n"
        e_w, wfn = _run_wicked_ducc(xyz, "sto-3g", nocc=1, noa=1, nva=1, ducc_level=1)
        e_ec = _run_exachem_ducc(wfn, ["H 0 0 0", "H 0 0 0.740848"], "sto-3g", nocc=1, noa=1, nva=1, ducc_level=1)
        assert abs(e_w - e_ec) < 1e-6, f"diff={abs(e_w - e_ec):.2e}"

    @_requires_exachem
    def test_lih_sto3g_cas11_level1(self):
        """LiH/STO-3G CAS(1,1) level 1 — frozen core + frozen virtuals."""
        xyz = "2\nLiH\nLi 0 0 0\nH 0 0 1.595\n"
        e_w, wfn = _run_wicked_ducc(xyz, "sto-3g", nocc=2, noa=1, nva=1, ducc_level=1)
        e_ec = _run_exachem_ducc(wfn, ["Li 0 0 0", "H 0 0 1.595"], "sto-3g", nocc=2, noa=1, nva=1, ducc_level=1)
        assert abs(e_w - e_ec) < 1e-6, f"diff={abs(e_w - e_ec):.2e}"

    @_requires_exachem
    def test_lih_sto3g_cas11_level2(self):
        """LiH/STO-3G CAS(1,1) level 2 — triple commutator."""
        xyz = "2\nLiH\nLi 0 0 0\nH 0 0 1.595\n"
        e_w, wfn = _run_wicked_ducc(xyz, "sto-3g", nocc=2, noa=1, nva=1, ducc_level=2)
        e_ec = _run_exachem_ducc(wfn, ["Li 0 0 0", "H 0 0 1.595"], "sto-3g", nocc=2, noa=1, nva=1, ducc_level=2)
        assert abs(e_w - e_ec) < 1e-6, f"diff={abs(e_w - e_ec):.2e}"

    @_requires_exachem
    @pytest.mark.parametrize("ducc_level", [1, 2])
    def test_h2o_ccpvdz_cas35(self, ducc_level):
        """H2O/cc-pVDZ CAS(3,5) — large active space (3136 determinants)."""
        xyz = "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n"
        atoms = ["O 0 0 0.117790", "H 0 0.756950 -0.471161", "H 0 -0.756950 -0.471161"]
        e_w, wfn = _run_wicked_ducc(xyz, "cc-pvdz", nocc=5, noa=3, nva=5, ducc_level=ducc_level)
        e_ec = _run_exachem_ducc(wfn, atoms, "cc-pvdz", nocc=5, noa=3, nva=5, ducc_level=ducc_level)
        assert abs(e_w - e_ec) < 1e-5, f"diff={abs(e_w - e_ec):.2e}"


class TestWickedDuccEnergyHierarchy:
    """Verify physical correctness without ExaChem."""

    def test_h2_sto3g_levels_monotonic(self):
        """DUCC levels 0 < 1 ≤ 2 in correlation recovered (H2 is exact at level 1)."""
        xyz = "2\nH2\nH 0 0 0\nH 0 0 0.740848\n"
        energies = []
        for lvl in [0, 1, 2]:
            e, _ = _run_wicked_ducc(xyz, "sto-3g", nocc=1, noa=1, nva=1, ducc_level=lvl)
            energies.append(e)
        # All levels give the same energy for H2/STO-3G (all orbitals active)
        assert abs(energies[0] - energies[1]) < 1e-10
        assert abs(energies[1] - energies[2]) < 1e-10

    def test_lih_sto3g_below_scf(self):
        """DUCC energy should be below SCF for non-trivial active space."""
        xyz = "2\nLiH\nLi 0 0 0\nH 0 0 1.595\n"
        structure = Structure.from_xyz(xyz)
        scf = create("scf_solver")
        e_scf, _ = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

        e_ducc, _ = _run_wicked_ducc(xyz, "sto-3g", nocc=2, noa=1, nva=1, ducc_level=2)
        assert e_ducc < e_scf - 0.001  # at least 1 mHa below SCF

    def test_h2o_ccpvdz_level2_below_level1(self):
        """Level 2 should recover more correlation than level 1."""
        xyz = "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n"
        e1, _ = _run_wicked_ducc(xyz, "cc-pvdz", nocc=5, noa=3, nva=5, ducc_level=1)
        e2, _ = _run_wicked_ducc(xyz, "cc-pvdz", nocc=5, noa=3, nva=5, ducc_level=2)
        assert e2 < e1 - 0.001  # level 2 gives significantly lower energy
