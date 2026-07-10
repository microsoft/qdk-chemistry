# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Integration tests for the ExaChem DUCC solver.

Compares two pipelines for each molecule/basis/active-space combination:

1. **qdk-chemistry SCF → ExaChem DUCC (noscf)**: Uses qdk-chemistry's SCF
   orbitals, exports them in ExaChem's serial-IO HDF5 format, then runs
   ExaChem's DUCC with ``noscf=true``.
2. **ExaChem end-to-end**: ExaChem runs its own SCF + DUCC from scratch.

Both pipelines' downfolded Hamiltonians are diagonalized with qdk-chemistry's
MACIS CI solver; their ground-state energies should match to within ~1e-5
Hartree (limited by Libint2 version differences).

Prerequisites:
    - ExaChem binary built with ``USE_SERIAL_IO`` at ``EXACHEM_PATH``
    - MPI runtime (``mpirun``)
    - ``h5py`` Python package

These tests are slow (each runs ExaChem twice via MPI) and require the
ExaChem binary. They are skipped if ``EXACHEM_PATH`` is not set or the
binary is not found.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import BasisSet, Element, Structure
from qdk_chemistry.plugins.exachem.conversion import (
    fcidump_to_hamiltonian,
    parse_ducc_results,
    parse_energy_shift,
)
from qdk_chemistry.plugins.exachem.ducc_solver import ExachemDuccSolver
from qdk_chemistry.plugins.exachem.scf_export import reorder_ao_qdk_to_exachem

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_exachem_path = os.environ.get("EXACHEM_PATH", "")
_has_exachem = _exachem_path and os.path.isfile(_exachem_path) and os.access(_exachem_path, os.X_OK)
_has_mpi = shutil.which("mpirun") is not None

try:
    import h5py  # noqa: F401

    _has_h5py = True
except ImportError:
    _has_h5py = False

try:
    from pyscf import fci as _pyscf_fci  # noqa: F401

    _has_pyscf = True
except ImportError:
    _has_pyscf = False

pytestmark = pytest.mark.skipif(
    not (_has_exachem and _has_mpi and _has_h5py),
    reason="Requires EXACHEM_PATH, mpirun, and h5py",
)


# ---------------------------------------------------------------------------
# Core (constant) energy helper for ExaChem standalone runs
# ---------------------------------------------------------------------------


def _core_energy_from_stdout(stdout: str, fcidump) -> float:
    """Reconstruct the qdk Hamiltonian core energy from an ExaChem run.

    ExaChem's energy shift excludes nuclear repulsion, so
    ``core_energy = energy_shift + V_nuc``. At ``ducc_lvl=0`` there is no
    "Total Energy Shift" line, so :func:`parse_energy_shift` falls back to
    ``Full SCF - Bare Active Space SCF`` (matching ExaChem's ``grab_data``).
    """
    shift = parse_energy_shift(stdout)
    return (shift + fcidump.nuclear_repulsion) if shift is not None else fcidump.nuclear_repulsion


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _run_ducc_comparison(
    atoms: list[str],
    coords_ang: list[list[float]],
    elements: list[Element],
    basis: str,
    noa: int,
    nva: int,
    charge: int = 0,
    multiplicity: int = 1,
    ducc_level: int = 2,
    atol: float = 1e-5,
) -> None:
    """Run both pipelines and assert eigenvalues match."""
    work_dir = tempfile.mkdtemp(prefix="ducc_test_")
    try:
        # ── Pipeline 1: qdk SCF → ExaChem DUCC (noscf) ──
        coords_bohr = np.array(coords_ang) / 0.529177249
        structure = Structure(coords_bohr, elements)
        scf = create("scf_solver")
        _, wfn = scf.run(structure, charge, multiplicity, basis)
        C = np.array(wfn.get_orbitals().get_coefficients()[0])
        nelec_alpha = (sum(e.value for e in elements) - charge + multiplicity - 1) // 2
        D = C[:, :nelec_alpha] @ C[:, :nelec_alpha].T

        solver = ExachemDuccSolver()
        s = solver.settings()
        s.set("atoms", atoms)
        s.set("basis", basis)
        s.set("charge", charge)
        s.set("multiplicity", multiplicity)
        s.set("nactive_oa", noa)
        s.set("nactive_ob", noa)
        s.set("nactive_va", nva)
        s.set("nactive_vb", nva)
        s.set("ducc_level", ducc_level)
        s.set("mpi_ranks", 2)
        s.set("exachem_binary", _exachem_path)
        s.set("work_dir", os.path.join(work_dir, "p1"))
        ducc_ham = solver.run(mo_coeff_alpha=C, density_alpha=D)

        # ── Pipeline 2: ExaChem standalone ──
        p2_dir = os.path.join(work_dir, "p2")
        os.makedirs(p2_dir, exist_ok=True)
        input_json = {
            "geometry": {"coordinates": atoms, "units": "angstrom"},
            "basis": {"basisset": basis},
            "common": {"maxiter": 100},
            "SCF": {"charge": charge, "multiplicity": multiplicity, "scf_type": "restricted"},
            "CD": {"diagtol": 1e-5},
            "CC": {
                "threshold": 1e-6,
                "nactive_oa": noa,
                "nactive_ob": noa,
                "nactive_va": nva,
                "nactive_vb": nva,
                "ducc_lvl": ducc_level,
                "writet": False,
            },
            "TASK": {"ducc": [True, "default"]},
        }
        input_path = os.path.join(p2_dir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_json, f)

        result = subprocess.run(
            ["mpirun", "-np", "2", _exachem_path, input_path],
            cwd=p2_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"ExaChem P2 failed: {result.stderr[:500]}"

        # ── CI via qdk-chemistry MACIS on both DUCC Hamiltonians ──
        macis = create("multi_configuration_calculator")

        # Pipeline 1: qdk SCF → ExaChem DUCC (noscf); solver.run already returns
        # a qdk Hamiltonian.
        e1, _ = macis.run(ducc_ham, noa, noa)

        # Pipeline 2: parse ExaChem standalone results → qdk Hamiltonian.
        b2 = f"input.{basis}"
        p2_base = os.path.join(p2_dir, f"{b2}_files", "restricted")
        fcidump2 = parse_ducc_results(
            os.path.join(p2_base, "ducc", f"{b2}.ducc.results.txt"),
            os.path.join(p2_base, "json", f"{b2}.ducc.json"),
        )
        core2 = _core_energy_from_stdout(result.stdout, fcidump2)
        ham2 = fcidump_to_hamiltonian(
            fcidump2, atoms=atoms, basis=basis, units="angstrom", core_energy_override=core2
        )
        e2, _ = macis.run(ham2, noa, noa)

        np.testing.assert_allclose(
            e1, e2, atol=atol, err_msg="DUCC+MACIS ground-state energies differ between pipelines"
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_energy_hierarchy(
    atoms: list[str],
    coords_ang: list[list[float]],
    elements: list[Element],
    basis: str,
    noa: int,
    nva: int,
    num_active_electrons: int,
    num_active_orbitals: int,
    charge: int = 0,
    multiplicity: int = 1,
    ducc_level: int = 2,
) -> None:
    """Verify SCF > CASCI >= DUCC+MACIS >= FCI (within tolerance).

    Runs four calculations:

    1. **SCF** — Hartree-Fock energy.
    2. **CASCI** — exact diagonalization in the active space without DUCC
       corrections (plain frozen-core + active-space integrals).
    3. **DUCC+MACIS** — DUCC downfolded Hamiltonian diagonalized by MACIS.
    4. **FCI** — full configuration interaction over all orbitals.

    Asserts the energy ordering and that DUCC improves over CASCI.
    """
    work_dir = tempfile.mkdtemp(prefix="ducc_hierarchy_")
    try:
        # ── SCF ──
        coords_bohr = np.array(coords_ang) / 0.529177249
        structure = Structure(coords_bohr, elements)
        scf = create("scf_solver")
        e_scf, wfn = scf.run(structure, charge, multiplicity, basis)

        C = np.array(wfn.get_orbitals().get_coefficients()[0])
        nelec_alpha = (sum(e.value for e in elements) - charge + multiplicity - 1) // 2
        D = C[:, :nelec_alpha] @ C[:, :nelec_alpha].T

        # ── CASCI (active space, no DUCC) ──
        selector = create(
            "active_space_selector",
            "qdk_valence",
            num_active_electrons=num_active_electrons,
            num_active_orbitals=num_active_orbitals,
        )
        active_wfn = selector.run(wfn)
        ham_builder = create("hamiltonian_constructor")
        cas_ham = ham_builder.run(active_wfn.get_orbitals())
        macis = create("multi_configuration_calculator")
        n_alpha, n_beta = active_wfn.get_active_num_electrons()
        e_casci, _ = macis.run(cas_ham, n_alpha, n_beta)

        # ── DUCC + MACIS ──
        solver = ExachemDuccSolver()
        s = solver.settings()
        s.set("atoms", atoms)
        s.set("basis", basis)
        s.set("charge", charge)
        s.set("multiplicity", multiplicity)
        s.set("nactive_oa", noa)
        s.set("nactive_ob", noa)
        s.set("nactive_va", nva)
        s.set("nactive_vb", nva)
        s.set("ducc_level", ducc_level)
        s.set("mpi_ranks", 2)
        s.set("exachem_binary", _exachem_path)
        s.set("work_dir", os.path.join(work_dir, "ducc"))
        ducc_ham = solver.run(mo_coeff_alpha=C, density_alpha=D)
        e_ducc, _ = macis.run(ducc_ham, noa, noa)

        # ── Full FCI ──
        full_ham = ham_builder.run(wfn.get_orbitals())
        tot_alpha, tot_beta = wfn.get_total_num_electrons()
        e_fci, _ = macis.run(full_ham, tot_alpha, tot_beta)

        # ── Assertions ──
        assert e_scf > e_casci, f"SCF ({e_scf:.8f}) should be above CASCI ({e_casci:.8f})"
        assert e_casci > e_fci, f"CASCI ({e_casci:.8f}) should be above FCI ({e_fci:.8f})"
        assert e_casci >= e_ducc - 1e-6, (
            f"CASCI ({e_casci:.8f}) should be >= DUCC ({e_ducc:.8f}) (DUCC captures extra correlation)"
        )
        # DUCC may slightly overshoot FCI due to BCH truncation + Libint2
        # version differences, but should be within ~2 mHa.
        assert e_ducc >= e_fci - 2e-3, f"DUCC ({e_ducc:.8f}) should not exceed FCI ({e_fci:.8f}) by more than 2 mHa"

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_H2O_ATOMS = ["O 0 0 0.117790", "H 0 0.756950 -0.471161", "H 0 -0.756950 -0.471161"]
_H2O_COORDS = [[0, 0, 0.117790], [0, 0.756950, -0.471161], [0, -0.756950, -0.471161]]
_H2O_ELEMS = [Element.O, Element.H, Element.H]


class TestDuccIntegration:
    """Compare qdk-chemistry SCF → ExaChem DUCC vs ExaChem end-to-end."""

    def test_h2_sto3g_1o1v(self):
        """H2 / STO-3G, s-only basis, minimal active space."""
        _run_ducc_comparison(
            ["H 0 0 0", "H 0 0 0.74"],
            [[0, 0, 0], [0, 0, 0.74]],
            [Element.H, Element.H],
            "sto-3g",
            noa=1,
            nva=1,
        )

    def test_lih_sto3g_1o2v(self):
        """LiH / STO-3G, s+p basis, asymmetric active space."""
        _run_ducc_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "sto-3g",
            noa=1,
            nva=2,
        )

    def test_h2o_sto3g_2o2v(self):
        """H2O / STO-3G, s+p basis, (2o,2v) active space."""
        _run_ducc_comparison(_H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "sto-3g", noa=2, nva=2)

    def test_h2o_sto3g_1o1v(self):
        """H2O / STO-3G, s+p basis, minimal active space."""
        _run_ducc_comparison(_H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "sto-3g", noa=1, nva=1)

    def test_h2o_ccpvdz_2o2v(self):
        """H2O / cc-pVDZ, s+p+d basis, tests d-shell passthrough."""
        _run_ducc_comparison(_H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "cc-pvdz", noa=2, nva=2)

    def test_h2_ccpvdz_1o1v(self):
        """H2 / cc-pVDZ, p-functions on H atoms."""
        _run_ducc_comparison(
            ["H 0 0 0", "H 0 0 0.74"],
            [[0, 0, 0], [0, 0, 0.74]],
            [Element.H, Element.H],
            "cc-pvdz",
            noa=1,
            nva=1,
        )

    def test_n2_sto3g_2o2v(self):
        """N2 / STO-3G, larger molecule."""
        _run_ducc_comparison(
            ["N 0 0 0", "N 0 0 1.098"],
            [[0, 0, 0], [0, 0, 1.098]],
            [Element.N, Element.N],
            "sto-3g",
            noa=2,
            nva=2,
        )

    def test_lih_ccpvdz_1o2v(self):
        """LiH / cc-pVDZ, s+p+d basis, asymmetric active space."""
        _run_ducc_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "cc-pvdz",
            noa=1,
            nva=2,
        )

    def test_lih_631g_1o1v(self):
        """LiH / 6-31G, Pople SP-shell basis, tests inter-shell AO reordering."""
        _run_ducc_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "6-31g",
            noa=1,
            nva=1,
        )

    def test_h2o_ccpvdz_3o3v(self):
        """H2O / cc-pVDZ, larger active space (924 determinants)."""
        _run_ducc_comparison(_H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "cc-pvdz", noa=3, nva=3)

    def test_ch4_sto3g_3o3v(self):
        """CH4 / STO-3G, tetrahedral symmetry — active space includes all degenerate t2 orbitals."""
        _run_ducc_comparison(
            [
                "C 0 0 0",
                "H 0.629 0.629 0.629",
                "H -0.629 -0.629 0.629",
                "H -0.629 0.629 -0.629",
                "H 0.629 -0.629 -0.629",
            ],
            [
                [0, 0, 0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ],
            [Element.C, Element.H, Element.H, Element.H, Element.H],
            "sto-3g",
            noa=3,
            nva=3,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_ducc_levels(self, ducc_level):
        """LiH / STO-3G across DUCC truncation levels 0, 1, 2 (both pipelines must agree per level)."""
        _run_ducc_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "sto-3g",
            noa=1,
            nva=2,
            ducc_level=ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_ducc_levels(self, ducc_level):
        """H2O / STO-3G (2o,2v) across DUCC truncation levels 0, 1, 2."""
        _run_ducc_comparison(_H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "sto-3g", noa=2, nva=2, ducc_level=ducc_level)


class TestDuccEnergyHierarchy:
    """Verify the DUCC energy hierarchy: SCF > CASCI >= DUCC+MACIS >= FCI.

    For a non-trivial active space (significantly smaller than the full
    orbital space), DUCC downfolding should recover correlation energy from
    outside the active space, giving an energy between plain CASCI and full
    FCI. The DUCC energy may slightly exceed FCI due to the truncated
    Baker-Campbell-Hausdorff expansion, but should be within ~1 mHa.
    """

    def test_lih_ccpvdz_energy_hierarchy(self):
        """LiH / cc-pVDZ, CAS(2,5) active space.

        4 electrons, 19 orbitals total.  Active space = 1 occupied + 4
        virtual = 5 orbitals, 2 electrons.  This leaves 1 frozen core
        orbital and 14 frozen virtual orbitals, so DUCC has substantial
        out-of-active-space correlation to capture.
        """
        _run_energy_hierarchy(
            atoms=["Li 0 0 0", "H 0 0 1.6"],
            coords_ang=[[0, 0, 0], [0, 0, 1.6]],
            elements=[Element.Li, Element.H],
            basis="cc-pvdz",
            noa=1,
            nva=4,
            num_active_electrons=2,
            num_active_orbitals=5,
        )

    def test_h2o_sto3g_energy_hierarchy(self):
        """H2O / STO-3G, CAS(4,4) active space.

        10 electrons, 7 orbitals total.  Active space = 2 occupied + 2
        virtual = 4 orbitals, 4 electrons.  With 3 frozen core orbitals,
        DUCC should improve over plain CASCI.
        """
        _run_energy_hierarchy(
            atoms=_H2O_ATOMS,
            coords_ang=_H2O_COORDS,
            elements=_H2O_ELEMS,
            basis="sto-3g",
            noa=2,
            nva=2,
            num_active_electrons=4,
            num_active_orbitals=4,
        )


def _pyscf_fci_energy(stdout_text: str, results_file: str) -> float:
    """Compute DUCC+PySCF-FCI energy using ExaChem's hamiltonian_extractor.

    Uses ExaChem's own ``grab_data`` module to parse the DUCC integrals
    into spatial orbitals and ExaChem's ``fci_solver`` (PySCF wrapper) to
    diagonalize.  Returns the total energy including the energy shift
    and nuclear repulsion.

    Both *stdout_text* and *results_file* must come from the **same**
    ExaChem run so that metadata (nuc_rep, energy_shift) and integrals
    are consistent.
    """
    import sys

    extractor_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.pardir,
        ".tmp_exachem",
        "exachem",
        "cc",
        "scripts",
        "hamiltonian_extractor",
    )
    if extractor_dir not in sys.path:
        sys.path.insert(0, extractor_dir)
    from fci_solver import solve_fci  # type: ignore[import-untyped]
    from grab_data import extract_hamiltonian_data  # type: ignore[import-untyped]

    stdout_path = results_file + ".stdout"
    with open(stdout_path, "w") as f:
        f.write(stdout_text)

    data = extract_hamiltonian_data(stdout_path, results_file)
    fci_results = solve_fci(data, n_roots=1)
    return fci_results[0]["energy"]


# Tolerance for cross-code end-to-end energy comparison.
# qdk-chemistry uses Libint2 v2.9.0; ExaChem uses Libint2 v2.11.2.
# The different Libint2 versions produce AO integrals that differ at
# ~1e-7 Ha, which propagates through CCSD + DUCC.  Tightening SCF
# convergence thresholds does not help because the floor comes from the
# AO integral library, not from SCF convergence.
# Measured max discrepancy across all test molecules: 9.89e-08 Ha.
_cross_code_energy_tolerance = 1e-7


def _run_ducc_hamiltonian_comparison(
    atoms: list[str],
    coords_ang: list[list[float]],
    elements: list[Element],
    basis: str,
    noa: int,
    nva: int,
    charge: int = 0,
    multiplicity: int = 1,
    ducc_level: int = 2,
    atol: float = _cross_code_energy_tolerance,
) -> None:
    """Assert qdk end-to-end DUCC energy matches ExaChem end-to-end DUCC energy.

    Two fully independent end-to-end pipelines — different SCF solvers
    (qdk Libint2 v2.9.0 vs ExaChem Libint2 v2.11.2), different DUCC runs,
    different spatial-integral converters, and different CI solvers:

    1. **qdk pipeline**: qdk SCF → ExaChem DUCC (noscf) →
       qdk ``spinorb_to_spatial`` + ``fcidump_to_hamiltonian`` → MACIS.
    2. **ExaChem pipeline**: ExaChem SCF → ExaChem DUCC →
       ExaChem ``grab_data`` spatial conversion → PySCF FCI.

    Each pipeline is self-consistent (SCF, DUCC, conversion, and CI all
    come from the same run).  The tolerance is limited by the Libint2
    version difference at ~1e-7 Ha.
    """
    work_dir = tempfile.mkdtemp(prefix="ducc_ham_cmp_")
    try:
        # ── Pipeline 1: qdk SCF → ExaChem DUCC (noscf) → qdk MACIS ──
        coords_bohr = np.array(coords_ang) / 0.529177249
        structure = Structure(coords_bohr, elements)
        scf_solver = create("scf_solver")
        _, wfn = scf_solver.run(structure, charge, multiplicity, basis)
        C = np.array(wfn.get_orbitals().get_coefficients()[0])
        nelec_alpha = (sum(e.value for e in elements) - charge + multiplicity - 1) // 2
        D = C[:, :nelec_alpha] @ C[:, :nelec_alpha].T

        solver = ExachemDuccSolver()
        s = solver.settings()
        s.set("atoms", atoms)
        s.set("basis", basis)
        s.set("charge", charge)
        s.set("multiplicity", multiplicity)
        s.set("nactive_oa", noa)
        s.set("nactive_ob", noa)
        s.set("nactive_va", nva)
        s.set("nactive_vb", nva)
        s.set("ducc_level", ducc_level)
        s.set("mpi_ranks", 2)
        s.set("exachem_binary", _exachem_path)
        s.set("work_dir", os.path.join(work_dir, "p1"))
        ducc_ham = solver.run(mo_coeff_alpha=C, density_alpha=D)

        macis = create("multi_configuration_calculator")
        e_qdk, _ = macis.run(ducc_ham, noa, noa)

        # ── Pipeline 2: ExaChem e2e SCF+DUCC → ExaChem conversion → PySCF FCI ──
        import glob

        p2_dir = os.path.join(work_dir, "p2")
        os.makedirs(p2_dir, exist_ok=True)
        input_json = {
            "geometry": {"coordinates": atoms, "units": "angstrom"},
            "basis": {"basisset": basis},
            "common": {"maxiter": 100},
            "SCF": {"charge": charge, "multiplicity": multiplicity, "scf_type": "restricted"},
            "CD": {"diagtol": 1e-5},
            "CC": {
                "threshold": 1e-6,
                "nactive_oa": noa,
                "nactive_ob": noa,
                "nactive_va": nva,
                "nactive_vb": nva,
                "ducc_lvl": ducc_level,
                "writet": False,
            },
            "TASK": {"ducc": [True, "default"]},
        }
        input_path = os.path.join(p2_dir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_json, f)

        result = subprocess.run(
            ["mpirun", "-np", "2", _exachem_path, input_path],
            cwd=p2_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"ExaChem e2e failed: {result.stderr[:500]}"

        p2_results = glob.glob(os.path.join(p2_dir, f"input.{basis}_files/restricted/ducc/*.ducc.results.txt"))[0]
        e_exachem = _pyscf_fci_energy(result.stdout, p2_results)

        np.testing.assert_allclose(
            e_qdk,
            e_exachem,
            atol=atol,
            err_msg=(f"qdk e2e DUCC+MACIS ({e_qdk:.12f}) differs from ExaChem e2e DUCC+PySCF ({e_exachem:.12f})"),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.skipif(not _has_pyscf, reason="Requires PySCF")
class TestDuccHamiltonianVsExachem:
    """Compare qdk end-to-end DUCC energy against ExaChem end-to-end.

    Each test runs two fully independent pipelines with different SCF
    solvers (qdk Libint2 v2.9.0 vs ExaChem Libint2 v2.11.2), different
    DUCC runs, different spatial-integral converters (qdk
    ``spinorb_to_spatial`` vs ExaChem ``grab_data``), and different CI
    solvers (MACIS vs PySCF FCI).

    Agreement to within 1e-7 Ha confirms the full qdk DUCC integration
    produces the same physics as ExaChem's own pipeline.  The tolerance
    floor is set by the Libint2 version difference in AO integrals.
    """

    def test_h2_sto3g_1o1v(self):
        _run_ducc_hamiltonian_comparison(
            ["H 0 0 0", "H 0 0 0.74"],
            [[0, 0, 0], [0, 0, 0.74]],
            [Element.H, Element.H],
            "sto-3g",
            noa=1,
            nva=1,
        )

    def test_lih_sto3g_1o2v(self):
        _run_ducc_hamiltonian_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "sto-3g",
            noa=1,
            nva=2,
        )

    def test_h2o_sto3g_2o2v(self):
        _run_ducc_hamiltonian_comparison(
            _H2O_ATOMS,
            _H2O_COORDS,
            _H2O_ELEMS,
            "sto-3g",
            noa=2,
            nva=2,
        )

    def test_h2o_sto3g_1o1v(self):
        _run_ducc_hamiltonian_comparison(
            _H2O_ATOMS,
            _H2O_COORDS,
            _H2O_ELEMS,
            "sto-3g",
            noa=1,
            nva=1,
        )

    def test_h2o_ccpvdz_2o2v(self):
        _run_ducc_hamiltonian_comparison(
            _H2O_ATOMS,
            _H2O_COORDS,
            _H2O_ELEMS,
            "cc-pvdz",
            noa=2,
            nva=2,
        )

    def test_h2_ccpvdz_1o1v(self):
        _run_ducc_hamiltonian_comparison(
            ["H 0 0 0", "H 0 0 0.74"],
            [[0, 0, 0], [0, 0, 0.74]],
            [Element.H, Element.H],
            "cc-pvdz",
            noa=1,
            nva=1,
        )

    def test_n2_sto3g_2o2v(self):
        _run_ducc_hamiltonian_comparison(
            ["N 0 0 0", "N 0 0 1.098"],
            [[0, 0, 0], [0, 0, 1.098]],
            [Element.N, Element.N],
            "sto-3g",
            noa=2,
            nva=2,
        )

    def test_lih_ccpvdz_1o2v(self):
        _run_ducc_hamiltonian_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "cc-pvdz",
            noa=1,
            nva=2,
        )

    def test_h2o_ccpvdz_3o3v(self):
        _run_ducc_hamiltonian_comparison(
            _H2O_ATOMS,
            _H2O_COORDS,
            _H2O_ELEMS,
            "cc-pvdz",
            noa=3,
            nva=3,
        )

    def test_ch4_sto3g_3o3v(self):
        _run_ducc_hamiltonian_comparison(
            [
                "C 0 0 0",
                "H 0.629 0.629 0.629",
                "H -0.629 -0.629 0.629",
                "H -0.629 0.629 -0.629",
                "H 0.629 -0.629 -0.629",
            ],
            [
                [0, 0, 0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ],
            [Element.C, Element.H, Element.H, Element.H, Element.H],
            "sto-3g",
            noa=3,
            nva=3,
        )


# ---------------------------------------------------------------------------
# Machine-precision check: ExaChem's OWN orbitals through both pipelines
# ---------------------------------------------------------------------------


def _read_exachem_matrix(path: str) -> np.ndarray:
    """Read a matrix from ExaChem's serial-IO HDF5 format (flat dataset + ``rdims``)."""
    with h5py.File(path, "r") as f:
        rdims = f["rdims"][:]
        name = next(k for k in f.keys() if k != "rdims")
        flat = f[name][:]
    return np.array(flat).reshape(int(rdims[0]), int(rdims[1]))


def _macis_from_ducc_dir(results_dir, basis, prefix, stdout, atoms, noa, macis):
    """Parse a DUCC results dir -> qdk Hamiltonian -> MACIS ground-state energy."""
    base = os.path.join(results_dir, f"{prefix}.{basis}_files", "restricted")
    fcidump = parse_ducc_results(
        os.path.join(base, "ducc", f"{prefix}.{basis}.ducc.results.txt"),
        os.path.join(base, "json", f"{prefix}.{basis}.ducc.json"),
    )
    core = _core_energy_from_stdout(stdout, fcidump)
    ham = fcidump_to_hamiltonian(fcidump, atoms=atoms, basis=basis, units="angstrom", core_energy_override=core)
    energy, _ = macis.run(ham, noa, noa)
    return energy


def _run_exachem_orbitals_comparison(
    atoms: list[str],
    coords_ang: list[list[float]],
    elements: list[Element],
    basis: str,
    noa: int,
    nva: int,
    charge: int = 0,
    multiplicity: int = 1,
    ducc_level: int = 2,
    atol: float = 1e-8,
) -> None:
    """Feed ExaChem's OWN converged orbitals into qdk's noscf pipeline.

    Unlike the cross-Libint tests (which use qdk's SCF orbitals), this feeds
    *ExaChem's* converged orbitals into both pipelines, so the AO-integral
    (Libint2 version) discrepancy is eliminated while DUCC still runs on both:

    * **Reference**: ExaChem standalone SCF -> DUCC -> qdk MACIS.
    * **qdk path**: extract ExaChem's converged movecs, inverse-reorder to qdk
      AO order, feed into ``ExachemDuccSolver`` (noscf) -> DUCC -> qdk MACIS.

    Any residual exercises only the qdk machinery: the AO-reorder round-trip,
    the HDF5 export, the spin-orbital->spatial conversion, the energy-shift core
    handling, and MACIS. Agreement is to ~machine precision (~1e-10).
    """
    work_dir = tempfile.mkdtemp(prefix="ducc_orb_")
    try:
        macis = create("multi_configuration_calculator")

        # ── Reference: ExaChem standalone SCF -> DUCC ──
        std_dir = os.path.join(work_dir, "std")
        os.makedirs(std_dir, exist_ok=True)
        input_json = {
            "geometry": {"coordinates": atoms, "units": "angstrom"},
            "basis": {"basisset": basis},
            "common": {"maxiter": 100},
            "SCF": {"charge": charge, "multiplicity": multiplicity, "scf_type": "restricted"},
            "CD": {"diagtol": 1e-5},
            "CC": {
                "threshold": 1e-6,
                "nactive_oa": noa,
                "nactive_ob": noa,
                "nactive_va": nva,
                "nactive_vb": nva,
                "ducc_lvl": ducc_level,
                "writet": False,
            },
            "TASK": {"ducc": [True, "default"]},
        }
        input_path = os.path.join(std_dir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_json, f)
        result = subprocess.run(
            ["mpirun", "-np", "2", _exachem_path, input_path],
            cwd=std_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"ExaChem standalone failed: {result.stderr[:500]}"
        e_ref = _macis_from_ducc_dir(std_dir, basis, "input", result.stdout, atoms, noa, macis)

        # ── Extract ExaChem's converged orbitals (ExaChem AO order) ──
        movecs = os.path.join(
            std_dir, f"input.{basis}_files", "restricted", "scf", f"input.{basis}.alpha.movecs"
        )
        c_exachem = _read_exachem_matrix(movecs)  # (nbf, nmo), ExaChem AO order

        # Inverse AO reorder ExaChem -> qdk; export_scf_files reapplies qdk -> ExaChem,
        # so ExaChem's DUCC receives exactly its own orbitals back.
        coords_bohr = np.array(coords_ang) / 0.529177249
        basis_set = BasisSet.from_basis_name(basis, Structure(coords_bohr, elements))
        perm = reorder_ao_qdk_to_exachem(basis_set)  # c_exachem = c_qdk[perm]
        inv = np.argsort(perm)
        c_qdk = c_exachem[inv, :]
        nelec_alpha = (sum(e.value for e in elements) - charge + multiplicity - 1) // 2
        d_qdk = c_qdk[:, :nelec_alpha] @ c_qdk[:, :nelec_alpha].T

        # ── qdk noscf integration fed ExaChem's orbitals ──
        solver = ExachemDuccSolver()
        s = solver.settings()
        s.set("atoms", atoms)
        s.set("basis", basis)
        s.set("charge", charge)
        s.set("multiplicity", multiplicity)
        s.set("nactive_oa", noa)
        s.set("nactive_ob", noa)
        s.set("nactive_va", nva)
        s.set("nactive_vb", nva)
        s.set("ducc_level", ducc_level)
        s.set("mpi_ranks", 2)
        s.set("exachem_binary", _exachem_path)
        s.set("work_dir", os.path.join(work_dir, "noscf"))
        ducc_ham = solver.run(mo_coeff_alpha=c_qdk, density_alpha=d_qdk)
        e_qdk, _ = macis.run(ducc_ham, noa, noa)

        np.testing.assert_allclose(
            e_ref,
            e_qdk,
            atol=atol,
            err_msg=(
                f"qdk noscf with ExaChem orbitals ({e_qdk:.12f}) differs from "
                f"ExaChem standalone ({e_ref:.12f}); the integral discrepancy is "
                f"eliminated, so this should match to ~machine precision"
            ),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


class TestDuccExachemOrbitals:
    """Machine-precision: feed ExaChem's OWN orbitals through qdk's noscf pipeline.

    With ExaChem's orbitals on both sides the Libint2 integral discrepancy is
    eliminated, so qdk's noscf DUCC must reproduce ExaChem standalone to
    ~machine precision (measured worst ~1.3e-10). This isolates and validates
    the qdk integration machinery (AO-reorder round-trip, HDF5 export,
    spin-orbital->spatial conversion, energy-shift core handling, MACIS),
    independent of the ~1e-7 cross-Libint floor.
    """

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_levels(self, ducc_level):
        """LiH / STO-3G (1o,2v) — exercises the p-shell AO-reorder round-trip."""
        _run_exachem_orbitals_comparison(
            ["Li 0 0 0", "H 0 0 1.6"],
            [[0, 0, 0], [0, 0, 1.6]],
            [Element.Li, Element.H],
            "sto-3g",
            noa=1,
            nva=2,
            ducc_level=ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_levels(self, ducc_level):
        """H2O / STO-3G (2o,2v) across DUCC levels."""
        _run_exachem_orbitals_comparison(
            _H2O_ATOMS, _H2O_COORDS, _H2O_ELEMS, "sto-3g", noa=2, nva=2, ducc_level=ducc_level
        )

    def test_h2_sto3g_1o1v(self):
        """H2 / STO-3G — s-only (no AO reorder)."""
        _run_exachem_orbitals_comparison(
            ["H 0 0 0", "H 0 0 0.74"],
            [[0, 0, 0], [0, 0, 0.74]],
            [Element.H, Element.H],
            "sto-3g",
            noa=1,
            nva=1,
        )

    def test_n2_sto3g_2o2v(self):
        """N2 / STO-3G — larger p-shell system."""
        _run_exachem_orbitals_comparison(
            ["N 0 0 0", "N 0 0 1.098"],
            [[0, 0, 0], [0, 0, 1.098]],
            [Element.N, Element.N],
            "sto-3g",
            noa=2,
            nva=2,
        )
