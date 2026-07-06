# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Integration tests for the ExaChem CCSD(T) calculator.

For each molecule/basis/reference combination, two fully independent pipelines
are compared:

1. **qdk-chemistry SCF -> ExaChem CCSD(T) (noscf)**: qdk-chemistry's SCF orbitals
   are exported in ExaChem's serial-IO HDF5 format and ExaChem runs CCSD(T) with
   ``noscf=true``. Energy parsed from ExaChem stdout.
2. **PySCF end-to-end CCSD(T)**: PySCF runs its own SCF + CCSD + (T) from scratch.

Both pipelines' CCSD(T) total energies should match to within ~1e-6 Hartree
(limited by AO-integral library differences: qdk/ExaChem use Libint2, PySCF uses
Libcint). Both restricted (RHF) and unrestricted (UHF) references are exercised.

Prerequisites:
    - ExaChem binary built with ``USE_SERIAL_IO`` at ``EXACHEM_PATH``
    - MPI runtime (``mpirun``)
    - ``h5py`` and ``pyscf`` Python packages

These tests are slow (each runs ExaChem via MPI) and are skipped if
``EXACHEM_PATH`` is not set or the binary is not found.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import qdk_chemistry.plugins.exachem as exachem_plugin
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Ansatz, Element, Structure
from qdk_chemistry.plugins.exachem.cli import CcsdtInputConfig
from qdk_chemistry.plugins.exachem.conversion import parse_ccsdt_energy

exachem_plugin.load()

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
    from pyscf import cc as _pyscf_cc
    from pyscf import gto as _pyscf_gto
    from pyscf import scf as _pyscf_scf

    _has_pyscf = True
except ImportError:
    _has_pyscf = False

pytestmark = pytest.mark.skipif(
    not (_has_exachem and _has_mpi and _has_h5py and _has_pyscf),
    reason="Requires EXACHEM_PATH, mpirun, h5py, and pyscf",
)

_ANGSTROM_TO_BOHR = 1.0 / 0.529177249

# Cross-code tolerance. qdk/ExaChem use Libint2 AO integrals; PySCF uses Libcint.
# Measured worst-case discrepancy across test molecules: ~6e-8 Ha.
_cross_code_tol = 1e-6


# ---------------------------------------------------------------------------
# Reference: PySCF end-to-end CCSD(T)
# ---------------------------------------------------------------------------


def _pyscf_ccsdt_total(atoms_ang, symbols, basis, charge, multiplicity):
    """Run PySCF SCF + CCSD(T) and return the total energy."""
    atom = [[sym, tuple(xyz)] for sym, xyz in zip(symbols, atoms_ang, strict=False)]
    mol = _pyscf_gto.M(atom=atom, basis=basis, charge=charge, spin=multiplicity - 1, unit="Angstrom")
    mf = (_pyscf_scf.UHF(mol) if multiplicity > 1 else _pyscf_scf.RHF(mol)).run()
    mycc = _pyscf_cc.CCSD(mf).run()
    et = mycc.ccsd_t()
    return mycc.e_tot + et


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _run_ccsdt_comparison(
    atoms_ang,
    symbols,
    elements,
    basis,
    charge=0,
    multiplicity=1,
    scf_type="restricted",
    atol=_cross_code_tol,
):
    """Run qdk SCF -> ExaChem CCSD(T) and assert it matches PySCF CCSD(T)."""
    coords_bohr = np.array(atoms_ang) * _ANGSTROM_TO_BOHR
    structure = Structure(coords_bohr, elements)

    scf_solver = create("scf_solver")
    scf_solver.settings().set("scf_type", scf_type)
    _, wavefunction = scf_solver.run(structure, charge, multiplicity, basis)

    if scf_type == "unrestricted":
        assert wavefunction.get_orbitals().is_unrestricted()

    ham = create("hamiltonian_constructor", "qdk").run(wavefunction.get_orbitals())
    ansatz = Ansatz(ham, wavefunction)

    calc = create("dynamical_correlation_calculator", "exachem_ccsd_t")
    calc.settings().set("mpi_ranks", 2)
    calc.settings().set("exachem_binary", _exachem_path)
    e_qdk, _, _ = calc.run(ansatz)

    e_pyscf = _pyscf_ccsdt_total(atoms_ang, symbols, basis, charge, multiplicity)

    np.testing.assert_allclose(
        e_qdk,
        e_pyscf,
        atol=atol,
        err_msg=f"qdk->ExaChem CCSD(T) ({e_qdk:.10f}) differs from PySCF CCSD(T) ({e_pyscf:.10f})",
    )


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

_H2O_COORDS = [[0, 0, 0.117790], [0, 0.756950, -0.471161], [0, -0.756950, -0.471161]]
_H2O_SYMBOLS = ["O", "H", "H"]
_H2O_ELEMS = [Element.O, Element.H, Element.H]


# ---------------------------------------------------------------------------
# Restricted (RHF) reference
# ---------------------------------------------------------------------------


class TestCcsdtRestricted:
    """Compare qdk-chemistry -> ExaChem CCSD(T) against PySCF for RHF references."""

    def test_h2_sto3g(self):
        """H2 / STO-3G, minimal closed shell."""
        _run_ccsdt_comparison(
            [[0, 0, 0], [0, 0, 0.74]],
            ["H", "H"],
            [Element.H, Element.H],
            "sto-3g",
        )

    def test_h2o_sto3g(self):
        """H2O / STO-3G, exercises p-shell AO reorder."""
        _run_ccsdt_comparison(_H2O_COORDS, _H2O_SYMBOLS, _H2O_ELEMS, "sto-3g")

    def test_h2o_ccpvdz(self):
        """H2O / cc-pVDZ, exercises d-shell passthrough."""
        _run_ccsdt_comparison(_H2O_COORDS, _H2O_SYMBOLS, _H2O_ELEMS, "cc-pvdz")

    def test_n2_sto3g(self):
        """N2 / STO-3G, larger p-shell system."""
        _run_ccsdt_comparison(
            [[0, 0, 0], [0, 0, 1.098]],
            ["N", "N"],
            [Element.N, Element.N],
            "sto-3g",
        )


# ---------------------------------------------------------------------------
# Unrestricted (UHF) reference
# ---------------------------------------------------------------------------


class TestCcsdtUnrestricted:
    """Compare qdk-chemistry -> ExaChem CCSD(T) against PySCF for UHF references.

    These exercise the per-spin (alpha/beta) MO-coefficient and density export
    that distinguishes the CCSD(T) integration from the closed-shell-only DUCC
    integration.
    """

    def test_oh_doublet_sto3g(self):
        """OH radical / STO-3G doublet."""
        _run_ccsdt_comparison(
            [[0, 0, 0], [0, 0, 0.97]],
            ["O", "H"],
            [Element.O, Element.H],
            "sto-3g",
            charge=0,
            multiplicity=2,
            scf_type="unrestricted",
        )

    def test_o2_triplet_sto3g(self):
        """O2 / STO-3G triplet (ground state)."""
        _run_ccsdt_comparison(
            [[0, 0, 0], [0, 0, 1.208]],
            ["O", "O"],
            [Element.O, Element.O],
            "sto-3g",
            charge=0,
            multiplicity=3,
            scf_type="unrestricted",
        )

    def test_oh_doublet_ccpvdz(self):
        """OH radical / cc-pVDZ doublet, exercises d-shell with UHF."""
        _run_ccsdt_comparison(
            [[0, 0, 0], [0, 0, 0.97]],
            ["O", "H"],
            [Element.O, Element.H],
            "cc-pvdz",
            charge=0,
            multiplicity=2,
            scf_type="unrestricted",
        )


# ---------------------------------------------------------------------------
# Lightweight tests that do not require ExaChem
# ---------------------------------------------------------------------------


class TestCcsdtParsing:
    """Unit tests for CCSD(T) config generation and stdout parsing (no ExaChem)."""

    pytestmark = pytest.mark.skipif(not _has_h5py, reason="Requires h5py")

    def test_config_task_is_ccsd_t(self):
        """CCSD(T) config selects the ccsd_t task and omits DUCC/active-space keys."""
        cfg = CcsdtInputConfig(atoms=["H 0 0 0", "H 0 0 1.4"], basis="sto-3g", noscf=True)
        j = cfg.to_json()
        assert j["TASK"]["ccsd_t"] is True
        assert "ducc" not in j["TASK"]
        assert "nactive_oa" not in j["CC"]
        assert "ducc_lvl" not in j["CC"]
        assert j["SCF"]["noscf"] is True

    def test_config_freeze_block(self):
        """Frozen core/virtual options are emitted as a CC freeze block."""
        cfg = CcsdtInputConfig(basis="cc-pvdz", freeze_core=1, freeze_virtual=2)
        j = cfg.to_json()
        assert j["CC"]["freeze"] == {"core": 1, "virtual": 2}

    def test_config_no_freeze_block_by_default(self):
        """No freeze block is emitted when no orbitals are frozen."""
        cfg = CcsdtInputConfig(basis="cc-pvdz")
        assert "freeze" not in cfg.to_json()["CC"]

    def test_parse_ccsdt_energy(self):
        """The stdout parser extracts CCSD and CCSD(T)/CCSD[T] energies."""
        stdout = "\n".join(
            [
                " CCSD correlation energy / hartree =                   -0.050000000000000",
                " CCSD total energy / hartree       =                  -75.000000000000000",
                "CCSD[T] correction energy / hartree  = -0.001000000000000",
                "CCSD[T] total energy / hartree       = -75.001000000000000",
                "CCSD(T) correction energy / hartree  = -0.002000000000000",
                "CCSD(T) correlation energy / hartree = -0.052000000000000",
                "CCSD(T) total energy / hartree       = -75.002000000000000",
            ]
        )
        e = parse_ccsdt_energy(stdout)
        assert e.ccsd_correlation == pytest.approx(-0.05)
        assert e.ccsd_total == pytest.approx(-75.0)
        assert e.ccsd_bracket_t_correction == pytest.approx(-0.001)
        assert e.ccsd_bracket_t_total == pytest.approx(-75.001)
        assert e.ccsd_pt_correction == pytest.approx(-0.002)
        assert e.ccsd_pt_total == pytest.approx(-75.002)

    def test_parse_ccsdt_energy_missing(self):
        """Missing energies return None instead of raising."""
        e = parse_ccsdt_energy("no energies here")
        assert e.ccsd_pt_total is None
