"""Integration tests for the ExaChem CCSD calculator with T-amplitude read-in.

Both the ExaChem and PySCF coupled-cluster plugins implement the same
:class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator` interface, so each
test runs them on an identical :class:`~qdk_chemistry.data.Ansatz` built from
qdk-chemistry's own SCF orbitals. The only difference between the two runs is the
coupled-cluster solver itself: ExaChem's Cholesky-decomposed TAMM implementation
versus PySCF's.

Because both consume the same MO coefficients there is no per-orbital phase
ambiguity, so the converged T1/T2 amplitudes are compared elementwise rather than
by magnitude. The two runs are not otherwise bit-identical: ExaChem is handed the
orbitals through its serial-IO restart files and rebuilds the integrals with its
own Libint2, so agreement is limited by that integral re-computation, by ExaChem's
Cholesky decomposition tolerance, and by the two codes' coupled-cluster
convergence thresholds. Both restricted (RHF) and unrestricted (UHF) references
are exercised.

Each test runs ExaChem via MPI, so they are marked ``slow``. The binary is passed to
the calculator through its ``exachem_binary`` setting; the tests locate one by looking
for ``ExaChem`` on ``PATH`` and are skipped when it or an MPI runtime is missing.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import shutil

import numpy as np
import pytest

import qdk_chemistry.plugins.exachem as exachem_plugin
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Ansatz, Structure

from .reference_tolerances import mp2_energy_tolerance, rdm_tolerance

exachem_plugin.load()

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

# The ExaChem binary is configuration, not environment: it is supplied through the
# calculator's ``exachem_binary`` setting. These tests discover one on PATH purely to
# decide whether to run, then pass the resolved path through that setting.
_EXACHEM_BINARY = shutil.which("ExaChem") or ""

try:
    import pyscf  # noqa: F401

    import qdk_chemistry.plugins.pyscf as pyscf_plugin

    pyscf_plugin.load()

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_EXACHEM_BINARY and exachem_plugin.QDK_CHEMISTRY_HAS_MPI and PYSCF_AVAILABLE),
        reason="Requires ExaChem on PATH, an MPI runtime, and PySCF",
    ),
]

# The two solvers are not converged to a common threshold: ExaChem's ccsd_threshold
# defaults to 1e-6 and its Cholesky cd_diagtol to 1e-5, while pyscf's conv_tol is
# 1e-7. These tolerances are therefore set by the looser code plus the integral
# re-computation noted in the module docstring, not by either solver alone.
# NOTE: they are estimates -- they have not been calibrated against a real ExaChem
# build, so revisit them the first time this suite runs in CI.
_energy_tolerance = 100 * mp2_energy_tolerance
# Amplitudes are wavefunction parameters and converge less tightly than the energy.
_amplitude_tolerance = 10 * rdm_tolerance

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"
LIH = "2\nLiH\nLi 0 0 0\nH 0 0 1.6\n"
H2O = "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n"
OH = "2\nOH\nO 0 0 0\nH 0 0 0.97\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_ansatz(xyz: str, basis: str, charge: int, multiplicity: int, scf_type: str) -> Ansatz:
    """Run qdk-chemistry SCF and wrap the result in an Ansatz.

    Args:
        xyz: Molecular geometry as an XYZ-format string.
        basis: Gaussian basis set name.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        scf_type: SCF type, ``"restricted"`` or ``"unrestricted"``.

    Returns:
        An Ansatz pairing the qdk-chemistry Hamiltonian with its SCF wavefunction.

    """
    scf_solver = create("scf_solver")
    scf_solver.settings().set("scf_type", scf_type)
    _, wavefunction = scf_solver.run(Structure.from_xyz(xyz), charge, multiplicity, basis)

    hamiltonian = create("hamiltonian_constructor").run(wavefunction.get_orbitals())
    return Ansatz(hamiltonian, wavefunction)


def run_coupled_cluster(variant: str, ansatz: Ansatz):
    """Run a coupled-cluster calculator variant with amplitude storage enabled.

    Args:
        variant: The registered calculator name, e.g. ``"exachem_ccsd"``.
        ansatz: The reference Ansatz to correlate.

    Returns:
        The tuple ``(total_energy, wavefunction, bra_wavefunction)`` from the calculator.

    """
    calculator = create("dynamical_correlation_calculator", variant)
    calculator.settings().set("store_amplitudes", True)
    if variant == "exachem_ccsd":
        calculator.settings().set("mpi_ranks", 2)
        calculator.settings().set("exachem_binary", _EXACHEM_BINARY)
    return calculator.run(ansatz)


def assert_amplitudes_close(container, reference_container) -> None:
    """Assert two amplitude containers hold the same T1/T2 blocks.

    Args:
        container: The amplitude container under test.
        reference_container: The amplitude container to compare against.

    """
    assert container.has_t1_amplitudes()
    assert container.has_t2_amplitudes()

    for block, reference_block in zip(
        container.get_t1_amplitudes(), reference_container.get_t1_amplitudes(), strict=True
    ):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference_block), atol=_amplitude_tolerance)

    for block, reference_block in zip(
        container.get_t2_amplitudes(), reference_container.get_t2_amplitudes(), strict=True
    ):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference_block), atol=_amplitude_tolerance)


def assert_exachem_matches_pyscf(
    xyz: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    scf_type: str = "restricted",
) -> None:
    """Assert the ExaChem and PySCF CCSD plugins agree on the same Ansatz.

    Args:
        xyz: Molecular geometry as an XYZ-format string.
        basis: Gaussian basis set name.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        scf_type: SCF type, ``"restricted"`` or ``"unrestricted"``.

    """
    ansatz = build_ansatz(xyz, basis, charge, multiplicity, scf_type)

    energy, wavefunction, bra_wavefunction = run_coupled_cluster("exachem_ccsd", ansatz)
    reference_energy, reference_wavefunction, _ = run_coupled_cluster("pyscf_coupled_cluster", ansatz)

    np.testing.assert_allclose(
        energy,
        reference_energy,
        atol=_energy_tolerance,
        err_msg=f"ExaChem CCSD energy ({energy:.10f}) differs from PySCF ({reference_energy:.10f})",
    )

    # The returned wavefunction carries the amplitudes; CCSD has no distinct bra.
    assert bra_wavefunction is None
    assert wavefunction.get_container_type() == "amplitude"
    assert_amplitudes_close(wavefunction.get_container(), reference_wavefunction.get_container())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCcsdRestricted:
    """Compare the ExaChem and PySCF CCSD plugins for RHF references."""

    @pytest.mark.parametrize(
        ("xyz", "basis"),
        [
            pytest.param(H2, "sto-3g", id="h2_sto3g"),
            pytest.param(LIH, "sto-3g", id="lih_sto3g"),
            pytest.param(H2O, "sto-3g", id="h2o_sto3g"),
            pytest.param(H2O, "cc-pvdz", id="h2o_ccpvdz"),
        ],
    )
    def test_energy_and_amplitudes_match_pyscf(self, xyz: str, basis: str):
        assert_exachem_matches_pyscf(xyz, basis)


class TestCcsdUnrestricted:
    """Compare the ExaChem and PySCF CCSD plugins for UHF references.

    These exercise the MSO-tensor spin-block decoding of the T amplitudes.
    """

    @pytest.mark.parametrize(
        ("xyz", "basis"),
        [
            pytest.param(OH, "sto-3g", id="oh_doublet_sto3g"),
            pytest.param(OH, "cc-pvdz", id="oh_doublet_ccpvdz"),
        ],
    )
    def test_energy_and_amplitudes_match_pyscf(self, xyz: str, basis: str):
        assert_exachem_matches_pyscf(xyz, basis, multiplicity=2, scf_type="unrestricted")
