"""Native MO-basis SCF, independent reference energies, and registry integration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.data._spin_channels import spin_channel_matrix
from qdk_chemistry.data.symmetry import SymmetryProduct, axes
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian

from .reference_tolerances import scf_energy_tolerance


@pytest.fixture
def integral_hamiltonian():
    """A real three-orbital interacting model, with a deliberately nonstationary reference."""
    one_body = np.array([[-1.3, 0.25, -0.1], [0.25, -0.7, 0.15], [-0.1, 0.15, 0.3]])
    factors = np.array(
        [
            [[0.6, 0.04, -0.03], [0.04, 0.5, 0.02], [-0.03, 0.02, 0.4]],
            [[0.1, -0.05, 0.02], [-0.05, 0.2, 0.03], [0.02, 0.03, 0.15]],
        ]
    )
    two_body = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    orbitals = data.ModelOrbitals(3, SymmetryProduct([axes.spin(1, True)]))
    return data.Hamiltonian(
        data.CanonicalFourCenterHamiltonianContainer(one_body, two_body.ravel(), orbitals, 0.4, one_body)
    )


def test_registry_and_shared_settings(integral_hamiltonian):
    """MO SCF is a native, hashable algorithm with the molecular solver's iteration controls."""
    assert "qdk" in algorithms.available("mo_scf_solver")
    assert algorithms.show_default("mo_scf_solver") == "qdk"
    solver = algorithms.create("mo_scf_solver")
    assert isinstance(solver, algorithms.MoScfSolver)
    assert isinstance(solver, algorithms.QdkMoScfSolver)
    molecular = algorithms.create("scf_solver")
    for setting in (
        "scf_algorithm",
        "enable_gdm",
        "level_shift",
        "max_iterations",
        "convergence_threshold",
        "energy_thresh_diis_switch",
        "gdm_max_diis_iteration",
        "gdm_bfgs_history_size_limit",
        "fock_reset_steps",
    ):
        assert solver.settings().get(setting) == molecular.settings().get(setting)
    initial_hash = solver.hash(integral_hamiltonian, 1, 1)
    solver.settings().set("scf_algorithm", "gdm")
    assert solver.hash(integral_hamiltonian, 1, 1) != initial_hash


@pytest.mark.parametrize("algorithm", ["diis", "gdm", "diis_gdm"])
def test_sparse_hubbard_model(algorithm):
    """A two-site Hubbard model has the analytic RHF energy 2*epsilon-2*t+U/2."""
    hamiltonian = create_hubbard_hamiltonian(data.LatticeGraph.chain(2), epsilon=-0.5, t=1.0, U=0.3)
    solver = algorithms.create("mo_scf_solver", scf_algorithm=algorithm, convergence_threshold=1e-8)
    energy, ansatz = solver.run(hamiltonian, 1, 1)
    assert energy == pytest.approx(-2.85, abs=scf_energy_tolerance)
    assert ansatz.calculate_energy() == pytest.approx(energy, abs=scf_energy_tolerance)
    assert not ansatz.get_orbitals().has_basis_set()


@pytest.mark.parametrize("algorithm", ["diis", "gdm", "diis_gdm"])
@pytest.mark.parametrize(
    ("reference", "nalpha", "nbeta"),
    [
        ("rhf", 1, 1),
        ("rohf", 2, 1),
        ("uhf", 2, 1),
        ("rohf", 1, 0),
        ("rohf", 2, 0),
        ("rohf", 3, 1),
        ("rohf", 3, 2),
    ],
)
def test_against_independent_pyscf_reference(integral_hamiltonian, algorithm, reference, nalpha, nbeta):
    """PySCF is used only as a test oracle; the calculation under test is native QDK."""
    pyscf = pytest.importorskip("pyscf")
    h1, _ = integral_hamiltonian.get_one_body_integrals()
    g_flat, _, _ = integral_hamiltonian.get_two_body_integrals()
    g = g_flat.reshape((3,) * 4)
    mol = pyscf.gto.M(verbose=0)
    mol.nelectron = nalpha + nbeta
    mol.spin = nalpha - nbeta
    # The one-electron shortcut reads the constant from Mole, not SCF.
    mol.energy_nuc = lambda *_: integral_hamiltonian.get_core_energy()
    reference_class = {"rhf": pyscf.scf.RHF, "rohf": pyscf.scf.ROHF, "uhf": pyscf.scf.UHF}[reference]
    mf = reference_class(mol)
    mf.chkfile = None
    mf.get_hcore = lambda *_: h1
    mf.get_ovlp = lambda *_: np.eye(3)
    mf._eri = pyscf.ao2mo.restore(8, g, 3)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    da, db = np.diag(np.arange(3) < nalpha), np.diag(np.arange(3) < nbeta)
    dm0 = da.astype(float) + db if reference == "rhf" else np.array([da, db], dtype=float)
    reference_energy = mf.kernel(dm0=dm0)
    assert mf.converged

    solver = algorithms.create(
        "mo_scf_solver",
        scf_algorithm=algorithm,
        scf_type="unrestricted" if reference == "uhf" else "restricted",
        max_iterations=300,
        convergence_threshold=1e-8,
        gdm_max_diis_iteration=2,
    )
    energy, ansatz = solver.run(integral_hamiltonian, nalpha, nbeta)
    assert energy == pytest.approx(reference_energy, abs=scf_energy_tolerance)
    assert ansatz.calculate_energy() == pytest.approx(energy, abs=scf_energy_tolerance)
    orbitals = ansatz.get_orbitals()
    assert orbitals.is_unrestricted() == (reference == "uhf")
    ca = spin_channel_matrix(orbitals.coefficients(), axes.alpha())
    cb = spin_channel_matrix(orbitals.coefficients(), axes.beta())
    spin = (nalpha - nbeta) / 2
    spin_squared = spin * (spin + 1) + nbeta - np.linalg.norm(ca[:, :nalpha].T @ cb[:, :nbeta]) ** 2
    assert spin_squared == pytest.approx(mf.spin_square()[0], abs=1e-8)
    transformed = ansatz.get_hamiltonian()
    for block, left, right in zip(transformed.get_two_body_integrals(), (ca, ca, cb), (ca, cb, cb), strict=True):
        actual = block.reshape((3,) * 4)
        expected = np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, left, left, right, right, optimize=True)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("scf_type", ["restricted", "unrestricted"])
@pytest.mark.parametrize(("filetype", "suffix"), [("hdf5", "h5"), ("json", "json")])
def test_result_serialization_and_cache(integral_hamiltonian, tmp_path, scf_type, filetype, suffix):
    """Optimized integrals and orbitals survive the standard data and cache boundaries."""
    solver = algorithms.create("mo_scf_solver", scf_type=scf_type, convergence_threshold=1e-8)
    energy, ansatz = solver.run(integral_hamiltonian, 2, 1, cache=tmp_path / "cache")
    filename = tmp_path / f"optimized.ansatz.{suffix}"
    ansatz.to_file(filename, filetype)
    restored = data.Ansatz.from_file(filename, filetype)
    assert restored.calculate_energy() == pytest.approx(energy, abs=scf_energy_tolerance)
    assert restored.get_hamiltonian().content_hash() == ansatz.get_hamiltonian().content_hash()
    energy_cached, cached = solver.run(integral_hamiltonian, 2, 1, cache=tmp_path / "cache")
    assert energy_cached == energy
    assert cached.content_hash() == ansatz.content_hash()


def test_failures_are_explicit(integral_hamiltonian):
    """Bad counts and nonconvergence never return a success-shaped result."""
    with pytest.raises(ValueError, match="electron counts"):
        algorithms.create("mo_scf_solver").run(integral_hamiltonian, 4, 1)
    with pytest.raises((TypeError, ValueError)):
        algorithms.create("mo_scf_solver").run(integral_hamiltonian, -1, 1)
    with pytest.raises(RuntimeError, match="failed to converge"):
        algorithms.create("mo_scf_solver", max_iterations=1).run(integral_hamiltonian, 1, 1)
    with pytest.raises(ValueError, match="Hartree-Fock"):
        algorithms.create("mo_scf_solver", method="pbe").run(integral_hamiltonian, 1, 1)
