"""Integration tests for the QdkQIOLocalizer Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms, data
from qdk_chemistry.algorithms import (
    OrbitalLocalizer,
    QdkQIOLocalizer,
    create,
)

from .reference_tolerances import (
    ci_energy_tolerance,
    orthonormality_error_tolerance,
    rdm_tolerance,
)


def _run_cas(orbitals, n_alpha, n_beta):
    """Run macis_cas (with spin-dependent RDMs); return (energy, wavefunction)."""
    hamil = algorithms.create("hamiltonian_constructor").run(orbitals)
    mc = algorithms.create("multi_configuration_calculator", "macis_cas")
    mc.settings().set("calculate_one_rdm", True)
    mc.settings().set("calculate_two_rdm", True)
    return mc.run(hamil, n_alpha, n_beta)


def _correlated_cas_wavefunction(structure_path, multiplicity, n_active_e, n_active_o, basis):
    """Restricted CAS(n_active_e, n_active_o) wavefunction with spin-dependent RDMs.

    Loads the geometry from a shared ``test_data`` .xyz file (charge 0).
    ``multiplicity`` 1 gives a closed-shell RHF reference; > 1 an open-shell
    ROHF reference (a single spatial orbital set with na != nb).
    Returns ``(active_wavefunction, cas_energy, cas_wavefunction)``.
    """
    mol = data.Structure.from_xyz_file(structure_path)

    scf = algorithms.create("scf_solver")
    scf.settings().set("method", "hf")
    scf.settings().set("scf_type", "restricted")
    _, hf_wfn = scf.run(mol, 0, multiplicity, basis)

    selector = algorithms.create("active_space_selector", "qdk_valence")
    selector.settings().set("num_active_electrons", n_active_e)
    selector.settings().set("num_active_orbitals", n_active_o)
    active_wfn = selector.run(hf_wfn)

    n_a, n_b = active_wfn.get_active_num_electrons()
    cas_energy, cas_wfn = _run_cas(active_wfn.get_orbitals(), n_a, n_b)
    return active_wfn, cas_energy, cas_wfn


class TestQIOLocalizerBindings:
    """Test that the QdkQIOLocalizer Python bindings work correctly."""

    def test_factory_registration(self):
        """The QIO localizer is registered in the factory."""
        available = algorithms.available("orbital_localizer")
        assert "qdk_qio" in available

    def test_factory_creation(self):
        """Creating the localizer via the factory."""
        localizer = algorithms.create("orbital_localizer", "qdk_qio")
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)

    def test_direct_construction(self):
        """Direct construction of QdkQIOLocalizer."""
        localizer = QdkQIOLocalizer()
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)
        assert localizer.name() == "qdk_qio"
        assert localizer.type_name() == "orbital_localizer"

    def test_settings(self):
        """The localizer provides a settings interface."""
        localizer = QdkQIOLocalizer()
        assert localizer.settings() is not None

    def test_reduces_single_orbital_entropy(self, test_data_files_path):
        """QIO rotation does not increase the total single-orbital entropy."""
        active_wfn, _, cas_wfn = _correlated_cas_wavefunction(
            test_data_files_path / "ethylene.structure.xyz", 1, 4, 4, "def2-svp"
        )
        active_orbitals = active_wfn.get_orbitals()

        alpha_indices, beta_indices = active_orbitals.get_active_space_indices()
        assert alpha_indices == beta_indices
        active_indices = list(alpha_indices)
        n = len(active_indices)
        n_a, n_b = active_wfn.get_active_num_electrons()

        # Input (canonical) single-orbital entropy sum, from the library method.
        entropy_before = float(np.sum(cas_wfn.get_single_orbital_entropies()))

        # Run the QIO localizer (single rotation).
        localizer = create("orbital_localizer", "qdk_qio")
        qio_wfn = localizer.run(cas_wfn, active_indices, active_indices)
        assert qio_wfn is not None

        # The active-space rotation U = Ca_can^T S Ca_qio is unitary and the QIO
        # orbitals are orthonormal.
        s = np.asarray(active_orbitals.get_overlap_matrix())
        ca_can = np.asarray(active_orbitals.get_coefficients()[0])[:, active_indices]
        ca_qio = np.asarray(qio_wfn.get_orbitals().get_coefficients()[0])[:, active_indices]
        u = ca_can.T @ s @ ca_qio
        np.testing.assert_allclose(u @ u.T, np.eye(n), atol=orthonormality_error_tolerance)
        np.testing.assert_allclose(ca_qio.T @ s @ ca_qio, np.eye(n), atol=orthonormality_error_tolerance)

        # Re-solve the CAS in the QIO-rotated basis and take the entropy from the
        # library method (get_single_orbital_entropies) rather than recomputing it.
        _, rotated_cas_wfn = _run_cas(qio_wfn.get_orbitals(), n_a, n_b)
        entropy_after = float(np.sum(rotated_cas_wfn.get_single_orbital_entropies()))

        # The QIO objective must not increase under the optimized rotation.
        assert entropy_after <= entropy_before + rdm_tolerance

    def test_open_shell_triplet_energy_invariant(self, test_data_files_path):
        """ROHF triplet (open-shell) is accepted; the CASCI energy is invariant.

        "Restricted" means a single spatial orbital set (RHF/ROHF), not
        closed-shell: an open-shell reference with na != nb is supported.
        """
        # Triplet O2 (ground state) via ROHF -> restricted open-shell orbitals.
        active_wfn, e_before, cas_wfn = _correlated_cas_wavefunction(
            test_data_files_path / "o2.structure.xyz", 3, 8, 6, "cc-pvdz"
        )
        active_orbs = active_wfn.get_orbitals()
        assert active_orbs.is_restricted()
        n_a, n_b = active_wfn.get_active_num_electrons()
        assert n_a != n_b  # genuinely open-shell

        idx = list(active_orbs.get_active_space_indices()[0])
        qio_wfn = create("orbital_localizer", "qdk_qio").run(cas_wfn, idx, idx)
        e_after, _ = _run_cas(qio_wfn.get_orbitals(), n_a, n_b)

        # A unitary rotation of the active orbitals leaves the CASCI energy
        # invariant, even for an open-shell (na != nb) reference.
        assert abs(e_before - e_after) < ci_energy_tolerance

    def test_settings_defaults_and_override(self):
        """The Jacobi-sweep controls are exposed with defaults and settable."""
        localizer = QdkQIOLocalizer()
        settings = localizer.settings()
        assert settings.get("max_cycles") == 200
        settings.set("max_cycles", 50)
        assert settings.get("max_cycles") == 50
