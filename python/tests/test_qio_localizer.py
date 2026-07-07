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


def _single_orbital_entropy_sum(ga, gb, g2):
    """Boguslawski & Tecmer (2015) single-orbital entropy sum from RDMs.

    ga, gb : (n, n) alpha/beta 1-RDMs.
    g2     : (n, n, n, n) alpha-beta (aabb) 2-RDM.
    """
    n = ga.shape[0]
    total = 0.0
    for i in range(n):
        d = g2[i, i, i, i]
        omega = np.array([1.0 - ga[i, i] - gb[i, i] + d, ga[i, i] - d, gb[i, i] - d, d])
        omega = omega[omega > 1e-14]
        total -= float(np.sum(omega * np.log(omega)))
    return total


def _correlated_water_wavefunction():
    """Restricted water CAS(6e, 6o) wavefunction carrying spin-dependent RDMs."""
    coords = np.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.467],
            [0.0, -0.757, -0.467],
        ]
    )
    mol = data.Structure(coords, [8, 1, 1])

    scf = algorithms.create("scf_solver")
    scf.settings().set("method", "hf")
    _, hf_wfn = scf.run(mol, 0, 1, "cc-pvdz")

    selector = algorithms.create("active_space_selector", "qdk_valence")
    selector.settings().set("num_active_electrons", 6)
    selector.settings().set("num_active_orbitals", 6)
    active_wfn = selector.run(hf_wfn)

    hamil_constructor = algorithms.create("hamiltonian_constructor")
    hamil = hamil_constructor.run(active_wfn.get_orbitals())
    mc = algorithms.create("multi_configuration_calculator", "macis_cas")
    mc.settings().set("calculate_one_rdm", True)
    mc.settings().set("calculate_two_rdm", True)
    n_a, n_b = active_wfn.get_active_num_electrons()
    _, cas_wfn = mc.run(hamil, n_a, n_b)
    return active_wfn.get_orbitals(), cas_wfn


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

    def test_reduces_single_orbital_entropy(self):
        """QIO rotation does not increase the total single-orbital entropy."""
        active_orbitals, cas_wfn = _correlated_water_wavefunction()

        alpha_indices, beta_indices = active_orbitals.get_active_space_indices()
        assert alpha_indices == beta_indices
        active_indices = list(alpha_indices)
        n = len(active_indices)

        # Input (canonical) single-orbital entropy sum.
        entropy_before = float(np.sum(cas_wfn.get_single_orbital_entropies()))

        # Cross-check the numpy entropy reproduces the library value at U = I.
        ga, gb = (np.asarray(m) for m in cas_wfn.get_active_one_rdm_spin_dependent())
        _, aabb, _ = cas_wfn.get_active_two_rdm_spin_dependent()
        g2 = np.asarray(aabb).reshape(n, n, n, n)
        np.testing.assert_allclose(
            _single_orbital_entropy_sum(ga, gb, g2), entropy_before, atol=1e-8
        )

        # Run the QIO localizer (single rotation).
        localizer = create("orbital_localizer", "qdk_qio")
        qio_wfn = localizer.run(cas_wfn, active_indices, active_indices)
        assert qio_wfn is not None

        # Recover the active-space rotation U = Ca_can^T S Ca_qio.
        s = np.asarray(active_orbitals.get_overlap_matrix())
        ca_can = np.asarray(active_orbitals.get_coefficients()[0])[:, active_indices]
        ca_qio = np.asarray(qio_wfn.get_orbitals().get_coefficients()[0])[:, active_indices]
        u = ca_can.T @ s @ ca_qio

        # U is unitary and the QIO orbitals are orthonormal.
        np.testing.assert_allclose(u @ u.T, np.eye(n), atol=1e-8)
        np.testing.assert_allclose(ca_qio.T @ s @ ca_qio, np.eye(n), atol=1e-8)

        # Transform the input RDMs into the QIO basis and recompute the entropy.
        ga_rot = u.T @ ga @ u
        gb_rot = u.T @ gb @ u
        g2_rot = np.einsum("pqrl,pi,qj,rk,lm->ijkm", g2, u, u, u, u, optimize=True)
        entropy_after = _single_orbital_entropy_sum(ga_rot, gb_rot, g2_rot)

        # The QIO objective must not increase under the optimized rotation.
        assert entropy_after <= entropy_before + 1e-9

    def test_open_shell_triplet_energy_invariant(self):
        """ROHF triplet (open-shell) is accepted; the CASCI energy is invariant.

        "Restricted" means a single spatial orbital set (RHF/ROHF), not
        closed-shell: an open-shell reference with na != nb is supported.
        """
        # Triplet O2 (ground state) via ROHF -> restricted open-shell orbitals.
        coords = np.array([[0.0, 0.0, 1.14], [0.0, 0.0, -1.14]])  # Bohr
        mol = data.Structure(coords, [8, 8])
        scf = algorithms.create("scf_solver")
        scf.settings().set("method", "hf")
        scf.settings().set("scf_type", "restricted")
        _, hf_wfn = scf.run(mol, 0, 3, "cc-pvdz")  # charge 0, multiplicity 3
        assert hf_wfn.get_orbitals().is_restricted()

        selector = algorithms.create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 8)
        selector.settings().set("num_active_orbitals", 6)
        active_wfn = selector.run(hf_wfn)
        active_orbs = active_wfn.get_orbitals()
        n_a, n_b = active_wfn.get_active_num_electrons()
        assert n_a != n_b  # genuinely open-shell

        idx = list(active_orbs.get_active_space_indices()[0])

        def cas(orbs):
            ham = algorithms.create("hamiltonian_constructor").run(orbs)
            mc = algorithms.create("multi_configuration_calculator", "macis_cas")
            mc.settings().set("calculate_one_rdm", True)
            mc.settings().set("calculate_two_rdm", True)
            return mc.run(ham, n_a, n_b)

        e_before, cas_wfn = cas(active_orbs)
        qio_wfn = create("orbital_localizer", "qdk_qio").run(cas_wfn, idx, idx)
        e_after, _ = cas(qio_wfn.get_orbitals())

        # A unitary rotation of the active orbitals leaves the CASCI energy
        # invariant, even for an open-shell (na != nb) reference.
        assert abs(e_before - e_after) < 1e-8

    def test_settings_defaults_and_override(self):
        """The Jacobi-sweep controls are exposed with defaults and settable."""
        localizer = QdkQIOLocalizer()
        settings = localizer.settings()
        assert settings.get("max_cycles") == 200
        settings.set("max_cycles", 50)
        assert settings.get("max_cycles") == 50
