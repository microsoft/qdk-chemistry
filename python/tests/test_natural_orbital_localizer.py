"""Integration tests for the QdkNaturalOrbitalLocalizer Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import (
    OrbitalLocalizer,
    QdkNaturalOrbitalLocalizer,
    create,
)
from qdk_chemistry.data import (
    Orbitals,
    Structure,
)


class TestNaturalOrbitalLocalizerBindings:
    """Test that the QdkNaturalOrbitalLocalizer Python bindings work correctly."""

    def test_factory_registration(self):
        """Test that the natural orbital localizer is registered in the factory."""
        available = algorithms.available("orbital_localizer")
        assert "qdk_natural_orbitals" in available

    def test_factory_creation(self):
        """Test creating the localizer via the factory."""
        localizer = algorithms.create("orbital_localizer", "qdk_natural_orbitals")
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)

    def test_direct_construction(self):
        """Test direct construction of QdkNaturalOrbitalLocalizer."""
        localizer = QdkNaturalOrbitalLocalizer()
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)
        assert localizer.name() == "qdk_natural_orbitals"
        assert localizer.type_name() == "orbital_localizer"

    def test_settings(self):
        """Test that the localizer provides a settings interface."""
        localizer = QdkNaturalOrbitalLocalizer()
        settings = localizer.settings()
        assert settings is not None

    def test_stretched_n2_has_fractional_occupations(self):
        """Stretched N2 with broken-symmetry UKS should yield fractional NOONs."""
        structure = Structure.from_xyz("2\nN2\nN 0 0 0\nN 0 0 2.0\n")

        # Restricted KS to get starting MOs, then break alpha/beta symmetry
        rks = create("scf_solver")
        rks.settings().set("method", "pbe")
        rks.settings().set("scf_type", "restricted")
        _, rks_wfn = rks.run(structure, 0, 1, "cc-pvdz")

        orbs = rks_wfn.get_orbitals()
        c_a, c_b = orbs.get_coefficients()
        c_a, c_b = c_a.copy(), c_b.copy()
        n_occ = rks_wfn.get_total_num_electrons()[0]
        ho, lu = n_occ - 1, n_occ  # HOMO, LUMO
        s2 = np.sqrt(0.5)  # 45-degree rotation
        c_a[:, ho], c_a[:, lu] = s2 * (c_a[:, ho] + c_a[:, lu]), s2 * (c_a[:, lu] - c_a[:, ho])
        c_b[:, ho], c_b[:, lu] = s2 * (c_b[:, ho] - c_b[:, lu]), s2 * (c_b[:, ho] + c_b[:, lu])
        guess = Orbitals(
            coefficients_alpha=c_a,
            coefficients_beta=c_b,
            ao_overlap=orbs.get_overlap_matrix(),
            basis_set=orbs.get_basis_set(),
        )

        # Unrestricted KS from broken-symmetry guess
        uks = create("scf_solver")
        uks.settings().set("method", "pbe")
        uks.settings().set("scf_type", "unrestricted")
        _, wfn = uks.run(structure, 0, 1, guess)

        # The UKS wavefunction's spin-traced 1-RDM (in the alpha MO basis)
        # should be non-diagonal — its eigenvalues are the NOONs.
        rdm = wfn.get_active_one_rdm_spin_traced()
        noons = np.sort(np.linalg.eigvalsh(rdm))[::-1]
        n_a, n_b = wfn.get_total_num_electrons()

        assert not wfn.get_orbitals().is_restricted()
        assert np.all(noons >= -1e-12)
        assert np.all(noons < 2.0 + 1e-8)
        np.testing.assert_allclose(noons.sum(), n_a + n_b, atol=1e-6)
        assert any(0.1 < n < 1.9 for n in noons), f"No fractional NOONs: {noons}"

        # Run the natural orbital localizer — should accept UHF and produce
        # restricted NOs.
        localizer = create("orbital_localizer", "qdk_natural_orbitals")
        n_mo = c_a.shape[1]
        indices = list(range(n_mo))
        no_wfn = localizer.run(wfn, indices, indices)

        assert no_wfn.get_orbitals().is_restricted()
