"""Integration tests for the QdkGaugeFixingLocalizer Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import (
    OrbitalLocalizer,
    QdkGaugeFixingLocalizer,
    create,
)
from qdk_chemistry.data import Structure
from qdk_chemistry.data._spin_channels import spin_channel_indices
from qdk_chemistry.data.symmetry import axes
from qdk_chemistry.utils import compute_valence_space_parameters


class TestGaugeFixingLocalizerBindings:
    """Test that the QdkGaugeFixingLocalizer Python bindings work correctly."""

    def test_factory_registration(self):
        """Test that the gauge-fixing localizer is registered in the factory."""
        assert "qdk_gauge_fixing" in algorithms.available("orbital_localizer")

    def test_factory_creation(self):
        """Test creating the localizer via the factory."""
        localizer = create("orbital_localizer", "qdk_gauge_fixing")
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)

    def test_direct_construction(self):
        """Test direct construction of QdkGaugeFixingLocalizer."""
        localizer = QdkGaugeFixingLocalizer()
        assert isinstance(localizer, OrbitalLocalizer)
        assert localizer.name() == "qdk_gauge_fixing"
        assert localizer.type_name() == "orbital_localizer"

    def test_settings_from_create_kwargs(self):
        """Test that create() keyword arguments configure the settings."""
        localizer = create("orbital_localizer", "qdk_gauge_fixing", max_sweeps=1)
        assert localizer.settings().get("max_sweeps") == 1
        assert localizer.settings().get("angle_samples") == 32

    def test_angle_samples_lower_bound(self):
        """Test that fewer than four angular samples is rejected by the settings constraint."""
        localizer = QdkGaugeFixingLocalizer()
        with pytest.raises(ValueError, match="out of allowed range"):
            localizer.settings().set("angle_samples", 3)

    def test_lih_gauge_fixing_preserves_occupations_and_energy(self):
        """Gauge fixing LiH must preserve the CASCI energy and the natural occupations."""
        structure = Structure.from_xyz("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n")
        _, hartree_fock_wavefunction = create("scf_solver", "qdk").run(
            structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
        )
        num_electrons, num_orbitals = compute_valence_space_parameters(hartree_fock_wavefunction, 0)
        valence_wavefunction = create(
            "active_space_selector",
            "qdk_valence",
            num_active_electrons=num_electrons,
            num_active_orbitals=num_orbitals,
        ).run(hartree_fock_wavefunction)
        indices = list(spin_channel_indices(valence_wavefunction.get_orbitals().active_indices(), axes.alpha()))
        num_alpha, num_beta = valence_wavefunction.get_active_num_electrons()

        hamiltonian_constructor = create("hamiltonian_constructor", "qdk")
        casci_solver = create("multi_configuration_calculator", "macis_cas", calculate_one_rdm=True)
        _, casci_wavefunction = casci_solver.run(
            hamiltonian_constructor.run(valence_wavefunction.get_orbitals()), num_alpha, num_beta
        )
        natural = create("orbital_localizer", "qdk_natural_orbitals").run(casci_wavefunction, indices, indices)
        energy_before, _ = casci_solver.run(hamiltonian_constructor.run(natural.get_orbitals()), num_alpha, num_beta)

        gauge_fixed = create("orbital_localizer", "qdk_gauge_fixing").run(natural, indices, indices)

        energy_after, _ = casci_solver.run(hamiltonian_constructor.run(gauge_fixed.get_orbitals()), num_alpha, num_beta)
        assert energy_after == pytest.approx(energy_before, abs=1e-9)
        np.testing.assert_allclose(
            np.diag(np.asarray(gauge_fixed.get_active_one_rdm_spin_traced())),
            np.diag(np.asarray(natural.get_active_one_rdm_spin_traced())),
            atol=1e-12,
        )

    def test_unsorted_indices_raise(self):
        """Test that unsorted orbital indices are rejected through the binding."""
        structure = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
        _, wavefunction = create("scf_solver", "qdk").run(
            structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
        )
        with pytest.raises(ValueError, match="sorted"):
            create("orbital_localizer", "qdk_gauge_fixing").run(wavefunction, [1, 0], [1, 0])
