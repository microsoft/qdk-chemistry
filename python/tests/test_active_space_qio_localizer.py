"""Integration tests for the QdkActiveSpaceQIOLocalizer Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.algorithms import (
    OrbitalLocalizer,
    QdkActiveSpaceQIOLocalizer,
    create,
)

from .reference_tolerances import (
    ci_energy_tolerance,
    entropy_tol,
    orthonormality_error_tolerance,
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


class TestActiveSpaceQIOLocalizerBindings:
    """Test that the QdkActiveSpaceQIOLocalizer Python bindings work correctly."""

    def test_factory_registration(self):
        """The active-space QIO localizer is registered in the factory."""
        available = algorithms.available("orbital_localizer")
        assert "qdk_active_space_qio" in available

    def test_factory_creation(self):
        """Creating the localizer via the factory."""
        localizer = algorithms.create("orbital_localizer", "qdk_active_space_qio")
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)

    def test_direct_construction(self):
        """Direct construction of QdkActiveSpaceQIOLocalizer."""
        localizer = QdkActiveSpaceQIOLocalizer()
        assert localizer is not None
        assert isinstance(localizer, OrbitalLocalizer)
        assert localizer.name() == "qdk_active_space_qio"
        assert localizer.type_name() == "orbital_localizer"

    def test_settings(self):
        """The localizer provides a settings interface."""
        localizer = QdkActiveSpaceQIOLocalizer()
        assert localizer.settings() is not None

    @pytest.mark.parametrize(
        (
            "structure_name",
            "multiplicity",
            "n_active_e",
            "n_active_o",
            "basis",
            "expected_entropy",
            "reference_tolerance",
        ),
        [
            ("ethylene.structure.xyz", 1, 4, 4, "def2-svp", 0.28733427, entropy_tol),
            ("o2.structure.xyz", 3, 8, 6, "cc-pvdz", 0.93445155, 1e-4),
        ],
        ids=["singlet", "multiplet"],
    )
    def test_reference_entropy_and_energy_invariant(
        self,
        test_data_files_path,
        structure_name,
        multiplicity,
        n_active_e,
        n_active_o,
        basis,
        expected_entropy,
        reference_tolerance,
    ):
        """QIO produces reference entropies and preserves CASCI energies."""
        active_wfn, energy_before, cas_wfn = _correlated_cas_wavefunction(
            test_data_files_path / structure_name, multiplicity, n_active_e, n_active_o, basis
        )
        active_orbitals = active_wfn.get_orbitals()

        alpha_indices, beta_indices = active_orbitals.get_active_space_indices()
        assert alpha_indices == beta_indices
        active_indices = list(alpha_indices)
        n = len(active_indices)
        n_a, n_b = active_wfn.get_active_num_electrons()
        if multiplicity == 1:
            assert n_a == n_b
        else:
            assert n_a != n_b

        entropy_before = float(np.sum(cas_wfn.get_single_orbital_entropies()))
        localizer = create("orbital_localizer", "qdk_active_space_qio")
        qio_wfn = localizer.run(cas_wfn, active_indices, active_indices)
        assert qio_wfn is not None

        s = np.asarray(active_orbitals.get_overlap_matrix())
        ca_can = np.asarray(active_orbitals.get_coefficients()[0])[:, active_indices]
        ca_qio = np.asarray(qio_wfn.get_orbitals().get_coefficients()[0])[:, active_indices]
        u = ca_can.conj().T @ s @ ca_qio
        np.testing.assert_allclose(u @ u.conj().T, np.eye(n), atol=orthonormality_error_tolerance)
        np.testing.assert_allclose(ca_qio.conj().T @ s @ ca_qio, np.eye(n), atol=orthonormality_error_tolerance)

        energy_after, rotated_cas_wfn = _run_cas(qio_wfn.get_orbitals(), n_a, n_b)
        entropy_after = float(np.sum(rotated_cas_wfn.get_single_orbital_entropies()))

        assert entropy_after <= entropy_before + entropy_tol
        np.testing.assert_allclose(entropy_after, expected_entropy, atol=reference_tolerance, rtol=0.0)
        assert abs(energy_before - energy_after) < ci_energy_tolerance

    def test_settings_defaults_and_override(self):
        """The Jacobi-sweep controls are exposed with defaults and settable."""
        localizer = QdkActiveSpaceQIOLocalizer()
        settings = localizer.settings()
        assert settings.get("max_cycles") == 200
        settings.set("max_cycles", 50)
        assert settings.get("max_cycles") == 50
