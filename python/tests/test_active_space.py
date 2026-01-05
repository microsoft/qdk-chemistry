"""Tests for wf based active space selector functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data


class TestWavefunctionBasedActiveSpaceSelector:
    """Test class for WavefunctionBasedActiveSpaceSelector functionality."""

    def test_wfn_selector_factory(self):
        """Test selector factory functionality."""
        available_selectors = algorithms.available()["active_space_selector"]
        assert isinstance(available_selectors, list)
        assert len(available_selectors) >= 1

        # Test creating default selector
        selector = algorithms.create("active_space_selector")
        assert selector is not None

        # Test creating selector by name
        selector = algorithms.create("active_space_selector")
        selector_default = algorithms.create("active_space_selector", "qdk_autocas")
        assert selector_default is not None

        # Test that nonexistent selector raises error
        with pytest.raises(KeyError):
            algorithms.create("active_space_selector", "nonexistent_selector")

    def test_autocas_selection(self):
        """Test that the autocas active space selector works as expected."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        nuclear_charges = [7, 7]
        mol = data.Structure(coords, nuclear_charges)

        scf = algorithms.create("scf_solver")
        scf.settings().set("method", "hf")
        _, det = scf.run(mol, 0, 1, "cc-pvdz")
        val_cas_selector = algorithms.create("active_space_selector", "qdk_valence")
        val_cas_selector.settings().set("num_active_electrons", 10)
        val_cas_selector.settings().set("num_active_orbitals", 8)
        cas_det = val_cas_selector.run(det)
        hamil_constructor = algorithms.create("hamiltonian_constructor")
        hamil = hamil_constructor.run(cas_det.get_orbitals())
        mc = algorithms.create("multi_configuration_calculator", "macis_cas")
        mc.settings().set("calculate_one_rdm", True)
        mc.settings().set("calculate_two_rdm", True)
        _, wfn = mc.run(hamil, 5, 5)

        # autocas
        selector = algorithms.create("active_space_selector", "qdk_autocas")
        selected_wfn = selector.run(wfn)
        selected_orbitals = selected_wfn.get_orbitals()
        alpha_indices, beta_indices = selected_orbitals.get_active_space_indices()
        assert alpha_indices == [4, 5, 6, 7, 8, 9]
        assert beta_indices == [4, 5, 6, 7, 8, 9]

        # entropy
        selector = algorithms.create("active_space_selector", "qdk_autocas_eos")
        selected_wfn = selector.run(wfn)
        selected_orbitals = selected_wfn.get_orbitals()
        alpha_indices, beta_indices = selected_orbitals.get_active_space_indices()
        assert alpha_indices == [4, 5, 6, 7, 8, 9]
        assert beta_indices == [4, 5, 6, 7, 8, 9]
