"""Tests for ProjectedMultiConfigurationCalculator functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Configuration, Structure

from .reference_tolerances import (
    ci_energy_tolerance,
    float_comparison_relative_tolerance,
)


def create_water_structure():
    """Create a water molecule structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["O", "H", "H"]
    coords = (
        np.array(
            [
                [0.000000000, -0.0757918436, 0.000000000000],
                [0.866811829, 0.6014357793, -0.000000000000],
                [-0.866811829, 0.6014357793, -0.000000000000],
            ]
        )
        * ANGSTROM_TO_BOHR
    )
    return Structure(symbols, coords)


class TestPMCCalculator:
    """Test class ProjectedMultiConfigurationCalculator functionality."""

    def test_projected_multi_configuration_calculator_factory(self):
        """Test ProjectedMultiConfigurationCalculator factory functionality."""
        available_calculators = algorithms.available()["projected_multi_configuration_calculator"]
        assert isinstance(available_calculators, list)
        assert len(available_calculators) >= 1
        assert "macis_pmc" in available_calculators

        # Test creating default calculator
        mc_calculator = algorithms.create("projected_multi_configuration_calculator")
        assert mc_calculator is not None

        # Test creating calculator by name
        mc_calculator_default = algorithms.create("projected_multi_configuration_calculator", "macis_pmc")
        assert mc_calculator_default is not None

        # Test that nonexistent calculator raises error
        with pytest.raises(KeyError):
            algorithms.create("projected_multi_configuration_calculator", "nonexistent_calculator")

    def test_projected_multi_configuration_calculator_water_hf(self):
        """Test ProjectedMultiConfigurationCalculator on water molecule with HF determinant."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "qdk")
        projected_multi_configuration_calculator = algorithms.create(
            "projected_multi_configuration_calculator", "macis_pmc"
        )
        ham_constructor = algorithms.create("hamiltonian_constructor", "qdk")
        as_selector = algorithms.create("active_space_selector", "qdk_valence")

        # Perform SCF calculation
        e_hf, wfn_hf = scf_solver.run(water, 0, 1, "def2-svp")

        # Select active space
        as_selector.settings().set("num_active_electrons", 6)
        as_selector.settings().set("num_active_orbitals", 6)
        active_wfn = as_selector.run(wfn_hf)
        active_orbitals = active_wfn.get_orbitals()

        # Compute the Hamiltonian
        ham = ham_constructor.run(active_orbitals)

        # Perform MC calculation
        e_pmc, wfn_pmc = projected_multi_configuration_calculator.run(
            ham, [Configuration("222000"), Configuration("220200"), Configuration("220020"), Configuration("220002")]
        )

        assert e_pmc < e_hf
        assert wfn_pmc.size() == 4
        e_pmc, wfn_pmc = projected_multi_configuration_calculator.run(ham, [Configuration("222000")])

        assert np.isclose(e_pmc, e_hf, rtol=float_comparison_relative_tolerance, atol=ci_energy_tolerance)
        assert wfn_pmc.size() == 1

    def test_projected_multi_configuration_calculator_water3det(self):
        """Test ProjectedMultiConfigurationCalculator on water molecule with 3 determinants."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
        projected_multi_configuration_calculator = algorithms.create(
            "projected_multi_configuration_calculator", "macis_pmc"
        )
        ham_constructor = algorithms.create("hamiltonian_constructor")
        as_selector = algorithms.create("active_space_selector", "qdk_valence")

        # Perform SCF calculation
        e_hf, wfn_hf = scf_solver.run(water, 0, 1, "def2-svp")

        # Select active space
        as_selector.settings().set("num_active_electrons", 6)
        as_selector.settings().set("num_active_orbitals", 6)
        active_wfn = as_selector.run(wfn_hf)
        active_orbitals = active_wfn.get_orbitals()

        # Compute the Hamiltonian
        ham = ham_constructor.run(active_orbitals)

        # Perform MC calculation
        e_pmc, wfn_pmc = projected_multi_configuration_calculator.run(
            ham, [Configuration("222000"), Configuration("22ud00"), Configuration("222000")]
        )

        assert e_pmc < e_hf
        assert wfn_pmc.size() == 3
