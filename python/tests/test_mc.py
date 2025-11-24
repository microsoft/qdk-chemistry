"""Tests for MultiConfigurationCalculator functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

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


class TestMCCalculator:
    """Test class MultiConfigurationCalculator functionality."""

    def test_mc_calculator_factory(self):
        """Test MultiConfigurationCalculator factory functionality."""
        available_calculators = algorithms.available("multi_configuration_calculator")
        assert isinstance(available_calculators, list)
        assert len(available_calculators) >= 2
        assert "macis_cas" in available_calculators
        assert "macis_asci" in available_calculators

        # Test creating default calculator
        mc_calculator = algorithms.create("multi_configuration_calculator")
        assert mc_calculator is not None

        # Test creating calculator by name
        mc_calculator_default = algorithms.create("multi_configuration_calculator", "macis_cas")
        assert mc_calculator_default is not None

        # Test that nonexistent calculator raises error
        with pytest.raises(KeyError):
            algorithms.create("multi_configuration_calculator", "nonexistent_calculator")

    def test_mc_calculator_water_fci(self):
        """Test MultiConfigurationCalculator on water molecule with default settings."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
        mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
        ham_constructor = algorithms.create("hamiltonian_constructor")

        # Perform SCF calculation
        _, wfn_hf = scf_solver.run(water, 0, 1, "sto-3g")

        # Compute the Hamiltonian
        ham = ham_constructor.run(wfn_hf.get_orbitals())

        # Perform MC calculation
        e_fci, wfn_fci = mc_calculator.run(ham, 5, 5)

        # Validate results
        assert np.isclose(
            e_fci - ham.get_core_energy(),
            -83.01534669468,
            rtol=float_comparison_relative_tolerance,
            atol=ci_energy_tolerance,
        )
        assert wfn_fci.size() == 441
