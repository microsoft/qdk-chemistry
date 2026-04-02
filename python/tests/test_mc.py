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
    entropy_tol,
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

    def test_mc_cas_entropies_doublet(self):
        """Test MACIS CAS entropy evaluation on NO doublet with full active space."""
        # Create NO molecule
        symbols = ["N", "O"]
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.173],
            ]
        )
        no = Structure(symbols, coords)

        # use pyscf for ROHF
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
        ham_constructor = algorithms.create("hamiltonian_constructor")

        # SCF on NO doublet
        _, wfn_hf = scf_solver.run(no, 0, 2, "sto-3g")

        # Configure entropy calculations
        settings = mc_calculator.settings()
        settings.set("ci_residual_tolerance", ci_energy_tolerance)
        settings.set("calculate_single_orbital_entropies", True)
        settings.set("calculate_two_orbital_entropies", True)
        settings.set("calculate_mutual_information", True)

        # Hamiltonian with full active space
        ham = ham_constructor.run(wfn_hf.get_orbitals())

        # Run FCI (8 alpha, 7 beta)
        _, wfn = mc_calculator.run(ham, 8, 7)

        # Retrieve computed entropies
        s1 = wfn.get_single_orbital_entropies()
        s2 = wfn.get_two_orbital_entropies()
        mi = wfn.get_mutual_information()

        # fmt: off
        s1_expected = np.array([
            0.0000436, 0.0000862, 0.0449884, 0.0435672, 0.1400889,
            0.2929436, 0.1108481, 0.1429517, 0.3093473, 0.1391263,
        ])

        s2_expected = np.array([
            [0.0, 0.0001288, 0.0450207, 0.0436004, 0.1401288, 0.2929853, 0.1108853, 0.1429946, 0.3093872, 0.1391532],
            [0.0001288, 0.0, 0.0450631, 0.0436226, 0.1401695, 0.2930270, 0.1109101, 0.1430363, 0.3094252, 0.1391841],
            [0.0450207, 0.0450631, 0.0, 0.0856590, 0.1811253, 0.3281337, 0.1448387, 0.1851851, 0.3449373, 0.1531420],
            [0.0436004, 0.0436226, 0.0856590, 0.0, 0.1820123, 0.3280457, 0.1460050, 0.1818233, 0.3249047, 0.1671517],
            [0.1401288, 0.1401695, 0.1811253, 0.1820123, 0.0, 0.3697555, 0.2377242, 0.1479504, 0.3856283, 0.2617133],
            [0.2929853, 0.2930270, 0.3281337, 0.3280457, 0.3697555, 0.0, 0.3834483, 0.3749362, 0.1930608, 0.4026182],
            [0.1108853, 0.1109101, 0.1448387, 0.1460050, 0.2377242, 0.3834483, 0.0, 0.2419231, 0.3905371, 0.1498425],
            [0.1429946, 0.1430363, 0.1851851, 0.1818233, 0.1479504, 0.3749362, 0.2419231, 0.0, 0.3911559, 0.2650915],
            [0.3093872, 0.3094252, 0.3449373, 0.3249047, 0.3856283, 0.1930608, 0.3905371, 0.3911559, 0.0, 0.4178743],
            [0.1391532, 0.1391841, 0.1531420, 0.1671517, 0.2617133, 0.4026182, 0.1498425, 0.2650915, 0.4178743, 0.0]
        ])

        mi_expected = np.array([
            [0.0, 0.0000009, 0.0000113, 0.0000104, 0.0000037, 0.0000018, 0.0000064, 0.0000007, 0.0000037, 0.0000167],
            [0.0000009, 0.0, 0.0000115, 0.0000308, 0.0000055, 0.0000028, 0.0000242, 0.0000016, 0.0000083, 0.0000284],
            [0.0000113, 0.0000115, 0.0, 0.0028966, 0.0039520, 0.0097983, 0.0109979, 0.0027550, 0.0093985, 0.0309728],
            [0.0000104, 0.0000308, 0.0028966, 0.0, 0.0016437, 0.0084651, 0.0084103, 0.0046955, 0.0280098, 0.0155418],
            [0.0000037, 0.0000055, 0.0039520, 0.0016437, 0.0, 0.0632769, 0.0132128, 0.1350902, 0.0638079, 0.0175019],
            [0.0000018, 0.0000028, 0.0097983, 0.0084651, 0.0632769, 0.0, 0.0203434, 0.0609591, 0.4092301, 0.0294517],
            [0.0000064, 0.0000242, 0.0109979, 0.0084103, 0.0132128, 0.0203434, 0.0, 0.0118767, 0.0296583, 0.1001320],
            [0.0000007, 0.0000016, 0.0027550, 0.0046955, 0.1350902, 0.0609591, 0.0118767, 0.0, 0.0611431, 0.0169865],
            [0.0000037, 0.0000083, 0.0093985, 0.0280098, 0.0638079, 0.4092301, 0.0296583, 0.0611431, 0.0, 0.0305994],
            [0.0000167, 0.0000284, 0.0309728, 0.0155418, 0.0175019, 0.0294517, 0.1001320, 0.0169865, 0.0305994, 0.0]
        ])
        # fmt: on

        np.testing.assert_allclose(s1, s1_expected, atol=entropy_tol)
        np.testing.assert_allclose(s2, s2_expected, atol=entropy_tol)
        np.testing.assert_allclose(mi, mi_expected, atol=entropy_tol)

    def test_mc_cas_entropies_singlet(self):
        """Test MACIS CAS entropy evaluation on H2O singlet with full active space."""
        h2o = create_water_structure()

        # use pyscf for ROHF
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("scf_type", "restricted")
        mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
        ham_constructor = algorithms.create("hamiltonian_constructor")

        # SCF on H2O singlet
        _, wfn_hf = scf_solver.run(h2o, 0, 1, "sto-3g")

        # Configure entropy calculations
        settings = mc_calculator.settings()
        settings.set("ci_residual_tolerance", ci_energy_tolerance)
        settings.set("calculate_single_orbital_entropies", True)
        settings.set("calculate_two_orbital_entropies", True)
        settings.set("calculate_mutual_information", True)

        # Hamiltonian with full active space
        ham = ham_constructor.run(wfn_hf.get_orbitals())

        # Run FCI (5 alpha, 5 beta)
        _, wfn = mc_calculator.run(ham, 5, 5)

        # Retrieve computed entropies
        s1 = wfn.get_single_orbital_entropies()
        s2 = wfn.get_two_orbital_entropies()
        mi = wfn.get_mutual_information()

        # fmt: off
        s1_expected = np.array([
            0.0000382, 0.0518392, 0.1805445, 0.1661148, 0.0073285, 0.1890896, 0.1810905
        ])

        s2_expected = np.array([
            [0.0, 0.0518586, 0.1805784, 0.1661472, 0.0073635, 0.1891212, 0.1811199],
            [0.0518586, 0.0, 0.2181955, 0.1908449, 0.0577654, 0.2074609, 0.2201824],
            [0.1805784, 0.2181955, 0.0, 0.2519576, 0.1859838, 0.2455184, 0.1672916],
            [0.1661472, 0.1908449, 0.2519576, 0.0, 0.1706681, 0.2032416, 0.2525337],
            [0.0073635, 0.0577654, 0.1859838, 0.1706681, 0.0, 0.1875032, 0.1866889],
            [0.1891212, 0.2074609, 0.2455184, 0.2032416, 0.1875032, 0.0, 0.2472759],
            [0.1811199, 0.2201824, 0.1672916, 0.2525337, 0.1866889, 0.2472759, 0.0],
        ])

        mi_expected = np.array([
            [0.0, 0.0000188, 0.0000043, 0.0000058, 0.0000033, 0.0000067, 0.0000088],
            [0.0000188, 0.0, 0.0141882, 0.0271091, 0.0014023, 0.033468 , 0.0127472],
            [0.0000043, 0.0141882, 0.0, 0.0947016, 0.0018892, 0.1241158, 0.1943434],
            [0.0000058, 0.0271091, 0.0947016, 0.0, 0.0027751, 0.1519628, 0.0946715],
            [0.0000033, 0.0014023, 0.0018892, 0.0027751, 0.0, 0.008915 , 0.00173  ],
            [0.0000067, 0.033468 , 0.1241158, 0.1519628, 0.008915 , 0.0, 0.1229042],
            [0.0000088, 0.0127472, 0.1943434, 0.0946715, 0.00173  , 0.1229042, 0.0],
        ])
        # fmt: on

        np.testing.assert_allclose(s1, s1_expected, atol=entropy_tol)
        np.testing.assert_allclose(s2, s2_expected, atol=entropy_tol)
        np.testing.assert_allclose(mi, mi_expected, atol=entropy_tol)
