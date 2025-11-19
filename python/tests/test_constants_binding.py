"""Test the pybind11 constants binding functionality.

This module contains focused tests for the C++ to Python binding layer,
specifically testing the pybind11 constants.cpp implementation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

try:
    import qdk_chemistry._core.constants as core_constants
    from qdk_chemistry import constants

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    pytest.skip("QDK/Chemistry core module not available", allow_module_level=True)


class TestCoreBinding:
    """Test the core pybind11 binding functionality."""

    def test_constant_attributes_exist(self):
        """Test that all expected constants are bound."""
        # All expected constants that should be available in both C++ and Python
        expected_constants = [
            "BOHR_TO_ANGSTROM",
            "ANGSTROM_TO_BOHR",
            "FINE_STRUCTURE_CONSTANT",
            "ELECTRON_MASS",
            "PROTON_MASS",
            "NEUTRON_MASS",
            "ATOMIC_MASS_CONSTANT",
            "AVOGADRO_CONSTANT",
            "BOLTZMANN_CONSTANT",
            "PLANCK_CONSTANT",
            "REDUCED_PLANCK_CONSTANT",
            "SPEED_OF_LIGHT",
            "ELEMENTARY_CHARGE",
            "HARTREE_TO_EV",
            "EV_TO_HARTREE",
            "HARTREE_TO_KCAL_PER_MOL",
            "KCAL_PER_MOL_TO_HARTREE",
            "HARTREE_TO_KJ_PER_MOL",
            "KJ_PER_MOL_TO_HARTREE",
        ]

        for const_name in expected_constants:
            # Test C++ binding has the constant
            assert hasattr(core_constants, const_name), f"Missing constant in C++: {const_name}"
            cpp_value = getattr(core_constants, const_name)
            assert isinstance(cpp_value, float), f"C++ constant {const_name} should be float, got {type(cpp_value)}"

            # Test Python module has the constant
            assert hasattr(constants, const_name), f"Missing constant in Python: {const_name}"
            python_value = getattr(constants, const_name)
            assert isinstance(python_value, float), (
                f"Python constant {const_name} should be float, got {type(python_value)}"
            )

            # Verify that C++ and Python values match
            assert abs(cpp_value - python_value) < 1e-15, (
                f"Value mismatch for {const_name}: {cpp_value} != {python_value}"
            )

    def test_documentation_functions_exist(self):
        """Test that documentation functions are bound."""
        assert hasattr(core_constants, "get_constants_info")
        assert hasattr(core_constants, "get_constant_info")
        assert callable(core_constants.get_constants_info)
        assert callable(core_constants.get_constant_info)

    def test_constant_info_class_exists(self):
        """Test that ConstantInfo class is bound."""
        assert hasattr(core_constants, "ConstantInfo")

        # Test creating an instance through get_constant_info
        info = core_constants.get_constant_info("bohr_to_angstrom")
        assert hasattr(info, "name")
        assert hasattr(info, "description")
        assert hasattr(info, "units")
        assert hasattr(info, "source")
        assert hasattr(info, "symbol")
        assert hasattr(info, "value")

    def test_get_constants_info_returns_dict(self):
        """Test that get_constants_info returns a proper dictionary."""
        all_info = core_constants.get_constants_info()

        assert isinstance(all_info, dict)
        assert len(all_info) > 0

        # Check a known constant
        assert "bohr_to_angstrom" in all_info
        info = all_info["bohr_to_angstrom"]

        assert info.name == "bohr_to_angstrom"
        assert isinstance(info.value, float)
        assert isinstance(info.description, str)
        assert isinstance(info.units, str)
        assert isinstance(info.source, str)
        assert isinstance(info.symbol, str)

    def test_get_constant_info_single(self):
        """Test getting info for a single constant."""
        info = core_constants.get_constant_info("fine_structure_constant")

        assert info.name == "fine_structure_constant"
        assert "fine-structure" in info.description.lower() or "electromagnetic" in info.description.lower()
        assert "CODATA" in info.source
        assert isinstance(info.value, float)
        assert 0.007 < info.value < 0.008  # Should be around 1/137

    def test_constant_info_repr(self):
        """Test the __repr__ method of ConstantInfo."""
        info = core_constants.get_constant_info("bohr_to_angstrom")
        repr_str = repr(info)

        assert "ConstantInfo" in repr_str
        assert "bohr_to_angstrom" in repr_str
        assert str(0.529) in repr_str

    def test_error_handling(self):
        """Test error handling for unknown constants."""
        with pytest.raises(KeyError):
            core_constants.get_constant_info("nonexistent_constant")


class TestValueConsistency:
    """Test that bound values are consistent and reasonable."""

    def test_reciprocal_relationships(self):
        """Test that reciprocal constants are actually reciprocals."""
        # Test length conversions using Python constants
        assert np.isclose(
            constants.BOHR_TO_ANGSTROM * constants.ANGSTROM_TO_BOHR,
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Test energy conversions using Python constants
        assert np.isclose(
            constants.HARTREE_TO_EV * constants.EV_TO_HARTREE,
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Also verify that C++ constants have the same reciprocal relationship
        bohr_to_ang = core_constants.BOHR_TO_ANGSTROM
        ang_to_bohr = core_constants.ANGSTROM_TO_BOHR
        assert np.isclose(
            bohr_to_ang * ang_to_bohr,
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        h2ev = core_constants.HARTREE_TO_EV
        ev2h = core_constants.EV_TO_HARTREE
        assert np.isclose(
            h2ev * ev2h, 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_documentation_value_consistency(self):
        """Test that ConstantInfo values match actual constants."""
        test_constants = {
            "bohr_to_angstrom": ("BOHR_TO_ANGSTROM", constants.BOHR_TO_ANGSTROM),
            "fine_structure_constant": ("FINE_STRUCTURE_CONSTANT", constants.FINE_STRUCTURE_CONSTANT),
            "hartree_to_ev": ("HARTREE_TO_EV", constants.HARTREE_TO_EV),
            "speed_of_light": ("SPEED_OF_LIGHT", constants.SPEED_OF_LIGHT),
        }

        for info_key, (cpp_attr_name, python_value) in test_constants.items():
            cpp_value = getattr(core_constants, cpp_attr_name)
            info = core_constants.get_constant_info(info_key)

            # Test that C++ constant matches Python constant
            assert np.isclose(
                cpp_value,
                python_value,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            ), f"C++ vs Python value mismatch for {cpp_attr_name}: {cpp_value} != {python_value}"

            # Test that ConstantInfo matches both
            assert np.isclose(
                cpp_value,
                info.value,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            ), f"C++ vs ConstantInfo value mismatch for {info_key}: {cpp_value} != {info.value}"

    def test_codata_version_consistency(self):
        """Test that all constants report the same CODATA version."""
        all_info = core_constants.get_constants_info()

        sources = {info.source for info in all_info.values()}
        assert len(sources) == 1, f"Found multiple CODATA versions: {sources}"

        source = next(iter(sources))
        assert "CODATA" in source
        assert any(year in source for year in ["2014", "2018"])
