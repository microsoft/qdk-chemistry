"""Unit tests for the QDK/Chemistry constants documentation system.

This module tests the enhanced documentation features for physical constants,
including version awareness, metadata access, and utility functions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

try:
    from qdk_chemistry.constants import (
        # Basic constants
        ANGSTROM_TO_BOHR,
        BOHR_TO_ANGSTROM,
        ELECTRON_MASS,
        EV_TO_HARTREE,
        FINE_STRUCTURE_CONSTANT,
        HARTREE_TO_EV,
        # Documentation functions
        find_constant,
        get_constant_info,
        get_constants_info,
        list_constants,
        show_constant_details,
    )

    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False
    pytest.skip("QDK/Chemistry constants module not available", allow_module_level=True)


class TestConstantValues:
    """Test that constant values are reasonable and consistent."""

    def test_conversion_factors_reciprocal(self):
        """Test that conversion factors are reciprocals of each other."""
        # Test length conversions
        assert np.isclose(
            BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR,
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Test energy conversions
        assert np.isclose(
            HARTREE_TO_EV * EV_TO_HARTREE,
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_constant_value_ranges(self):
        """Test that constants are within expected physical ranges."""
        # Bohr radius should be around 0.5 Angstrom
        assert 0.5 < BOHR_TO_ANGSTROM < 0.6

        # Fine structure constant should be around 1/137
        assert 0.007 < FINE_STRUCTURE_CONSTANT < 0.008

        # Electron mass should be around 9e-31 kg
        assert 9e-32 < ELECTRON_MASS < 1e-30

        # Hartree to eV should be around 27 eV
        assert 25 < HARTREE_TO_EV < 30

    def test_constant_types(self):
        """Test that constants are the correct type (float)."""
        assert isinstance(BOHR_TO_ANGSTROM, float)
        assert isinstance(FINE_STRUCTURE_CONSTANT, float)
        assert isinstance(ELECTRON_MASS, float)
        assert isinstance(HARTREE_TO_EV, float)


class TestConstantInfo:
    """Test the ConstantInfo structure and metadata access."""

    def test_get_constant_info_basic(self):
        """Test getting info for a single constant."""
        info = get_constant_info("bohr_to_angstrom")

        assert info.name == "bohr_to_angstrom"
        assert "Bohr radius" in info.description
        assert info.units in ["Å/bohr", "A/bohr"]  # Handle Unicode variations
        assert "CODATA" in info.source
        assert info.symbol in {"a₀", "a0"}  # Handle Unicode variations
        assert np.isclose(
            info.value,
            BOHR_TO_ANGSTROM,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_get_constant_info_unknown(self):
        """Test error handling for unknown constants."""
        with pytest.raises(KeyError, match="Unknown constant"):
            get_constant_info("nonexistent_constant")

    def test_get_constants_info_complete(self):
        """Test getting info for all constants."""
        all_info = get_constants_info()

        assert isinstance(all_info, dict)
        assert len(all_info) > 15  # Should have at least the constants we know about

        # Check that all expected constants are present
        expected_constants = {
            "bohr_to_angstrom",
            "angstrom_to_bohr",
            "fine_structure_constant",
            "electron_mass",
            "proton_mass",
            "neutron_mass",
            "hartree_to_ev",
            "ev_to_hartree",
            "planck_constant",
            "speed_of_light",
        }

        for const_name in expected_constants:
            assert const_name in all_info
            info = all_info[const_name]
            assert hasattr(info, "name")
            assert hasattr(info, "description")
            assert hasattr(info, "units")
            assert hasattr(info, "source")
            assert hasattr(info, "symbol")
            assert hasattr(info, "value")

    def test_constant_info_consistency(self):
        """Test that ConstantInfo values match actual constants."""
        all_info = get_constants_info()

        # Test a few key constants
        test_cases = [
            ("bohr_to_angstrom", BOHR_TO_ANGSTROM),
            ("angstrom_to_bohr", ANGSTROM_TO_BOHR),
            ("fine_structure_constant", FINE_STRUCTURE_CONSTANT),
            ("hartree_to_ev", HARTREE_TO_EV),
        ]

        for const_name, const_value in test_cases:
            info = all_info[const_name]
            assert np.isclose(
                info.value,
                const_value,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            ), f"ConstantInfo value for {const_name} doesn't match actual constant"

    def test_codata_version_consistency(self):
        """Test that all constants report the same CODATA version."""
        all_info = get_constants_info()

        versions = {info.source for info in all_info.values()}
        assert len(versions) == 1, f"Multiple CODATA versions found: {versions}"

        version = next(iter(versions))
        assert "CODATA" in version
        assert ("2014" in version) or ("2018" in version)


class TestDocumentationFunctions:
    """Test the utility functions for documentation access."""

    def test_find_constant_by_name(self):
        """Test finding constants by name substring."""
        results = find_constant("bohr")

        assert isinstance(results, dict)
        assert len(results) >= 2  # Should find bohr_to_angstrom and angstrom_to_bohr
        assert "bohr_to_angstrom" in results
        assert "angstrom_to_bohr" in results

    def test_find_constant_by_description(self):
        """Test finding constants by description content."""
        results = find_constant("mass")

        assert isinstance(results, dict)
        assert len(results) >= 3  # Should find electron_mass, proton_mass, neutron_mass

        # Check that we found mass-related constants
        mass_constants = {"electron_mass", "proton_mass", "neutron_mass", "atomic_mass_constant"}
        found_constants = set(results.keys())

        assert len(mass_constants.intersection(found_constants)) >= 3

    def test_find_constant_case_insensitive(self):
        """Test that search is case insensitive."""
        results_lower = find_constant("bohr")
        results_upper = find_constant("BOHR")
        results_mixed = find_constant("Bohr")

        for k, v in results_lower.items():
            assert k in results_upper
            assert k in results_mixed
            assert v.value == results_upper[k].value == results_mixed[k].value

    def test_find_constant_empty_search(self):
        """Test behavior with empty search term."""
        results = find_constant("")

        # Empty search should return all constants
        all_info = get_constants_info()
        assert len(results) == len(all_info)

    def test_show_constant_details_output(self, capsys: pytest.CaptureFixture[str]):
        """Test that show_constant_details produces output."""
        show_constant_details("bohr_to_angstrom")

        captured = capsys.readouterr()
        assert "bohr_to_angstrom" in captured.out
        assert "Bohr radius" in captured.out
        assert "CODATA" in captured.out

    def test_show_constant_details_unknown(self, capsys: pytest.CaptureFixture[str]):
        """Test error handling in show_constant_details."""
        show_constant_details("nonexistent_constant")

        captured = capsys.readouterr()
        assert "Unknown constant" in captured.out

    def test_list_constants_output(self, capsys: pytest.CaptureFixture[str]):
        """Test that list_constants produces categorized output."""
        list_constants(show_values=False)

        captured = capsys.readouterr()
        output = captured.out

        # Check for expected categories
        assert "Length Conversion" in output
        assert "Fundamental Constants" in output
        assert "Particle Masses" in output
        assert "Energy Conversion" in output

        # Check for some expected constants
        assert "bohr_to_angstrom" in output
        assert "fine_structure_constant" in output

    def test_list_constants_with_values(self, capsys: pytest.CaptureFixture[str]):
        """Test list_constants with values shown."""
        list_constants(show_values=True, show_units=True)

        captured = capsys.readouterr()
        output = captured.out

        # Should contain numerical values
        assert str(BOHR_TO_ANGSTROM)[:6] in output  # First 6 digits
        assert "CODATA" in output


class TestVersionAwareness:
    """Test that the system properly handles CODATA version information."""

    def test_version_detection(self):
        """Test that we can detect the current CODATA version."""
        info = get_constant_info("bohr_to_angstrom")

        assert "CODATA" in info.source
        # Should be either 2014 or 2018
        assert ("2014" in info.source) or ("2018" in info.source)

    def test_expected_values_codata_2018(self):
        """Test expected values if using CODATA 2018."""
        info = get_constant_info("bohr_to_angstrom")

        if "2018" in info.source:
            # CODATA 2018 value
            assert np.isclose(
                BOHR_TO_ANGSTROM,
                0.529177210903,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        elif "2014" in info.source:
            # CODATA 2014 value
            assert np.isclose(
                BOHR_TO_ANGSTROM,
                0.52917721067,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

    def test_documentation_reflects_version(self):
        """Test that documentation correctly reflects the version in use."""
        all_info = get_constants_info()

        # All constants should report the same version
        versions = {info.source for info in all_info.values()}
        assert len(versions) == 1

        version = next(iter(versions))

        # Check that the reported version matches the actual values
        bohr_info = all_info["bohr_to_angstrom"]

        if "2018" in version:
            expected_bohr = 0.529177210903
        elif "2014" in version:
            expected_bohr = 0.52917721067
        else:
            pytest.fail(f"Unexpected CODATA version: {version}")

        assert np.isclose(
            bohr_info.value,
            expected_bohr,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_symbol_constants(self):
        """Test constants that don't have mathematical symbols."""
        # Energy conversion factors typically don't have standard symbols
        info = get_constant_info("hartree_to_ev")

        # Should have empty or reasonable symbol
        assert info.symbol == "" or len(info.symbol) < 10

    def test_unicode_in_symbols(self):
        """Test that Unicode symbols are handled properly."""
        info = get_constant_info("fine_structure_constant")

        # Should be able to handle Greek alpha (use ASCII for tests)
        assert info.symbol in ["alpha", "a"] or len(info.symbol) == 1

    def test_long_descriptions(self):
        """Test constants with long descriptions."""
        info = get_constant_info("fine_structure_constant")

        # Should have a meaningful description
        assert len(info.description) > 20
        assert "constant" in info.description.lower()


# Integration test
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test a complete workflow using the constants system."""
        # 1. Get all constants
        all_constants = get_constants_info()
        assert len(all_constants) > 10

        # 2. Find conversion constants
        conversions = find_constant("conversion")
        assert len(conversions) > 0

        # 3. Use a constant for calculation
        length_bohr = 2.5
        length_angstrom = length_bohr * BOHR_TO_ANGSTROM
        length_bohr_back = length_angstrom * ANGSTROM_TO_BOHR

        assert np.isclose(
            length_bohr,
            length_bohr_back,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # 4. Verify documentation matches
        info = get_constant_info("bohr_to_angstrom")
        assert np.isclose(
            info.value,
            BOHR_TO_ANGSTROM,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_scientific_accuracy(self):
        """Test that the constants maintain scientific accuracy."""
        # Test fundamental relationships
        # c = speed of light should be exactly 299792458 m/s (defined)
        c_info = get_constant_info("speed_of_light")
        assert c_info.value == 299792458.0

        # Test that derived constants are consistent
        # Check that reciprocal relationships hold
        conversions = [
            ("bohr_to_angstrom", "angstrom_to_bohr"),
            ("hartree_to_ev", "ev_to_hartree"),
        ]

        for forward, reverse in conversions:
            forward_info = get_constant_info(forward)
            reverse_info = get_constant_info(reverse)

            product = forward_info.value * reverse_info.value
            assert np.isclose(
                product,
                1.0,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            ), f"{forward} * {reverse} should equal 1.0, got {product}"
