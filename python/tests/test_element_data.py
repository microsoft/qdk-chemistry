"""Test the element_data module including Element and Isotope enums."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Element, Isotope, get_current_ciaaw_version


class TestElementEnum:
    """Test cases for the Element enum."""

    def test_element_values_exist(self):
        """Test that elements are defined."""
        assert Element.H.value == 1
        assert Element.He.value == 2
        assert Element.Og.value == 118

    def test_element_names(self):
        """Test that element names are correct."""
        assert Element.H.name == "H"
        assert Element.He.name == "He"
        assert Element.Og.name == "Og"

    def test_element_enum_type(self):
        """Test that Element is a proper enum."""
        assert isinstance(Element.H, Element)
        assert isinstance(Element.He, Element)
        assert isinstance(Element.Og, Element)

    def test_element_comparison(self):
        """Test that elements can be compared."""
        assert Element.H == Element.H
        assert Element.He != Element.H
        assert Element.C.value < Element.O.value

    def test_all_elements_present(self):
        """Test that all 118 elements are accessible."""
        expected_elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]

        for symbol in expected_elements:
            assert hasattr(Element, symbol), f"Element.{symbol} not found"
            element = getattr(Element, symbol)
            assert isinstance(element, Element)


class TestIsotopeEnum:
    """Test cases for the Isotope enum."""

    def test_isotope_standard_elements(self):
        """Test that isotopes include standard atomic weight values."""
        # Isotopes should have element symbols as well
        assert hasattr(Isotope, "H")
        assert hasattr(Isotope, "He")
        assert hasattr(Isotope, "Og")

        # Standard atomic weights should match element values
        assert Isotope.H.value == 1
        assert Isotope.He.value == 2
        assert Isotope.Og.value == 118

    def test_specific_isotopes(self):
        """Test that specific isotopes are defined."""
        assert hasattr(Isotope, "H1")
        assert hasattr(Isotope, "H2")
        assert hasattr(Isotope, "Og295")

    def test_isotope_values_encoded(self):
        """Test that isotope values are properly encoded."""
        # Specific isotopes should have different values than elements
        assert Isotope.H1.value != Isotope.H.value
        assert Isotope.C12.value != Isotope.C.value
        assert Isotope.O16.value != Isotope.O.value

        # But standard isotope references should match
        assert Isotope.H.value == Element.H.value
        assert Isotope.C.value == Element.C.value
        assert Isotope.O.value == Element.O.value

    def test_isotope_enum_type(self):
        """Test that Isotope is a proper enum."""
        assert isinstance(Isotope.H, Isotope)
        assert isinstance(Isotope.H1, Isotope)
        assert isinstance(Isotope.Og295, Isotope)

    def test_isotope_comparison(self):
        """Test that isotopes can be compared."""
        assert Isotope.H1 == Isotope.H1
        assert Isotope.H1 != Isotope.H2
        assert Isotope.C != Isotope.C12

    def test_element_isotope_value_consistency(self):
        """Test that standard isotope values match element values."""
        # For standard atomic weights, isotope and element should have same value
        test_elements = ["H", "C", "N", "O"]

        for symbol in test_elements:
            element = getattr(Element, symbol)
            isotope = getattr(Isotope, symbol)
            assert element.value == isotope.value, (
                f"Element.{symbol}.value ({element.value}) != Isotope.{symbol}.value ({isotope.value})"
            )

    def test_element_isotope_name_consistency(self):
        """Test that element and isotope names match for standard values."""
        test_elements = ["H", "C", "N", "O"]

        for symbol in test_elements:
            element = getattr(Element, symbol)
            isotope = getattr(Isotope, symbol)
            assert element.name == isotope.name, (
                f"Element.{symbol}.name ({element.name}) != Isotope.{symbol}.name ({isotope.name})"
            )


class TestCIAAWVersion:
    """Test cases for CIAAW version information."""

    def test_get_current_ciaaw_version_exists(self):
        """Test that get_current_ciaaw_version function exists."""
        assert callable(get_current_ciaaw_version)

    def test_get_current_ciaaw_version_returns_string(self):
        """Test that get_current_ciaaw_version returns a valid string."""
        version = get_current_ciaaw_version()

        assert isinstance(version, str)
        assert len(version) > 0

    def test_ciaaw_version_format(self):
        """Test that CIAAW version has expected format."""
        version = get_current_ciaaw_version()

        assert "CIAAW" in version
        # Should include a year
        assert any(str(year) in version for year in [2024])
