"""Test the element_data module including Element enum."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Element, get_current_ciaaw_version


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
