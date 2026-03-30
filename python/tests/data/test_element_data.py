"""Test the element_data module — quantitative checks only."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Element, get_current_ciaaw_version


def test_element_count():
    """All 118 elements must be accessible."""
    elements = [e for e in Element.__members__.values()]
    assert len(elements) == 118


def test_element_atomic_numbers():
    """Spot-check atomic numbers against known values."""
    assert Element.H.value == 1
    assert Element.C.value == 6
    assert Element.Fe.value == 26
    assert Element.Og.value == 118


def test_ciaaw_version_format():
    """CIAAW version string must contain identifier and year."""
    version = get_current_ciaaw_version()
    assert "CIAAW" in version
    assert any(str(year) in version for year in range(2020, 2030))
