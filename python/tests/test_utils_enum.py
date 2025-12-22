"""Tests for utility enums in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.data.noise_models import SupportedErrorTypes, SupportedGate
from qdk_chemistry.utils.enum import CaseInsensitiveStrEnum


def test_case_insensitive_str_enum_basic():
    """Test basic CaseInsensitiveStrEnum functionality."""

    class TestEnum(CaseInsensitiveStrEnum):
        """Test enum for case-insensitive string comparison."""

        OPTION_A = "option_a"
        OPTION_B = "option_b"

    # Test lowercase
    assert TestEnum("option_a") == TestEnum.OPTION_A
    assert TestEnum("option_b") == TestEnum.OPTION_B

    # Test uppercase
    assert TestEnum("OPTION_A") == TestEnum.OPTION_A
    assert TestEnum("OPTION_B") == TestEnum.OPTION_B

    # Test mixed case
    assert TestEnum("OpTiOn_A") == TestEnum.OPTION_A
    assert TestEnum("oPtIoN_b") == TestEnum.OPTION_B


def test_case_insensitive_str_enum_invalid():
    """Test that CaseInsensitiveStrEnum raises ValueError for invalid values."""

    class TestEnum(CaseInsensitiveStrEnum):
        """Test enum for invalid value handling."""

        OPTION_A = "option_a"
        OPTION_B = "option_b"

    with pytest.raises(ValueError, match="invalid_option is not a valid TestEnum"):
        TestEnum("invalid_option")


def test_case_insensitive_str_enum_in_noise_models():
    """Test that SupportedGate and SupportedErrorTypes use CaseInsensitiveStrEnum."""
    # Verify they inherit from CaseInsensitiveStrEnum
    assert issubclass(SupportedGate, CaseInsensitiveStrEnum)
    assert issubclass(SupportedErrorTypes, CaseInsensitiveStrEnum)

    # Verify case-insensitive behavior works
    assert SupportedGate("h") == SupportedGate.H
    assert SupportedGate("H") == SupportedGate.H
    assert SupportedGate("cx") == SupportedGate.CX
    assert SupportedGate("CX") == SupportedGate.CX

    assert SupportedErrorTypes("depolarizing_error") == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert SupportedErrorTypes("DEPOLARIZING_ERROR") == SupportedErrorTypes.DEPOLARIZING_ERROR
