"""Tests for plugin discovery via dir()."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import qdk_chemistry.plugins
import qdk_chemistry.plugins.pyscf
import qdk_chemistry.plugins.qiskit


def test_plugins_visible_in_dir():
    """Test that plugin modules are visible in dir() output.

    This test verifies the fix for the issue where dir(qdk_chemistry.plugins)
    did not list the pyscf and qiskit submodules.
    """
    dir_output = dir(qdk_chemistry.plugins)

    # Both plugins should be visible in dir() output
    assert "pyscf" in dir_output, "pyscf should be visible in dir(qdk_chemistry.plugins)"
    assert "qiskit" in dir_output, "qiskit should be visible in dir(qdk_chemistry.plugins)"


def test_plugins_accessible_as_attributes():
    """Test that plugin modules are accessible as attributes."""
    # Plugins should be accessible as attributes
    assert hasattr(qdk_chemistry.plugins, "pyscf"), "pyscf should be accessible as qdk_chemistry.plugins.pyscf"
    assert hasattr(qdk_chemistry.plugins, "qiskit"), "qiskit should be accessible as qdk_chemistry.plugins.qiskit"


def test_plugins_can_be_imported():
    """Test that plugin modules can be imported directly."""
    # Verify they are actually modules
    assert qdk_chemistry.plugins.pyscf is not None
    assert qdk_chemistry.plugins.qiskit is not None


def test_plugins_all_export():
    """Test that __all__ is properly defined."""
    # __all__ should be defined
    assert hasattr(qdk_chemistry.plugins, "__all__"), "__all__ should be defined"

    # __all__ should contain both plugins
    assert "pyscf" in qdk_chemistry.plugins.__all__, "pyscf should be in __all__"
    assert "qiskit" in qdk_chemistry.plugins.__all__, "qiskit should be in __all__"
