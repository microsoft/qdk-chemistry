"""Tests for Ansatz property bindings and immutability."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .test_helpers import create_test_ansatz


def test_ansatz_property_bindings():
    """Test that Ansatz getters work as both methods and properties."""
    ansatz = create_test_ansatz()

    # Test get_hamiltonian as both method and property
    h_method = ansatz.get_hamiltonian()
    h_property = ansatz.hamiltonian
    assert h_method is h_property, "Method and property should return the same object"

    # Test get_wavefunction as both method and property
    w_method = ansatz.get_wavefunction()
    w_property = ansatz.wavefunction
    assert w_method is w_property, "Method and property should return the same object"

    # Test get_orbitals as both method and property
    o_method = ansatz.get_orbitals()
    o_property = ansatz.orbitals
    assert o_method is o_property, "Method and property should return the same object"

    # Test get_summary as both method and property
    s_method = ansatz.get_summary()
    s_property = ansatz.summary
    assert s_method == s_property, "Method and property should return the same value"
    assert isinstance(s_property, str)


def test_ansatz_property_immutability():
    """Test that Ansatz properties are read-only."""
    ansatz = create_test_ansatz()

    # Test that properties are read-only
    try:
        ansatz.hamiltonian = None
        raise AssertionError("Should not be able to set hamiltonian property")
    except AttributeError:
        pass  # Expected - property is read-only

    try:
        ansatz.wavefunction = None
        raise AssertionError("Should not be able to set wavefunction property")
    except AttributeError:
        pass  # Expected - property is read-only

    try:
        ansatz.orbitals = None
        raise AssertionError("Should not be able to set orbitals property")
    except AttributeError:
        pass  # Expected - property is read-only


def test_property_bindings_backward_compatibility():
    """Test that old method-based API still works alongside new property API."""
    ansatz = create_test_ansatz()

    # Old API (method calls) should work
    h1 = ansatz.get_hamiltonian()
    w1 = ansatz.get_wavefunction()
    o1 = ansatz.get_orbitals()

    # New API (property access) should work
    h2 = ansatz.hamiltonian
    w2 = ansatz.wavefunction
    o2 = ansatz.orbitals

    # Both should return the same objects
    assert h1 is h2
    assert w1 is w2
    assert o1 is o2

    # Both APIs should coexist without issues
    assert ansatz.get_summary() == ansatz.summary
